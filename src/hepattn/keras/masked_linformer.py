"""Keras/HGQ2 port of masked-linformer attention (compressed-maskformer-reco/masked-linformer).

KerasLinformerAttention applies the attention mask in the PROJECTED space, which is only
meaningful for k >= kv_len (no compression) and ignores key padding altogether. This module
masks where the original positions still exist, so the projection rank k may be smaller than
the sequence:

* no attn_mask (encoder self-attention, decoder query self-attention): padded key/value rows
  are zeroed BEFORE the sequence projection, and each projection column j is rescaled by
  ||E[:, j]|| / ||E[valid, j]|| so the projected norm does not shrink with the number of
  valid constituents. Attention then runs over the k projected keys.
* attn_mask (the decoder's mask attention): each query i attends to its own projections
  E^T diag(M_i) K and F^T diag(M_i) V. They are never built: with S = q k^T the scores are
  (S * M_i) E and the output is ((a F^T) * M_i) v. The mask is exact, but this path costs
  as much as ordinary attention -- only the softmax is k wide instead of kv_len wide.

Quantizable leaves are factory-built (QDense / QEinsum / QSoftmax). The two sequence
projections, the mask selects and the column rescale stay float torch ops: the same kind of
conversion boundary as the norms. The two paths use SEPARATE einsum/softmax leaves because
HGQ2 sizes a leaf's bitwidths from the static shape of its first call, and the two paths
contract tensors of different shapes. A module only builds the leaves of the path it runs.

The projections index sequence SLOTS, so constituents must arrive in a consistent order
with padding last (data option ``sort_nodes_by``, or ``kv_sort_idx`` at call time).
"""

import math

import torch
from torch import Tensor, nn

from hepattn.keras.factory import LayerFactory, apply_softmax


def column_rescale(proj: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """||E[:, j]|| / ||E[valid, j]|| for a bool mask (..., n) -> (..., k).

    The numerator runs over all seq_len rows so the result does not depend on how far a
    batch was padded. The clamp sits before the sqrt: sqrt'(0) is inf, and inf * 0 would
    poison the gradient of an all-False row.
    """
    p = proj.to(torch.promote_types(proj.dtype, torch.float32))
    full = p.pow(2).sum(0).sqrt()
    kept = (mask.to(p.dtype) @ p[: mask.shape[-1]].pow(2)).clamp_min(eps**2).sqrt()
    return full / kept


class KerasMaskedLinformerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        seq_len: int = 168,
        k: int = 64,
        bias: bool = True,
        attn_type: str = "masked-linformer",
        renorm: bool = True,
        eps: float = 1e-8,
        factory: LayerFactory | None = None,
        name: str | None = None,
        **_ignored,
    ) -> None:
        """Args mirror KerasLinformerAttention; ``k`` may be smaller than the key sequence.

        Extra Attention kwargs (qkv_norm, value_residual, ...) are accepted and ignored, as
        in KerasLinformerAttention.

        Raises:
            ValueError: If attn_type is not 'masked-linformer'.
        """
        super().__init__()
        factory = factory or LayerFactory()
        if attn_type != "masked-linformer":
            raise ValueError(f"KerasMaskedLinformerAttention only supports attn_type='masked-linformer', got '{attn_type}'")
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.k = k
        self.renorm = renorm
        self.eps = eps
        self.attn_type = "masked-linformer"

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        self.to_q = factory.dense(dim, use_bias=False, name=sub("to_q"))
        self.to_k = factory.dense(dim, use_bias=False, name=sub("to_k"))
        self.to_v = factory.dense(dim, use_bias=False, name=sub("to_v"))
        self.to_out = factory.dense(dim, use_bias=True, name=sub("to_out"))
        for layer in (self.to_q, self.to_k, self.to_v, self.to_out):
            if not factory.quantize:
                layer.build((None, dim))

        std = 1.0 / math.sqrt(k)
        self.proj_k = nn.Parameter(torch.empty(seq_len, k).uniform_(-std, std))
        self.proj_v = nn.Parameter(torch.empty(seq_len, k).uniform_(-std, std))

        # shared-projection path (no attn_mask): attention over the k projected keys
        self.seq_proj_k = factory.einsum("bnd,nk->bkd", name=sub("seqproj_k"))
        self.seq_proj_v = factory.einsum("bnd,nk->bkd", name=sub("seqproj_v"))
        self.scores_einsum = factory.einsum("bhnd,bhkd->bhnk", name=sub("scores"))
        self.attn_softmax = factory.softmax(axis=-1, name=sub("softmax"))
        self.values_einsum = factory.einsum("bhnk,bhkd->bhnd", name=sub("values"))

        # per-query-mask path: full scores, projected to k for the softmax, unprojected for the values
        self.m_scores_einsum = factory.einsum("bhnd,bhmd->bhnm", name=sub("mscores"))
        self.m_score_proj = factory.einsum("bhnm,mk->bhnk", name=sub("mscoreproj"))
        self.m_attn_softmax = factory.softmax(axis=-1, name=sub("msoftmax"))
        self.m_attn_unproj = factory.einsum("bhnk,mk->bhnm", name=sub("munproj"))
        self.m_values_einsum = factory.einsum("bhnm,bhmd->bhnd", name=sub("mvalues"))

    def set_backend(self, attn_type: str, **kwargs) -> str:
        if attn_type != "masked-linformer":
            raise ValueError(f"KerasMaskedLinformerAttention cannot switch to backend '{attn_type}'")
        return self.attn_type

    def _heads(self, t: Tensor) -> Tensor:
        return t.reshape(t.shape[0], t.shape[1], self.num_heads, -1).transpose(1, 2)  # (b, s, d) -> (b, h, s, dh)

    def _project(self, t: Tensor, leaf, proj: Tensor, kv_mask: Tensor | None) -> Tensor:
        """(b, n, d) -> (b, k, d) along the sequence axis; padded rows contribute nothing."""
        n = t.shape[1]
        if kv_mask is None:
            return leaf([t, proj[:n]], training=self.training)
        # where, not a multiply: a NaN in a padded row would survive 0 * nan
        out = leaf([torch.where(kv_mask[..., None], t, 0.0), proj[:n]], training=self.training)
        if self.renorm:
            out = out * column_rescale(proj, kv_mask, self.eps).to(out.dtype)[..., None]
        return out

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        attn_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        kv_sort_idx: Tensor | None = None,
        **_ignored,
    ) -> Tensor:
        """Masked low-rank attention. Masks follow the hepattn convention: True = participates.

        Args:
            q: Queries (b, nq, d).
            k: Keys (b, n, d); None for self-attention.
            v: Values (b, n, d); None for self-attention.
            attn_mask: Optional per-query mask (b, nq, n).
            kv_mask: Optional key/value validity mask (b, n).
            kv_sort_idx: Optional (b, n) order in which the keys enter the sequence projection.
        """
        if k is None and v is None:  # self-attention
            k = v = q
        assert k is not None and v is not None
        n = k.shape[1]
        assert n <= self.seq_len, f"kv_len {n} exceeds seq_len {self.seq_len}"

        queries = self._heads(self.to_q(q, training=self.training))
        keys = self.to_k(k, training=self.training)
        values = self.to_v(v, training=self.training)

        # the projections index key slots, so keys enter in kv_sort_idx order; queries keep theirs
        if kv_sort_idx is not None:
            idx = kv_sort_idx[..., None].expand_as(keys)
            keys, values = keys.gather(1, idx), values.gather(1, idx)
            kv_mask = None if kv_mask is None else kv_mask.gather(-1, kv_sort_idx)
            attn_mask = None if attn_mask is None else attn_mask.gather(-1, kv_sort_idx[:, None].expand_as(attn_mask))

        scale = self.head_dim**-0.5
        if attn_mask is None:
            keys = self._heads(self._project(keys, self.seq_proj_k, self.proj_k, kv_mask))
            values = self._heads(self._project(values, self.seq_proj_v, self.proj_v, kv_mask))
            scores = self.scores_einsum([queries, keys], training=self.training) * scale
            attn = apply_softmax(self.attn_softmax, scores, attn_mask=None, training=self.training)
            out = self.values_einsum([attn, values], training=self.training)
        else:
            mask = attn_mask.to(torch.bool)
            if kv_mask is not None:
                mask = mask & kv_mask[:, None]
            m = mask[:, None]  # broadcast over heads
            # keys no query may see can hold NaN (padding); zero them before any matmul
            seen = mask.any(-2)[..., None]
            keys = self._heads(torch.where(seen, keys, 0.0))
            values = self._heads(torch.where(seen, values, 0.0))
            scores = self.m_scores_einsum([queries, keys], training=self.training)
            scores = torch.where(m, scores, 0.0) * scale
            scores = self.m_score_proj([scores, self.proj_k[:n]], training=self.training)
            if self.renorm:
                scores = scores * column_rescale(self.proj_k, mask, self.eps).to(scores.dtype)[:, None]
            attn = apply_softmax(self.m_attn_softmax, scores, attn_mask=None, training=self.training)
            if self.renorm:
                attn = attn * column_rescale(self.proj_v, mask, self.eps).to(attn.dtype)[:, None]
            attn = self.m_attn_unproj([attn, self.proj_v[:n]], training=self.training)
            # a query with an all-False row gets a zero value contribution (to_out's bias only)
            out = self.m_values_einsum([torch.where(m, attn, 0.0), values], training=self.training)

        out = out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)
        return self.to_out(out, training=self.training)
