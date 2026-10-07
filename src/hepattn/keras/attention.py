"""Keras/HGQ2 twin of hepattn.models.attention.Attention (torch-SDPA semantics).

The torch Attention packs Q/K/V into one (3*dim, dim) projection and calls a fused
scaled_dot_product_attention. This twin uses three separate factory-built projections
(so each can be quantized independently) and an explicit einsum -> masked softmax ->
einsum attention core (so the scores and attention weights are quantizable tensors).
In float mode this reproduces the torch backend numerically within fp32 tolerance;
the exact measured deviations are documented in docs/hgq/PARITY.md.

Only the dense-mask "torch" attention semantics are implemented: flash/flex/varlen
backends, attention biases and score mods are training-time performance features of
the torch model with no keras/HGQ2 counterpart.
"""

from torch import Tensor, nn

from hepattn.keras.factory import LayerFactory, apply_softmax
from hepattn.models.attention import merge_masks
from hepattn.models.norm import NORM_TYPES


class KerasAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        attn_type: str = "torch",
        torch_compile: bool = False,
        window_size: int | None = None,
        qkv_norm: bool = False,
        norm: str | None = None,
        value_residual: bool = False,
        is_first_layer: bool = False,
        factory: LayerFactory | None = None,
        name: str | None = None,
    ) -> None:
        """Multi-head attention from factory-built (quantizable) leaves.

        Accepts the same arguments as hepattn Attention so torch attn_kwargs from the
        YAML configs can be passed through; backend-specific options are rejected.

        Raises:
            ValueError: If a non-torch backend or a backend-specific option is requested.
        """
        super().__init__()
        factory = factory or LayerFactory()

        assert dim % num_heads == 0, "num_heads must divide dim."
        if attn_type != "torch":
            raise ValueError(f"KerasAttention implements torch-SDPA semantics only, got attn_type='{attn_type}'")
        if torch_compile or window_size is not None:
            raise ValueError("torch_compile / window_size are not supported by KerasAttention")
        if qkv_norm and not norm:
            raise ValueError("norm must be provided when qkv_norm is True")
        if norm is not None and norm not in NORM_TYPES:
            raise ValueError(f"Unsupported norm: {norm}. Must be one of {list(NORM_TYPES.keys())}")

        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_type = "torch"
        self.qkv_norm = qkv_norm
        self.value_residual = value_residual
        self.is_first_layer = is_first_layer

        def sub(part: str) -> str | None:
            return f"{name}_{part}" if name else None

        # float: eager build; quantized: lazy build on first forward (see KerasDense)
        self.q_proj = factory.dense(dim, use_bias=bias, name=sub("q_proj"))
        self.k_proj = factory.dense(dim, use_bias=bias, name=sub("k_proj"))
        self.v_proj = factory.dense(dim, use_bias=bias, name=sub("v_proj"))
        self.out_proj = factory.dense(dim, use_bias=bias, name=sub("out_proj"))
        if not factory.quantize:
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
                layer.build((None, dim))

        if self.value_residual and not self.is_first_layer:
            self.value_residual_mix = factory.dense(num_heads, activation="sigmoid", use_bias=True, name=sub("value_residual_mix"))
            if not factory.quantize:
                self.value_residual_mix.build((None, dim))

        if self.qkv_norm:
            assert norm is not None
            norm_cls = NORM_TYPES[norm]
            self.q_norm = norm_cls(dim)
            self.k_norm = norm_cls(dim)
            self.v_norm = norm_cls(dim)

        # explicit attention core: quantizable scores einsum, softmax, values einsum
        self.scores_einsum = factory.einsum("bhqd,bhkd->bhqk", name=sub("scores"))
        self.attn_softmax = factory.softmax(axis=-1, name=sub("softmax"))
        self.values_einsum = factory.einsum("bhqk,bhkd->bhqd", name=sub("values"))

    def set_backend(self, attn_type: str, **kwargs) -> str:
        """Duck-typed counterpart of Attention.set_backend; only 'torch' is expressible.

        Raises:
            ValueError: If a non-torch backend is requested.
        """
        if attn_type != "torch":
            raise ValueError(f"KerasAttention cannot switch to backend '{attn_type}'")
        return self.attn_type

    def separate_heads(self, x: Tensor) -> Tensor:
        return x.unflatten(-1, (self.num_heads, -1)).transpose(-3, -2)  # B S D -> B H S Dh

    def recombine_heads(self, x: Tensor) -> Tensor:
        return x.transpose(-3, -2).flatten(-2)  # B H S Dh -> B S D

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        q_mask: Tensor | None = None,
        kv_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
        attn_bias: Tensor | None = None,
        score_mod=None,
        initial_values: dict | None = None,
        **kwargs,
    ) -> Tensor:
        """Mirror of hepattn Attention.forward for the torch backend path.

        Masks follow the hepattn convention: True = participates in attention.

        Raises:
            NotImplementedError: If attn_bias or score_mod are requested.
        """
        if attn_bias is not None:
            raise NotImplementedError("attn_bias is not supported by KerasAttention")
        if score_mod is not None:
            raise NotImplementedError("score_mod is a flex-attention feature, not supported by KerasAttention")

        q_shape = q.shape
        if k is None and v is None:  # self-attention
            k = v = q
            kv_shape = q.shape
        else:  # cross-attention
            assert k is not None, "k must be provided for cross-attention"
            if v is None:
                v = k
            kv_shape = k.shape
            assert k.shape == v.shape, f"Shape mismatch: k.shape={k.shape} vs v.shape={v.shape}"

        # value-residual mix is computed from the raw (pre-projection) queries, as in torch
        mix = None
        if self.value_residual and not self.is_first_layer:
            mix = self.value_residual_mix(q, training=self.training)  # (B, S, H)
            mix = mix.unsqueeze(-1).transpose(-2, -3)  # -> (B, H, S, 1)

        qp = self.q_proj(q, training=self.training)
        kp = self.k_proj(k, training=self.training)
        vp = self.v_proj(v, training=self.training)

        if self.qkv_norm:
            qp = self.q_norm(qp)
            kp = self.k_norm(kp)
            vp = self.v_norm(vp)

        qp = self.separate_heads(qp)
        kp = self.separate_heads(kp)
        vp = self.separate_heads(vp)

        if self.value_residual and initial_values is not None:
            if self.is_first_layer:
                initial_values["v"] = vp
            else:
                assert mix is not None
                vp = vp * mix + initial_values["v"] * (1.0 - mix)

        mask = merge_masks(q_mask, kv_mask, attn_mask, q_shape, kv_shape, qp.device)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(-3)  # broadcast over heads

        # scale queries before the matmul, matching the order of operations in torch SDPA
        qp = qp * (self.head_dim**-0.5)
        scores = self.scores_einsum([qp, kp], training=self.training)
        attn = apply_softmax(self.attn_softmax, scores, attn_mask=mask, training=self.training)
        out = self.values_einsum([attn, vp], training=self.training)

        out = self.recombine_heads(out)
        return self.out_proj(out, training=self.training)


def build_keras_attention(dim: int, attn_type: str = "torch", *, factory: "LayerFactory | None" = None, name: str | None = None, **attn_kwargs):
    """Dispatch to the keras attention module for the requested backend.

    torch/flash/flash-varlen -> KerasAttention (dense-SDPA semantics);
    linformer -> KerasLinformerAttention (low-rank, projected-space mask, needs k >= kv_len);
    masked-linformer -> KerasMaskedLinformerAttention (low-rank, exact masks, any k).
    Backend-specific kwargs (linformer_seq_len/linformer_proj_dim) are consumed here.
    """
    if attn_type == "linformer":
        from hepattn.keras.linformer import KerasLinformerAttention  # noqa: PLC0415  (avoids an import cycle)

        seq_len = attn_kwargs.pop("linformer_seq_len", 256)
        k = attn_kwargs.pop("linformer_proj_dim", 256)
        return KerasLinformerAttention(dim, seq_len=seq_len, k=k, factory=factory, name=name, **attn_kwargs)
    if attn_type == "masked-linformer":
        from hepattn.keras.masked_linformer import KerasMaskedLinformerAttention  # noqa: PLC0415  (avoids an import cycle)

        # no defaults: k < seq_len is the point, and seq_len must cover the longest key axis
        seq_len = attn_kwargs.pop("linformer_seq_len")
        k = attn_kwargs.pop("linformer_proj_dim")
        return KerasMaskedLinformerAttention(dim, seq_len=seq_len, k=k, factory=factory, name=name, **attn_kwargs)
    return KerasAttention(dim, attn_type=attn_type, factory=factory, name=name, **attn_kwargs)
