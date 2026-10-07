"""KerasMaskedLinformerAttention: masks stay exact with a projection rank k < sequence length.

Two groups. The parity tests compare against the reference `masked_linformer` package and
skip when it is not installed
(pip install git+https://github.com/compressed-maskformer-reco/masked-linformer). The
property tests need nothing extra: padding must not leak, a query allowed nothing must not
depend on the keys, and the key order must be the only thing kv_sort_idx changes.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from parity_utils import assert_parity, make_padded_batch  # ty: ignore [unresolved-import]

from hepattn.keras.attention import build_keras_attention
from hepattn.keras.factory import LayerFactory
from hepattn.keras.masked_linformer import KerasMaskedLinformerAttention

DIM = 64
HEADS = 8
N = 168  # encoder sequence: 160 nodes + 8 register tokens
NQ = 150
K = 32  # real compression: K < N, which KerasLinformerAttention cannot mask

QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 12, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 6, "f0": 10},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
}


def build(seed: int = 0, n: int = N) -> KerasMaskedLinformerAttention:
    torch.manual_seed(seed)
    return KerasMaskedLinformerAttention(DIM, num_heads=HEADS, seq_len=n, k=K).eval()


def reference(klin, q, kv, kv_mask=None, attn_mask=None):
    """The same numbers from the reference package, using the keras module's own weights."""
    ml = pytest.importorskip("masked_linformer", reason="reference masked-linformer package not installed")
    with torch.no_grad():
        qh, kh, vh = (klin._heads(t) for t in (klin.to_q(q), klin.to_k(kv), klin.to_v(kv)))  # noqa: SLF001
        if attn_mask is None:
            out = F.scaled_dot_product_attention(qh, ml.project(kh, klin.proj_k, kv_mask), ml.project(vh, klin.proj_v, kv_mask))
        else:
            if kv_mask is not None:
                attn_mask = attn_mask & kv_mask[:, None]
            out = ml.attend(qh, kh, vh, klin.proj_k, klin.proj_v, attn_mask)
        return klin.to_out(out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1))


def test_self_attention_parity_padded():
    klin = build(1)
    x, mask = make_padded_batch(2, N, DIM, seed=2)
    with torch.no_grad():
        out = klin(x, kv_mask=mask)
    assert_parity("masked_linformer.self", f"dim{DIM}h{HEADS}k{K}", reference(klin, x, x, kv_mask=mask), out, atol=2e-5, rtol=2e-4)


def test_self_attention_parity_unmasked():
    klin = build(3)
    x, _ = make_padded_batch(2, NQ, DIM, seed=4)  # query self-attention: shorter than seq_len, no mask
    with torch.no_grad():
        out = klin(x)
    assert_parity("masked_linformer.self_nomask", f"dim{DIM}h{HEADS}k{K}", reference(klin, x, x), out, atol=2e-5, rtol=2e-4)


def test_cross_attention_parity_per_query_mask():
    klin = build(5, n=160)
    q, _ = make_padded_batch(2, NQ, DIM, seed=6)
    kv, kv_mask = make_padded_batch(2, 160, DIM, seed=7)
    attn_mask = torch.rand(2, NQ, 160, generator=torch.Generator().manual_seed(8)) > 0.4
    with torch.no_grad():
        out = klin(q, k=kv, v=kv, attn_mask=attn_mask, kv_mask=kv_mask)
    ref = reference(klin, q, kv, kv_mask=kv_mask, attn_mask=attn_mask)
    assert_parity("masked_linformer.cross_masked", f"dim{DIM}h{HEADS}k{K}", ref, out, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("per_query", [False, True])
def test_padded_keys_do_not_leak(per_query):
    """Padded key rows -- even NaN -- must not change any output, on either path."""
    klin = build(9, n=160)
    q, _ = make_padded_batch(2, NQ, DIM, seed=10)
    kv, kv_mask = make_padded_batch(2, 160, DIM, seed=11)
    assert not kv_mask.all(), "fixture must contain padding"
    attn_mask = torch.ones(2, NQ, 160, dtype=torch.bool) if per_query else None
    poisoned = kv.clone()
    poisoned[~kv_mask] = float("nan")
    with torch.no_grad():
        clean = klin(q, k=kv, v=kv, attn_mask=attn_mask, kv_mask=kv_mask)
        dirty = klin(q, k=poisoned, v=poisoned, attn_mask=attn_mask, kv_mask=kv_mask)
        no_mask = klin(q, k=kv, v=kv, attn_mask=attn_mask)
    assert torch.isfinite(dirty).all(), "NaN in a padded key reached the output"
    assert torch.allclose(clean, dirty, atol=1e-6), "padded key content changed the output"
    assert not torch.allclose(clean, no_mask, atol=1e-4), "kv_mask had no effect (fixture sanity)"


def test_query_allowed_nothing_is_key_independent():
    """An all-False attn_mask row gives to_out's bias only, whatever the keys hold."""
    klin = build(12, n=160)
    q, _ = make_padded_batch(2, NQ, DIM, seed=13)
    kv1, _ = make_padded_batch(2, 160, DIM, seed=14)
    kv2, _ = make_padded_batch(2, 160, DIM, seed=15)
    attn_mask = torch.ones(2, NQ, 160, dtype=torch.bool)
    attn_mask[:, 0] = False
    with torch.no_grad():
        out1 = klin(q, k=kv1, v=kv1, attn_mask=attn_mask)
        out2 = klin(q, k=kv2, v=kv2, attn_mask=attn_mask)
    assert torch.isfinite(out1).all()
    assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-6), "a query allowed nothing depends on the keys"
    assert not torch.allclose(out1[:, 1], out2[:, 1], atol=1e-4), "an unmasked query should depend on the keys (fixture sanity)"


@pytest.mark.parametrize("per_query", [False, True])
def test_kv_sort_idx_equals_presorted_keys(per_query):
    klin = build(16, n=160)
    q, _ = make_padded_batch(2, NQ, DIM, seed=17)
    kv, kv_mask = make_padded_batch(2, 160, DIM, seed=18)
    gen = torch.Generator().manual_seed(19)
    attn_mask = torch.rand(2, NQ, 160, generator=gen) > 0.4 if per_query else None
    idx = torch.stack([torch.randperm(160, generator=gen) for _ in range(2)])
    kv_sorted = kv.gather(1, idx[..., None].expand_as(kv))
    mask_sorted = kv_mask.gather(-1, idx)
    attn_sorted = None if attn_mask is None else attn_mask.gather(-1, idx[:, None].expand_as(attn_mask))
    with torch.no_grad():
        by_idx = klin(q, k=kv, v=kv, attn_mask=attn_mask, kv_mask=kv_mask, kv_sort_idx=idx)
        by_hand = klin(q, k=kv_sorted, v=kv_sorted, attn_mask=attn_sorted, kv_mask=mask_sorted)
        unsorted = klin(q, k=kv, v=kv, attn_mask=attn_mask, kv_mask=kv_mask)
    assert torch.allclose(by_idx, by_hand, atol=1e-6)
    assert not torch.allclose(by_idx, unsorted, atol=1e-4), "key order had no effect (fixture sanity)"


def test_factory_dispatch_and_required_kwargs():
    attn = build_keras_attention(DIM, attn_type="masked-linformer", num_heads=HEADS, linformer_seq_len=N, linformer_proj_dim=K)
    assert isinstance(attn, KerasMaskedLinformerAttention)
    assert attn.proj_k.shape == (N, K)
    with pytest.raises(KeyError):
        build_keras_attention(DIM, attn_type="masked-linformer", num_heads=HEADS)


def test_quantized_both_paths_train():
    """HGQ2 leaves: both paths run in training mode with k < n and every built weight gets a finite gradient.

    One module per path, as in the decoder (q_sa never sees an attn_mask; q_ca always does),
    so each module leaves the other path's leaves unbuilt -- that must not break a backward.
    """
    factory = LayerFactory(QUANT)
    torch.manual_seed(20)
    with factory.scopes():
        shared = KerasMaskedLinformerAttention(DIM, num_heads=HEADS, seq_len=N, k=K, factory=factory, name="t_shared").train()
        per_query = KerasMaskedLinformerAttention(DIM, num_heads=HEADS, seq_len=160, k=K, factory=factory, name="t_perq").train()
    x, x_mask = make_padded_batch(2, N, DIM, seed=21)
    q, _ = make_padded_batch(2, NQ, DIM, seed=22)
    kv, kv_mask = make_padded_batch(2, 160, DIM, seed=23)
    attn_mask = torch.rand(2, NQ, 160, generator=torch.Generator().manual_seed(24)) > 0.4

    out_a = shared(x, kv_mask=x_mask)
    out_b = per_query(q, k=kv, v=kv, attn_mask=attn_mask, kv_mask=kv_mask)
    assert out_a.shape == x.shape and out_b.shape == q.shape
    assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all()
    (out_a.square().mean() + out_b.square().mean()).backward()
    for module in (shared, per_query):
        for name in ("proj_k", "proj_v"):
            grad = getattr(module, name).grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, f"{name} got no gradient"
        for leaf in (module.to_q, module.to_k, module.to_v, module.to_out):
            grads = [w.value.grad for w in leaf.trainable_weights if w.value.grad is not None]
            assert grads, f"{leaf.name}: no trainable weight received a gradient"
            assert all(torch.isfinite(g).all() for g in grads), f"{leaf.name}: non-finite gradient"
    assert not shared.m_scores_einsum.built and not per_query.scores_einsum.built, "each module builds only its own path"
