import pytest
import torch
from masked_linformer import LinformerSelfAttention

from hepattn.models.attention import Attention
from hepattn.models.decoder import MaskFormerDecoderLayer
from hepattn.models.encoder import Encoder

DIM, HEADS, SEQ, K = 32, 4, 24, 8


def linformer_attention(**kwargs):
    torch.manual_seed(0)
    return Attention(DIM, num_heads=HEADS, attn_type="linformer", linformer_seq_len=SEQ, linformer_k=K, **kwargs).double()


def padded_batch(lengths, n=SEQ):
    torch.manual_seed(1)
    x = torch.randn(len(lengths), n, DIM, dtype=torch.float64)
    mask = torch.arange(n)[None] < torch.tensor(lengths)[:, None]
    return x, mask


def test_matches_masked_linformer():
    attn = linformer_attention(bias=False)
    ref = LinformerSelfAttention(DIM, SEQ, k=K, heads=HEADS).double()
    w_q, w_k, w_v = attn.in_proj_weight.data.chunk(3)
    ref.to_q.weight.data.copy_(w_q)
    ref.to_k.weight.data.copy_(w_k)
    ref.to_v.weight.data.copy_(w_v)
    ref.proj_k.data.copy_(attn.proj_k.data)
    ref.proj_v.data.copy_(attn.proj_v.data)
    ref.to_out.weight.data.copy_(attn.out_proj.weight.data)
    ref.to_out.bias.data.zero_()

    x, mask = padded_batch([7, 24, 13])
    out = attn(x, kv_mask=mask)
    torch.testing.assert_close(out[mask], ref(x, mask=mask)[mask])
    # control: without the mask the padded rows would be projected in
    assert (out[mask] - attn(x)[mask]).abs().max() > 1e-3


def test_padded_keys_do_not_leak():
    attn = linformer_attention()
    x, mask = padded_batch([7, 24, 13])
    out = attn(x, kv_mask=mask)
    for fill in (1e3, float("nan")):
        garbage = torch.where(mask[..., None], x, fill)
        assert torch.equal(attn(garbage, kv_mask=mask)[mask], out[mask])
    x2 = x.clone()
    x2[0, 3] += 1.0
    assert (attn(x2, kv_mask=mask)[mask] - out[mask]).abs().max() > 1e-3


@pytest.mark.parametrize(("built", "switched"), [("linformer", "torch"), ("torch", "linformer")])
def test_backend_switch_refused(built, switched):
    attn = Attention(DIM, num_heads=HEADS, attn_type=built, linformer_seq_len=SEQ, linformer_k=K)
    with pytest.raises(ValueError, match="linformer"):
        attn.set_backend(switched)


def test_sort_puts_padding_last():
    torch.manual_seed(0)
    enc = Encoder(num_layers=2, dim=DIM, attn_type="linformer", attn_kwargs={"num_heads": HEADS, "linformer_seq_len": SEQ, "linformer_k": K})
    enc = enc.double().eval()
    x, _ = padded_batch([SEQ])
    phi = torch.rand(1, SEQ) * 2 - 1
    # padding interleaved, with a sort value that falls in the middle of the real ones
    mask = torch.rand(1, SEQ) < 0.6
    phi_padded = phi.masked_fill(~mask, 0.0)
    out = enc(x, x_sort_value=phi_padded, kv_mask=mask)

    # the same real tokens, compacted to the front
    order = torch.argsort((~mask).int(), dim=-1, stable=True)
    x_c, phi_c, mask_c = x[:, order[0]], phi[:, order[0]], mask[:, order[0]]
    out_c = enc(x_c, x_sort_value=phi_c.masked_fill(~mask_c, 5.0), kv_mask=mask_c)
    torch.testing.assert_close(out[mask], out_c[mask_c])


def test_mask_attention_matches_masked_linformer():
    attn = linformer_attention(bias=False)
    ref = LinformerSelfAttention(DIM, SEQ, k=K, heads=HEADS).double()
    w_q, w_k, w_v = attn.in_proj_weight.data.chunk(3)
    for lin, w in ((ref.to_q, w_q), (ref.to_k, w_k), (ref.to_v, w_v)):
        lin.weight.data.copy_(w)
    ref.proj_k.data.copy_(attn.proj_k.data)
    ref.proj_v.data.copy_(attn.proj_v.data)
    ref.to_out.weight.data.copy_(attn.out_proj.weight.data)
    ref.to_out.bias.data.zero_()

    kv, kv_mask = padded_batch([7, 24, 13])
    q = torch.randn(3, 5, DIM, dtype=torch.float64)
    attn_mask = torch.rand(3, 5, SEQ) < 0.4
    out = attn(q, k=kv, v=kv, kv_mask=kv_mask, attn_mask=attn_mask)
    torch.testing.assert_close(out, ref(q, context=kv, context_mask=kv_mask, attn_mask=attn_mask))
    # control: the per-query mask matters
    assert (out - attn(q, k=kv, v=kv, kv_mask=kv_mask)).abs().max() > 1e-3


@pytest.mark.parametrize("masked", [False, True])
def test_kv_sort_idx_equals_presorted_keys(masked):
    attn = linformer_attention()
    kv, kv_mask = padded_batch([7, 24, 13])
    q = torch.randn(3, 5, DIM, dtype=torch.float64)
    attn_mask = torch.rand(3, 5, SEQ) < 0.4 if masked else None
    idx = torch.stack([torch.randperm(SEQ) for _ in range(3)])
    out = attn(q, k=kv, v=kv, kv_mask=kv_mask, attn_mask=attn_mask, kv_sort_idx=idx)

    kv_s = kv.gather(1, idx[..., None].expand_as(kv))
    am_s = None if attn_mask is None else attn_mask.gather(-1, idx[:, None].expand_as(attn_mask))
    torch.testing.assert_close(out, attn(q, k=kv_s, v=kv_s, kv_mask=kv_mask.gather(-1, idx), attn_mask=am_s))
    assert (out - attn(q, k=kv, v=kv, kv_mask=kv_mask, attn_mask=attn_mask)).abs().max() > 1e-3


def test_decoder_layer_all_linformer_trains():
    torch.manual_seed(0)
    attn_kwargs = {"attn_type": "linformer", "num_heads": HEADS, "linformer_seq_len": SEQ, "linformer_k": K}
    layer = MaskFormerDecoderLayer(dim=DIM, attn_kwargs=attn_kwargs, depth=1, hybrid_norm=True)
    q = torch.randn(2, 10, DIM)
    kv, kv_mask = padded_batch([20, 15])
    kv = kv.float()
    attn_mask = torch.rand(2, 10, SEQ) < 0.3
    idx = torch.stack([torch.randperm(SEQ) for _ in range(2)])
    q_out, kv_out = layer(q, kv, attn_mask=attn_mask, kv_mask=kv_mask, kv_sort_idx=idx)
    (q_out.square().sum() + kv_out[kv_mask].square().sum()).backward()
    assert q_out.isfinite().all()
    assert kv_out[kv_mask].isfinite().all()
    no_grad = [n for n, p in layer.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not no_grad, no_grad
    assert sum(n.endswith("proj_k") for n, _ in layer.named_parameters()) == 3
