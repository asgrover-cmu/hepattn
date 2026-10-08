"""A whole KerasMaskFormer with masked-linformer in the encoder AND the decoder.

test_masked_linformer.py exercises the attention block alone, which is how a model that
could not even be constructed (the decoder's parent class rejected the attention type)
got through CI. These tests build, run and backpropagate the full model.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from test_maskformer_parity import (  # ty: ignore [unresolved-import]
    DIM,
    MAX_NODES,
    NUM_QUERIES,
    clic_dummy_batch,
    make_input_nets,
    make_matcher,
    make_tasks,
)
from test_quantized import HIGH_QUANT  # ty: ignore [unresolved-import]

from hepattn.keras.masked_linformer import KerasMaskedLinformerAttention
from hepattn.keras.maskformer import KerasMaskFormer

K = 16  # well below both key axes (168 in the encoder, 160 in the decoder)
REGISTERS = 8


def make_model(quant: dict | None, seed: int = 70) -> KerasMaskFormer:
    torch.manual_seed(seed)
    encoder_attn = {"num_heads": 16, "linformer_seq_len": MAX_NODES + REGISTERS, "linformer_proj_dim": K}
    decoder_attn = {"num_heads": 16, "attn_type": "masked-linformer", "linformer_seq_len": MAX_NODES, "linformer_proj_dim": K}
    return KerasMaskFormer(
        input_nets=make_input_nets(),
        encoder={
            "num_layers": 2,
            "attn_type": "masked-linformer",
            "hybrid_norm": True,
            "value_residual": True,
            "num_register_tokens": REGISTERS,
            "attn_kwargs": encoder_attn,
        },
        decoder={
            "num_decoder_layers": 2,
            "num_queries": NUM_QUERIES,
            "mask_attention": True,
            "use_query_masks": False,
            "decoder_layer_config": {"dim": DIM, "hybrid_norm": True, "attn_kwargs": decoder_attn},
        },
        tasks=make_tasks(),
        dim=DIM,
        matcher=make_matcher(),
        quant=quant,
    )


@pytest.mark.parametrize("quant", [None, {**HIGH_QUANT, "ebops": {"beta0": 0.0}}], ids=["float", "quantized"])
def test_full_model_builds_runs_and_backpropagates(quant):
    model = make_model(quant)
    inputs, targets = clic_dummy_batch()

    attns = [m for m in model.modules() if isinstance(m, KerasMaskedLinformerAttention)]
    assert len(attns) == 2 + 2 * 3, "expected masked-linformer in 2 encoder layers and in q_ca, q_sa, kv_ca of 2 decoder layers"

    # materialize lazy (quantized) layers, then publish keras weights as MPflowHGQ.setup does;
    # each attention leaves the other path's leaves unbuilt, which neither call may trip over
    model.eval()
    with torch.no_grad():
        model(inputs)
    model.register_keras_parameters()
    weights, quantizers = model.trainable_parameter_groups()
    assert weights, "no trainable weights collected"
    assert bool(quantizers) == (quant is not None), "quantizer group should be non-empty exactly when quantizing"

    model.train()
    outputs = model(inputs)
    _, _, loss_dict = model.loss(outputs, dict(targets))
    total = sum(v for layer in loss_dict.values() for task in layer.values() for v in task.values() if torch.isfinite(v))
    total = total + model.quant_losses()
    assert torch.isfinite(total), "non-finite training loss"
    total.backward()

    for attn in attns:
        grads = [g for g in (attn.proj_k.grad, attn.proj_v.grad) if g is not None]
        assert grads, "a sequence projection received no gradient"
        assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient in a sequence projection"
