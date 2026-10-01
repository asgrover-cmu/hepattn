"""quant_scope: per-component quantization for encoder/decoder precision ablations.

The switch must (1) quantize exactly the leaves a preset names and nothing else,
(2) leave the float twins numerically identical to the all-float model, and (3) keep a
mixed float/HGQ2 model trainable with every weight reaching the optimizer.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

from hgq.layers.core.base import QLayerBase
from integration_utils import materialize  # ty: ignore [unresolved-import]
from test_maskformer_parity import (  # ty: ignore [unresolved-import]
    DECODER_CFG,
    ENCODER_CFG,
    clic_dummy_batch,
    flatten_outputs,
    make_input_nets,
    make_matcher,
    make_tasks,
)

from hepattn.keras.maskformer import KerasMaskFormer, resolve_quant_scope
from hepattn.keras.porting import port_keras_to_keras

QUANT = {"weight": {"default_q_type": "kbi", "b0": 8, "i0": 2}, "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8}, "ebops": {"beta0": 1e-12}}
TASK_NAMES = ["classification", "mask", "incidence", "regression"]


def build(quant=QUANT, scope=None, seed=0):
    torch.manual_seed(seed)
    model = KerasMaskFormer(
        input_nets=make_input_nets(),
        encoder=ENCODER_CFG,
        decoder=DECODER_CFG,
        tasks=make_tasks(),
        dim=32,
        matcher=make_matcher(),
        quant=quant,
        quant_scope=scope,
    )
    return materialize(model)


def leaves(model):
    """(name, is_quantized) for every factory-built leaf."""
    return {layer.name: isinstance(layer, QLayerBase) for layer in model.keras_layers()}


@pytest.fixture(scope="module")
def all_leaves():
    return leaves(build(scope="all"))


def quantized(scope):
    return {n for n, q in leaves(build(scope=scope)).items() if q}


def test_encoder_and_decoder_presets_partition_the_model(all_leaves):
    assert all(all_leaves.values()), "scope='all' must quantize every leaf"
    enc, dec = quantized("encoder"), quantized("decoder")
    assert enc and dec
    assert not enc & dec
    assert enc | dec == set(all_leaves)
    assert all(n.startswith(("innet", "encoder_l")) for n in enc)
    assert quantized("none") == set()


def test_decoder_stages_are_cumulative_and_isolated():
    groups = {
        "A": ("_ffn_", "_out_proj", "_to_out"),
        "B": ("_q_proj", "_k_proj", "_v_proj", "_to_q", "_to_k", "_to_v", "_seqproj_"),
        "C": ("_scores", "_values"),
        "D": ("_softmax",),
    }
    prev: set[str] = set()
    for stage in "ABCDEF":
        cur = quantized(f"dec_{stage}")
        new = cur - prev
        assert prev <= cur, f"dec_{stage} must include every earlier stage"
        assert new, f"dec_{stage} adds nothing"
        assert not any(n.startswith(("innet", "encoder_l")) for n in cur), "decoder stages must keep the encoder float"
        if stage in groups:
            assert all(n.startswith("decoder_l") and any(g in n for g in groups[stage]) for n in new), (stage, sorted(new))
        elif stage == "E":
            assert all(n.startswith("task1_") for n in new), sorted(new)  # task1 is the "mask" head
        prev = cur
    assert prev == quantized("decoder")


def test_resolve_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown quant_scope"):
        resolve_quant_scope("dec_Z", TASK_NAMES)


def test_empty_scope_reproduces_the_float_model():
    """Quant set but nothing in scope: the graph must be the float model, bit for bit up to fp32."""
    ref = build(quant=None).eval()
    model = build(scope="none").eval()
    port_keras_to_keras(ref, model)
    inputs, _ = clic_dummy_batch(2)
    with torch.no_grad():
        out_ref, out = flatten_outputs(ref(inputs)), flatten_outputs(model(inputs))
    assert set(out_ref) == set(out)
    for key, t in out_ref.items():
        if t.dtype == torch.bool:
            assert torch.equal(t, out[key]), key
        else:
            finite = torch.isfinite(t)
            assert torch.equal(finite, torch.isfinite(out[key])), key
            torch.testing.assert_close(out[key][finite], t[finite], atol=1e-5, rtol=1e-5, msg=key)


@pytest.mark.parametrize("scope", ["encoder", "dec_C"])
def test_mixed_model_trains_and_optimizer_sees_every_weight(scope):
    model = build(scope=scope)
    model.register_keras_parameters()
    decay, quant = model.trainable_parameter_groups()
    assert decay and quant, "a mixed model has both float/network and quantizer parameters"
    inputs, targets = clic_dummy_batch(2)
    outputs = model(inputs)
    _, _, losses = model.loss(outputs, targets)
    # dummy-data mask_bce is +inf for the all-float model too (finfo.min logits on padded
    # nodes); it is precision-independent, so sum only the finite terms
    terms = [v for layer in losses.values() for task in layer.values() for v in task.values()]
    total = sum(v for v in terms if torch.isfinite(v)) + model.quant_losses()
    assert torch.isfinite(total)
    total.backward()
    kernels = [p for p in decay if p.dim() == 2]
    assert kernels and all(p.grad is not None for p in kernels), "every Dense kernel (float or HGQ2) must get a gradient"
