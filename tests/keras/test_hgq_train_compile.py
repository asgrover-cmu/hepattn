"""Compiled HGQ2 training path (MPflowHGQ hgq_train_compile / HGQ2 set_train_compile).

HGQ2 owns the quantization code; hepattn only switches on HGQ2's own `set_train_compile()`,
which compiles HGQ2's pure training-mode quantizer functions while HGQ2 still assigns the
quantizer state eagerly, outside any compiled graph. The compile tests need the HGQ2 build
pinned in pyproject.toml (they skip on a stock HGQ2) and an Inductor-capable C++ compiler
(Polaris login nodes: CC=gcc-14 CXX=g++-14).

* test_compiled_train_fns_do_not_mutate_state -- the WRAP integer-bit variable is not
  written inside the compiled functions (its tensor version counter is unchanged across
  the compiled call) and is written by HGQ2's eager `assign` afterwards. Writing a variable
  inside a compiled graph that also reads it back is what breaks Inductor's backward
  ("modified by an inplace operation").
* test_set_train_compile_matches_eager -- two training steps of a small fully quantized
  MaskFormer compiled by Inductor equal the same HGQ2 run eagerly (loss, every gradient,
  every keras variable).
* test_option_* -- the hepattn option: refuses to run under TORCHDYNAMO_DISABLE, names the
  missing HGQ2 build, and enables HGQ2's compile with hepattn's loss functions kept eager.

The compile tests assert that Inductor actually generated code: tests/keras/conftest.py
forces the `force_eager` stance, which would otherwise turn every compile into eager silently.
"""

import pytest
import torch

pytest.importorskip("hepattn.keras", reason="hgq dependency group not installed")

import hgq.quantizer.internal.fixed_point_quantizer as fpq
from hgq.config import QuantizerConfig
from hgq.quantizer import Quantizer
from hgq.quantizer.internal.fixed_point_quantizer import FixedPointQuantizerBase
from integration_utils import materialize  # ty: ignore [unresolved-import]
from test_maskformer_parity import (  # ty: ignore [unresolved-import]
    DECODER_CFG,
    ENCODER_CFG,
    clic_dummy_batch,
    make_input_nets,
    make_matcher,
    make_tasks,
)

from hepattn.experiments.clic import lightning_module_hgq
from hepattn.keras.maskformer import KerasMaskFormer

PATCHED = hasattr(fpq, "set_train_compile")

QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1e-12},
}


def _inductor_builds():
    try:
        torch.compile(lambda t: t * 2 + 1, backend="inductor")(torch.ones(4))
    except Exception:  # noqa: BLE001  (old host g++ cannot build Inductor's CPU kernels)
        return False
    return True


def build():
    """Small fully quantized MaskFormer; bitwidths perturbed so rounding and SAT saturation are exercised."""
    torch.manual_seed(0)
    model = KerasMaskFormer(
        input_nets=make_input_nets(),
        encoder=ENCODER_CFG,
        decoder=DECODER_CFG,
        tasks=make_tasks(),
        dim=32,
        matcher=make_matcher(),
        quant=QUANT,
        quant_scope="all",
    )
    model = materialize(model)
    model.register_keras_parameters()
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, FixedPointQuantizerBase) and m.built:
                for name in ("_b", "_f"):
                    v = getattr(m, name, None)
                    if v is not None:
                        v.value.add_(torch.rand(v.value.shape, generator=gen) * 3.0 - 1.5)
                if m.overflow_mode in {"SAT", "SAT_SYM"}:
                    m._i.value.sub_(torch.rand(m._i.value.shape, generator=gen) * 3.0)  # noqa: SLF001
    return model.train()


def run_step(model, inputs, targets):
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    _, _, losses = model.loss(outputs, targets)
    terms = [v for layer in losses.values() for task in layer.values() for v in task.values()]
    total = sum(v for v in terms if torch.isfinite(v)) + model.quant_losses()
    total.backward()
    grads = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    state = [v.value.detach().clone() for v in model.keras_variables()]
    return total.detach(), grads, state


@pytest.fixture(autouse=True)
def _reset_train_compile():
    yield
    if PATCHED:
        fpq.set_train_compile(False)


@pytest.fixture
def inductor():
    torch.compiler.set_stance("default")  # conftest forces eager; its fixture restores it
    if not _inductor_builds():
        pytest.skip("Inductor cannot build CPU kernels here (set CC/CXX to a newer gcc)")
    torch._dynamo.reset()  # noqa: SLF001
    yield
    if PATCHED:
        fpq.set_train_compile(False)
    torch._dynamo.reset()  # noqa: SLF001


def _inductor_ran() -> bool:
    from torch._dynamo.utils import counters  # noqa: PLC0415, PLC2701

    return sum(counters["inductor"].values()) > 0


@pytest.mark.skipif(not PATCHED, reason="needs the patched HGQ2 tree (set_train_compile) on PYTHONPATH")
@pytest.mark.parametrize("q_type", ["kif", "kbi"])
@pytest.mark.usefixtures("inductor")
def test_compiled_train_fns_do_not_mutate_state(q_type):
    kw = {"b0": 6, "i0": 1} if q_type == "kbi" else {"i0": 1, "f0": 5}
    q = Quantizer(QuantizerConfig(q_type, "datalane", overflow_mode="WRAP", round_mode="RND", homogeneous_axis=(0,), **kw))
    x = torch.randn(6, 5, 7) * 4
    q.build(tuple(x.shape))
    inner = q.quantizer
    fpq.set_train_compile(True, backend="inductor")
    fn = fpq._train_fn(fpq._kif_wrap_train if q_type == "kif" else fpq._kbi_wrap_train)  # noqa: SLF001
    i_var = inner._i.value  # noqa: SLF001
    v0 = i_var._version  # noqa: SLF001
    bits = (inner._f if q_type == "kif" else inner._b).value  # noqa: SLF001
    extra = () if q_type == "kif" else (inner._i.constraint,)  # noqa: SLF001
    args = (i_var, bits, inner._i_decay_speed.value, inner.bw_mapper)  # noqa: SLF001
    out, _ = fn(x.requires_grad_(True), *args, inner.stateless_quantizer, inner.symmetric, *extra, False, None)
    out.sum().backward()
    assert _inductor_ran()
    assert i_var._version == v0, "compiled function wrote the state variable"  # noqa: SLF001
    q(x, training=True)  # HGQ2's call: compiled pure fn, then its own eager assign
    assert i_var._version > v0  # noqa: SLF001


@pytest.mark.skipif(not PATCHED, reason="needs the patched HGQ2 tree (set_train_compile) on PYTHONPATH")
@pytest.mark.usefixtures("inductor")
def test_set_train_compile_matches_eager():
    model = build()
    inputs, targets = clic_dummy_batch(2)
    s0 = [v.value.detach().clone() for v in model.keras_variables()]
    p0 = {n: p.detach().clone() for n, p in model.named_parameters()}

    def reset():
        with torch.no_grad():
            for v, t in zip(model.keras_variables(), s0, strict=True):
                v.value.copy_(t)
            for n, p in model.named_parameters():
                p.copy_(p0[n])

    ref = [run_step(model, inputs, dict(targets))]
    ref.append(run_step(model, inputs, dict(targets)))  # second step continues from the updated WRAP state

    fpq.set_train_compile(True, backend="inductor")
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)  # noqa: SLF001
    reset()
    new = [run_step(model, inputs, dict(targets)), run_step(model, inputs, dict(targets))]

    assert _inductor_ran(), "Inductor generated nothing: the compiled path did not run"
    assert set(fpq._compiled_train_fns) >= {fpq._kif_wrap_train, fpq._fixed_train}  # noqa: SLF001
    for (loss_r, grads_r, state_r), (loss_n, grads_n, state_n) in zip(ref, new, strict=True):
        torch.testing.assert_close(loss_n, loss_r, rtol=1e-6, atol=0)
        assert grads_r.keys() == grads_n.keys()
        for name, g in grads_r.items():
            if g is None:
                assert grads_n[name] is None, name
            else:
                torch.testing.assert_close(grads_n[name], g, rtol=1e-5, atol=1e-7, msg=name)
        for a, b in zip(state_r, state_n, strict=True):
            assert torch.equal(a, b)


def test_option_refuses_when_dynamo_disabled(monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    with pytest.raises(RuntimeError, match="TORCHDYNAMO_DISABLE"):
        lightning_module_hgq.enable_hgq_train_compile()


@pytest.mark.skipif(PATCHED, reason="checks the error path of an unpatched HGQ2")
def test_option_needs_pinned_hgq2(monkeypatch):
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    with pytest.raises(ImportError, match="set_train_compile"):
        lightning_module_hgq.enable_hgq_train_compile()


@pytest.mark.skipif(not PATCHED, reason="needs the pinned HGQ2 build (set_train_compile)")
def test_option_enables_hgq2_compile_and_eager_losses(monkeypatch):
    from hepattn.models import loss as loss_mod  # noqa: PLC0415

    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    for table in (loss_mod.cost_fns, loss_mod.loss_fns):  # restore the compiled wrappers afterwards
        monkeypatch.setattr(loss_mod, "cost_fns" if table is loss_mod.cost_fns else "loss_fns", dict(table))
    for k in ("recompile_limit", "accumulated_recompile_limit"):
        monkeypatch.setattr(torch._dynamo.config, k, getattr(torch._dynamo.config, k))  # noqa: SLF001  restored after the test
    lightning_module_hgq.enable_hgq_train_compile()
    assert fpq._train_compile is not None  # noqa: SLF001
    assert torch._dynamo.config.recompile_limit >= lightning_module_hgq.HGQ_COMPILE_RECOMPILE_LIMIT  # noqa: SLF001
    for table in (loss_mod.cost_fns, loss_mod.loss_fns):
        assert all(not hasattr(fn, "_torchdynamo_orig_callable") for fn in table.values())
