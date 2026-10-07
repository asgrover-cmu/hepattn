"""PeriodicEBOPs: skip the EBOPs resource penalty on all but every N-th step.

Discriminating properties, each with the wrong-implementation it catches:
- schedule: layers toggle ON exactly when global_step % N == 0 — catches an off-by-one
  or a callback that never finds the quantized layers (it would silently do nothing).
- effect: the penalty actually collapses on skipped steps — catches toggling an
  attribute the forward does not read (the flag would flip but nothing would change).
- safety: a skipped step still produces a finite, differentiable total loss — catches
  the failure mode where an empty loss list breaks the training step.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("hgq", reason="hgq dependency group not installed")

from parity_utils import make_padded_batch  # noqa: F401  # ty: ignore [unresolved-import]  (pins test conftest paths)
from test_maskformer_parity import clic_dummy_batch, make_keras_model  # ty: ignore

from hepattn.keras.callbacks import PeriodicEBOPs

QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0e-12},
}
EVERY = 8


@pytest.fixture(scope="module")
def built():
    """Quantized model with its lazily-built layers materialized, plus a stub module."""
    model = make_keras_model(seed=0, quant=QUANT)
    inputs, _ = clic_dummy_batch(2)
    with torch.no_grad():
        model.eval()(inputs)  # quantized layers build lazily at first forward
    model.train()
    pl = SimpleNamespace(model=model, log=lambda *_a, **_k: None)
    return model, pl, inputs


def _fire(cb, pl, step):
    cb.on_train_batch_start(SimpleNamespace(global_step=step), pl, None, 0)


def test_finds_quantized_layers(built):
    """A callback that finds zero layers would silently be a no-op."""
    _, pl, _ = built
    cb = PeriodicEBOPs(every_n_steps=EVERY)
    assert len(cb._quantized_layers(pl)) > 0, "no quantized layers found — callback would do nothing"  # noqa: SLF001


@pytest.mark.parametrize("step", [0, 1, 7, 8, 9, 16])
def test_schedule(built, step):
    _, pl, _ = built
    cb = PeriodicEBOPs(every_n_steps=EVERY)
    _fire(cb, pl, step)
    flags = {layer._enable_ebops for layer in cb._quantized_layers(pl)}  # noqa: SLF001
    assert len(flags) == 1, "quantized layers disagree about the EBOPs flag"
    assert flags.pop() is (step % EVERY == 0)


def test_penalty_collapses_on_skipped_steps(built):
    """The flag must actually change what the forward computes, not just be set."""
    model, pl, inputs = built
    cb = PeriodicEBOPs(every_n_steps=EVERY)

    _fire(cb, pl, 0)  # active
    model(inputs)
    active = float(model.quant_losses())

    _fire(cb, pl, 3)  # skipped
    model(inputs)
    skipped = float(model.quant_losses())

    assert active > 0, "active step registered no EBOPs penalty at all"
    assert skipped < 0.01 * active, f"skipped step kept {100 * skipped / active:.2f}% of the penalty"


def test_skipped_step_still_trains(built):
    """An empty loss list must not break the step: the total must stay finite and
    differentiable, otherwise every skipped step would crash training.
    """
    model, pl, inputs = built
    cb = PeriodicEBOPs(every_n_steps=EVERY)
    _fire(cb, pl, 3)  # skipped

    outputs = model(inputs)
    total = model.quant_losses()
    assert torch.isfinite(torch.as_tensor(total)), "non-finite penalty on a skipped step"

    # any differentiable model output will do; avoid pinning a task-specific key
    head = next(t for t in outputs["final"]["classification"].values() if torch.is_tensor(t) and t.dtype.is_floating_point and t.requires_grad)
    (head.float().pow(2).mean() + torch.as_tensor(total)).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed on a skipped step"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradients on a skipped step"
