"""Tests for the Lightning port of HGQ2's BetaPID controller.

The load-bearing test is `test_matches_hgq2_reference`: it drives the *real*
`hgq.utils.sugar.BetaPID` and our port over the same EBOPs sequence and requires the
beta trajectories to agree to floating-point tolerance. A port that silently deviates
from the published control law would make our runs incomparable to Laatu, Sun et al.
(arXiv:2510.24784), which is the entire reason for porting rather than inventing one.

`test_reference_test_is_discriminating` proves that test is not vacuous by perturbing
each gain and the integral seeding in turn and asserting the comparison then fails.
"""

import math
from typing import ClassVar

import pytest

from hepattn.keras.callbacks import BetaPID, BetaScheduler

TARGET = 1.0e15
INIT_BETA = 2.08e-9
WARMUP = 2


def drive_reference(ebops_seq, target=TARGET, init_beta=INIT_BETA, warmup=WARMUP, p=1.0, i=0.5, d=0.0, max_beta=1e-7, min_beta=0.0, damp=0.0):
    """Run HGQ2's own BetaPID over `ebops_seq`, returning the beta after each epoch.

    `get_ebops`/`set_beta` are stubbed so no Keras model is needed; everything else --
    the PID object, the integral seeding, the log-space transform, the clamping -- is
    HGQ2's unmodified code.
    """
    hgq_pid = pytest.importorskip("hgq.utils.sugar.beta_pid")
    cb = hgq_pid.BetaPID(
        target_ebops=target,
        init_beta=init_beta,
        p=p,
        i=i,
        d=d,
        warmup=warmup,
        log=True,
        max_beta=max_beta,
        min_beta=min_beta,
        damp_beta_on_target=damp,
    )
    written: list[float] = []
    feed = iter(ebops_seq)
    current = {"ebops": next(feed)}
    cb.get_ebops = lambda: current["ebops"]
    cb.set_beta = written.append

    cb.on_train_begin()
    out = []
    for epoch in range(len(ebops_seq)):
        logs: dict = {}
        cb.on_epoch_begin(epoch, logs)
        out.append(cb.beta if epoch >= warmup else init_beta)
        # HGQ2 refreshes _ebops in on_epoch_end and consumes it at the next
        # epoch_begin -- the same "most recent measurement" semantics Lightning gets
        # from reading layer.ebops at epoch start. Advance before the call so both
        # implementations consume seq[e] on epoch e.
        current["ebops"] = next(feed, current["ebops"])
        cb.on_epoch_end(epoch, logs)
    return out


def drive_port(ebops_seq, target=TARGET, init_beta=INIT_BETA, warmup=WARMUP, p=1.0, i=0.5, d=0.0, max_beta=1e-7, min_beta=0.0, damp=0.0):
    cb = BetaPID(
        target_ebops=target, init_beta=init_beta, p=p, i=i, d=d, warmup_epochs=warmup, max_beta=max_beta, min_beta=min_beta, damp_beta_on_target=damp
    )
    out = []
    for epoch, ebops in enumerate(ebops_seq):
        out.append(init_beta if epoch < warmup else cb.step(ebops))
    return out


# EBOPs sequences: over target, under target, and the one we actually measured --
# a metric that does not move at all (1.6755e15, flat to 0.0002% over nine epochs).
SEQUENCES = {
    "decaying": [1.9e15, 1.7e15, 1.5e15, 1.3e15, 1.1e15, 0.9e15, 0.8e15, 0.95e15, 1.05e15, 1.0e15],
    "below_target": [4.0e14] * 10,
    "unresponsive": [1.6755e15] * 10,
    "noisy": [1.6e15, 1.65e15, 1.58e15, 1.71e15, 1.62e15, 1.55e15, 1.6e15, 1.59e15, 1.63e15, 1.6e15],
}


@pytest.mark.parametrize("name", sorted(SEQUENCES))
def test_matches_hgq2_reference(name):
    """Our control law is arithmetically identical to HGQ2's."""
    seq = SEQUENCES[name]
    ref, ours = drive_reference(seq), drive_port(seq)
    assert len(ref) == len(ours) == len(seq)
    for epoch, (r, o) in enumerate(zip(ref, ours, strict=True)):
        assert r == pytest.approx(o, rel=1e-12), f"{name} epoch {epoch}: hgq2={r!r} port={o!r}"


def test_reference_test_is_discriminating():
    """The agreement test above fails under any perturbation of the control law.

    Without this, `test_matches_hgq2_reference` could pass for a trivial reason (e.g.
    both sides clamping to max_beta every epoch).
    """
    seq = SEQUENCES["decaying"]
    baseline = drive_reference(seq)
    assert baseline == pytest.approx(drive_port(seq), rel=1e-12)

    perturbations = {
        "p": {"p": 1.05},
        "i": {"i": 0.55},
        "d": {"d": 0.05},
        "init_beta": {"init_beta": INIT_BETA * 1.05},
        "target": {"target": TARGET * 1.05},
        "warmup": {"warmup": 3},
    }
    for label, kwargs in perturbations.items():
        perturbed = drive_port(seq, **kwargs)
        assert len(perturbed) == len(baseline)
        differs = any(b != pytest.approx(q, rel=1e-9) for b, q in zip(baseline, perturbed, strict=True))
        assert differs, f"perturbing {label} did not change the trajectory — test is vacuous"

    # And the seeding specifically: an unseeded integral must not reproduce init_beta.
    cb = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, i=0.5, warmup_epochs=0)
    cb._seeded = True  # noqa: SLF001  (deliberately skip seeding, as a naive port would)
    unseeded = cb.step(1.6755e15)
    assert unseeded != pytest.approx(INIT_BETA, rel=1e-6), "integral seeding is not load-bearing — the first output matched init_beta without it"


def test_first_post_warmup_output_reproduces_init_beta():
    """The controller starts from init_beta rather than jumping."""
    for ebops in (1.6755e15, 4.0e14, 1.0e15):
        cb = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, i=0.5, warmup_epochs=0)
        assert cb.step(ebops) == pytest.approx(INIT_BETA, rel=1e-9), f"at ebops={ebops:.3e}"


def test_beta_moves_in_the_correct_direction():
    over = drive_port([1.6755e15] * 6)
    under = drive_port([4.0e14] * 6)
    assert over[-1] > over[WARMUP], "over target: beta must increase"
    assert under[-1] < under[WARMUP], "under target: beta must decrease"


def test_unresponsive_metric_saturates_at_max_beta_and_stops():
    """Our measured situation: EBOPs does not move, so the integral runs away.

    max_beta must bound it — this is the safety valve the docstring promises.
    """
    betas = drive_port(SEQUENCES["unresponsive"] * 5, max_beta=1e-7)
    assert max(betas) <= 1e-7 + 1e-18
    assert betas[-1] == pytest.approx(1e-7, rel=1e-9), "should have reached the ceiling"
    # Without the clamp it would blow far past it, i.e. the clamp is doing work.
    unclamped = drive_port(SEQUENCES["unresponsive"] * 5, max_beta=float("inf"))
    assert max(unclamped) > 1e-5, f"clamp test is vacuous; unclamped max was {max(unclamped):.3e}"


def test_warmup_holds_init_beta():
    betas = drive_port([1.6755e15] * 6, warmup=4)
    assert betas[:4] == [INIT_BETA] * 4
    assert betas[4] == pytest.approx(INIT_BETA, rel=1e-9)  # seeded, so continuous
    assert betas[5] != betas[4]  # then it moves


def test_rejects_beta_scheduler_in_the_same_trainer():
    """BetaScheduler writes _beta every batch and would erase the controller."""

    class FakeTrainer:
        callbacks: ClassVar[list] = [BetaScheduler(beta_start=0.0, beta_end=1e-12), BetaPID(TARGET, INIT_BETA)]

    cb = FakeTrainer.callbacks[1]
    with pytest.raises(ValueError, match="BetaScheduler"):
        cb.setup(FakeTrainer(), None, "fit")

    class LoneTrainer:
        callbacks: ClassVar[list] = [BetaPID(TARGET, INIT_BETA)]

    LoneTrainer.callbacks[0].setup(LoneTrainer(), None, "fit")  # must not raise


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_ebops": 0.0, "init_beta": 1e-9}, "target_ebops"),
        ({"target_ebops": -1.0, "init_beta": 1e-9}, "target_ebops"),
        ({"target_ebops": 1e15, "init_beta": 0.0}, "init_beta"),
        ({"target_ebops": 1e15, "init_beta": 1e-9, "i": 0.0}, "integral gain"),
    ],
)
def test_rejects_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BetaPID(**kwargs)


def test_log_space_law_is_what_the_docstring_claims():
    """Independent re-derivation of one step, not a re-run of the implementation."""
    cb = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, p=1.0, i=0.5, warmup_epochs=0)
    e0, e1 = 1.6755e15, 1.5e15
    cb.step(e0)  # seeds; returns INIT_BETA
    got = cb.step(e1)

    err0 = math.log10(e0 / TARGET + 1e-9)
    integral = (math.log10(INIT_BETA) - 1.0 * err0) / 0.5  # post-seed, post first +=
    err1 = math.log10(e1 / TARGET)
    integral += err1
    expected = 10.0 ** (1.0 * err1 + 0.5 * integral)
    assert got == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------------------
# The Lightning hook itself. The tests above exercise `step()` (the control law); these
# exercise `on_train_epoch_start`, which owns the warmup branch, the EBOPs read, the
# write-back to every quantized layer, and the not-yet-populated guard.
# --------------------------------------------------------------------------------------


class FakeBetaVar:
    def __init__(self):
        self.value = None

    def assign(self, v):
        self.value = float(v)


class FakeLayer:
    def __init__(self, ebops):
        self.ebops = ebops
        self._beta = FakeBetaVar()


class FakeKerasModel:
    def __init__(self, layer_ebops):
        self._layers = [FakeLayer(e) for e in layer_ebops]

    def keras_layers(self):
        return self._layers


class FakeModule:
    """Stands in for the LightningModule.

    The nesting matters: the callback reaches the quantized layers through
    `pl_module.model.keras_layers()`, exactly as `EBOPsMonitor` and `BetaScheduler` do.
    """

    def __init__(self, layer_ebops):
        self.model = FakeKerasModel(layer_ebops)
        self.logged: dict[str, float] = {}

    def keras_layers(self):
        return self.model.keras_layers()

    def log(self, key, value, **_):
        self.logged[key] = float(value)


class FakeTrainerAt:
    def __init__(self, epoch):
        self.current_epoch = epoch
        self.callbacks = []


def run_hook(cb, module, epoch):
    cb.on_train_epoch_start(FakeTrainerAt(epoch), module)
    return [layer._beta.value for layer in module.keras_layers()]  # noqa: SLF001  (our own stub)


def test_hook_holds_init_beta_through_warmup_then_engages():
    cb = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, i=0.5, warmup_epochs=3)
    module = FakeModule([1.6755e15 / 4] * 4)  # sums to the measured 1.6755e15

    for epoch in range(3):
        written = run_hook(cb, module, epoch)
        assert written == [INIT_BETA] * 4, f"epoch {epoch} must hold init_beta, got {written}"

    seeded = run_hook(cb, module, 3)  # first controlled epoch == init_beta
    assert seeded == pytest.approx([INIT_BETA] * 4, rel=1e-9)
    moved = run_hook(cb, module, 4)  # then it must actually move
    assert moved[0] > INIT_BETA
    assert moved == pytest.approx([moved[0]] * 4), "every layer must get the same beta"


def test_hook_sums_ebops_over_all_layers():
    cb = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, i=0.5, warmup_epochs=0)
    module = FakeModule([3e14, 4e14, 5e14, 4.755e14])
    run_hook(cb, module, 0)
    # rel=1e-6, not tighter: the read goes through `float(torch.as_tensor(...))` and
    # torch's default dtype is float32, whose resolution at 1.6755e15 is ~1.3e8
    # (0.00001%). Worth recording, because it bounds what "val/ebops is flat" can mean
    # -- the 0.0002% we measured across the beta sweep is ~20x above this floor, so it
    # is a real observation and not a precision artifact.
    assert module.logged["train/pid_ebops"] == pytest.approx(1.6755e15, rel=1e-6)


def test_hook_holds_beta_when_ebops_not_yet_populated():
    """A pre-forward epoch reports 0 EBOPs; log10(0) would blow up."""
    cb = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, i=0.5, warmup_epochs=0)
    module = FakeModule([0.0, 0.0])
    written = run_hook(cb, module, 0)
    assert written == [INIT_BETA] * 2
    assert module.logged["train/pid_ebops"] == 0.0
    # ...and the controller is still unseeded, so it engages cleanly once EBOPs appear.
    module = FakeModule([1.6755e15 / 2] * 2)
    assert run_hook(cb, module, 1) == pytest.approx([INIT_BETA] * 2, rel=1e-9)


def test_hook_reports_saturation():
    cb = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, i=0.5, warmup_epochs=0, max_beta=1e-7)
    module = FakeModule([1.6755e15])
    for epoch in range(40):
        run_hook(cb, module, epoch)
    assert module.logged["train/pid_saturated"] == 1.0
    assert module.logged["train/quant_beta"] == pytest.approx(1e-7, rel=1e-9)

    calm_module = FakeModule([1.0e15])
    calm = BetaPID(target_ebops=TARGET, init_beta=INIT_BETA, i=0.5, warmup_epochs=0, max_beta=1e-7)
    run_hook(calm, calm_module, 0)
    assert calm_module.logged["train/pid_saturated"] == 0.0, "must not report saturation at init_beta"


def test_damping_applies_only_below_target():
    """`damp_beta_on_target` is off by default; prove the parameter is not dead code."""
    under = SEQUENCES["below_target"]
    assert drive_port(under, damp=0.5)[-1] < drive_port(under, damp=0.0)[-1]
    # ...and has no effect while the model is still over budget.
    over = SEQUENCES["unresponsive"]
    assert drive_port(over, damp=0.5) == pytest.approx(drive_port(over, damp=0.0), rel=1e-12)
    # Matching HGQ2 with damping on is the real check.
    assert drive_reference(under, damp=0.5) == pytest.approx(drive_port(under, damp=0.5), rel=1e-12)
