"""Steady-state seconds per micro-batch, measured INSIDE one run.

12_ddp_epoch_time tried to get this by differencing two whole-process wall times, one
short and one long, to divide out startup. That failed: the devices=1 control produced a
NEGATIVE slope (t20=241s, t120=174s) because the first run cold-loads the ROOT file and
compiles the loss functions with inductor while the second finds both cached, and that
one-off saving is larger than 100 batches of compute.

So time the batches themselves, discard a warmup window, and report the median. Startup,
page cache and compile cache all fall out because they are simply not inside the window.

    PYTHONPATH=src python polaris/13_step_time_fit.py fit --config ... --trainer.devices 4
"""

import os
import statistics
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import lightning.pytorch as pl
import torch

WARMUP = int(os.environ.get("STEP_WARMUP", "25"))
TRAIN_EVENTS = int(os.environ.get("STEP_TRAIN_EVENTS", "994400"))


class StepTimer(pl.Callback):
    """Wall time per training micro-batch, synchronised so GPU work is actually included."""

    def __init__(self) -> None:
        self.dt: list[float] = []
        self._t0: float | None = None

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_start(self, *_args, **_kwargs) -> None:
        self._sync()
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, *_args, **_kwargs) -> None:
        if self._t0 is None:
            return
        self._sync()
        self.dt.append(time.perf_counter() - self._t0)

    def on_train_end(self, trainer: "pl.Trainer", _module) -> None:
        rank = trainer.global_rank
        kept = self.dt[WARMUP:]
        if len(kept) < 10:
            print(f"[STEP-TIME rank={rank}] only {len(self.dt)} batches, {len(kept)} after {WARMUP} warmup -- run more", flush=True)
            return
        med = statistics.median(kept)
        devices = trainer.world_size
        per_rank = TRAIN_EVENTS / (devices * trainer.datamodule.batch_size)
        print(
            f"[STEP-TIME rank={rank}] batches={len(self.dt)} warmup={WARMUP} "
            f"median={med:.4f}s mean={statistics.mean(kept):.4f}s "
            f"p10={sorted(kept)[len(kept) // 10]:.4f}s "
            f"first={self.dt[0]:.2f}s",
            flush=True,
        )
        if rank == 0:
            print(
                f"[STEP-TIME] devices={devices} batch={trainer.datamodule.batch_size} "
                f"-> {per_rank:.0f} micro-batches/rank/epoch "
                f"-> EPOCH {per_rank * med / 3600:.2f} h",
                flush=True,
            )


_orig_trainer_init = pl.Trainer.__init__


def _init_with_timer(self, *args, **kwargs):
    cbs = kwargs.get("callbacks") or []
    if not isinstance(cbs, list):
        cbs = [cbs]
    kwargs["callbacks"] = [*cbs, StepTimer()]
    return _orig_trainer_init(self, *args, **kwargs)


pl.Trainer.__init__ = _init_with_timer
print(f"[STEP-TIME] StepTimer injected (warmup={WARMUP})", flush=True)

from hepattn.experiments.clic.main_hgq import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
