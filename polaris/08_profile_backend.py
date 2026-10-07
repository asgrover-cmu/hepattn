"""Is the QAT slowness the Keras/torch backend, PyTorch Lightning, the batch size, or
the Hungarian matcher?

07_profile.py already established, in a BARE torch loop with no Lightning:
  quant step = 1140 ms at batch 32, QAT tax 2.3x, EBOPs bookkeeping 16.1%.

Arithmetic on those numbers: 994,400 train events / 32 = 31,075 fwd+bwd per epoch,
x 1.140 s = 9.8 h. The reported wall clock is 12.7 h. So the bare loop ALREADY accounts
for ~78% of the epoch and Lightning can be at most ~1.3x -- not the 12x. This script
tests that directly and looks for the cost that IS avoidable.

Run via polaris/08_profile_backend.pbs (debug queue, ~20 min).
"""

import importlib.util
import os
import pathlib
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("prof07", HERE / "07_profile.py")
p7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p7)  # reuses its model/task/data builders verbatim

DEVICE = p7.DEVICE
TRAIN_EVENTS = 994_400
REPORTED_HOURS = 12.7
# Smoke knobs: PROF_DUMMY=1 uses synthetic events so the script can be validated on a
# laptop without the CLIC ROOT files; PROF_ITERS shortens every timing loop.
DUMMY = os.environ.get("PROF_DUMMY", "0") == "1"
ITERS = int(os.environ.get("PROF_ITERS", "12"))
BATCHES = tuple(int(x) for x in os.environ.get("PROF_BATCHES", "4,8,16,32,64").split(","))


def step(model, inp, tgt):
    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        out = model(inp)
        out, _, losses = model.loss(out, dict(tgt))
    p7.total_loss(losses).backward()


def slice_batch(inp, tgt, n):
    return ({k: v[:n] for k, v in inp.items()}, {k: v[:n] for k, v in tgt.items()})


def free():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0


def timed(model, inp, tgt, iters=None, warmup=None):
    iters = ITERS if iters is None else iters
    warmup = max(1, ITERS // 3) if warmup is None else warmup
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        step(model, inp, tgt)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        step(model, inp, tgt)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    model.zero_grad(set_to_none=True)
    return ms


def epoch_hours(ms_per_step, batch):
    return (TRAIN_EVENTS / batch) * (ms_per_step / 1e3) / 3600


def phase_breakdown(model, inp, tgt, iters=None):
    iters = ITERS if iters is None else iters
    """Forward / loss+matcher / backward, each synchronized."""
    acc = {"forward": 0.0, "loss+matcher": 0.0, "backward": 0.0}

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        sync()
        t0 = time.perf_counter()
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model(inp)
            sync()
            t1 = time.perf_counter()
            out, _, losses = model.loss(out, dict(tgt))
            sync()
            t2 = time.perf_counter()
        p7.total_loss(losses).backward()
        sync()
        t3 = time.perf_counter()
        acc["forward"] += t1 - t0
        acc["loss+matcher"] += t2 - t1
        acc["backward"] += t3 - t2
    return {k: v / iters * 1e3 for k, v in acc.items()}


def main():
    if DEVICE == "cuda":
        print(f"device   {torch.cuda.get_device_name(0)}")
    print(f"torch    {torch.__version__}\n")

    if DUMMY:
        from hepattn.experiments.clic.pflow_data import CLICDataset  # noqa: PLC0415

        ds = CLICDataset(
            filepath="",
            inputs={"node": ["features"]},
            targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
            scale_dict_path=p7.SCALE,
            num_events=max(BATCHES),
            num_objects=p7.NQ,
            max_nodes=p7.MAX_NODES,
            dummy_data=True,
        )
        ev = [ds[i] for i in range(max(BATCHES))]

        def stack(j):
            return {k: torch.stack([e[j][k] for e in ev]).to(DEVICE) for k in ev[0][j]}

        inp, tgt = stack(0), stack(1)
    else:
        inp, tgt = p7.get_batch()
    have = next(iter(inp.values())).shape[0]
    model = p7.build(p7.QUANT).train()
    with torch.no_grad():
        model.eval()(inp)
    model.train()

    print("================ E. BATCH SCALING (is the step launch-bound?) ================")
    print(f"{'batch':>6s} {'ms/step':>10s} {'ms/event':>10s} {'epoch h':>9s} {'peak GB':>9s}")
    base_per_event = None
    for b in BATCHES:
        if b > have:
            print(f"{b:>6d}  (only {have} events staged — raise PROF_EVENTS)")
            continue
        bi, bt = slice_batch(inp, tgt, b)
        free()
        try:
            ms = timed(model, bi, bt)
        except torch.OutOfMemoryError:
            print(f"{b:>6d}  OOM")
            free()
            break
        per_ev = ms / b
        base_per_event = base_per_event or per_ev
        print(f"{b:>6d} {ms:10.1f} {per_ev:10.2f} {epoch_hours(ms, b):9.2f} {peak_gb():9.2f}")
    free()
    print("\n  Flat ms/step across batch => launch/dispatch bound: the GPU is idling")
    print("  between kernels and a bigger micro-batch is free throughput.")
    print("  ms/event falling with batch => same conclusion, stated as throughput.")

    b = min(int(os.environ.get("PROF_MAIN_BATCH", "8")), have)
    bi, bt = slice_batch(inp, tgt, b)

    def section(name, fn):
        """Run one section; an OOM here must not kill the sections after it."""
        free()
        try:
            fn()
        except torch.OutOfMemoryError:
            print(f"  section {name}: OOM at batch {b} -- rerun with PROF_MAIN_BATCH smaller")
        except Exception as exc:  # noqa: BLE001
            print(f"  section {name} failed: {type(exc).__name__}: {exc}")
        free()

    print(f"\n================ F. PHASE BREAKDOWN (batch {b}) ================")

    def _f():
        ph = phase_breakdown(model, bi, bt)
        tot = sum(ph.values())
        for k, v in ph.items():
            print(f"  {k:16s} {v:8.1f} ms  {100 * v / tot:5.1f}%")
        print(f"  {'total':16s} {tot:8.1f} ms")

    section("F", _f)

    print(f"\n================ G. HUNGARIAN MATCHER ABLATION (batch {b}) ================")

    def _g():
        ms_on = timed(model, bi, bt)
        real_matcher = p7.Matcher.forward

        def identity_match(self, costs, target_valid, query_valid):  # noqa: ARG001
            n_pred = costs.shape[-1]
            idx = torch.arange(n_pred, device=costs.device)
            return idx.unsqueeze(0).expand(costs.shape[0], n_pred).clone()

        p7.Matcher.forward = identity_match
        try:
            ms_off = timed(model, bi, bt)
        finally:
            p7.Matcher.forward = real_matcher
        share = 100 * (ms_on - ms_off) / ms_on
        print(f"  matcher on   {ms_on:8.1f} ms")
        print(f"  matcher off  {ms_off:8.1f} ms")
        print(f"  matcher share of step time  {share:5.1f}%")

    section("G", _g)
    print("  (scipy linear_sum_assignment runs on CPU per event, every step;")
    print("   if this is large it is a pure-CPU serialisation, not an HGQ2 cost)")

    print(f"\n================ H. LIGHTNING A/B (batch {b}) ================")
    try:
        import lightning as L  # noqa: PLC0415
        from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415

        class OneBatch(Dataset):
            def __len__(self):
                return max(4, ITERS // 2)

            def __getitem__(self, i):
                return 0

        class Wrap(L.LightningModule):
            def __init__(self, m):
                super().__init__()
                self.m = m
                self.automatic_optimization = True

            def training_step(self, _batch, _idx):
                out = self.m(bi)
                out, _, losses = self.m.loss(out, dict(bt))
                return p7.total_loss(losses)

            def configure_optimizers(self):
                return torch.optim.AdamW(self.m.parameters(), lr=1e-6)

        dl = DataLoader(OneBatch(), batch_size=1)
        trainer = L.Trainer(
            accelerator="gpu" if DEVICE == "cuda" else "cpu",
            devices=1,
            precision="bf16-mixed" if DEVICE == "cuda" else "32",
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        t0 = time.perf_counter()
        trainer.fit(Wrap(model), dl)
        ms_lightning = (time.perf_counter() - t0) / len(dl.dataset) * 1e3

        opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

        def bare():
            for _ in range(len(dl.dataset)):
                opt.zero_grad(set_to_none=True)
                step(model, bi, bt)
                opt.step()

        bare()  # warm
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        bare()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        ms_bare = (time.perf_counter() - t0) / len(dl.dataset) * 1e3

        print(f"  bare torch loop   {ms_bare:8.1f} ms/step")
        print(f"  lightning Trainer {ms_lightning:8.1f} ms/step")
        print(f"  lightning overhead {100 * (ms_lightning - ms_bare) / ms_bare:6.1f}%")
        print("  (includes trainer setup amortised over few steps -- treat as an")
        print("   upper bound on Lightning's cost, not a precise figure)")
    except torch.OutOfMemoryError:
        print(f"  OOM at batch {b} -- AdamW state plus the graph does not fit; rerun with PROF_MAIN_BATCH smaller")
    except Exception as exc:  # noqa: BLE001
        print(f"  skipped: {type(exc).__name__}: {exc}")
    finally:
        # Lightning's teardown moves the module back to CPU. Without this every
        # later section dies with "found at least two devices, cuda:0 and cpu".
        model.to(DEVICE)
    free()

    print(f"\n================ I. WHERE THE 12.7 h GOES (batch {b}) ================")

    def _i():
        ms = timed(model, bi, bt)
        h = epoch_hours(ms, b)
        print(f"  measured step               {ms:8.1f} ms at batch {b}")
        print(f"  implied epoch (compute only){h:8.2f} h")
        print(f"  reported epoch              {REPORTED_HOURS:8.2f} h")
        print(f"  unexplained by compute      {REPORTED_HOURS - h:8.2f} h  ({100 * (REPORTED_HOURS - h) / REPORTED_HOURS:.0f}%)")
        print("  The remainder is data loading, optimizer, validation, DDP and Lightning.")
        print("  If it is small, no framework change can win more than that fraction.")

    section("I", _i)

    print("\nPROFILE08-DONE", flush=True)


if __name__ == "__main__":
    main()
