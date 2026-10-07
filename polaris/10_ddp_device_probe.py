"""Which tensors are still on CPU when DDP tries to broadcast them?

06_ddp_validate.pbs died in DDP.__init__ with
    RuntimeError: No backend type associated with device type cpu
raised from _sync_module_states, which broadcasts module.parameters() + module.buffers().
So at least one of those is on CPU after Lightning has moved the module. This finds it by
name instead of guessing, and checks the keras Variables separately -- the lightning
module notes that keras weights are NOT in named_parameters(), so they may be reached by
a different path (or not at all).

    PYTHONPATH=src python polaris/10_ddp_device_probe.py
"""

import os
from collections import Counter

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch

import hepattn.keras  # noqa: F401  pins the keras backend
from hepattn.keras import set_keras_default_device

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"probe device: {DEV}")


def report(model, tag):
    par = [(n, p.device.type) for n, p in model.named_parameters()]
    buf = [(n, b.device.type) for n, b in model.named_buffers()]
    kv = []
    for m in model.modules():
        for attr in ("_ebops", "_beta"):
            v = getattr(m, attr, None)
            if v is not None:
                try:
                    t = v.value if hasattr(v, "value") else v
                    kv.append((f"{type(m).__name__}.{attr}", torch.as_tensor(t).device.type))
                except Exception as exc:  # noqa: BLE001
                    print(f"    (skipped {type(m).__name__}.{attr}: {type(exc).__name__})")
    print(f"\n--- {tag} ---")
    for label, items in (("parameters", par), ("buffers", buf), ("keras vars (_ebops/_beta)", kv)):
        c = Counter(d for _, d in items)
        print(f"  {label:26s} {len(items):>6d}  devices={dict(c)}")
        bad = [n for n, d in items if d == "cpu"] if DEV != "cpu" else []
        if bad:
            print(f"    ON CPU ({len(bad)}): {bad[:8]}{' ...' if len(bad) > 8 else ''}")
    return par, buf, kv


# Build exactly as lightning_module_hgq.setup() does: keras default device cpu, then a
# materialising forward, then Lightning's .to(device).
import importlib.util  # noqa: E402
import pathlib  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("prof07", HERE / "07_profile.py")
p7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p7)

set_keras_default_device("cpu")
model = p7.build(p7.QUANT)
model = model.cpu()
if os.environ.get("PROBE_DUMMY"):
    from hepattn.experiments.clic.pflow_data import CLICDataset

    ds = CLICDataset(
        filepath="",
        inputs={"node": ["features"]},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=p7.SCALE,
        num_events=4,
        num_objects=p7.NQ,
        max_nodes=p7.MAX_NODES,
        dummy_data=True,
    )
    ev = [ds[i] for i in range(4)]
    inp = {k: torch.stack([e[0][k] for e in ev]) for k in ev[0][0]}
else:
    inp, _tgt = p7.get_batch()
inp_cpu = {k: v.cpu() for k, v in inp.items()}
model.eval()
with torch.no_grad():
    model(inp_cpu)
report(model, "after CPU materialisation (what setup() leaves behind)")

model = model.to(DEV)
par, buf, kv = report(model, f"after .to({DEV}) (what DDP would broadcast)")

# DDP collects exactly parameters + buffers (torch/distributed/utils.py _sync_module_states).
# Also check for a SPLIT across cuda indices, which produces a similar broadcast failure.
devs = {}
for n, t in list(model.named_parameters()) + list(model.named_buffers()):
    devs.setdefault(str(t.device), []).append(n)
print(f"\nexact device set across parameters+buffers: { {k: len(v) for k, v in devs.items()} }")
if len(devs) > 1:
    for k, v in devs.items():
        if len(v) < 12:
            print(f"  minority device {k}: {v}")
if DEV == "cuda":
    print(f"  torch.cuda.current_device()={torch.cuda.current_device()}  count={torch.cuda.device_count()}")

stragglers = [n for n, d in par + buf if d == "cpu"]
print(f"\nVERDICT: {len(stragglers)} parameter/buffer(s) still on cpu after .to({DEV})")
print(
    "  -> these are what DDP._sync_module_states chokes on"
    if stragglers
    else "  -> parameters/buffers are clean; the CPU tensor must come from elsewhere"
)
