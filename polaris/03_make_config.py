#!/usr/bin/env python
"""Generate the Polaris HGQ2 training config from the repo's base.yaml.

Run from the repo root, inside the venv:
    python 03_make_config.py

Writes ./polaris_hgq.yaml. Everything Polaris-specific (data paths, checkpoint dir,
batch size, precision) is set here so the repo's committed configs stay clean.

Env overrides:
    HGQ_BATCH      micro-batch per GPU            (default 32)
    HGQ_ACCUM      gradient accumulation steps    (default 4)
    HGQ_DEVICES    GPUs for one DDP run           (default 4; validated run 7598515)
    HGQ_PRECISION  bf16-mixed | 32                (default bf16-mixed)
    HGQ_BETA_MODE  pid | schedule                 (default pid)
    HGQ_BETA_END   final EBOPs penalty weight     (default 4.0e-15, schedule mode only)
    HGQ_TARGET_EBOPS   EBOPs setpoint for pid mode    (default 1.0e15)
    HGQ_INIT_BETA      starting beta for pid mode     (default 2.6e-10)
    HGQ_MAX_BETA       pid beta ceiling               (default 1.0e-7)
    HGQ_PID_I          pid integral gain              (default 0.5)
    HGQ_PID_WARMUP_EPOCHS  epochs before pid engages  (default 2)
    HGQ_EPOCHS     max epochs                     (default 20)
    HGQ_WARMUP_EPOCHS  beta ramp length, in epochs (default 1.0)
    DATA_DIR       CLIC ROOT directory            (default $HGQ_DATA or ./data/clic)
    CKPT_DIR       checkpoint output directory    (default ./hgq_ckpts)
"""

import math
import os
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent  # polaris/ -> repo root
SRC = REPO / "src/hepattn/experiments/clic/configs/base.yaml"
DST = REPO / "polaris_hgq.yaml"
if not SRC.exists():
    raise SystemExit(f"cannot find {SRC} — run this from the repo (it lives in polaris/)")

DATA_DIR = os.environ.get("DATA_DIR", os.environ.get("HGQ_DATA", str(REPO / "data/clic")))
CKPT_DIR = os.environ.get("CKPT_DIR", str(REPO / "hgq_ckpts"))
SCALE = str(REPO / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml")

BATCH = int(os.environ.get("HGQ_BATCH", "32"))
ACCUM = int(os.environ.get("HGQ_ACCUM", "4"))
PRECISION = os.environ.get("HGQ_PRECISION", "bf16-mixed")
BETA_MODE = os.environ.get("HGQ_BETA_MODE", "pid").lower()
if BETA_MODE not in {"pid", "schedule"}:
    raise SystemExit(f"HGQ_BETA_MODE must be 'pid' or 'schedule', got {BETA_MODE!r}")
BETA_END = float(os.environ.get("HGQ_BETA_END", "4.0e-15"))

# --- pid mode -------------------------------------------------------------------
# Closed-loop control of beta against a fixed EBOPs setpoint, as in Laatu, Sun et al.
# (arXiv:2510.24784), which trains every model to 350k EBOPs. Open-loop scheduling
# needs the right beta known in advance; we do not have it, and calibrating it from
# the reported total_ebops overstated the loss slope by 10,410x.
#
# Setpoint: the measured total is 1.6755e15, so 1.0e15 asks for a 40% cut. This is
# deliberately NOT the hardware budget -- an XCVU13P holds ~2.15e6 EBOPs under
# LUTs ~ exp(0.985*log(EBOPs)) against its 1.728M LUTs, i.e. 8e8x below where we are.
# A setpoint that far away saturates the controller on its first epoch and degenerates
# into training at max_beta. 1.0e15 is the setpoint that asks a question we can answer
# in one run: does closed-loop pressure move the reported cost AT ALL?
TARGET_EBOPS = float(os.environ.get("HGQ_TARGET_EBOPS", "1.0e15"))
# Top of the recalibrated sweep -- the beta that moved /f furthest (0.165 bits off its
# 8-bit init) without destabilising the task loss.
INIT_BETA = float(os.environ.get("HGQ_INIT_BETA", "2.6e-10"))
# Safety valve. total_ebops is a step function here (~0.6 bits of headroom before it
# moves), so an unresponsive process variable WILL wind the integral up indefinitely.
# Saturating this ceiling is an expected and informative outcome, not a failure.
MAX_BETA = float(os.environ.get("HGQ_MAX_BETA", "1.0e-7"))
# HGQ2's default integral gain of 2e-3 is tuned for hundred-epoch runs: at an error of
# log10(1.6755) = 0.224 it moves beta by a factor of 1.001 per epoch. QAT costs 12.7
# h/epoch here, so gain for roughly one decade per ten epochs instead.
PID_I = float(os.environ.get("HGQ_PID_I", "0.5"))
PID_WARMUP_EPOCHS = int(os.environ.get("HGQ_PID_WARMUP_EPOCHS", "2"))
EPOCHS = int(os.environ.get("HGQ_EPOCHS", "20"))

# Evaluate the EBOPs resource penalty every N steps rather than every step. Measured
# on an A100: EBOPs bookkeeping is 16.1% of step time, and EBOPs itself drifts 0.03%
# over 14k steps — so sampling it costs essentially nothing. Set to 1 to disable.
EBOPS_EVERY = int(os.environ.get("HGQ_EBOPS_EVERY", "8"))
# Skipped steps contribute no penalty, so the time-averaged pressure would drop by
# 1/N. Scale beta_end by N so the user-facing beta_end keeps its meaning.
BETA_END_EFFECTIVE = BETA_END * EBOPS_EVERY
# Same 1/N argument applies to the controller's beta: it would find the right value on
# its own, but starting and bounding it in the same units keeps every beta in this file
# comparable to the BetaScheduler runs.
INIT_BETA_EFFECTIVE = INIT_BETA * EBOPS_EVERY
MAX_BETA_EFFECTIVE = MAX_BETA * EBOPS_EVERY

# Beta ramp length. This was hardcoded at 42000 optimizer steps, which at 7769
# steps/epoch meant beta only reached full strength at epoch 5.4 — a diagnostic sweep
# had to run for days before it could show anything. Scale it to epochs instead.
# TRAIN_EVENTS is measured: 31075 micro-batches x 32 at batch 32, i.e. after the
# loader drops the ~1% of events exceeding the node/particle caps.
TRAIN_EVENTS = 994_400
STEPS_PER_EPOCH = math.ceil(TRAIN_EVENTS / (BATCH * ACCUM))
WARMUP_EPOCHS = float(os.environ.get("HGQ_WARMUP_EPOCHS", "1.0"))
WARMUP_STEPS = max(1, round(WARMUP_EPOCHS * STEPS_PER_EPOCH))

# Losses are mean-reduced, so Lightning SUMS the accumulated micro-batch gradients.
# Keep each optimizer step invariant to ACCUM: divide LR, multiply the clip threshold.
# Reference point: 5e-5 at effective batch 128 (matches the group's float baseline).
LR_MAX = 5.0e-5 / ACCUM
CLIP = 0.1 * ACCUM

with SRC.open() as f:
    cfg = yaml.safe_load(f)

cfg["name"] = "CLIC_pflow_HGQ_polaris"

d = cfg["data"]
d["train_path"] = f"{DATA_DIR}/train_clic_fix.root"
d["valid_path"] = f"{DATA_DIR}/val_clic_fix.root"
# NOTE: test_clic_common_infer.root contains -9999 sentinel indices that crash the
# loader; point test at the val file until that is patched.
d["test_path"] = f"{DATA_DIR}/val_clic_fix.root"
d["scale_dict_path"] = SCALE
d["batch_size"] = BATCH
d["num_workers"] = 8
d["num_val"] = 5000

t = cfg["trainer"]
# DDP is now validated on this stack (run 7598515, after the keras-device fix in
# lightning_module_hgq.setup): two-rank fit completes, checkpoint has 7095 tensors /
# 6176 quantizer vars / 0 non-finite, and DDP reaches val/loss 40.29 against the
# single-GPU 39.61 -- a 1.72% difference on 10 batches with different sharding.
#
# LR is deliberately NOT rescaled for devices. Lightning's DDP AVERAGES gradients across
# ranks, so the gradient magnitude at each optimizer step is unchanged; what grows is the
# number of samples behind it (devices x BATCH x ACCUM), i.e. less gradient noise. That
# permits a larger LR but does not require one, so raising it is left as a tuning
# decision rather than smuggled in with a throughput change. Contrast ACCUM below, where
# Lightning SUMS and the division IS required for correctness.
DEVICES = int(os.environ.get("HGQ_DEVICES", "4"))
t["devices"] = DEVICES
# Pin the strategy. Lightning auto-selects ddp for devices>1, but the validation ran with
# it explicit, and an implicit choice is one more thing that can change under us.
if DEVICES > 1:
    t["strategy"] = "ddp"
t["accelerator"] = "gpu"
t["max_epochs"] = EPOCHS
t["precision"] = PRECISION
t["accumulate_grad_batches"] = ACCUM
t["gradient_clip_val"] = CLIP
t["gradient_clip_algorithm"] = "norm"
t["log_every_n_steps"] = 200

# Drop the Compile callback: torch.compile is measured NET-SLOWER (0.77x) on this
# stack because the Keras-torch dispatch layer breaks the graph at every layer.
cbs = [c for c in t["callbacks"] if "Compile" not in c.get("class_path", "")]
# The repo's Checkpoint callback monitors val/loss, which never fires mid-epoch. On a
# walltime-limited cluster that loses everything since the last epoch boundary, so add
# a plain step-interval ModelCheckpoint with save_last for resumable chaining.
cbs.append({
    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
    "init_args": {
        "dirpath": CKPT_DIR,
        "every_n_train_steps": 1000,
        "save_last": True,
        "save_top_k": 1,
        "monitor": None,
    },
})
t["callbacks"] = cbs

# Comet is the repo default; compute nodes have no outbound network, so force it offline.
# NOTE: trainer.logger must stay a SINGLE entry. The repo's CLI hardcodes
# `trainer.logger.init_args.offline_directory` and links `name` into
# `trainer.logger.init_args.name`; a list of loggers makes jsonargparse replace the list
# with a bare Namespace and the run dies at instantiation. Readable on-disk metrics come
# from the MetricsCsvWriter callback below instead.
if isinstance(t.get("logger"), dict):
    t["logger"].setdefault("init_args", {})["online"] = False

m = cfg["model"]
m["optimizer"] = "AdamW"
m["lrs_config"]["max"] = LR_MAX

net = m["model"]["init_args"]
net_cls = m["model"]
net_cls["class_path"] = "hepattn.keras.maskformer.KerasMaskFormer"

# Quantized encoder/decoder with the linformer attention used by the group's baseline.
net["encoder"] = {
    "num_layers": 6,
    "attn_type": "linformer",
    "hybrid_norm": True,
    "value_residual": True,
    "num_register_tokens": 8,
    "attn_kwargs": {"num_heads": 16, "linformer_seq_len": 256, "linformer_proj_dim": 256},
}
net["decoder"]["decoder_layer_config"]["attn_kwargs"].update({
    "attn_type": "linformer",
    "linformer_seq_len": 256,
    "linformer_proj_dim": 256,
})

# HGQ2 quantization spec. q_type/place are SELECTORS in HGQ2 scopes; the type is
# chosen via default_q_type.
net["quant"] = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 0.0},
}

# BetaPID and BetaScheduler both write every quantized layer's _beta, and BetaScheduler
# writes per batch -- i.e. after BetaPID's epoch-start write -- so exactly one of them
# may be registered. BetaPID.setup() raises if both are; drop the other here so the
# config is never generated in that state.
t["callbacks"] = [c for c in t["callbacks"] if not c.get("class_path", "").endswith(("BetaScheduler", "BetaPID"))]
if BETA_MODE == "pid":
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.BetaPID",
        "init_args": {
            "target_ebops": TARGET_EBOPS,
            "init_beta": INIT_BETA_EFFECTIVE,
            "i": PID_I,
            "warmup_epochs": PID_WARMUP_EPOCHS,
            "max_beta": MAX_BETA_EFFECTIVE,
        },
    })
else:
    # Open-loop ramp: kept for reproducing the earlier sweeps.
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.BetaScheduler",
        "init_args": {"beta_start": 0.0, "beta_end": BETA_END_EFFECTIVE, "warmup_steps": WARMUP_STEPS},
    })
if not any("EBOPsMonitor" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({"class_path": "hepattn.keras.callbacks.EBOPsMonitor"})
# The learned bitwidths are the only direct readout of whether QAT is doing anything.
# Without this, an inert beta sweep looks healthy on every other metric.
if not any("BitwidthMonitor" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({"class_path": "hepattn.keras.callbacks.BitwidthMonitor"})
# Must come after BitwidthMonitor: Lightning runs callbacks in order, and this writes
# whatever is in callback_metrics at that moment.
if not any("MetricsCsvWriter" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.MetricsCsvWriter",
        "init_args": {"path": f"{CKPT_DIR}/metrics.csv"},
    })
if EBOPS_EVERY > 1 and not any("PeriodicEBOPs" in c.get("class_path", "") for c in t["callbacks"]):
    t["callbacks"].append({
        "class_path": "hepattn.keras.callbacks.PeriodicEBOPs",
        "init_args": {"every_n_steps": EBOPS_EVERY},
    })

for task in net["tasks"]["init_args"]["modules"]:
    ia = task.get("init_args", {})
    if "scale_dict_path" in ia:
        ia["scale_dict_path"] = SCALE

with DST.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"WROTE {DST}")
print(f"  data       {DATA_DIR}")
print(f"  ckpts      {CKPT_DIR}")
print(f"  batch      {BATCH} x accum {ACCUM} x devices {DEVICES} = effective {BATCH * ACCUM * DEVICES}")
print(f"  precision  {PRECISION}")
print(f"  lr max     {LR_MAX:.3e}   clip {CLIP}   (invariant to accum)")
print(f"  epochs     {EPOCHS}")
if BETA_MODE == "pid":
    print(f"  beta       PID -> target {TARGET_EBOPS:.3e} EBOPs (measured now: 1.6755e15)")
    print(f"             init {INIT_BETA_EFFECTIVE:.2e}  ceiling {MAX_BETA_EFFECTIVE:.2e}  i={PID_I}  warmup {PID_WARMUP_EPOCHS} epoch(s)")
    headroom = math.log10(MAX_BETA_EFFECTIVE / INIT_BETA_EFFECTIVE)
    err = math.log10(1.6755e15 / TARGET_EBOPS)
    print(
        f"             {headroom:.2f} decades of headroom; at a constant error of {err:.3f}"
        f" the ceiling is reached in ~{headroom / (PID_I * err):.0f} epochs"
    )
else:
    print(f"  beta_end   {BETA_END:.2e}  (open-loop schedule)")
    print(f"  beta ramp  {WARMUP_STEPS} steps = {WARMUP_EPOCHS:g} epoch(s) at {STEPS_PER_EPOCH} steps/epoch")
if EBOPS_EVERY > 1:
    scaled = f"beta_end scaled to {BETA_END_EFFECTIVE:.2e}" if BETA_MODE != "pid" else "pid betas scaled by 8"
    print(f"  EBOPs      every {EBOPS_EVERY} steps ({scaled} so time-averaged pressure is unchanged); ~16% faster steps")
