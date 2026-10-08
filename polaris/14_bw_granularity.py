"""Study 1: how much of the QAT step cost is the GRANULARITY of the learned bitwidths?

HGQ2's default for activation ("datalane") quantizers is homogeneous_axis=(0,): one
learned bitwidth per element of the tensor, shared only across the batch. This script
times the same model with coarser sharing and reports, per variant:

    quantizer params | ms/step | ms/event | peak GB        at each batch size

Variants (datalane quantizers only; weight quantizers are left alone):
    per-element   HGQ2 default            one bitwidth per (position, channel) element
    per-last-axis heterogeneous_axis=(-1,) one per channel (per key for attention scores)
    per-layer     heterogeneous_axis=()    one per quantizer

Each is timed under two attention setups (PROF_ATTN=k256|mlin64|both, default both):
    k256    the production config: linformer in encoder and decoder with k=256. The sequence
            is 168 long (160 nodes + 8 register tokens), so k=256 compresses nothing.
    mlin64  masked-linformer in every attention with k=64 (seq_len 168 in the encoder, 160
            in the decoder), the setup of compressed-maskformer-reco's configs/linformer.yaml.
            The decoder's mask attention is exact and costs as much as ordinary attention;
            only its softmax narrows to 64.

Two controls separate what changed between those two (PROF_ATTN takes a comma list, or "all"):
    mlin256 masked-linformer with k=256: the new implementation without compression. Against
            mlin64 it isolates the effect of k; against k256, the effect of the implementation.
    plain   ordinary attention in encoder and decoder, no sequence projection at all.

The fixed batch is in file order, not phi order. That does not change step time; a real
training run with masked-linformer needs data.sort_nodes_by: phi.

Timing only: 5 warmup + PROF_ITERS steps on one fixed batch. It says nothing about accuracy.

    DATA_ROOT=/path/to/clic PYTHONPATH=src python polaris/14_bw_granularity.py
"""

import importlib.util
import os
import pathlib
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PROF_EVENTS", "64")  # read by 07_profile at import; must be >= largest batch

import torch

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("prof07", HERE / "07_profile.py")
p7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p7)  # reuses its model/task/data builders verbatim

ITERS = int(os.environ.get("PROF_ITERS", "12"))
WARMUP = 5
BATCHES = tuple(int(x) for x in os.environ.get("PROF_BATCHES", "32,64").split(","))
TRAIN_EVENTS = 994_400


def _mlin(k: int) -> tuple[dict, dict]:
    """masked-linformer in every attention with projection rank k (seq_len 168 encoder, 160 decoder)."""
    enc = {"num_heads": 16, "linformer_seq_len": 168, "linformer_proj_dim": k}
    dec = {"num_heads": 16, "attn_type": "masked-linformer", "linformer_seq_len": 160, "linformer_proj_dim": k}
    return (
        {**p7.ENCODER, "attn_type": "masked-linformer", "attn_kwargs": enc},
        {**p7.DECODER, "decoder_layer_config": {"dim": p7.DIM, "hybrid_norm": True, "attn_kwargs": dec}},
    )


ATTN = {
    "k256": None,  # p7's ENCODER / DECODER as they are
    "mlin64": _mlin(64),
    "mlin256": _mlin(256),
    "plain": (
        {**p7.ENCODER, "attn_type": "torch", "attn_kwargs": {"num_heads": 16}},
        {**p7.DECODER, "decoder_layer_config": {"dim": p7.DIM, "hybrid_norm": True, "attn_kwargs": {"num_heads": 16}}},
    ),
}
_PROD = (p7.ENCODER, p7.DECODER)
ATTN_SEL = os.environ.get("PROF_ATTN", "both")
# "both" keeps its original meaning (the two setups of the first study); "all" adds the controls
ATTN_RUN = {"both": ["k256", "mlin64"], "all": list(ATTN)}.get(ATTN_SEL) or ATTN_SEL.split(",")

VARIANTS = {
    "per-element (default)": {},
    "per-last-axis": {"homogeneous_axis": None, "heterogeneous_axis": (-1,)},
    "per-layer": {"homogeneous_axis": None, "heterogeneous_axis": ()},
}


def quant_spec(extra: dict) -> dict:
    q = {k: dict(v) for k, v in p7.QUANT.items()}
    q["datalane"].update(extra)
    return q


def timed(model, inp, tgt) -> float:
    for _ in range(WARMUP):
        model.zero_grad(set_to_none=True)
        p7.step(model, inp, tgt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model.zero_grad(set_to_none=True)
        p7.step(model, inp, tgt)
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    return (time.perf_counter() - t0) / ITERS * 1e3


def main() -> None:
    print(f"device {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    inp_all, tgt_all = p7.get_batch()
    have = next(iter(inp_all.values())).shape[0]
    rows = []
    for attn, (gran, extra) in ((a, v) for a in ATTN_RUN for v in VARIANTS.items()):
        p7.ENCODER, p7.DECODER = ATTN[attn] or _PROD  # p7.build() reads these module globals
        tag = f"{attn} {gran}"
        for b in BATCHES:
            if b > have:
                print(f"[{tag}] batch {b}: only {have} events loaded, skipped", flush=True)
                continue
            inp = {k: v[:b] for k, v in inp_all.items()}
            tgt = {k: v[:b] for k, v in tgt_all.items()}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = None
            try:
                model = p7.build(quant_spec(extra)).train()
                with torch.no_grad():
                    model.eval()(inp)  # materialize the lazy HGQ2 layers
                model.train()
                weights, quant = model.trainable_parameter_groups()
                n_w, n_q = sum(p.numel() for p in weights), sum(p.numel() for p in quant)
                ms = timed(model, inp, tgt)
                gb = torch.cuda.max_memory_allocated() / 2**30
                hours = TRAIN_EVENTS / b * ms / 1e3 / 3600
                rows.append((tag, b, n_w, n_q, ms, ms / b, gb, hours))
                print(
                    f"[{tag:28s}] batch {b:3d}  weights {n_w / 1e6:6.2f}M  quantizer {n_q / 1e6:7.3f}M  "
                    f"{ms:8.1f} ms/step  {ms / b:6.2f} ms/event  {gb:5.1f} GB  ~{hours:5.2f} h/epoch(1 GPU)",
                    flush=True,
                )
            except torch.OutOfMemoryError:
                print(f"[{tag:28s}] batch {b:3d}  OUT OF MEMORY", flush=True)
            except (RuntimeError, ValueError, TypeError, KeyError, AssertionError) as e:  # a variant HGQ2 rejects must not hide the others
                print(f"[{tag:28s}] batch {b:3d}  FAILED: {type(e).__name__}: {e}", flush=True)
            finally:
                del model
                torch.cuda.empty_cache()

    ref = f"{ATTN_RUN[0]} per-element"
    base = {b: ms for tag, b, _nw, _nq, ms, *_ in rows if tag.startswith(ref)}
    print(f"\nspeedup vs '{ref}' at the same batch (step time only, no accuracy claim):")
    for tag, b, _nw, _nq, ms, *_ in rows:
        if b in base and not tag.startswith(ref):
            print(f"  {tag:28s} batch {b:3d}  {base[b] / ms:5.2f}x")
    print("BWGRAN-DONE")


if __name__ == "__main__":
    main()
