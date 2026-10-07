"""Decompose the QAT cost on real CUDA hardware.

Answers three questions that MPS profiling could not:
  1. What is the true float-vs-quant step-time ratio on a DEDICATED A100?
     (The 18 h/epoch figure from the previous cluster was measured on a shared GPU.)
  2. How much of the QAT tax is the EBOPs resource-penalty bookkeeping, which is a
     REGULARIZER and therefore does not have to be recomputed every step?
  3. Which ops actually dominate, and are any of them host-sync ops that should not
     be in a training step at all?

Run via polaris/07_profile.pbs (debug queue, ~20 min).
"""

import os
import pathlib
import time

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from torch import nn

import hepattn.keras  # noqa: F401  pins the keras backend
from hepattn.experiments.clic.pflow_data import CLICDataset
from hepattn.keras import set_keras_default_device
from hepattn.keras.maskformer import KerasMaskFormer
from hepattn.models import Dense, InputNet
from hepattn.models.matcher import Matcher
from hepattn.models.posenc import FourierPositionEncoder
from hepattn.models.task import (
    IncidenceBasedRegressionTask,
    IncidenceRegressionTask,
    ObjectClassificationTask,
    ObjectHitMaskTask,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_keras_default_device(DEVICE)

DIM, NQ, MAX_NODES = 256, 150, 160
N_EVENTS = int(os.environ.get("PROF_EVENTS", "32"))
WARMUP, ITERS = 5, 20
DATA = os.environ["DATA_ROOT"]
SCALE = str(pathlib.Path(__file__).resolve().parent.parent / "src/hepattn/experiments/clic/configs/clic_var_transform.yaml")

LINF = {"num_heads": 16, "linformer_seq_len": 256, "linformer_proj_dim": 256}
ENCODER = {
    "num_layers": 6,
    "attn_type": "linformer",
    "hybrid_norm": True,
    "value_residual": True,
    "num_register_tokens": 8,
    "attn_kwargs": dict(LINF),
}
DECODER = {
    "num_decoder_layers": 4,
    "num_queries": NQ,
    "mask_attention": True,
    "use_query_masks": False,
    "decoder_layer_config": {"dim": DIM, "hybrid_norm": True, "attn_kwargs": {**LINF, "attn_type": "linformer"}},
}
QUANT = {
    "weight": {"default_q_type": "kbi", "b0": 8, "i0": 2},
    "datalane": {"default_q_type": "kif", "i0": 4, "f0": 8},
    "table": {"default_q_type": "kif", "i0": 2, "f0": 10},
    "ebops": {"beta0": 1.0e-12},
}


def make_input_nets():
    return nn.ModuleList([
        InputNet(
            input_name="node",
            fields=["features"],
            net=Dense(input_size=27, output_size=DIM),
            posenc=FourierPositionEncoder(input_name="node", dim=DIM, fields=["eta", "phi"], scale=0.1),
        )
    ])


def make_tasks():
    return nn.ModuleList([
        ObjectClassificationTask(
            name="classification",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            num_classes=5,
            losses={"object_ce": 2},
            costs={"object_ce": 2},
            net=Dense(input_size=DIM, output_size=6, hidden_layers=[256, 128, 32], activation=nn.SiLU()),
            null_weight=0.5,
            class_weights=[1.0, 3.0, 8.0, 1.5, 1.0],
            mask_queries=False,
            has_intermediate_loss=True,
        ),
        ObjectHitMaskTask(
            name="mask",
            input_constituent="node",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            pred_threshold=0.1,
            logit_scale=4,
            losses={"mask_bce": 5.0, "mask_dice": 1.0},
            costs={"mask_dice": 1.0},
            dim=DIM,
            null_weight=1.0,
            has_intermediate_loss=True,
        ),
        IncidenceRegressionTask(
            name="incidence",
            input_constituent="node",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            losses={"kl_div": 1.0},
            costs={"kl_div": 1.0},
            net=Dense(input_size=DIM, hidden_layers=2, activation=nn.SiLU()),
            node_net=Dense(input_size=DIM, hidden_layers=1),
            has_intermediate_loss=False,
        ),
        IncidenceBasedRegressionTask(
            name="regression",
            fields=["e", "pt", "eta", "sinphi", "cosphi"],
            input_constituent="node",
            input_object="query",
            output_object="pflow",
            target_object="particle",
            loss="l1",
            loss_weight=10.0,
            cost_weight=10.0,
            use_incidence=True,
            use_nodes=True,
            cost="new",
            mode="scale",
            scale_dict_path=SCALE,
            net=Dense(input_size=518, output_size=5, hidden_layers=[512, 256, 128, 64, 32], activation=nn.SiLU()),
            has_intermediate_loss=False,
        ),
    ])


def build(quant):
    torch.manual_seed(0)
    return KerasMaskFormer(
        input_nets=make_input_nets(),
        encoder=dict(ENCODER),
        decoder=dict(DECODER),
        tasks=make_tasks(),
        dim=DIM,
        matcher=Matcher(default_solver="scipy", adaptive_solver=False, parallel_solver=False),
        quant=quant,
    ).to(DEVICE)


def get_batch():
    ds = CLICDataset(
        filepath=f"{DATA}/val_clic_fix.root",
        inputs={"node": ["features"]},
        targets={"particle": ["e", "pt", "eta", "sinphi", "cosphi"]},
        scale_dict_path=SCALE,
        num_events=N_EVENTS,
        num_objects=NQ,
        max_nodes=MAX_NODES,
        dummy_data=False,
    )
    ev = [ds[i] for i in range(min(N_EVENTS, len(ds)))]

    def stack(idx):  # idx 0 = inputs, 1 = targets
        return {k: torch.stack([e[idx][k] for e in ev]).to(DEVICE) for k in ev[0][idx]}

    return stack(0), stack(1)


def iter_tensors(node):
    """Yield every tensor in a nested loss dict."""
    for v in node.values():
        if isinstance(v, dict):
            yield from iter_tensors(v)
        elif torch.is_tensor(v):
            yield v


def total_loss(losses):
    s = None
    for v in iter_tensors(losses):
        if v.dtype.is_floating_point:
            m = v.mean()
            if torch.isfinite(m):  # skip the known finfo.min mask_bce artifact
                s = m if s is None else s + m
    if s is None:
        raise RuntimeError("no finite loss terms — cannot profile a backward pass")
    return s


def step(model, inp, tgt):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(inp)
        out, _tgt2, losses = model.loss(out, dict(tgt))
    total_loss(losses).backward()


def timeit(model, inp, tgt, tag):
    for _ in range(WARMUP):
        model.zero_grad(set_to_none=True)
        step(model, inp, tgt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model.zero_grad(set_to_none=True)
        step(model, inp, tgt)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1e3
    print(f"[{tag:28s}] {ms:8.1f} ms/step  (batch {N_EVENTS})", flush=True)
    return ms


def set_ebops(model, on: bool) -> int:
    """EBOPs is a REGULARIZER: the forward math does not depend on it. Toggling it
    isolates the cost of the resource-penalty bookkeeping (an extra matmul +
    add_loss per quantized layer, every step).
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "_enable_ebops"):
            m._enable_ebops = on  # noqa: SLF001  (the attribute the forward reads)
            n += 1
    return n


def main():
    print(f"device   {torch.cuda.get_device_name(0)}")
    print(f"torch    {torch.__version__} (cuda {torch.version.cuda})\n")
    inp, tgt = get_batch()

    print("================ A. STEP-TIME DECOMPOSITION ================")
    mf = build(None).train()
    with torch.no_grad():
        mf.eval()(inp)
    mf.train()
    t_float = timeit(mf, inp, tgt, "float (no quantizers)")
    del mf
    torch.cuda.empty_cache()

    mq = build(QUANT).train()
    with torch.no_grad():
        mq.eval()(inp)
    mq.train()
    t_quant = timeit(mq, inp, tgt, "quant, EBOPs on (prod)")

    n = set_ebops(mq, False)
    t_noeb = timeit(mq, inp, tgt, "quant, EBOPs OFF")
    set_ebops(mq, True)

    print(f"\n  QAT tax vs float          {t_quant / t_float:6.2f}x")
    print(f"  EBOPs share of step time  {100 * (t_quant - t_noeb) / t_quant:6.1f}%  ({n} quantized layers)")
    print(f"  tax without EBOPs         {t_noeb / t_float:6.2f}x")
    if (t_quant - t_noeb) / t_quant > 0.15:
        print("  => EBOPs is a large share. It is only a regularizer, so computing it")
        print("     every N steps instead of every step is a safe, real speedup.")
    else:
        print("  => EBOPs is not the bottleneck; the cost is core fake-quantization.")

    print("\n================ B. TOP CUDA OPS (production config) ================")
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            mq.zero_grad(set_to_none=True)
            step(mq, inp, tgt)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=22))

    print("\n================ C. LAUNCH-BOUND CHECK ================")
    ka = prof.key_averages()

    def dev_us(e):  # torch >= 2.9 renamed self_cuda_time_total -> self_device_time_total
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)

    launches = sum(e.count for e in ka if dev_us(e) > 0)
    cuda_us = sum(dev_us(e) for e in ka)
    cpu_us = sum(e.self_cpu_time_total for e in ka)
    print(f"  CPU self time  : {cpu_us / 1e3:>10.1f} ms over 3 steps")
    print(f"  GPU self time  : {cuda_us / 1e3:>10.1f} ms over 3 steps")
    print("  (CPU > GPU means the step is launch/dispatch bound, not compute bound)")
    print(f"  distinct CUDA-active op invocations : {launches:>10,} over 3 steps  ({launches // 3:,}/step)")
    print(f"  total CUDA self time                : {cuda_us / 1e3:>10.1f} ms over 3 steps")
    print(f"  mean CUDA time per invocation       : {cuda_us / max(launches, 1):>10.1f} us")
    print("  (a few us per op => launch-bound, not FLOP-bound)")

    print("\n================ D. HOST-SYNC OPS IN THE STEP ================")
    suspects = ("nonzero", "item", "_local_scalar_dense", "sync")
    hits = [e for e in ka if any(s in e.key.lower() for s in suspects)]
    hits.sort(key=lambda e: -e.self_cpu_time_total)
    if hits:
        print(f"  {'op':40s} {'count':>8s} {'self CPU ms':>12s}")
        for e in hits[:8]:
            print(f"  {e.key[:40]:40s} {e.count:>8d} {e.self_cpu_time_total / 1e3:>12.1f}")
        print("  NOTE: aten::nonzero appeared at 17.6% on MPS. If it is absent or tiny")
        print("        here, that was an MPS-fallback artifact and not a real problem.")
    else:
        print("  none of the suspect ops appear — no host syncs in the step")

    print("\nPROFILE-DONE", flush=True)


if __name__ == "__main__":
    main()
