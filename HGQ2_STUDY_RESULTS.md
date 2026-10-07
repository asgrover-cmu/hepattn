# HGQ2 training-speed study: results and what to try next

Run on Polaris on 2026-10-07 in two 1 h `debug` jobs (7723432, 7723647), one A100 40 GB
per measurement. Branch `akum/hgq2-studies`. Session log with every command:
`polaris/STUDY_PROGRESS.md`. Raw output: `bwgran_*.log`, `test_*.log` in the repo root.

## Short version

1. **Bitwidth granularity is not a speed lever.** Sharing one bitwidth per channel or per
   layer halves the quantizer parameters and changes step time and memory by ~1%.
2. **Masked-linformer at k=64 halves memory** (30.3 -> 14.7 GB at batch 32) and is 15%
   faster per step. The port matches the reference package (parity tests pass).
3. **A training step costs ~900 ms no matter how many events are in it.** Batch 8, 16 and
   32 take the same time. So the cheapest speedup measured is a bigger batch: masked-linformer
   at batch 80 implies 4.2 h/epoch on one GPU against 9.4 h for the current setup.
4. Everything here is step time on one fixed batch. None of it says anything about
   accuracy or about the bitwidths a real training run would learn.

## Caveat on the baseline

`k256 per-element` at batch 32 measured 1087.9 ms and 1100.8 ms (repeat), against the
1.008 s in `polaris/README.md`: 8-9% slower. The model also counts 12.42M weights and
41.80M quantizer parameters here, against 11.6M and 35.3M in the handoff. Neither gap is
explained. Comparisons inside the tables below are like for like; treat the absolute
numbers as provisional until this is reconciled.

## Results

`polaris/14_bw_granularity.py`: bare torch loop, 5 warmup + 12 timed steps on one fixed
batch (30 timed steps for the repeat). h/epoch assumes 994,400 training events.

### Granularity and attention

| Attention | Granularity | Batch | Weights | Quantizer params | ms/step | Peak GB | h/epoch |
|---|---|---|---|---|---|---|---|
| k256 | per-element (default) | 32 | 12.42M | 41.80M | 1087.9 | 30.3 | 9.39 |
| k256 | per-last-axis | 32 | 12.42M | 20.07M | 1109.1 | 30.6 | 9.57 |
| k256 | per-layer | 32 | 12.42M | 20.01M | 1084.8 | 30.0 | 9.36 |
| k256 | per-element, 30 iters | 32 | 12.42M | 41.80M | 1100.8 | 30.5 | 9.50 |
| k256 | per-last-axis, 30 iters | 32 | 12.42M | 20.07M | 1108.2 | 30.2 | 9.57 |
| k256 | per-layer, 30 iters | 32 | 12.42M | 20.01M | 1098.5 | 30.1 | 9.48 |
| k256 | all three | 64 | | | out of memory | | |
| mlin64 | per-element (default) | 32 | 10.44M | 37.16M | 924.9 | 14.7 | 7.98 |
| mlin64 | per-last-axis | 32 | 10.44M | 20.06M | 944.4 | 14.4 | 8.15 |
| mlin64 | per-layer | 32 | 10.44M | 20.01M | 936.8 | 14.4 | 8.09 |
| mlin64 | per-element (default) | 64 | 10.44M | 37.16M | 1104.6 | 28.3 | 4.77 |
| mlin64 | per-last-axis | 64 | 10.44M | 20.06M | 1119.1 | 27.9 | 4.83 |
| mlin64 | per-layer | 64 | 10.44M | 20.01M | 1102.1 | 27.9 | 4.76 |

- `k256`: the production config, `linformer` with k=256 against a sequence of 168.
- `mlin64`: `masked-linformer` with k=64 in every attention.
- Run-to-run noise is about 1% (compare the two k256 blocks), so the 1-2% differences
  between granularities are noise.
- About 20.0M quantizer parameters remain at per-layer in both setups. The script only
  changes the activation ("datalane") quantizers, so these are presumably the weight
  quantizers. Not checked layer by layer.
- mlin64 has 1.98M fewer weights because the Linformer projections shrink from 256x256
  to 168x64 (encoder) and 160x64 (decoder).

### Batch scan, mlin64 per-element

| Batch | ms/step | ms/event | Peak GB | h/epoch (1 GPU) |
|---|---|---|---|---|
| 8 | 912.6 | 114.1 | 4.4 | 31.51 |
| 16 | 926.8 | 57.9 | 7.8 | 16.00 |
| 32 | 924.9 | 28.9 | 14.7 | 7.98 |
| 64 | 1104.6 | 17.3 | 28.3 | 4.77 |
| 80 | 1216.0 | 15.2 | 34.6 | 4.20 |

Memory is linear in batch (~0.43 GB per event). Time is flat up to 32 and grows by about
6 ms per event after that. Per-last-axis and per-layer follow the same curve.

### Tests

- `tests/keras/test_masked_linformer.py`: 10 passed, 0 skipped, including the 3 parity
  tests against the reference package (`atol=2e-5`, `rtol=2e-4`).
- All of `tests/keras` after the decoder fix: 127 passed, 7 skipped. That run was before
  the reference package was installed, so 3 of the 7 skips were the parity tests.

## What changed in the code

One commit, local only, not pushed: `4628cae fix(keras): let KerasMaskFormerDecoder build
with attn_type masked-linformer` (`src/hepattn/keras/decoder.py`, 7 lines). Without it a
full MaskFormer with `masked-linformer` in the decoder fails at construction with
`AssertionError: Invalid attention type: masked-linformer`. Linted with ruff 0.12.4 (CI
uses 0.13.0); `ty` was not run.

## How to rerun

The shared venv's interpreter is a symlink into another user's home and cannot be
executed. Its packages work with Akum's own Python 3.12:

```bash
P=/eagle/hgcal-maskformer-fpga/akum_pixi/envs/hepattn-15564266159871646574/envs/default/bin/python
SP=/eagle/hgcal-maskformer-fpga/hepattn-hgq/hepattn/.venv/lib/python3.12/site-packages
REF=/eagle/hgcal-maskformer-fpga/akum_pixi/masked_linformer_ref   # masked-linformer 0.1.0
export PYTHONNOUSERSITE=1 KERAS_BACKEND=torch TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=0
export DATA_ROOT=/eagle/hgcal-maskformer-fpga/clic_data PYTHONPATH=$PWD/src:$SP:$REF

$P -m pytest tests/keras/test_masked_linformer.py -v
PROF_ATTN=mlin64 PROF_BATCHES=32,64 $P polaris/14_bw_granularity.py
PROF_ATTN=mlin64 PROF_BATCHES=8,16,80 PROF_EVENTS=80 $P polaris/14_bw_granularity.py
```

Versions: torch 2.10.0+cu128, keras 3.15.0, hgq2 0.1.9 (stock, not the fork pinned in
`pyproject.toml`), lightning 2.5.2. Three builds at one batch size take about 90 s.

## What to try next

Ordered by how much each is likely to tell you per GPU-hour. Items marked *inferred* are
reasoning from the numbers above, not measurements.

1. **Find out what the 900 ms floor is.** Run the `07_profile.py` profiler at batch 8 and
   batch 64 with mlin64 and compare. Whatever does not grow with batch is the floor.
   Candidates, none checked: per-operation Python and dispatch overhead across the many
   quantizer layers, the EBOPs bookkeeping, the scipy matcher. This decides which of the
   items below is worth doing, and it fits in one debug job.
2. **Try HGQ2's `set_train_compile` on top of mlin64.** It was measured earlier at
   769 -> 466 ms/step with the decoder quantized. If the floor is per-operation overhead,
   compilation is the tool aimed at it (*inferred*). It needs the pinned HGQ2 fork, which
   the shared venv does not have, so it means building an environment (see item 7).
3. **Train with masked-linformer at a larger micro-batch.** `env.sh.example` uses batch 32
   with accumulation 4. Batch 64 with accumulation 2 keeps the effective batch at 128 and
   would cut step time per epoch from ~9.4 h to ~4.8 h on one GPU (*inferred from step
   timing*). Batch 80 fits 40 GB with 5 GB to spare, which is thin for a real run with
   validation. Requirements: `data.sort_nodes_by: phi`, and a short run first to check
   the loss and the learned bitwidths against the k256 baseline.
4. **Scale to 4 GPUs after that.** The existing 4-GPU DDP figure is ~3 h/epoch at batch 32
   k256. If the same scaling held for mlin64 at batch 64-80 it would be roughly
   1.5 h/epoch. That is an extrapolation, and the ~4 h of non-step wall clock per epoch
   noted in the README would then dominate, so look at where that goes too.
5. **Stop pursuing granularity for speed.** It may still be worth choosing for other
   reasons (fewer bitwidths to learn, simpler firmware), but it buys no time or memory.
6. **Fix the two quantization concerns before any long run.** From the handoff, not
   re-examined here: `quantizer_grad_clip: global` clips bitwidth gradients far below
   AdamW's eps, so bitwidths barely move (`separate` exists for this); and reported EBOPs
   is ~99.98% `QSoftmax` and stayed at 1.6755e15 across nine epochs and four beta values.
   A faster epoch is of little use if the quantizers are not actually training.
7. **Build your own environment on `/eagle`.** The current one depends on a venv you can
   read but not run or modify, with stock hgq2. A venv of your own with the pinned fork
   removes the `PYTHONPATH` workaround and unblocks item 2.
8. **Reconcile the baseline.** Find why this run counts 41.8M quantizer parameters and
   ~1.09 s where the earlier measurement had 35.3M and 1.008 s. Likely suspects are the
   hgq2 build and config differences between `07_profile.py` then and now; unverified.

## Not done

- No profiling of the floor, no `set_train_compile` run, no training run.
- No k256 number above batch 32 (does not fit 40 GB).
- Nothing pushed. This file, `polaris/STUDY_PROGRESS.md` and the logs are untracked.
