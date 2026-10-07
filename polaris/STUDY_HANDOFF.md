# Handoff: why HGQ2 training is slow, and the first study to run

Written for a Claude Code session on Polaris that has no memory of the conversation
this came from. Read all of it before running anything.

## The question

Quantization-aware training of the CLIC MaskFormer through Keras/HGQ2 takes hours per
epoch. Akum wants to know where the time goes and which changes are worth a real
training run. This handoff covers the first study only: step time against **bitwidth
granularity** and against **real Linformer compression**.

## What is already measured (from `polaris/README.md` and the commit log, not re-run)

- 1.008 s per micro-batch of 32 on one A100; 31,075 micro-batches per epoch = 8.7 h of
  step time. The 12.7 h wall clock had ~4 h outside the steps. 4-GPU DDP: ~3 h per epoch.
- Batch 32 fits 40 GB; batch 128 ran out of memory on 80 GB. The quantizers hold fp32 state.
- EBOPs bookkeeping is 16% of the step; the config already evaluates it every 8 steps.
- Whole-model `torch.compile` is slower (0.77x). HGQ2's own `set_train_compile` measured
  769 -> 466 ms/step, but only with the decoder quantized.
- Every one of those numbers used `attn_type: linformer` with k=256 against a sequence of
  168. That compresses nothing, ignores key padding, and applies the decoder's mask in
  projected space (which is what forces k >= sequence length).
- The model has ~35.3M quantizer parameters against ~11.6M weights. HGQ2's datalane
  default is `homogeneous_axis=(0,)`: one learned bitwidth per element of each activation.

## What this branch adds on top of `helenllii/hepattn@perf/hgq2-train-compile`

| File | What it is |
|---|---|
| `polaris/14_bw_granularity.py` | The timing study. Bare torch loop, one fixed batch, reuses `07_profile.py`'s builders. |
| `src/hepattn/keras/masked_linformer.py` | Keras/HGQ2 port of `compressed-maskformer-reco/masked-linformer`; `attn_type: masked-linformer`. Masks stay exact with k smaller than the sequence. |
| `tests/keras/test_masked_linformer.py` | 9 tests for the port. |
| `pflow_data.py` `sort_nodes_by` | Phi ordering with padding last, from `asgrover-cmu/hepattn@03bf509`. A trained masked-linformer model needs `data.sort_nodes_by: phi`; the timing script does not. |
| lint / type fixes | The branch now passes `ruff` 0.13.0 and `ty`. No behaviour change, except `models/linformer.py` no longer crashes on `k=None`. |

## What is verified, and what is not

- GitHub CI on this branch: lint, type_check, unit and integration pass. The unit job runs
  `tests/keras/` on CPU with the HGQ dependencies, so 6 of the 9 masked-linformer tests
  have passed there (padding and NaN leakage, fully masked query, key sort, factory
  dispatch, quantized forward+backward on both paths).
- **Not verified:** the 3 parity tests against the reference package. They skip unless
  `masked_linformer` is importable. Until they pass, the port is self-consistent but not
  shown to match the original.
- **Not verified:** `14_bw_granularity.py` has never run on a GPU. Expect to fix things.
- **Not verified:** that HGQ2 accepts `heterogeneous_axis` overrides through
  `QuantizerConfigScope(place="datalane", ...)` the way the script passes them. A variant
  HGQ2 rejects prints `FAILED` with the reason and the others still run.

## Where you are running

You are running **on a compute node, inside an interactive `debug` job** (`qsub -I`,
1 h maximum) that Akum started from `tmux` on a login node. That shapes everything:

- **You already have the GPU.** Run the tests and the timing script directly in this
  shell. Do not `qsub` anything: it would queue behind this job and may hit the per-user
  debug limit.
- **The job's walltime is a hard stop, and it kills you too.** Check the time left first
  (`qstat -f $PBS_JOBID | grep -E "walltime"`) and again before each long step. Do not
  start a step that cannot finish.
- **Write progress to disk as you go**, in `polaris/STUDY_PROGRESS.md` (untracked): what
  is done, what you found, what is next, and the exact command in flight. The next
  session starts cold in a new job and reads that file first. If it already exists, you
  are that next session: read it and continue from where it stops.
- **Keep logs on disk** (`tee bwgran_<attn>.log`), never only in the terminal.
- **Outbound network goes through the ALCF proxy.** If a download fails, export
  `http_proxy` and `https_proxy` (`http://proxy.alcf.anl.gov:3128`) and retry.

## Step 0: Akum's home directory is over quota

On 2026-10-06 `git status` in `~/hepattn-hgq` failed with `disk quota exceeded` while Git
LFS wrote a temp file. Home is limited to ~45 GB. This checkout lives under `/eagle` for
that reason, but your own session state is in `~/.claude`, so fix the quota before the
study. Do it in two phases and stop at the end of each.

**Phase A: find what is using the space. Read-only.**

- `myquota`, then `du -sh ~/* ~/.cache ~/.local ~/.pixi ~/.claude 2>/dev/null | sort -rh | head -20`.
- Go one level deeper into the largest few (checkpoints, logs, `.pixi` / `.venv`
  environments, `.git/lfs`, package caches, data files).
- Give Akum a table of path, size, what it is and how you know, grouped as:
  - **A. Regenerable:** pip / uv / pixi caches, `__pycache__`, LFS temp files.
  - **B. Rebuildable with effort:** pixi or venv environments, duplicate git clones.
  - **C. Possibly irreplaceable:** checkpoints, training logs, prediction files, and
    anything you cannot identify. Say whether a copy already exists under `/eagle`.
- Stop and ask which items to remove or move. Ask item by item for B and C.

**Phase B: free only what Akum approved.**

- Delete nothing he has not named. No wildcards beyond what he approved.
- For group C, move to `/eagle/hgcal-maskformer-fpga/` instead of deleting, and verify the
  copy (file count and total size) before removing the original.
- Never touch `~/.ssh`, `~/.claude`, or anything under `/eagle` that is not his.
- Before he decides on a git clone, show its branch, remotes, and any unpushed commits or
  uncommitted changes. Known clones: `~/hepattn`, `~/hepattn-linformer`, `~/hepattn-hgq`
  (on `keras-hgq2`), `~/hepattn-clic-paper`. `~/hepattn-linformer/.pixi/envs/clic` is the
  environment the Lion-vs-AdamW runs use, and that clone has a local branch
  `backup/polaris-configs-old` holding a commit that exists nowhere else.
- Report `myquota` again and stop.

## Polaris facts (some are guesses; check them first)

- You cannot ssh to a compute node. If this session dies, the job is lost with it.
- The `debug` queue allows 1-2 nodes for 1 h, with a per-user limit on queued jobs.
- Project `hgcal-maskformer-fpga`; data in `/eagle/hgcal-maskformer-fpga/clic_data/`
  (`val_clic_fix.root` is the file the profiler reads).
- This checkout should be under `/eagle`, not `$HOME`. If `pwd` says otherwise, tell Akum.
- The repo's Polaris kit expects a clone and venv under `$WORK_ROOT/hepattn`, with
  `hepattn` itself NOT installed in the venv, so `PYTHONPATH=<checkout>/src` decides which
  code runs. A guess from an old `env.sh` line: the venv is
  `/eagle/hgcal-maskformer-fpga/hepattn-hgq/hepattn/.venv`, and the clone beside it sits on
  an older branch (`keras-hgq2`). **Run from the checkout that has this file**, and confirm
  with `python -c "import hepattn; print(hepattn.__file__)"`.
- `polaris/env.sh` is gitignored, so this fresh clone has none. Look in
  `~/hepattn-hgq/polaris/` and `~/hepattn-linformer/polaris/`, show Akum the contents
  before copying one, or build it from `env.sh.example`. Ask for values you cannot discover.
- The venv's HGQ2 may be older than the build pinned in `pyproject.toml`. The pin is only
  needed for `hgq_train_compile`, which this study does not use. Report the installed
  version (`pip show hgq2`); do not reinstall without asking.

## Do this, in order

0. **Quota.** Step 0 above, both phases, before anything below.
1. **Orient.** Confirm the branch and commit, find `env.sh`, find the venv python, and
   confirm on a compute node that `torch`, `keras` and `hgq` import and CUDA is available.
   Report what you found before going further.
2. **Reference package.** Ask Akum before installing anything. If he agrees:
   `<venv>/bin/python -m pip install git+https://github.com/compressed-maskformer-reco/masked-linformer`
   (through the proxy, see above).
3. **Tests.** In this shell:
   `PYTHONPATH=$PWD/src KERAS_BACKEND=torch <venv>/bin/python -m pytest tests/keras/test_masked_linformer.py -v`
   Report passed / failed / skipped per test. If a parity test fails, that is a real bug in
   the port: show the measured error and the failing case, propose a fix, and wait.
4. **Timing study.** In this shell, on one GPU, one attention setup at a time so each
   piece fits the time left and leaves its own log:
   ```bash
   export PYTHONNOUSERSITE=1 KERAS_BACKEND=torch TORCHDYNAMO_DISABLE=1 CUDA_VISIBLE_DEVICES=0
   export DATA_ROOT=/eagle/hgcal-maskformer-fpga/clic_data PYTHONPATH=$PWD/src
   PROF_ATTN=k256   <venv>/bin/python polaris/14_bw_granularity.py 2>&1 | tee bwgran_k256.log
   PROF_ATTN=mlin64 <venv>/bin/python polaris/14_bw_granularity.py 2>&1 | tee bwgran_mlin64.log
   ```
   Each call builds 3 granularities at batch 32 and 64 (6 builds). Nobody has timed a
   build, so run `k256` with `PROF_BATCHES=32` first, see how long it takes, and size the
   rest from that. Run it in the background and poll the log; do not block on it. What
   does not fit goes into `STUDY_PROGRESS.md` for the next session.
5. **Report** the table it prints, plus:
   - Does `k256 per-element` at batch 32 land near the 1.008 s already measured? If not,
     say so first: nothing else in the table can be trusted until that is explained.
   - Quantizer parameter count, ms/step, peak GB and implied epoch hours for each row.
   - Which variants failed or ran out of memory, with the message.

## How to read the result

- Granularity changes how many bitwidths are stored, differentiated and updated. It does
  not change the rounding work, which still scales with the activation size. So the
  memory saving is the more certain effect; the speedup is the open question.
- `mlin64` speeds up the encoder and the query self-attention. The decoder's mask
  attention costs as much as ordinary attention by construction (the reference README
  says so); only its softmax narrows from 160 to 64.
- This study measures step time only. A faster variant is a candidate, not a result: it
  still needs a training run to show the accuracy and the learned bitwidths hold up.

## Stop and ask Akum before

- Deleting or moving anything in his home directory.
- Submitting anything outside the `debug` queue, or any multi-GPU or multi-hour job.
- Installing or upgrading packages in the venv.
- Changing a config's physics (model size, loss, optimizer, learning rate, data).
- Pushing commits. Commit locally if a fix is needed and show the diff.

## Worth knowing before any long run (not part of this study)

Two notes in the code suggest earlier long runs did little quantization:
`quantizer_grad_clip: global` (the default) clips bitwidth gradients far below AdamW's
eps, so bitwidths barely move; `separate` exists for that. And reported EBOPs is ~99.98%
`QSoftmax` and held at 1.6755e15 across nine epochs and four beta values. Raise both with
Akum before anyone spends GPU-days.
