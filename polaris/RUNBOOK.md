# Polaris runbook: stock MaskFormer, Lion vs AdamW

Experiment: `stock_lion_polaris.yaml` vs `stock_adamw_polaris.yaml` in
`src/hepattn/experiments/clic/configs/`. They differ in exactly `name`, `optimizer` and
`lrs_config.max` (8e-5 vs 3e-4, **Option B: tuned LR, not a matched-LR comparison**).
Global batch is 1024 (256 x 4 GPUs), not the 8192 that Lion's 8e-5 was tuned at.
Full rationale is in the commit message of `540614f`.

## Prompt for a Claude Code session on Polaris

> Read `polaris/RUNBOOK.md`, `polaris/train.pbs`, `polaris/submit_train.sh` and the last
> three commits. Find the hepattn checkout in `$HOME` (not the one under `/eagle` on
> `keras-hgq2`), create `polaris/env.sh` if missing, run the flash-attn check on a debug
> node, then smoke-test both `stock_*_polaris.yaml` configs one after the other and report
> the trainable params, whether it ran clean, and it/s. Do not submit the full runs
> without asking me.

## `polaris/env.sh` (gitignored; create by hand)

The scripts read these. Discover each value as shown; nothing here is guessed at run time.

```bash
export PBS_PROJECT=hgcal-maskformer-fpga        # sbank-list-allocations
export PBS_FILESYSTEMS=eagle:home               # df -h /eagle/hgcal-maskformer-fpga ~
export WORK_ROOT=/eagle/hgcal-maskformer-fpga   # venv is ${WORK_ROOT}/hepattn/.venv
# optional:
# export QUEUE_PROD=preemptable                 # qstat -Q
# export PROXY=http://proxy.alcf.anl.gov:3128   # curl -sI --proxy $PROXY https://github.com
```

`WORK_ROOT` must make `ls ${WORK_ROOT}/hepattn/.venv/bin/python` succeed (or set `VENV_DIR` to the venv).
Smokes always go to the `debug` queue; `DATA_ROOT` and `QUEUE_DEBUG` are no longer read.

## Preconditions (each has bitten this cluster before)

1. Two clones exist: the venv's clone under `/eagle` sits on `keras-hgq2`; train from the
   `$HOME` checkout on `akum/polaris-configs`. `train.pbs` asserts `hepattn` resolves inside
   `$REPO_DIR/src`.
2. `ls ~/.local/lib/python3.12/site-packages` must not contain a stray torch.
3. Do installs and builds on a compute node; the login node has process limits (Errno 11).
4. Data files `train_clic_fix.root` and `val_clic_fix.root` in
   `/eagle/hgcal-maskformer-fpga/clic_data/`; preflight aborts the job if they are missing.
5. `gcc` on PATH (Triton); flash_attn absent means `attn_type: torch`, already set.

## Sequence

1. flash-attn check on a debug node, then `exit` so the debug slot is free:
   ```bash
   qsub -I -l select=1:ngpus=1 -l walltime=00:15:00 -l filesystems=eagle:home -A hgcal-maskformer-fpga -q debug
   source polaris/env.sh && export PYTHONNOUSERSITE=1 && export PATH=${WORK_ROOT}/hepattn/.venv/bin:$PATH   # activate script is unreliable
   python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
   ```
   If it imports cleanly, `attn_type` may be switched to `flash-varlen` in both configs.
   Otherwise keep `torch` (exact, just slower).
2. Smoke Lion, wait for it to leave `qstat -u $USER`, then smoke AdamW (debug limits how
   many jobs you can queue):
   ```bash
   SMOKE=1 CFG=src/hepattn/experiments/clic/configs/stock_lion_polaris.yaml bash polaris/submit_train.sh
   SMOKE=1 CFG=src/hepattn/experiments/clic/configs/stock_adamw_polaris.yaml bash polaris/submit_train.sh
   ```
   Pass criteria in each `polaris/logs/*_smoke.log`: `preflight OK`, **10,126,115**
   trainable params, no NaN/inf. Note it/s.
3. Only if both pass, submit the real runs:
   ```bash
   CFG=src/hepattn/experiments/clic/configs/stock_lion_polaris.yaml bash polaris/submit_train.sh
   CFG=src/hepattn/experiments/clic/configs/stock_adamw_polaris.yaml bash polaris/submit_train.sh
   ```
4. Commit the run records: `git add polaris/runs && git commit && git push`.
5. Report per arm: final `val/loss`, whether training was stable (NaN/inf/divergence), and
   wall-clock time per epoch.

Each submission is recorded in `polaris/runs/<timestamp>_<arm>/` (`config.yaml`,
`meta.txt`, `dirty.diff` if any). The job log starts with a dump of both.
