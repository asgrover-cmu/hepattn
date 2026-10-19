#!/usr/bin/env bash
# Submit a CLIC paper-tag training run with polaris/clic_paper_train.pbs.
#
#   SMOKE=1 CONFIG=/eagle/.../<run>/config.yaml bash polaris/submit_clic_paper.sh   # debug queue, ~10 min
#   CONFIG=/eagle/.../<run>/config.yaml bash polaris/submit_clic_paper.sh           # preemptable, 25 h
#
# Optional env: QUEUE, WALLTIME (hh:mm:ss), CHAIN_LEFT (follow-up jobs after preemption or
# walltime, default 2), REPO (paper-tag checkout, default /home/akum/hepattn-clic-paper).
# Always run the smoke first: it trains, checkpoints and resumes in the real environment.
# NEVER put a trailing comment on a #PBS line -- qsub parses the rest as args.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:?set CONFIG=/eagle/.../<run>/config.yaml}"
CONFIG="$(readlink -f "${CONFIG}")"
[ -f "${CONFIG}" ] || { echo "FATAL: no config at ${CONFIG}" >&2; exit 1; }
REPO="${REPO:-/home/akum/hepattn-clic-paper}"
RUN_DIR="$(dirname "${CONFIG}")"
NAME="$(grep -m1 '^name:' "${CONFIG}" | awk '{print $2}')"

if [ "${SMOKE:-0}" = "1" ]; then
  QUEUE="${QUEUE:-debug}"; WALLTIME="${WALLTIME:-00:45:00}"; CHAIN_LEFT=0; LOG="${RUN_DIR}/smoke_$(date +%Y%m%d-%H%M%S).log"
else
  QUEUE="${QUEUE:-preemptable}"; WALLTIME="${WALLTIME:-25:00:00}"; CHAIN_LEFT="${CHAIN_LEFT:-2}"; LOG="${RUN_DIR}/train_$(date +%Y%m%d-%H%M%S).log"
fi

JOB=$(qsub -q "${QUEUE}" -l "walltime=${WALLTIME}" -N "${NAME:0:15}" -o "${LOG}" \
  -v "CONFIG=${CONFIG},SMOKE=${SMOKE:-0},CHAIN_LEFT=${CHAIN_LEFT},WALLTIME=${WALLTIME},REPO=${REPO},SCRIPT=${HERE}/clic_paper_train.pbs" \
  "${HERE}/clic_paper_train.pbs")
echo "submitted ${JOB}  queue=${QUEUE}  walltime=${WALLTIME}  chain=${CHAIN_LEFT}"
echo "log: ${LOG}"
