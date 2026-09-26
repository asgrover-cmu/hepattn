#!/usr/bin/env bash
# Submit a CLIC training run and record exactly what was submitted.
#
#   CFG=src/hepattn/experiments/clic/configs/stock_lion_polaris.yaml bash polaris/submit_train.sh
#   SMOKE=1 CFG=... bash polaris/submit_train.sh     # 100-batch gate, always the debug queue (SMOKE_QUEUE to change)
#
# Every submission creates polaris/runs/<timestamp>_<arm>[_smoke]/ holding
#   config.yaml   byte-for-byte copy of the config; the job trains from THIS file
#   meta.txt      git SHA/branch, dirty flag, overrides, queue, and the job id
#   dirty.diff    only if the tree had uncommitted changes (tracked files)
# That folder is tracked in git (not ignored): commit it so the history of what
# was run lives next to the code. The job log goes to polaris/logs/ (ignored) and
# starts with a full dump of config.yaml.
#
# Optional env: SMOKE=1, EPOCHS=n, RESUME_CKPT=/abs/path, WALLTIME=hh:mm:ss.
# NEVER put a trailing comment on a #PBS line -- qsub parses the rest as args.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${CFG:?set CFG=src/hepattn/experiments/clic/configs/<arm>_polaris.yaml}"
[ -f "${REPO_DIR}/${CFG}" ] || { echo "FATAL: no config at ${REPO_DIR}/${CFG}" >&2; exit 1; }
source "${REPO_DIR}/polaris/env.sh"

ARM="$(basename "${CFG}" _polaris.yaml)"
STAMP="$(date +%Y%m%d-%H%M%S)"
if [ "${SMOKE:-0}" = "1" ]; then
  # Hard-wired: env.sh is sourced above, so an env.sh QUEUE_DEBUG must not redirect a smoke.
  QUEUE="${SMOKE_QUEUE:-debug}"; WALL="00:30:00"; RUN_ID="${STAMP}_${ARM}_smoke"; JOB_NAME="clic-${ARM}-smoke"
else
  QUEUE="${QUEUE_PROD:-preemptable}"; WALL="${WALLTIME:-48:00:00}"; RUN_ID="${STAMP}_${ARM}"; JOB_NAME="clic-${ARM}"
fi

RUN_DIR="${REPO_DIR}/polaris/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}" "${REPO_DIR}/polaris/logs"
cp "${REPO_DIR}/${CFG}" "${RUN_DIR}/config.yaml"

DIRTY="clean"
if ! git -C "${REPO_DIR}" diff --quiet HEAD; then
  DIRTY="DIRTY (see dirty.diff)"
  git -C "${REPO_DIR}" diff HEAD > "${RUN_DIR}/dirty.diff"
fi
{
  echo "run_id:      ${RUN_ID}"
  echo "submitted:   $(date -Is)"
  echo "host:        $(hostname)"
  echo "config_src:  ${CFG}"
  echo "git_commit:  $(git -C "${REPO_DIR}" rev-parse HEAD)"
  echo "git_branch:  $(git -C "${REPO_DIR}" rev-parse --abbrev-ref HEAD)"
  echo "git_tree:    ${DIRTY}"
  echo "queue:       ${QUEUE}"
  echo "walltime:    ${WALL}"
  echo "smoke:       ${SMOKE:-0}"
  echo "epochs_override: ${EPOCHS:-none}"
  echo "resume_ckpt: ${RESUME_CKPT:-none}"
} > "${RUN_DIR}/meta.txt"

LOG="${REPO_DIR}/polaris/logs/${RUN_ID}.log"
: > "${LOG}"
VARS="REPO_DIR=${REPO_DIR},RUN_DIR=${RUN_DIR},SMOKE=${SMOKE:-0}"
VARS+="${EPOCHS:+,EPOCHS=${EPOCHS}}${RESUME_CKPT:+,RESUME_CKPT=${RESUME_CKPT}}"

set -x
JOB_ID="$(qsub -A "${PBS_PROJECT}" \
     -q "${QUEUE}" \
     -l "filesystems=${PBS_FILESYSTEMS}" \
     -l "walltime=${WALL}" \
     -N "${JOB_NAME}" \
     -o "${LOG}" \
     -v "${VARS}" \
     "${REPO_DIR}/polaris/train.pbs")"
set +x
echo "job_id:      ${JOB_ID}" >> "${RUN_DIR}/meta.txt"
echo "log:         ${LOG}" >> "${RUN_DIR}/meta.txt"
echo "submitted ${JOB_ID}; record in ${RUN_DIR}"
