#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run_model.sh MODEL_KEY [CONFIG]" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_KEY="$1"
CONFIG="${2:-${REPO_DIR}/configs/experiment.yaml}"
VENV_DIR="${REDEEP_VENV:-${REPO_DIR}/.venv}"
LOG_DIR="${REDEEP_LOG_DIR:-${REPO_DIR}/outputs/logs}"

cd "${REPO_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
mkdir -p "${LOG_DIR}"

redeep --config "${CONFIG}" run-all --model "${MODEL_KEY}" \
  2>&1 | tee -a "${LOG_DIR}/${MODEL_KEY}.log"
