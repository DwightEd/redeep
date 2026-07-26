#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${REPO_DIR}/configs/experiment.yaml}"
VENV_DIR="${REDEEP_VENV:-${REPO_DIR}/.venv}"
LOG_DIR="${REDEEP_LOG_DIR:-${REPO_DIR}/outputs/logs}"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

bash scripts/run_model.sh llama31 "${CONFIG}"
bash scripts/run_model.sh qwen3 "${CONFIG}"

# Model runs are process-isolated; compare only after both result sets exist.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
redeep --config "${CONFIG}" compare --first llama31 --second qwen3 \
  2>&1 | tee -a "${LOG_DIR}/comparison.log"
