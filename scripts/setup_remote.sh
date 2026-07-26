#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REDEEP_VENV:-${REPO_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAGTRUTH_COMMIT="${RAGTRUTH_COMMIT:-c103204b9ce28d6bbad859304bf30de72b8ed8fe}"

cd "${REPO_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

if [[ ! -d "${REPO_DIR}/external/RAGTruth/.git" ]]; then
  mkdir -p "${REPO_DIR}/external"
  git clone https://github.com/ParticleMedia/RAGTruth.git \
    "${REPO_DIR}/external/RAGTruth"
fi
git -C "${REPO_DIR}/external/RAGTruth" checkout --detach "${RAGTRUTH_COMMIT}"

redeep --config "${REPO_DIR}/configs/experiment.yaml" audit-data

echo "Environment ready: ${VENV_DIR}"
echo "Next: redeep --config configs/experiment.yaml doctor"
