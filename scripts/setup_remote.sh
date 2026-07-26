#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REDEEP_VENV:-${REPO_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RAGTRUTH_DIR="/share/home/tm902089733300000/a903202310/lys/data/RAGTruth"
RESPONSE_FILE="${RAGTRUTH_DIR}/dataset/response.jsonl"
SOURCE_FILE="${RAGTRUTH_DIR}/dataset/source_info.jsonl"

cd "${REPO_DIR}"

for required_file in "${RESPONSE_FILE}" "${SOURCE_FILE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required RAGTruth file: ${required_file}" >&2
    exit 2
  fi
done

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

redeep --config "${REPO_DIR}/configs/experiment.yaml" audit-data

echo "Environment ready: ${VENV_DIR}"
echo "RAGTruth data: ${RAGTRUTH_DIR}"
echo "Next: redeep --config configs/experiment.yaml doctor"
