#!/usr/bin/env bash
set -euo pipefail

LYS_ROOT="${LYS_ROOT:-/share/home/tm902089733300000/a903202310/lys}"
PROJECT_DIR="${PROJECT_DIR:-${LYS_ROOT}/research/ReDEeP-ICLR}"
MODEL_DIR="${MODEL_DIR:-${LYS_ROOT}/models/Meta-Llama-3.1-8B-Instruct}"
DATA_DIR="${DATA_DIR:-${LYS_ROOT}/data/RAGTruth/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${LYS_ROOT}/results/redeep_token/llama31_on_llama2}"
PYTHON_BIN="${PYTHON_BIN:-${LYS_ROOT}/venvs/lumina-ragtruth/bin/python}"
MODE="${1:-${MODE:-full}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to an environment containing requirements-token-eval.txt." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" run_redeep_token_eval.py \
  --mode "${MODE}" \
  --model-name-or-path "${MODEL_DIR}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --generator-model llama-2-7b-chat \
  --tasks QA Summary Data2txt \
  --configuration-mode train-transfer \
  --allow-checkpoint-transfer \
  --quality-cohort all \
  --dtype float16 \
  --attention-implementation eager \
  --device cuda:0 \
  --selection-unit token \
  --head-counts 3 \
  --layer-counts 30 \
  --beta-values 0.4 \
  --logit-chunk-size "${LOGIT_CHUNK_SIZE:-32}" \
  --cosine-chunk-size "${COSINE_CHUNK_SIZE:-16}" \
  2>&1 | tee "${OUTPUT_DIR}/${MODE}.log"
