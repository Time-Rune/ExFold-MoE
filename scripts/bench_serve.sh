#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME}"
PORT="${PORT:-8000}"
INPUT_LEN="${INPUT_LEN:?Set INPUT_LEN}"
OUTPUT_LEN="${OUTPUT_LEN:?Set OUTPUT_LEN}"
NUM_PROMPTS="${NUM_PROMPTS:?Set NUM_PROMPTS}"
REQUEST_RATE="${REQUEST_RATE:?Set REQUEST_RATE}"
RESULT_DIR="${RESULT_DIR:?Set RESULT_DIR}"
RESULT_FILE="${RESULT_FILE:?Set RESULT_FILE}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
mkdir -p "${RESULT_DIR}"

common=(
    -m vllm.entrypoints.cli.main bench serve
    --backend vllm
    --host 127.0.0.1
    --port "${PORT}"
    --endpoint /v1/completions
    --model "${MODEL_NAME}"
    --served-model-name "${MODEL_NAME}"
    --tokenizer "${MODEL_PATH}"
    --dataset-name random
    --random-input-len "${INPUT_LEN}"
    --random-output-len "${OUTPUT_LEN}"
    --random-range-ratio 0
    --ignore-eos
    --temperature 0
    --percentile-metrics ttft,tpot,itl,e2el
    --metric-percentiles 50,90,95,99
    --disable-tqdm
    --seed "${SEED:-20260809}"
)
if [[ -n "${MAX_CONCURRENCY:-}" ]]; then
    common+=(--max-concurrency "${MAX_CONCURRENCY}")
fi
if [[ "${TOKENIZER_MODE:-}" == deepseek_v4 ]]; then
    common+=(--tokenizer-mode deepseek_v4)
fi

if (( NUM_WARMUPS > 0 )); then
    "${PYTHON}" "${common[@]}" \
        --num-prompts "${NUM_WARMUPS}" --request-rate inf >/dev/null
fi
"${PYTHON}" "${common[@]}" \
    --num-prompts "${NUM_PROMPTS}" \
    --request-rate "${REQUEST_RATE}" \
    --save-result \
    --result-dir "${RESULT_DIR}" \
    --result-filename "${RESULT_FILE}"
