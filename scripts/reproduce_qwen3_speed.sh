#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to Qwen3-30B-A3B}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/qwen3-speed}"
PORT="${PORT:-18080}"
MODEL_NAME=qwen3-30b-a3b
mkdir -p "${RESULT_ROOT}"/{logs,original,exfold}

run_profile() {
    local profile="$1" variant="$2" python="$3"
    local server_log="${RESULT_ROOT}/logs/${variant}_${profile}.log"
    start_server "${PORT}" "${MODEL_NAME}" "${server_log}" \
        env PYTHON="${python}" MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
        PORT="${PORT}" bash "${ROOT}/scripts/serve_qwen3.sh" "${profile}" "${variant}"

    if [[ "${profile}" == prefill-speed ]]; then
        for qps in 1 2 3 4 5 6 7 8; do
            env PYTHON="${python}" MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
                PORT="${PORT}" INPUT_LEN=8192 OUTPUT_LEN=1 NUM_PROMPTS=192 \
                NUM_WARMUPS=16 REQUEST_RATE="${qps}" MAX_CONCURRENCY=96 \
                RESULT_DIR="${RESULT_ROOT}/${variant}" \
                RESULT_FILE="${variant}_prefill_qps${qps}_in8192_out1.json" \
                bash "${ROOT}/scripts/bench_serve.sh"
        done
    else
        for qps in 2 4 6 8 10 12; do
            env PYTHON="${python}" MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
                PORT="${PORT}" INPUT_LEN=1 OUTPUT_LEN=256 NUM_PROMPTS=512 \
                NUM_WARMUPS=32 REQUEST_RATE="${qps}" \
                RESULT_DIR="${RESULT_ROOT}/${variant}" \
                RESULT_FILE="${variant}_decode_qps${qps}_in1_out256.json" \
                bash "${ROOT}/scripts/bench_serve.sh"
        done
    fi
    stop_server
}

for variant in original exfold; do
    run_profile prefill-speed "${variant}" "${QWEN3_PREFILL_PYTHON:-python3}"
done
for variant in original exfold; do
    run_profile decode-speed "${variant}" "${QWEN3_DECODE_PYTHON:-python3}"
done
python3 "${ROOT}/evaluation/summarize_speed.py" "${RESULT_ROOT}" --ttft-stat mean
