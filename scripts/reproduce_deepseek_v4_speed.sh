#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to DeepSeek-V4-Flash}"
PYTHON="${DEEPSEEK_V4_PYTHON:-python3}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/deepseek-v4-speed}"
PORT="${PORT:-18080}"
MODEL_NAME=deepseek-v4-flash
mkdir -p "${RESULT_ROOT}"/{logs,original,exfold}

run_profile() {
    local profile="$1" variant="$2"
    local server_log="${RESULT_ROOT}/logs/${variant}_${profile}.log"
    start_server "${PORT}" "${MODEL_NAME}" "${server_log}" \
        env PYTHON="${PYTHON}" MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
        PORT="${PORT}" bash "${ROOT}/scripts/serve_deepseek_v4.sh" "${profile}" "${variant}"
    if [[ "${profile}" == prefill-speed ]]; then
        env PYTHON="${PYTHON}" MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
            TOKENIZER_MODE=deepseek_v4 PORT="${PORT}" INPUT_LEN=8192 OUTPUT_LEN=1 \
            NUM_PROMPTS=128 NUM_WARMUPS=32 REQUEST_RATE=inf MAX_CONCURRENCY=1 \
            RESULT_DIR="${RESULT_ROOT}/${variant}" \
            RESULT_FILE="${variant}_prefill_qpsinf_in8192_out1.json" \
            bash "${ROOT}/scripts/bench_serve.sh"
    else
        env PYTHON="${PYTHON}" MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
            TOKENIZER_MODE=deepseek_v4 PORT="${PORT}" INPUT_LEN=1 OUTPUT_LEN=256 \
            NUM_PROMPTS=1024 NUM_WARMUPS=32 REQUEST_RATE=64 MAX_CONCURRENCY=128 \
            RESULT_DIR="${RESULT_ROOT}/${variant}" \
            RESULT_FILE="${variant}_decode_qps64_in1_out256.json" \
            bash "${ROOT}/scripts/bench_serve.sh"
    fi
    stop_server
}

for variant in original exfold; do run_profile prefill-speed "${variant}"; done
for variant in original exfold; do run_profile decode-speed "${variant}"; done
python3 "${ROOT}/evaluation/summarize_speed.py" "${RESULT_ROOT}"
