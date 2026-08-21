#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
MODEL="${1:?Usage: reproduce_quality.sh <qwen3|deepseek-v4> <original|exfold> [task,...]}"
VARIANT="${2:?Usage: reproduce_quality.sh <qwen3|deepseek-v4> <original|exfold> [task,...]}"
shift 2
TASKS="${1:-paper}"
if (( $# > 0 )); then shift; fi
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
PYTHON="${PYTHON:-python3}"
OPENCOMPASS_DIR="${OPENCOMPASS_DIR:-${ROOT}/.deps/opencompass}"
PORT="${PORT:-18080}"

case "${MODEL}" in
    qwen3)
        MODEL_NAME="${MODEL_NAME:-qwen3-30b-a3b}"
        SERVE_SCRIPT="${ROOT}/scripts/serve_qwen3.sh"
        ;;
    deepseek-v4)
        MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"
        SERVE_SCRIPT="${ROOT}/scripts/serve_deepseek_v4.sh"
        ;;
    *) echo "Unknown model: ${MODEL}" >&2; exit 2 ;;
esac
case "${VARIANT}" in original|exfold) ;; *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;; esac

if [[ ! -f "${OPENCOMPASS_DIR}/run.py" ]]; then
    env PYTHON="${PYTHON}" OPENCOMPASS_DIR="${OPENCOMPASS_DIR}" \
        bash "${ROOT}/scripts/setup_opencompass.sh"
fi

RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/${MODEL}-quality/${VARIANT}}"
mkdir -p "${RESULT_ROOT}/logs"
start_server "${PORT}" "${MODEL_NAME}" "${RESULT_ROOT}/logs/server.log" \
    env PYTHON="${PYTHON}" MODEL_PATH="${MODEL_PATH}" MODEL_NAME="${MODEL_NAME}" \
    PORT="${PORT}" bash "${SERVE_SCRIPT}" quality "${VARIANT}"

"${PYTHON}" "${ROOT}/evaluation/opencompass_eval.py" \
    --protocol "${MODEL}" \
    --opencompass-dir "${OPENCOMPASS_DIR}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --api-base "http://127.0.0.1:${PORT}/v1" \
    --tasks "${TASKS}" \
    --work-dir "${RESULT_ROOT}/opencompass" \
    "$@"
