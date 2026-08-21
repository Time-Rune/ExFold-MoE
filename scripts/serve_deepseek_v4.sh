#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?Usage: serve_deepseek_v4.sh <quality|prefill-speed|decode-speed> <original|exfold>}"
VARIANT="${2:?Usage: serve_deepseek_v4.sh <quality|prefill-speed|decode-speed> <original|exfold>}"
shift 2
PYTHON="${PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to DeepSeek-V4-Flash}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4-flash}"
ARTIFACT="${ARTIFACT:-${ROOT}/artifacts/deepseek-v4-flash.pt}"
PORT="${PORT:-8000}"

case "${PROFILE}" in
    quality)
        TP="${TP:-8}"
        GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
        ;;
    prefill-speed)
        TP="${TP:-4}"
        GPU_IDS="${GPU_IDS:-0,1,2,3}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
        ;;
    decode-speed)
        TP="${TP:-4}"
        GPU_IDS="${GPU_IDS:-0,1,2,3}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
        ;;
    *) echo "Unknown DeepSeek-V4 profile: ${PROFILE}" >&2; exit 2 ;;
esac

case "${VARIANT}" in
    original) EXFOLD_ENABLE=0 ;;
    exfold) EXFOLD_ENABLE=1 ;;
    *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing model: ${MODEL_PATH}" >&2; exit 2; }
[[ "${VARIANT}" == original || -f "${ARTIFACT}" ]] || { echo "Missing artifact: ${ARTIFACT}" >&2; exit 2; }

if [[ "${VARIANT}" == exfold && ! -f "${ROOT}/exfold/deepseek_v4/csrc/build/libdsv4_exfold_cuda.so" ]]; then
    PYTHONPATH="${ROOT}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
        "${PYTHON}" -m exfold.deepseek_v4.csrc.build
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export EXFOLD_MODEL=deepseek-v4
export EXFOLD_ENABLE
export DEEPSEEK_V4_RUNTIME_ENABLE="${EXFOLD_ENABLE}"
export DEEPSEEK_V4_ROUTING_MODE=exfold
export EXFOLD_CALIBRATION_PATH="${ARTIFACT}"
export EXFOLD_PREFILL_TOPK="${EXFOLD_PREFILL_TOPK:-3}"
export EXFOLD_PREFILL_RENORM="${EXFOLD_PREFILL_RENORM:-similarity_blend}"
export EXFOLD_DECODE_EXPERT_BUDGET="${EXFOLD_DECODE_EXPERT_BUDGET:-64}"
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_DEEP_GEMM_WARMUP=relax

args=(
    -m vllm.entrypoints.openai.api_server
    --model "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --tensor-parallel-size "${TP}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --kv-cache-dtype fp8
    --block-size 256
    --tokenizer-mode deepseek_v4
    --reasoning-parser deepseek_v4
    --tool-call-parser deepseek_v4
    --enable-auto-tool-choice
    --compilation-config '{"custom_ops":["all"],"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --no-enable-chunked-prefill
    --no-enable-prefix-caching
    --no-enable-log-requests
    --host "${HOST:-127.0.0.1}"
    --port "${PORT}"
)
if [[ "${PROFILE}" != quality ]]; then args+=(--moe-backend "${MOE_BACKEND:-marlin}"); fi
exec "${PYTHON}" "${args[@]}" "$@"
