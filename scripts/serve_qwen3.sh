#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:?Usage: serve_qwen3.sh <quality|prefill-speed|decode-speed> <original|exfold>}"
VARIANT="${2:?Usage: serve_qwen3.sh <quality|prefill-speed|decode-speed> <original|exfold>}"
PYTHON="${PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to Qwen3-30B-A3B}"
MODEL_NAME="${MODEL_NAME:-qwen3-30b-a3b}"
ARTIFACT="${ARTIFACT:-${ROOT}/artifacts/qwen3-30b-a3b.pt}"
PORT="${PORT:-8000}"

case "${PROFILE}" in
    quality)
        export VLLM_USE_V1=0
        TP="${TP:-4}"
        GPU_IDS="${GPU_IDS:-0,1,2,3}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
        PREFILL_BUDGET="${PREFILL_BUDGET:-4}"
        DECODE_BUDGET="${DECODE_BUDGET:-64}"
        DECODE_POLICY="${DECODE_POLICY:-dynamic}"
        ;;
    prefill-speed)
        export VLLM_USE_V1=0
        TP="${TP:-4}"
        GPU_IDS="${GPU_IDS:-0,1,2,3}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-8193}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-96}"
        PREFILL_BUDGET="${PREFILL_BUDGET:-4}"
        DECODE_BUDGET=128
        DECODE_POLICY=dynamic
        ;;
    decode-speed)
        export VLLM_USE_V1=1
        TP="${TP:-1}"
        GPU_IDS="${GPU_IDS:-0}"
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-273}"
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
        PREFILL_BUDGET=8
        DECODE_BUDGET="${DECODE_BUDGET:-32}"
        DECODE_POLICY=static_norm
        export EXFOLD_FORCE_PHASE=decode
        ;;
    *) echo "Unknown Qwen3 profile: ${PROFILE}" >&2; exit 2 ;;
esac

case "${VARIANT}" in
    original) export EXFOLD_ENABLE=0 ;;
    exfold) export EXFOLD_ENABLE=1 ;;
    *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;;
esac

[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing model: ${MODEL_PATH}" >&2; exit 2; }
[[ "${VARIANT}" == original || -f "${ARTIFACT}" ]] || { echo "Missing artifact: ${ARTIFACT}" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export EXFOLD_MODEL=qwen3
export EXFOLD_ARTIFACT="${ARTIFACT}"
export EXFOLD_PREFILL_BUDGET="${PREFILL_BUDGET}"
export EXFOLD_DECODE_BUDGET="${DECODE_BUDGET}"
export EXFOLD_QWEN3_DECODE_POLICY="${DECODE_POLICY}"
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0
if [[ "${DISABLE_NCCL_NVLS:-0}" == 1 ]]; then export NCCL_NVLS_ENABLE=0; fi

args=(
    -m vllm.entrypoints.openai.api_server
    --model "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --tensor-parallel-size "${TP}"
    --dtype bfloat16
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --host "${HOST:-127.0.0.1}"
    --port "${PORT}"
    --generation-config vllm
    --disable-log-requests
)
if [[ "${PROFILE}" != decode-speed ]]; then
    args+=(--no-enable-chunked-prefill)
fi
exec "${PYTHON}" "${args[@]}" "$@"
