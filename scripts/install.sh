#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:?Usage: install.sh <qwen3|deepseek-v4|all>}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"

install_env() {
    local environment="$1" requirements="$2"
    if [[ ! -x "${environment}/bin/python" ]]; then
        "${BOOTSTRAP_PYTHON}" -m venv "${environment}"
    fi
    "${environment}/bin/python" -m pip install -r "${ROOT}/${requirements}"
    "${environment}/bin/python" -m pip install --no-build-isolation -e "${ROOT}"
}

install_qwen3() {
    install_env "${ROOT}/.venv-qwen010" requirements/qwen3-vllm010.txt
    install_env "${ROOT}/.venv-qwen011" requirements/qwen3-vllm011.txt
}

install_deepseek_v4() {
    install_env "${ROOT}/.venv-dsv4" requirements/deepseek-v4-vllm026.txt
}

case "${TARGET}" in
    qwen3) install_qwen3; VERIFY_PYTHON="${ROOT}/.venv-qwen010/bin/python" ;;
    deepseek-v4) install_deepseek_v4; VERIFY_PYTHON="${ROOT}/.venv-dsv4/bin/python" ;;
    all) install_qwen3; install_deepseek_v4; VERIFY_PYTHON="${ROOT}/.venv-dsv4/bin/python" ;;
    *) echo "Unknown target: ${TARGET}" >&2; exit 2 ;;
esac

PYTHONPATH="${ROOT}" "${VERIFY_PYTHON}" \
    "${ROOT}/scripts/validate_artifacts.py"
