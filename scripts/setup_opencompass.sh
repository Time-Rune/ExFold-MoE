#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OPENCOMPASS_DIR="${OPENCOMPASS_DIR:-${ROOT}/.deps/opencompass}"
OPENCOMPASS_COMMIT=4df8be67e3b3c67929f7730614c4e5444b62d6a3

if [[ ! -d "${OPENCOMPASS_DIR}/.git" ]]; then
    [[ ! -e "${OPENCOMPASS_DIR}" ]] || {
        echo "OPENCOMPASS_DIR exists but is not a git checkout: ${OPENCOMPASS_DIR}" >&2
        exit 2
    }
    mkdir -p "$(dirname "${OPENCOMPASS_DIR}")"
    git clone https://github.com/open-compass/opencompass.git "${OPENCOMPASS_DIR}"
    git -C "${OPENCOMPASS_DIR}" checkout --detach "${OPENCOMPASS_COMMIT}"
fi

revision="$(git -C "${OPENCOMPASS_DIR}" rev-parse HEAD)"
[[ "${revision}" == "${OPENCOMPASS_COMMIT}" ]] || {
    echo "OpenCompass must be at ${OPENCOMPASS_COMMIT}; found ${revision}" >&2
    exit 2
}

"${PYTHON}" -m pip install -r "${ROOT}/requirements/evaluation.txt"
"${PYTHON}" -m pip install -e "${OPENCOMPASS_DIR}"
printf 'OpenCompass ready: %s (%s)\n' "${OPENCOMPASS_DIR}" "${revision}"
