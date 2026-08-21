#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_PID=""

stop_server() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}

wait_for_server() {
    local port="$1"
    local model_name="$2"
    local log_file="$3"
    for _ in $(seq 1 "${SERVER_WAIT_STEPS:-360}"); do
        if curl -fsS --max-time 5 "http://127.0.0.1:${port}/v1/models" \
            | grep -Fq "${model_name}"; then
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            tail -120 "${log_file}" >&2 || true
            return 1
        fi
        sleep 5
    done
    tail -120 "${log_file}" >&2 || true
    return 1
}

start_server() {
    local port="$1"
    local model_name="$2"
    local log_file="$3"
    shift 3
    mkdir -p "$(dirname "${log_file}")"
    "$@" >"${log_file}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${port}" "${model_name}" "${log_file}"
}

trap stop_server EXIT INT TERM
