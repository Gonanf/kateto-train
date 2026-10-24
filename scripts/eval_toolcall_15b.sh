#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="http://127.0.0.1:8207"
MODEL="kateto-15b"
GGUF="${KATETO_15B_GGUF:-$REPO_ROOT/out/kateto-15b-Q8_0.gguf}"
BIN="${LLAMA_SERVER_BIN:-/home/chaos/.local/bin/llama-server}"
LOG="${KATETO_15B_LOG:-$REPO_ROOT/out/kateto-15b-8207.log}"

PID=""

cleanup() {
    if [ -n "$PID" ]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=""
    fi
}
trap cleanup EXIT INT TERM

ready() {
    local resp
    resp="$(curl -sf -m 5 "$URL/v1/models" 2>/dev/null || true)"
    if [ -n "$resp" ] && [[ "$resp" != *"Loading model"* ]]; then
        return 0
    fi
    return 1
}

if ready; then
    PID=""
    echo "wrapper: :8207 ya respondia (no lo bajo)" >&2
else
    if [ ! -f "$GGUF" ]; then
        echo "wrapper: archivo GGUF no encontrado: $GGUF" >&2
        exit 3
    fi
    if [ ! -f "$BIN" ]; then
        echo "wrapper: binario llama-server no encontrado: $BIN" >&2
        exit 3
    fi

    mkdir -p "$(dirname "$LOG")"
    "$BIN" --model "$GGUF" --alias kateto-15b --host 127.0.0.1 --port 8207 --ctx-size 4096 --jinja --threads 4 >"$LOG" 2>&1 &
    PID=$!
    echo "wrapper: levante :8207 pid $PID, lo bajo al terminar" >&2

    is_ready=0
    for ((i=0; i<300; i++)); do
        if ready; then
            is_ready=1
            break
        fi
        sleep 1
    done

    if [ "$is_ready" -eq 0 ]; then
        echo "wrapper: :8207 no quedo listo en 300s" >&2
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=""
        exit 4
    fi
fi

cd "$REPO_ROOT"
set +e
python3 scripts/eval_harness_toolcall.py --base-url "$URL" --model "$MODEL" --timeout 600 --out out/eureka-toolcall-15b.json "$@"
rc=$?
set -e
exit "$rc"
