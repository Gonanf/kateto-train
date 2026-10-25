#!/usr/bin/env bash
# B1 experiment: state round-trip over GGUF.
# Default backend is CPU (the GPU is busy with another job). To try Vulkan:
#   BACKEND=vulkan bash scripts/probar-gguf-estado.sh
set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-cpu}"
MODEL="${MODEL:-out/kateto-rwkv29-Q4_K_M.gguf}"
PROMPT="${PROMPT:-Hola, soy Kateto.}"
BIN="scripts/gguf_state_roundtrip"

echo "=== build ==="
g++ -O2 -std=c++17 scripts/gguf_state_roundtrip.cpp -o "$BIN" -lllama -lggml -lggml-base
rc=$?
if [ $rc -ne 0 ]; then
  echo "BUILD FAILED (exit $rc)"
  exit 1
fi
echo "build ok"

echo
echo "=== llama.cpp build info ==="
if command -v llama-cli >/dev/null 2>&1; then
  llama-cli --version 2>&1 | head -5
fi
if command -v llama-bench >/dev/null 2>&1; then
  llama-bench --version 2>&1 | head -3
fi

echo
echo "=== round-trip (backend=$BACKEND) ==="
"$BIN" --model "$MODEL" --prompt "$PROMPT" --backend "$BACKEND"
rc=$?

echo
# The client already prints its own RESULTADO line; rc==2 means a llama.cpp API
# call failed, so echo the raw reason came from the client, not this script.
echo "script_exit: $rc"
exit $rc