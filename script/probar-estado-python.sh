#!/usr/bin/env bash
# B2: el state buffer de llama.cpp se construye desde Python.
#
#   1. dump del estado con el cliente de B1
#   2. rebuild del mismo buffer con rwkv_pipeline/state_bytes.py
#   3. cmp byte a byte
#   4. inyeccion del buffer reconstruido + RESULTADO: PASS del cliente
#   5. RESULTADO B2: PASS|FAIL
#
# No compila nada: usa el binario de B1 (tools/gguf_state_roundtrip) tal cual.
# CPU por default (la placa esta ocupada). BACKEND=vulkan para GPU.
set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-cpu}"
MODEL="${MODEL:-out/kateto-rwkv29-Q4_K_M.gguf}"
PROMPT="${PROMPT:-Hola, soy Kateto.}"
BIN="tools/gguf_state_roundtrip"
ORIG="${ORIG:-/tmp/estado-orig.bin}"
REBUILD="${REBUILD:-/tmp/estado-python.bin}"
PY="${PY:-venv/bin/python}"
[ -x "$PY" ] || PY="python3"

fallar() { echo; echo "RESULTADO B2: FAIL"; echo "motivo: $1"; exit 1; }

echo "=== preflight ==="
[ -x "$BIN" ]   || fallar "falta $BIN (correo: bash script/probar-gguf-estado.sh lo compila)"
[ -f "$MODEL" ] || fallar "falta el modelo $MODEL"
"$PY" -c "import numpy" 2>/dev/null || fallar "$PY no tiene numpy"
echo "bin:    $BIN"
echo "modelo: $MODEL"
echo "python: $PY ($("$PY" -c 'import sys;print(sys.version.split()[0])'))"

echo
echo "=== 1/4 dump del estado (cliente B1) ==="
"$BIN" --model "$MODEL" --prompt "$PROMPT" --backend "$BACKEND" --dump-state "$ORIG" 2>&1 \
  | grep -v '^ggml_vulkan' | grep -v '^\.*$'
[ -f "$ORIG" ] || fallar "el cliente no escribio $ORIG"
ORIG_BYTES=$(stat -c%s "$ORIG")
echo "dump: $ORIG ($ORIG_BYTES bytes)"

echo
echo "=== 2/4 rebuild desde Python ==="
"$PY" -m rwkv_pipeline.state_bytes "$ORIG" "$REBUILD" || fallar "el rebuild de Python fallo"
[ -f "$REBUILD" ] || fallar "no se escribio $REBUILD"

echo
echo "=== 3/4 cmp byte a byte ==="
echo "\$ cmp $ORIG $REBUILD"
if cmp "$ORIG" "$REBUILD"; then
  echo "cmp: identicos (0 diferencias)"
else
  fallar "los buffers difieren"
fi

echo
echo "=== 4/4 inyeccion del buffer reconstruido ==="
OUT=$("$BIN" --model "$MODEL" --prompt "$PROMPT" --backend "$BACKEND" --inject-state "$REBUILD" 2>&1)
echo "$OUT" | grep -v '^ggml_vulkan' | grep -v '^\.*$'
grep -q '^RESULTADO: PASS$' <<<"$OUT" || fallar "el cliente no dio RESULTADO: PASS sobre el buffer de Python"

echo
echo "RESULTADO B2: PASS"
