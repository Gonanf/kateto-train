#!/usr/bin/env bash
# B3: el estado nativo de RWKV (.pth) se convierte en el byte-stream que
# llama.cpp acepta como estado.
#
#   1. --report: cuantas capas trae el .pth, que forma tiene su time_state, que
#      forma espera el layout de llama.cpp por capa, y si calzan.
#   2. si CALZA: arma el buffer, lo inyecta con scripts/gguf_state_roundtrip y
#      exige state_seq_set_data_bytes != 0 y test1_identidad: OK -> PASS
#   3. si NO CALZA: imprime los numeros y sale con RESULTADO B3: SHAPES_NO_CALZAN
#      (exit 3). Eso es una respuesta valida: un .pth de otro modelo no se
#      salva inventando tensores.
#
# No compila nada: usa el binario de B1 tal cual. CPU por default.
set -uo pipefail
cd "$(dirname "$0")/.."

BACKEND="${BACKEND:-cpu}"
PTH="${PTH:-out/rwkv_states/seco/rwkv-3.pth}"
MODELOS="${MODELOS:-out/kateto-rwkv29-Q4_K_M.gguf out/base-rwkv29-f16.gguf}"
PROMPT="${PROMPT:-Hola, soy Kateto.}"
BIN="scripts/gguf_state_roundtrip"
BUFFER="${BUFFER:-/tmp/estado-desde-pth.bin}"
PY="${PY:-venv/bin/python}"
[ -x "$PY" ] || PY="python3"

fallar() { echo; echo "RESULTADO B3: FAIL"; echo "motivo: $1"; exit 1; }

echo "=== preflight ==="
[ -f "$PTH" ] || fallar "falta el estado nativo $PTH"
"$PY" -c "import numpy, torch" 2>/dev/null || fallar "$PY no tiene numpy+torch"
echo "pth:    $PTH"
echo "modelos: $MODELOS"
echo "python: $PY ($("$PY" -c 'import sys;print(sys.version.split()[0])'))"

echo
echo "=== 1/3 --report (no inyecta, no fabrica) ==="
# shellcheck disable=SC2086
"$PY" -m rwkv_pipeline.state_from_pth --pth "$PTH" --modelo $MODELOS --report
REPORT_RC=$?
echo "report_exit: $REPORT_RC   (0 = calza, 3 = no calza)"

if [ "$REPORT_RC" -ne 0 ]; then
  echo
  echo "=== 2/3 saltado: las formas no calzan ==="
  echo "Las formas NO calzan con ningun modelo de la lista. Se reportan arriba y"
  echo "no se fabrica ningun buffer: un .pth de estado de otro modelo no se"
  echo "arregla inventando tensores."
  echo
  echo "RESULTADO B3: SHAPES_NO_CALZAN"
  exit 3
fi

echo
echo "=== 2/3 armar el buffer desde el .pth ==="
# shellcheck disable=SC2086
"$PY" -m rwkv_pipeline.state_from_pth --pth "$PTH" --modelo $MODELOS --out "$BUFFER" \
  || fallar "el armado del buffer fallo"
[ -f "$BUFFER" ] || fallar "no se escribio $BUFFER"
echo "buffer: $BUFFER ($(stat -c%s "$BUFFER") bytes)"

echo
echo "=== 3/3 inyeccion del buffer desde el .pth ==="
[ -x "$BIN" ] || fallar "falta $BIN (bash scripts/probar-gguf-estado.sh lo compila)"
MODELO_1=${MODELOS%% *}
OUT=$("$BIN" --model "$MODELO_1" --prompt "$PROMPT" --backend "$BACKEND" --inject-state "$BUFFER" 2>&1)
echo "$OUT" | grep -v '^ggml_vulkan' | grep -v '^\.*$'

BYTES=$(grep -oP '^state_seq_set_data_bytes: \K\d+' <<<"$OUT")
[ -n "${BYTES:-}" ] && [ "$BYTES" -ne 0 ] \
  || fallar "llama_state_seq_set_data devolvio ${BYTES:-<nada>}: no acepto el estado"
grep -q '^test1_identidad: OK' <<<"$OUT" \
  || fallar "test1_identidad no fue OK: llama.cpp no devolvio el buffer identico"
echo "state_seq_set_data_bytes: $BYTES (distinto de cero)"
echo "test1_identidad: OK"

echo
echo "RESULTADO B3: PASS"
