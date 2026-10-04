#!/bin/bash
# Post-training: merge LoRA -> pth -> GGUF -> imatrix -> Q4_K_M -> validacion.
# Uso: ./export_gguf.sh <adapter_dir> <nombre_salida>
#
# ============================================================================
# OJO — FUERA DEL CAMINO CRITICO (ver docs/decision-gguf-vs-nativo.md).
# Este export es solo BENCH/experimento en otra maquina. NO sirve para Kateto:
#   * GGUF no acepta estados externos -> las voces de la Capa 2 (state-tuning)
#     son inutilizables aca; no hay forma de inyectar el S_0 entrenado.
#   * ROSA es estructura de runtime (rosa_module.py), no se serializa a GGUF.
#   * Cambiar entre agentes exige recargar el modelo entero (segundos) en vez
#     del swap nativo medido en 48-135 ms.
# El camino de servicio y de verificacion es el nativo:
#   rwkv_pipeline/infer_kateto.py (.pth + out/rwkv_states/<voz>/ + ROSA)
# ----------------------------------------------------------------------------
# CORREGIDO 2026-10-01: esto es falso a nivel API. La inyeccion de estado existe
# (llama_state_seq_set_data) y esta medida: ver docs/gguf-estado-roundtrip.md (RESULTADO: PASS).
# El texto de arriba se conserva como historia de la premisa, no como estado actual.
# ----------------------------------------------------------------------------
# ============================================================================
set -e
TDIR=/run/media/chaos/terciario/proyectos/kateto-train
ADAPTER="$1"
NAME="${2:-kateto-rwkv29}"
MERGED="$TDIR/out/${NAME}-merged.pth"
F16="$TDIR/out/${NAME}-f16.gguf"
Q4="$TDIR/out/${NAME}-Q4_K_M.gguf"

source "$TDIR/venv/bin/activate"
echo "### 1/5 merge LoRA"
python "$TDIR/merge_lora.py" "$TDIR/base/rwkv7-g1j-2.9b-20260831-ctx16384.pth" "$ADAPTER" "$MERGED"

echo "### 2/5 pth a GGUF"
if [ ! -d "$TDIR/rwkv-mobile" ]; then
  git clone --depth 1 https://github.com/MollySophia/rwkv-mobile "$TDIR/rwkv-mobile"
fi
python "$TDIR/rwkv-mobile/converter/convert_rwkv_pth_to_gguf.py" "$MERGED" "$TDIR/rwkv-mobile/assets/rwkv_vocab_v20230424.txt" --outfile "$F16" --outtype f16

echo "### 3/5 quant Q4_K_M"
llama-quantize "$F16" "$Q4" Q4_K_M

echo "### 4/5 smoke test"
llama-cli -m "$Q4" -p "User: Che Kateto, contame un chiste corto de programadores.\n\nAssistant:" -n 64 --temp 0.7 2>&1 | tail -20
echo "Listo: $Q4"
