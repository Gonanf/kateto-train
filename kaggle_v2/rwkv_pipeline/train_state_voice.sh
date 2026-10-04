#!/bin/bash
set -e

# ==============================================================================
# PIPELINE RWKV: CAPA 2 - STATE TUNING (VOCES / PERSONALIDADES)
# Entrena ÚNICAMENTE el vector de estado inicial S_0 (100% de pesos congelados)
# Uso: bash rwkv_pipeline/train_state_voice.sh [seco|streamer|jane|doktor|whisperer] [ruta_al_modelo_base.pth]
# ==============================================================================

VOICE="${1:-seco}"
case "$VOICE" in
    seco|streamer|jane|doktor|whisperer) ;;
    *) echo "ERROR: voz '$VOICE' inválida. Uso: [seco|streamer|jane|doktor|whisperer]"; exit 1 ;;
esac
# Épocas por voz: contrato config/state_tuning.yaml (SPEC §2.3)
case "$VOICE" in
    seco) EPOCHS=5 ;;
    streamer|doktor) EPOCHS=15 ;;
    jane) EPOCHS=12 ;;
    whisperer) EPOCHS=10 ;;
esac
PROJECT="${PROJECT:-/kaggle/working/kateto-train}"
if [ ! -d "$PROJECT" ]; then
    PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$PROJECT"

# Modelo base: usa el resultado de Capa 1 si existe, o el base descargado
if [ -d "$PROJECT/out/rwkv_kateto_base" ] && ls "$PROJECT/out/rwkv_kateto_base"/*.pth 1> /dev/null 2>&1; then
    DEFAULT_MODEL=$(ls -t "$PROJECT/out/rwkv_kateto_base"/*.pth | head -n 1)
elif [ -d "/kaggle/working/out/rwkv_kateto_base" ] && ls "/kaggle/working/out/rwkv_kateto_base"/*.pth 1> /dev/null 2>&1; then
    DEFAULT_MODEL=$(ls -t "/kaggle/working/out/rwkv_kateto_base"/*.pth | head -n 1)
elif [ -f "/kaggle/working/models/rwkv7-g1j-2.9b-20260831-ctx16384.pth" ]; then
    DEFAULT_MODEL="/kaggle/working/models/rwkv7-g1j-2.9b-20260831-ctx16384.pth"
else
    DEFAULT_MODEL="$PROJECT/models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth"
fi

MODEL_PATH="${2:-$DEFAULT_MODEL}"
DATA_FILE="$PROJECT/data/rwkv_kateto_${VOICE}.jsonl"
PROJ_OUT="$PROJECT/out/rwkv_states/$VOICE"

mkdir -p "$PROJ_OUT"

if [ ! -f "$DATA_FILE" ]; then
    echo "ERROR: No existe el dataset para la voz: $DATA_FILE"
    echo "Ejecutá primero: python3 rwkv_pipeline/prepare_datasets.py"
    exit 1
fi

# Detección automática de arquitectura según el archivo de pesos o variables de entorno
if [ -n "$N_LAYER" ] && [ -n "$N_EMBD" ]; then
    MICRO_BSZ="${MICRO_BSZ:-1}"
elif [[ "$MODEL_PATH" == *"0.4b"* || "$MODEL_PATH" == *"0.4B"* ]]; then
    N_LAYER=24
    N_EMBD=1024
    MICRO_BSZ=1
elif [[ "$MODEL_PATH" == *"1.5b"* || "$MODEL_PATH" == *"1.5B"* ]]; then
    N_LAYER=24
    N_EMBD=2048
    MICRO_BSZ=1
elif [[ "$MODEL_PATH" == *"2.9b"* || "$MODEL_PATH" == *"2.9B"* || "$MODEL_PATH" == *"rwkv_kateto_base"* || "$MODEL_PATH" == *"kateto"* ]]; then
    N_LAYER=32
    N_EMBD=2560
    MICRO_BSZ=1
elif [[ "$MODEL_PATH" == *"7.2b"* || "$MODEL_PATH" == *"7.2B"* ]]; then
    N_LAYER=32
    N_EMBD=4096
    MICRO_BSZ=1
else
    N_LAYER=32
    N_EMBD=2560
    MICRO_BSZ=1
fi

# Saltear si la voz ya fue entrenada completamente
FINAL_CKPT="$PROJ_OUT/rwkv-$((EPOCHS - 1)).pth"
if [ -f "$FINAL_CKPT" ] || [ -f "$PROJ_OUT/rwkv-${EPOCHS}.pth" ]; then
    echo "================================================================="
    echo ">>> VOZ [$VOICE] YA ENTRENADA ($FINAL_CKPT) — SALTEANDO <<<"
    echo "================================================================="
    exit 0
fi

# Micro batch size: 4 para optimizar tensor cores y reducir pasos
MICRO_BSZ="${MICRO_BSZ:-4}"
GRAD_CP="${GRAD_CP:-0}"
DEVICES="${DEVICES:-1}"
STRATEGY="${STRATEGY:-auto}"

echo "================================================================="
echo ">>> INICIANDO CAPA 2: STATE-TUNING PARA VOZ [$VOICE] <<<"
echo "Modelo Base: $MODEL_PATH"
echo "Layers:      $N_LAYER | Embd: $N_EMBD | MicroBSZ: $MICRO_BSZ | GradCP: $GRAD_CP"
echo "Dataset Voz: $DATA_FILE"
echo "Épocas:      $EPOCHS"
echo "Output:      $PROJ_OUT"
echo "================================================================="

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

if [ -x "$PROJECT/venv-unsloth-qwen/bin/python" ]; then
    PYTHON_EXEC="$PROJECT/venv-unsloth-qwen/bin/python"
else
    PYTHON_EXEC="${PYTHON_EXEC:-python3}"
fi

cd "$PROJECT/RWKV-PEFT"
"$PYTHON_EXEC" train.py \
    --load_model "$MODEL_PATH" \
    --proj_dir "$PROJ_OUT" \
    --data_file "$DATA_FILE" \
    --data_type jsonl \
    --loss_mask qa \
    --vocab_size 65536 \
    --n_layer $N_LAYER --n_embd $N_EMBD \
    --ctx_len 512 --micro_bsz $MICRO_BSZ \
    --epoch_steps 500 --epoch_count $EPOCHS --epoch_save 1 \
    --lr_init 1e-2 --lr_final 1e-4 \
    --accelerator gpu --precision bf16 \
    --devices $DEVICES --strategy $STRATEGY --grad_cp $GRAD_CP \
    --my_testing "x070" \
    --op fla \
    --peft state

echo "================================================================="
echo ">>> STATE-TUNING COMPLETADO <<<"
echo "Archivo de estado generado en: $PROJ_OUT"
echo "================================================================="
