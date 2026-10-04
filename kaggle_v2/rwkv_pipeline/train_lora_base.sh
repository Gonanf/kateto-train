#!/bin/bash
set -e

# ==============================================================================
# PIPELINE RWKV: CAPA 1 - FINE-TUNING BASE (LORA)
# Aprende: sintaxis de Tool Calling, base Hermes, dialecto rioplatense general
# ==============================================================================

PROJECT="/run/media/chaos/terciario/proyectos/kateto-train"
cd "$PROJECT"

MODEL_PATH="${1:-$PROJECT/models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth}"
PROJ_OUT="$PROJECT/out/rwkv_kateto_base"
DATA_FILE="$PROJECT/data/rwkv_kateto_base_train.jsonl"

mkdir -p "$PROJ_OUT"

# Detección automática de arquitectura según el archivo de pesos
if [[ "$MODEL_PATH" == *"0.4b"* || "$MODEL_PATH" == *"0.4B"* ]]; then
    N_LAYER=24
    N_EMBD=1024
    MICRO_BSZ=1
elif [[ "$MODEL_PATH" == *"1.5b"* || "$MODEL_PATH" == *"1.5B"* ]]; then
    N_LAYER=24
    N_EMBD=2048
    MICRO_BSZ=1
elif [[ "$MODEL_PATH" == *"2.9b"* || "$MODEL_PATH" == *"2.9B"* ]]; then
    N_LAYER=32
    N_EMBD=2560
    MICRO_BSZ=1
elif [[ "$MODEL_PATH" == *"7.2b"* || "$MODEL_PATH" == *"7.2B"* ]]; then
    N_LAYER=32
    N_EMBD=4096
    MICRO_BSZ=1
else
    echo "Modelo no reconocido por nombre, usando defaults para 0.4B"
    N_LAYER=24
    N_EMBD=1024
    MICRO_BSZ=1
fi

echo "================================================================="
echo ">>> INICIANDO CAPA 1: RWKV-7 BASE LORA FINE-TUNING <<<"
echo "Modelo:   $MODEL_PATH"
echo "Layers:   $N_LAYER | Embd: $N_EMBD | MicroBSZ: $MICRO_BSZ"
echo "Output:   $PROJ_OUT"
echo "Dataset:  $DATA_FILE"
echo "================================================================="

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

PYTHON_EXEC="${PYTHON_EXEC:-$PROJECT/venv-unsloth-qwen/bin/python}"

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
    --epoch_steps 750 --epoch_count 2 --epoch_save 1 \
    --lr_init 2e-4 --lr_final 2e-5 \
    --accelerator gpu --precision bf16 \
    --devices 1 --strategy auto --grad_cp 1 \
    --my_testing "x070" \
    --op triton \
    --peft lora \
    --peft_config "{\"r\":16, \"lora_alpha\":32, \"lora_dropout\":0.01}" \
    --merge 1

echo ">>> CAPA 1 COMPLETADA CON ÉXITO: Modelo fusionado en $PROJ_OUT <<<"
