#!/bin/bash
set -e

# ==============================================================================
# PIPELINE RWKV: CAPA 1 - FINE-TUNING BASE (PiSSA r=16 niter=4)
# Régimen G1k 1.5B (ctx25600): lr=1.0e-05, ~400 steps, una pasada.
# infctx con chunk_ctx: 256 y ctx_len: 1024 (detach entre chunks).
# Compuertas de decaimiento temporal (w0, w1, w2 / time_decay): CONGELADAS.
# ==============================================================================

PROJECT="/run/media/chaos/terciario/proyectos/kateto-train"
cd "$PROJECT"

MODEL_PATH="${1:-/run/media/chaos/secundario/models/rwkv7-1.5b-g1k/rwkv7-g1k-1.5b-20260930-ctx25600.pth}"
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
    echo "Modelo no reconocido por nombre, usando defaults para 1.5B G1k"
    N_LAYER=24
    N_EMBD=2048
    MICRO_BSZ=1
fi

echo "================================================================="
echo ">>> INICIANDO CAPA 1: RWKV-7 G1k BASE PiSSA FINE-TUNING <<<"
echo "Modelo:   $MODEL_PATH"
echo "Layers:   $N_LAYER | Embd: $N_EMBD | MicroBSZ: $MICRO_BSZ"
echo "Output:   $PROJ_OUT"
echo "Dataset:  $DATA_FILE"
echo "================================================================="

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

PYTHON_EXEC="${PYTHON_EXEC:-$PROJECT/venv-unsloth-qwen/bin/python}"

cd "$PROJECT/RWKV-PEFT"
if ! "$PYTHON_EXEC" train.py \
    --load_model "$MODEL_PATH" \
    --proj_dir "$PROJ_OUT" \
    --data_file "$DATA_FILE" \
    --data_type jsonl \
    --loss_mask qa \
    --vocab_size 65536 \
    --n_layer $N_LAYER --n_embd $N_EMBD \
    --train_type infctx --chunk_ctx 256 --ctx_len 1024 --micro_bsz $MICRO_BSZ \
    --epoch_steps 400 --epoch_count 1 --epoch_save 1 \
    --lr_init 1e-5 --lr_final 1e-6 \
    --accelerator gpu --precision bf16 \
    --devices 1 --strategy auto --grad_cp 1 \
    --my_testing "x070" \
    --op triton \
    --peft pissa \
    --peft_config "{}" \
    --pissa_config '{"pissa_load":"", "pissa_init":"", "pissa_r":16, "svd_niter":4}' \
    --merge 1; then
    echo "pissa dispatch failed at $(date -u +%FT%TZ), degrading to LoRA-r16" > "$PROJ_OUT/pissa_fallback_reason.txt"
    echo ">>> PISSA FALLÓ, REINTENTANDO COMO LoRA-r16 (razón en pissa_fallback_reason.txt) <<<"
    "$PYTHON_EXEC" train.py \
        --load_model "$MODEL_PATH" \
        --proj_dir "$PROJ_OUT" \
        --data_file "$DATA_FILE" \
        --data_type jsonl \
        --loss_mask qa \
        --vocab_size 65536 \
        --n_layer $N_LAYER --n_embd $N_EMBD \
        --train_type infctx --chunk_ctx 256 --ctx_len 1024 --micro_bsz $MICRO_BSZ \
        --epoch_steps 400 --epoch_count 1 --epoch_save 1 \
        --lr_init 1e-5 --lr_final 1e-6 \
        --accelerator gpu --precision bf16 \
        --devices 1 --strategy auto --grad_cp 1 \
        --my_testing "x070" \
        --op triton \
        --peft lora \
        --peft_config "{\"r\":16, \"lora_alpha\":32, \"lora_dropout\":0.01}" \
        --merge 1
fi

echo ">>> CAPA 1 COMPLETADA CON ÉXITO: Modelo fusionado en $PROJ_OUT <<<"
