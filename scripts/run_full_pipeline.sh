#!/bin/bash
set -e

PROJECT="/run/media/chaos/terciario/proyectos/kateto-train"
cd "$PROJECT"

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UNSLOTH_CE_LOSS_N_CHUNKS=16
PYTHON="$PROJECT/venv-unsloth-qwen/bin/python"

echo "========================================================"
echo ">>> PASO 1/3: CONTINUANDO INGESTA Y ASR DE YOUTUBE <<<"
echo "========================================================"
python3 "$PROJECT/scripts/ingest_youtube_streams.py"

echo "========================================================"
echo ">>> PASO 2/3: RECONSOLIDANDO DATASET COMBINADO V2 <<<"
echo "========================================================"
python3 "$PROJECT/scripts/build_dataset_v2.py"

echo "========================================================"
echo ">>> PASO 3/3: INICIANDO ENTRENAMIENTO STAGE 2 LORA <<<"
echo "========================================================"
"$PYTHON" "$PROJECT/train_qwen_2stage.py" --all

echo "========================================================"
echo ">>> PIPELINE COMPLETO FINALIZADO CON ÉXITO <<<"
echo "========================================================"
