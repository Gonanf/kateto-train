#!/bin/bash
set -e

PROJECT="${KATETO_HOME:-/run/media/chaos/terciario/proyectos/kateto-train}"
cd "$PROJECT"

echo "================================================================="
echo ">>> MASTER PIPELINE RWKV-7: BASE FT + STATE VOICES + ROSA <<<"
echo "================================================================="

PYTHON_EXEC="${PYTHON_EXEC:-$PROJECT/venv-unsloth-qwen/bin/python}"

# Paso 1: Preparar datasets
echo -e "\n>>> 1/4 PREPARANDO DATASETS <<<"
"$PYTHON_EXEC" "$PROJECT/rwkv_pipeline/prepare_datasets.py"

# Paso 2: Entrenar Capa 1 (Base LoRA)
echo -e "\n>>> 2/4 ENTRENANDO CAPA 1: BASE LORA (TOOL CALLING + DOMINIO) <<<"
bash "$PROJECT/rwkv_pipeline/train_lora_base.sh"

# Paso 3: Entrenar Capa 2 (Voces)
echo -e "\n>>> 3/4 ENTRENANDO CAPA 2: STATE-TUNING DE VOCES <<<"
echo "--- Entrenando Voz: Kateto Seco ---"
bash "$PROJECT/rwkv_pipeline/train_state_voice.sh" seco

echo "--- Entrenando Voz: Kateto Streamer ---"
bash "$PROJECT/rwkv_pipeline/train_state_voice.sh" streamer

# Paso 4: Test de Inferencia
echo -e "\n>>> 4/4 EJECUTANDO TEST DE INFERENCIA Y SWAP DE ESTADOS <<<"
"$PYTHON_EXEC" "$PROJECT/rwkv_pipeline/infer_kateto.py" --voice seco --prompt "Che Kateto, qué hora es y qué hacés acá?"
"$PYTHON_EXEC" "$PROJECT/rwkv_pipeline/infer_kateto.py" --voice streamer --prompt "Che Kateto, qué hora es y qué hacés acá?"

echo "\n================================================================="
echo ">>> PIPELINE RWKV COMPLETO FINALIZADO CON ÉXITO <<<"
echo "================================================================="
