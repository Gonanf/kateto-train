#!/bin/bash
# Arranque ROCm para inferencia Kateto: el override de gfx DEBE exportarse
# ANTES de que python importe torch (torch cachea is_available al importar).
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/venv-unsloth-qwen/bin/python" "$DIR/rwkv_pipeline/infer_kateto.py" "$@"
