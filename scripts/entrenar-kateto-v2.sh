#!/usr/bin/env bash
set -euo pipefail

T="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$T"

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1

CFG="$T/config/kateto_v2.yaml"
CKPT="$(yq -r '.model.base_checkpoint' "$CFG")"
OUT="$(yq -r '.training.output_dir' "$CFG")"
mkdir -p "$OUT"

PY="$T/venv-unsloth-qwen/bin/python"

echo "=== Kateto v2 training start ==="
echo "Config: $CFG"
echo "Base: $CKPT"
echo "Out: $OUT"

# 1. PiSSA SFT
echo "[1/3] PiSSA SFT"
$PY rwkv_pipeline/train_pissa.py \
  --config "$CFG" \
  --output_dir "$OUT/sft"

# 2. ROSA head training
echo "[2/3] ROSA head"
$PY rwkv_pipeline/rosa_train_corto.py \
  --ckpt "$OUT/sft/rwkv-0.pth" \
  --data "$(yq -r '.data.sft.path' "$CFG")" \
  --out "$OUT/rosa_head" \
  --steps 200 \
  --ctx-len 512 \
  --lr 3e-3

# 3. ORPO alignment
echo "[3/3] ORPO alignment"
$PY rwkv_pipeline/orpo_trainer.py \
  --model "$OUT/sft/rwkv-0.pth" \
  --pairs "$(yq -r '.data.orpo_pairs' "$CFG")" \
  --lambda_or "$(yq -r '.orpo.lambda_or' "$CFG")" \
  --epochs "$(yq -r '.orpo.epochs' "$CFG")"

echo "=== Kateto v2 done ==="
echo "Final artifacts in $OUT"
