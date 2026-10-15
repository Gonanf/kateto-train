#!/usr/bin/env bash
# ============================================================================
# Modo contexto largo (infctx) + modo clasico, mismo script.
# Régimen G1k 1.5B (ctx25600): chunk_ctx=256, ctx_len=1024, lr=1e-5, r=16, 400 steps.
# Compuertas de decaimiento congeladas (sin --train_parts time) para preservar
# el contexto preentrenado de 25.600 tokens.
# ============================================================================
set -u
if [ -n "${PROJECT:-}" ] && [ -d "$PROJECT" ]; then
  T="$PROJECT"
else
  T="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
DATA="${1:-${DATA:-$T/data/rwkv_kateto_base_train_plus.jsonl}}"
PROJ="${2:-${PROJ:-$T/out/rwkv_kateto_base_plus_1.5b_g1k}}"
BASE="${BASE:-/run/media/chaos/secundario/models/rwkv7-1.5b-g1k/rwkv7-g1k-1.5b-20260930-ctx25600.pth}"
N_LAYER="${N_LAYER:-24}"
N_EMBD="${N_EMBD:-2048}"
EPOCH_STEPS="${EPOCH_STEPS:-400}"
EPOCHS="${EPOCHS:-1}"
CTX_LEN="${CTX_LEN:-${CTX:-1024}}"
TRAIN_TYPE="${TRAIN_TYPE:-infctx}"
CHUNK_CTX="${CHUNK_CTX:-256}"
LOSS_MASK="${LOSS_MASK:-qa}"
MICRO_BSZ="${MICRO_BSZ:-1}"
TAG="${TAG:-1.5b_g1k}"
LOG="${LOG:-$T/out/entrenar-largo-$TAG.log}"

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

if [ -x "$T/venv-unsloth-qwen/bin/python" ]; then
  PY="$T/venv-unsloth-qwen/bin/python"
else
  PY="${PYTHON_EXEC:-python3}"
fi

mkdir -p "$PROJ" "$PROJ/hist"

STAMP=$(date +%Y%m%d-%H%M)
for f in "$PROJ"/rwkv-*.pth; do
  [ -e "$f" ] || continue
  mv -f "$f" "$PROJ/hist/$(basename "$f" .pth)-$STAMP.pth"
done
ls -1t "$PROJ"/hist/*.pth 2>/dev/null | tail -n +3 | xargs -r rm -f

EXTRA=()
if [ -n "$TRAIN_TYPE" ]; then
  EXTRA+=(--train_type "$TRAIN_TYPE" --chunk_ctx "$CHUNK_CTX")
fi

: > "$LOG"
{
  echo "=== INICIO $(date -Is) ==="
  echo "dataset  : $DATA ($(wc -l < "$DATA") lineas)"
  echo "proj_dir : $PROJ"
  echo "base     : $BASE"
  echo "arch     : n_layer=$N_LAYER n_embd=$N_EMBD"
  echo "steps    : epoch_steps=$EPOCH_STEPS x $EPOCHS epochs"
  echo "ctx      : ctx_len=$CTX_LEN train_type=${TRAIN_TYPE:-<legacy>} chunk_ctx=$CHUNK_CTX"
} >> "$LOG"

cd "$T/RWKV-PEFT" || exit 1
"$PY" -u train.py \
  --load_model "$BASE" \
  --proj_dir "$PROJ" \
  --data_file "$DATA" \
  --data_type jsonl --loss_mask "$LOSS_MASK" --vocab_size 65536 \
  --n_layer "$N_LAYER" --n_embd "$N_EMBD" \
  --ctx_len "$CTX_LEN" --micro_bsz "$MICRO_BSZ" \
  "${EXTRA[@]}" \
  --epoch_steps "$EPOCH_STEPS" --epoch_count "$EPOCHS" --epoch_save 1 \
  --lr_init 1e-5 --lr_final 1e-6 \
  --accelerator gpu --precision bf16 --devices 1 --strategy auto --grad_cp 1 \
  --my_testing x070 --op triton \
  --peft pissa --pissa_config '{"pissa_r":16,"svd_niter":4}' \
  --merge 1 >> "$LOG" 2>&1
RC=$?
echo "=== FIN $(date -Is) RC=$RC ===" >> "$LOG"
exit $RC
