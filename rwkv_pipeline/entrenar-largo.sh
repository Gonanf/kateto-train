#!/usr/bin/env bash
# ============================================================================
# Modo contexto largo (infctx) + modo clasico, mismo script.
# Por default replica el comportamiento de hoy (SFT corto, sin infctx).
# Modo largo:
#   TRAIN_TYPE=infctx CHUNK_CTX=512 CTX_LEN=1048576 \
#     DATA=data/largo/rwkv_kateto_largo_train.jsonl \
#     bash rwkv_pipeline/entrenar-largo.sh [ruta_dataset] [proj_dir]
# Knobs (env, todos opcionales):
#   TRAIN_TYPE=infctx|""  CHUNK_CTX=512  CTX_LEN=512  EPOCH_STEPS=1500  EPOCHS=2
#   BASE=...pth  N_LAYER=24  N_EMBD=1024  LOSS_MASK=qa  TAG=0.4b  MICRO_BSZ=1
# Portable: PROJECT se autodetecta (env PROJECT o raiz del repo); no hay rutas
# de maquina hardcodeadas salvo los defaults que se resuelven contra PROJECT.
# En Kaggle: PROJECT=/kaggle/working/kateto-train (o export PROJECT antes).
# ============================================================================
set -u
if [ -n "${PROJECT:-}" ] && [ -d "$PROJECT" ]; then
  T="$PROJECT"
else
  T="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
DATA="${1:-${DATA:-$T/data/rwkv_kateto_base_train_plus.jsonl}}"
PROJ="${2:-${PROJ:-$T/out/rwkv_kateto_base_plus_0.4b}}"
BASE="${BASE:-$T/models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth}"
N_LAYER="${N_LAYER:-24}"
N_EMBD="${N_EMBD:-1024}"
EPOCH_STEPS="${EPOCH_STEPS:-1500}"
EPOCHS="${EPOCHS:-2}"
CTX_LEN="${CTX_LEN:-${CTX:-512}}"
TRAIN_TYPE="${TRAIN_TYPE:-}"
CHUNK_CTX="${CHUNK_CTX:-512}"
LOSS_MASK="${LOSS_MASK:-qa}"
MICRO_BSZ="${MICRO_BSZ:-1}"
TAG="${TAG:-0.4b}"
LOG="${LOG:-$T/out/entrenar-largo-$TAG.log}"

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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
  --lr_init 2e-4 --lr_final 2e-5 \
  --accelerator gpu --precision bf16 --devices 1 --strategy auto --grad_cp 1 \
  --my_testing x070 --op triton \
  --peft pissa --pissa_config '{"pissa_r":32,"svd_niter":4}' \
  --train_parts '["time","ln"]' \
  --merge 1 >> "$LOG" 2>&1
RC=$?
echo "=== FIN $(date -Is) RC=$RC ===" >> "$LOG"
exit $RC
