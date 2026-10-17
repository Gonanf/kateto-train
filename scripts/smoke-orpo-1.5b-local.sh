#!/usr/bin/env bash
# ==============================================================================
# scripts/smoke-orpo-1.5b-local.sh
# ORPO quirurgico sobre un checkpoint mergeado (receta del Senior, 2026-10-17).
# Base por defecto: el v7. 360 pares / 180 steps / ~6-10 min en RX 6500 XT.
# ==============================================================================
set -u
set -o pipefail

SMOKE_CHUNK="${SMOKE_CHUNK:-256}"
SMOKE_CTX="${SMOKE_CTX:-512}"
SMOKE_STEPS="${SMOKE_STEPS:-180}"
SMOKE_BSZ="${SMOKE_BSZ:-1}"
SMOKE_ACCUM="${SMOKE_ACCUM:-4}"
SMOKE_LR="${SMOKE_LR:-5e-6}"
SMOKE_LR_FINAL="${SMOKE_LR_FINAL:-1e-6}"
SMOKE_LAMBDA="${SMOKE_ORPO_LAMBDA:-0.2}"
SMOKE_OUT="${SMOKE_OUT:-/run/media/chaos/secundario/kateto-orpo-v7}"
SMOKE_PROJ="${SMOKE_PROJ:-$SMOKE_OUT/proj-c$SMOKE_CHUNK}"
SMOKE_LOG="${SMOKE_LOG:-$SMOKE_OUT/orpo-c$SMOKE_CHUNK.log}"

REPO_DIR="${REPO_DIR:-/run/media/chaos/terciario/proyectos/kateto-train}"
PY="$REPO_DIR/venv-unsloth-qwen/bin/python"
BASE_MODEL="${BASE_MODEL:-/run/media/chaos/secundario/kateto-v7-epoca1/proj-c256/rwkv-0.pth}"
SMOKE_DATA="${SMOKE_DATA:-$REPO_DIR/data/kateto_orpo_v7.jsonl}"
SCRIPT_PATH="$(realpath "$0" 2>/dev/null || echo "$0")"

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1

mkdir -p "$SMOKE_OUT"
if [ -e "$SMOKE_PROJ" ]; then
  SMOKE_PROJ="${SMOKE_PROJ}_$(date +%Y%m%d_%H%M%S)_$$"
fi
mkdir -p "$SMOKE_PROJ"
mkdir -p "$(dirname "$SMOKE_LOG")"

if [ -f "$SMOKE_DATA" ]; then
  DATASET_LINES=$(wc -l < "$SMOKE_DATA" | tr -d ' ')
else
  DATASET_LINES=0
fi
if [ ! -f "$BASE_MODEL" ]; then
  echo "FALTA el checkpoint base: $BASE_MODEL" >&2
  exit 2
fi

emit_result() {
  local rc_val="$1"
  local wall_val="$2"
  "$PY" -c "
import json, sys
data = {'script': sys.argv[1], 'steps': int(sys.argv[2]), 'rc': None if sys.argv[3]=='null' else int(sys.argv[3]),
        'wall_s': None if sys.argv[4]=='null' else int(sys.argv[4]), 'log': sys.argv[5],
        'proj_dir': sys.argv[6], 'pairs': int(sys.argv[7]), 'base': sys.argv[8], 'lambda_or': sys.argv[9]}
line = 'RESULT_JSON: ' + json.dumps(data)
print(line)
with open(sys.argv[5], 'a') as f:
    f.write(line + '\n')
" "$SCRIPT_PATH" "$SMOKE_STEPS" "$rc_val" "$wall_val" "$SMOKE_LOG" "$SMOKE_PROJ" "$DATASET_LINES" "$BASE_MODEL" "$SMOKE_LAMBDA"
}

emit_result "null" "0"

cd "$REPO_DIR/RWKV-PEFT" || exit 1
START_TIME=$(date +%s)
set +e
"$PY" -u train.py \
  --load_model "$BASE_MODEL" \
  --proj_dir "$SMOKE_PROJ" \
  --data_file "$SMOKE_DATA" \
  --data_type orpo --vocab_size 65536 \
  --orpo_lambda "$SMOKE_LAMBDA" --orpo_detach_rejected ${SMOKE_DETACH:-1} \
  --n_layer 24 --n_embd 2048 \
  --ctx_len "$SMOKE_CTX" --train_type ${SMOKE_TRAINTYPE:-none} \
  --micro_bsz "$SMOKE_BSZ" --epoch_steps "$SMOKE_STEPS" --epoch_count 1 --epoch_save 1 \
  --lr_init "$SMOKE_LR" --lr_final "$SMOKE_LR_FINAL" \
  --accelerator gpu --precision ${SMOKE_PREC:-fp32} --devices 1 --strategy auto --grad_cp ${SMOKE_GRADCP:-0} \
  --my_testing x070 --op triton --num_workers 2 \
  --peft pissa --pissa_config '{"pissa_r":16,"svd_niter":4}' \
  --train_parts '["time","ln"]' --freeze_decay 1 --merge ${SMOKE_MERGE:-0} \
  --accumulate_grad_batches "$SMOKE_ACCUM" 2>&1 | tee -a "$SMOKE_LOG"
RC=${PIPESTATUS[0]}
END_TIME=$(date +%s)
WALL_S=$(( END_TIME - START_TIME ))

emit_result "$RC" "$WALL_S"
exit "$RC"
