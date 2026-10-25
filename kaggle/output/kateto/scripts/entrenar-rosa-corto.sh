#!/usr/bin/env bash
# ============================================================================
# ROSA corto: run ACOTADO para que el head ROSA tenga pesos con senal.
#
#   bash scripts/entrenar-rosa-corto.sh smoke     # 2 pasos, prueba que arranca
#   bash scripts/entrenar-rosa-corto.sh run       # el run corto de verdad
#   bash scripts/entrenar-rosa-corto.sh daemon    # lanza `run` detached
#
# El `daemon` es el que se usa para dejar el run sobrevivir la sesion:
#   systemd-run --user --unit=rosa-train --collect -p MemoryMax=infinity
# MemoryMax=infinity porque el cgroup de background de este host capea a 4 GiB
# (salida 137 = cgroup, no bug del run) y este run pide ~2.5 GB. TasksMax
# infinito porque triton abre hilos.
#
# Knobs (env): STEPS=200 SUBSET=64 CTX_LEN=512 LR=3e-3 DEVICE=cuda
#              OUT=out/rosa_head_corto UNIT=rosa-train
#              CKPT=<base .pth>  SMOKE_TIMEOUT=900  RUN_TIMEOUT=21600
#
# No toca el checkpoint ni el dataset: solo los lee. Escribe unicamente en $OUT.
# ============================================================================
set -euo pipefail

T="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$T"

MODE="${1:-run}"
CKPT="${CKPT:-$T/models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth}"
DATA="${DATA:-$T/data/rwkv_kateto_base_train_plus.jsonl}"
OUT="${OUT:-$T/out/rosa_head_corto}"
STEPS="${STEPS:-200}"
SUBSET="${SUBSET:-64}"
CTX_LEN="${CTX_LEN:-512}"
LR="${LR:-3e-3}"
DEVICE="${DEVICE:-cuda}"
UNIT="${UNIT:-rosa-train}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-900}"
RUN_TIMEOUT="${RUN_TIMEOUT:-21600}"

# El unico python con torch+peft+triton para ROCm en este repo (mismo que
# entrenar-largo.sh y out/opd/_run_generic.sh).
PY="${PY:-$T/venv-unsloth-qwen/bin/python}"
[ -x "$PY" ] || { echo "FALTA python: $PY" >&2; exit 1; }
[ -f "$CKPT" ] || { echo "FALTA checkpoint: $CKPT" >&2; exit 1; }
[ -f "$DATA" ] || { echo "FALTA dataset: $DATA" >&2; exit 1; }

rosa_run() {  # $1=steps  $2=subset  $3=out  $4=timeout
  mkdir -p "$3"
  timeout "$4" "$PY" -u rwkv_pipeline/rosa_train_corto.py \
    --ckpt "$CKPT" --data "$DATA" --out "$3" --rosa \
    --steps "$1" --subset "$2" --ctx-len "$CTX_LEN" --lr "$LR" --device "$DEVICE"
}

case "$MODE" in
  smoke)
    # 2 pasos sobre el mismo camino de codigo del run real (mismo student, mismo
    # loss-mask, mismo --rosa). Es la prueba de que arranca sin NaN; la
    # separacion con el run es solo el out dir, para no pisar el artefacto.
    rosa_run 2 8 "$T/out/rosa_head_smoke" "$SMOKE_TIMEOUT"
    ;;
  run)
    rosa_run "$STEPS" "$SUBSET" "$OUT" "$RUN_TIMEOUT"
    ;;
  daemon)
    if systemctl --user is-active --quiet "$UNIT"; then
      echo "ABORTA: la unit $UNIT ya esta activa" >&2
      exit 1
    fi
    mkdir -p "$OUT"
    : > "$OUT/train.log"
    # Los knobs se pasan explicitamente: systemd-run no hereda el env de este
    # shell, asi que sin esto la unit corre con los defaults (medido: pedimos
    # 300 pasos y corrieron 200).
    systemd-run --user --unit="$UNIT" --collect \
      -p MemoryMax=infinity -p TasksMax=infinity \
      -p WorkingDirectory="$T" \
      bash -lc "cd $T && MODE=run \
        STEPS=$STEPS SUBSET=$SUBSET CTX_LEN=$CTX_LEN LR=$LR \
        DEVICE=$DEVICE OUT=$OUT DATA=$DATA CKPT=$CKPT RUN_TIMEOUT=$RUN_TIMEOUT \
        bash scripts/entrenar-rosa-corto.sh run \
        >> $OUT/train.log 2>&1; echo \"EXIT=\$?\" >> $OUT/train.log"
    echo "lanzada $UNIT -> $OUT/train.log (steps=$STEPS subset=$SUBSET lr=$LR device=$DEVICE)"
    ;;
  *)
    echo "modo desconocido: $MODE (usar smoke|run|daemon)" >&2
    exit 2
    ;;
esac
