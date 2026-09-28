#!/usr/bin/env bash
# ============================================================================
# Loop de entrenamiento local (Capa 1 sobre datos ya generados).
# Gatilla un reentrenamiento SOLO si:
#   - hay >= MIN_NUEVOS muestras nuevas respecto de la ultima corrida
#   - no hay ya una corrida activa
#   - la GPU esta libre (nadie mas usando VRAM)
# Pensado para systemd timer (kateto-entrenar.timer, cada 4h).
# ============================================================================
set -u
MED="${KATETO_MEDICION:-/run/media/chaos/terciario/proyectos/kateto-medicion}"
T="${KATETO_HOME:-/run/media/chaos/terciario/proyectos/kateto-train}"
STATE="$MED/.entrenar-state.json"
LOG="$MED/entrenar-loop.log"
LOCK="$MED/.entrenar-loop.lock"
MIN_NUEVOS="${MIN_NUEVOS:-300}"   # 50 era ridiculo: disparaba 2 epocas por 65 muestras
VRAM_MAX="${VRAM_MAX:-600}"          # MiB ocupados que toleramos antes de arrancar

# --check: solo evalua el guard de tamano vs VRAM (no lanza, no toca el dataset).
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

log() { echo "[$(date -Is)] $*" >> "$LOG"; }

exec 9>"$LOCK"
flock -n 9 || { log "SKIP: otra instancia del loop corriendo"; exit 0; }

if systemctl --user is-active --quiet kateto-capa1-plus && [ "$CHECK_ONLY" -eq 0 ]; then
    log "SKIP: kateto-capa1-plus ya esta activa"
    exit 0
fi

# --- rearmar dataset (atomico) ---------------------------------------------
if [ "$CHECK_ONLY" -eq 0 ]; then
    if ! python3 "$MED/build-train-set.py" >> "$LOG" 2>&1; then
        log "ERROR: fallo build-train-set.py"
        exit 1
    fi
    TOTAL=$(python3 -c "
print(sum(1 for l in open('$T/data/rwkv_kateto_base_train_plus.jsonl',encoding='utf-8') if l.strip()))
")
    PREV=$(python3 -c "
import json,os
p='$STATE'
print(json.load(open(p)).get('total',0) if os.path.exists(p) else 0)
")
    NUEVOS=$(( TOTAL - PREV ))

    if [ "$NUEVOS" -lt "$MIN_NUEVOS" ]; then
        log "SKIP: $NUEVOS nuevas (< $MIN_NUEVOS). total=$TOTAL prev=$PREV"
        exit 0
    fi

    # --- GPU libre? -----------------------------------------------------------
    VRAM=$(python3 -c "
import glob
print(max([int(open(p).read())/1048576 for p in glob.glob('/sys/class/drm/card*/device/mem_info_vram_used')] or [0]))
" 2>/dev/null || echo 0)
    VRAM=${VRAM%%.*}
    if [ "$VRAM" -gt "$VRAM_MAX" ]; then
        log "SKIP: GPU ocupada (${VRAM} MiB > ${VRAM_MAX}). Se reintenta en el proximo tick."
        exit 0
    fi
fi

# --- guard fix-126: elegir el modelo por tamano vs VRAM total -----------------
# La 1.5B (ckpt 2914 MiB) NO entra en los 4080 MiB del box (pico medido 4071;
# OOM a las 17:31). La 0.4B (ckpt 860 MiB) entra holgada. Elegimos el modelo mas
# grande cuyo footprint (checkpoint + overhead de entrenamiento) caiga dentro de
# la VRAM total menos el margen. Si ninguno entra, loguea un SKIP explicito y no
# lanza nada (asi no quema ~7 min por intento ni alimenta el retry del OOM).
TRAIN_OVH="${TRAIN_OVH:-1400}"   # MiB de overhead: ctx hip + gradientes + activaciones
guard_sel() {
    python3 - <<PYEOF
import glob, os, sys
TRAIN_OVH=$TRAIN_OVH
VRAM_MAX=$VRAM_MAX
def _mib(p):
    try: return int(open(p).read()) // (1024*1024)
    except Exception: return 0
tots=[_mib(p) for p in glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')]
used=max([_mib(p) for p in glob.glob('/sys/class/drm/card*/device/mem_info_vram_used')] or [0])
max_fit = (max(tots)-VRAM_MAX) if tots else 0
print(f"reporte: VRAM total={max(tots) if tots else 0} MiB, usada={used} MiB, max_fit={max_fit} MiB")
if not tots:
    print("SKIP: modelo no entra en la VRAM (no se pudo leer la VRAM total)")
    sys.exit(2)
cands = [
  # nombre, base pth, n_embd, n_layer, tag, proj, ctx
  ("1.5B", "$T/models/rwkv7-1.5b/rwkv7-g1j-1.5b-20260831-ctx16384.pth", 2048, 24, "1.5b", "$T/out/rwkv_kateto_base_plus_1.5b", 256),
  ("0.4B", "$T/models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth",    1024, 24, "0.4b", "$T/out/rwkv_kateto_base_plus_0.4b", 512),
]
selected = None
for name, base, emb, layer, tag, proj, ctx in cands:
    if not os.path.exists(base):
        continue
    ckpt_mib = os.path.getsize(base) // (1024*1024)
    need = ckpt_mib + TRAIN_OVH
    if need <= max_fit:
        print(f"  [{name}] ckpt={ckpt_mib} MiB + ovh={TRAIN_OVH} = {need} MiB <= {max_fit} MiB libres -> OK")
        if selected is None:
            selected = (name, base, emb, layer, tag, proj, ctx, need, max_fit, ckpt_mib)
    else:
        print(f"SKIP: {name} no entra en la VRAM ({need/1024:.2f} GB > {max_fit/1024:.2f} GB libres)")
if selected is None:
    print("PYNONE")
    sys.exit(2)
print("PYOK")
for f in selected:
    print(f)
PYEOF
}
MODEL_SEL=$(guard_sel)
GRC=$?
# loguear solo las lineas legibles (reporte / [modelo] OK / SKIP), no el bloque PYOK
echo "$MODEL_SEL" | grep -E 'reporte:|^[[:space:]]*\[|^SKIP:' | while IFS= read -r l; do log "$l"; done

if [ "$CHECK_ONLY" -eq 1 ]; then
    exit 0
fi
if [ $GRC -ne 0 ]; then
    exit 0   # el SKIP por tamano ya se logueo; no se lanza nada este ciclo
fi
mapfile -t SEL < <(echo "$MODEL_SEL" | awk '/^PYOK$/{f=1; next} f')
NAME="${SEL[0]}"; BASE="${SEL[1]}"; EMBD="${SEL[2]}"; LAYER="${SEL[3]}"
TAG="${SEL[4]}"; PROJ="${SEL[5]}"; CTX="${SEL[6]}"; NEED="${SEL[7]}"; MAXFIT="${SEL[8]}"

# no reintentar el mismo modelo que acaba de morir por OOM (mismo ciclo)
TRAIN_LOG="$T/out/entrenar-local-$TAG.log"
if [ -f "$TRAIN_LOG" ] \
   && ! systemctl --user is-active --quiet kateto-capa1-plus \
   && tail -120 "$TRAIN_LOG" 2>/dev/null | grep -qE 'OutOfMemoryError|HIP out of memory|CUDA out of memory|out of memory'; then
    log "SKIP: $NAME acaba de fallar por OOM (ultimo log) — no se reintenta en este ciclo"
    exit 0
fi

# --- lanzar el modelo elegido por tamano ------------------------------------
log "LANZO Capa 1 $NAME: nuevas=$NUEVOS total=$TOTAL vram=${VRAM:-?}MiB footprint=${NEED}MiB (max ${MAXFIT}MiB)"
systemd-run --user --unit=kateto-capa1-plus --collect -p MemoryMax=infinity \
    --setenv=BASE="$BASE" \
    --setenv=N_LAYER="$LAYER" --setenv=N_EMBD="$EMBD" \
    --setenv=EPOCH_STEPS=1500 --setenv=EPOCHS=2 --setenv=TAG="$TAG" --setenv=CTX="$CTX" \
    /usr/bin/bash "$MED/entrenar-local.sh" \
    "$T/data/rwkv_kateto_base_train_plus.jsonl" \
    "$PROJ" >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
    python3 -c "
import json
json.dump({'total':$TOTAL,'ts':'$(date -Is)','last_model':'$NAME'},open('$STATE','w'))
"
    log "OK: lanzado, total registrado=$TOTAL"
else
    log "ERROR: systemd-run rc=$RC"
fi
exit $RC
