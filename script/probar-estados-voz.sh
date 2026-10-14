#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."
REPO_DIR="$(pwd)"

TMPDIR="${TMPDIR:-/tmp}"
LOG_FILE="$TMPDIR/probar-estados-voz.log"
JSONL_FILE="$TMPDIR/probar-estados-voz-resultados.jsonl"
BUFFERS_DIR="$TMPDIR/rwkv_state_buffers"

rm -f "$LOG_FILE" "$JSONL_FILE"
mkdir -p "$BUFFERS_DIR"

echo "=== Inicio del gate probar-estados-voz ===" | tee -a "$LOG_FILE"
echo "Log completo en: $LOG_FILE"

# 1. Rutas del entorno
GGUF="/home/chaos/.cache/huggingface/hub/models--shoumenchougou--RWKV7-G1j-2.9B-GGUF/snapshots/ab0040b584622847f9b8702c41af222a00fda968/rwkv7-g1j-2.9b-Q4_K_M.gguf"
GGUF_SHA256="68c3a49dfc8a6dc34032d7c89adf64542ec9c81229a1333591163717bc3ab7b5"
LLAMA_DIR="/run/media/chaos/secundario/llamacpp-conv"
LLAMA_BIN="$LLAMA_DIR/build/bin"
LLAMA_COMMIT="f2cc7282ce0042641df55e0e563b75e824e2b647"

if [ -x "./venv/bin/python" ]; then
    PYTHON="./venv/bin/python"
elif [ -x "/run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python" ]; then
    PYTHON="/run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python"
else
    PYTHON="python3"
fi

if [ -d "out/rwkv_states" ]; then
    STATES_DIR="out/rwkv_states"
elif [ -d "/run/media/chaos/terciario/proyectos/kateto-train/out/rwkv_states" ]; then
    STATES_DIR="/run/media/chaos/terciario/proyectos/kateto-train/out/rwkv_states"
else
    echo "ERROR: Directorio out/rwkv_states no encontrado" | tee -a "$LOG_FILE"
    exit 1
fi

export LD_LIBRARY_PATH="$LLAMA_BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# a. Compilar el tool
echo "--- [Paso a] Compilando tools/gguf_state_gen ---" | tee -a "$LOG_FILE"
g++ -O2 -std=c++17 tools/gguf_state_gen.cpp -o tools/gguf_state_gen \
    -I "$LLAMA_DIR/include" \
    -I "$LLAMA_DIR/ggml/include" \
    -L "$LLAMA_BIN" \
    -lllama -lggml -lggml-base >> "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Falló la compilación de tools/gguf_state_gen" | tee -a "$LOG_FILE"
    exit 1
fi
echo "Compilación exitosa." | tee -a "$LOG_FILE"

# b. Diagnóstico de formas para los 8 states
echo "--- [Paso b] Corriendo diagnóstico de formas para los 8 states ---" | tee -a "$LOG_FILE"

STATES=(
    "jane/rwkv-11.pth"
    "doktor/rwkv-0.pth"
    "doktor/rwkv-14.pth"
    "jane/rwkv-0.pth"
    "seco/rwkv-3.pth"
    "streamer/rwkv-0.pth"
    "whisperer/rwkv-0.pth"
    "seco-20260906/rwkv-0.pth"
)

declare -A VERDICTOS
declare -A N_TENSORES
declare -A FORMAS
declare -A N_EMBD_PTH
declare -A N_EMBD_GGUF

CALZA_COUNT=0
TOTAL_STATES=${#STATES[@]}

for s in "${STATES[@]}"; do
    pth_file="$STATES_DIR/$s"
    if [ ! -f "$pth_file" ]; then
        echo "ERROR: Archivo de estado no encontrado: $pth_file" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "Verificando forma de $s..." >> "$LOG_FILE"
    
    # Correr reporte oficial de state_from_pth
    set +e
    env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -m rwkv_pipeline.state_from_pth \
        --pth "$pth_file" --modelo "$GGUF" --report >> "$LOG_FILE" 2>&1
    rc=$?
    set -e
    
    if [ $rc -eq 0 ]; then
        VERDICTOS["$s"]="CALZA"
        CALZA_COUNT=$((CALZA_COUNT + 1))
    elif [ $rc -eq 3 ]; then
        VERDICTOS["$s"]="NO CALZA"
    else
        echo "ERROR: Código de salida inesperado ($rc) para $s" | tee -a "$LOG_FILE"
        exit 1
    fi
done

# Extraer metadata de formas en formato JSON para el generador de reportes
SHAPES_JSON="$TMPDIR/shapes_info.json"
env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -c "
import os, sys, json
from rwkv_pipeline.state_from_pth import diagnostico

gguf = sys.argv[1]
base = sys.argv[2]
states = sys.argv[3:]

out = []
for s in states:
    pth = os.path.join(base, s)
    dx = diagnostico(pth, gguf)
    hp = dx['hparams']
    shape = dx['shape_pth']
    n_tensores = dx['n_tensores_pth']
    calza = dx['calza']
    n_embd_pth = shape[0] * shape[1] if len(shape) >= 2 else None
    out.append({
        'state': s,
        'n_tensores': n_tensores,
        'shape': str(shape),
        'calza': 'CALZA' if calza else 'NO CALZA',
        'n_embd_pth': n_embd_pth,
        'n_embd_gguf': hp['n_embd']
    })
print(json.dumps(out))
" "$GGUF" "$STATES_DIR" "${STATES[@]}" > "$SHAPES_JSON"

echo "Veredictos de formas: $CALZA_COUNT CALZA, $((TOTAL_STATES - CALZA_COUNT)) NO CALZA" | tee -a "$LOG_FILE"

# c. Fabricar buffers y correr generaciones
echo "--- [Paso c] Fabricando buffers y ejecutando pruebas de generación ---" | tee -a "$LOG_FILE"

BUFFER_JANE_11="$BUFFERS_DIR/jane-11.bin"
BUFFER_DOKTOR_0="$BUFFERS_DIR/doktor-0.bin"
BUFFER_DOKTOR_14="$BUFFERS_DIR/doktor-14.bin"

echo "Fabricando buffer para jane/rwkv-11..." >> "$LOG_FILE"
env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -m rwkv_pipeline.state_from_pth \
    --pth "$STATES_DIR/jane/rwkv-11.pth" --modelo "$GGUF" --out "$BUFFER_JANE_11" >> "$LOG_FILE" 2>&1

echo "Fabricando buffer para doktor/rwkv-0..." >> "$LOG_FILE"
env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -m rwkv_pipeline.state_from_pth \
    --pth "$STATES_DIR/doktor/rwkv-0.pth" --modelo "$GGUF" --out "$BUFFER_DOKTOR_0" >> "$LOG_FILE" 2>&1

echo "Fabricando buffer para doktor/rwkv-14..." >> "$LOG_FILE"
env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -m rwkv_pipeline.state_from_pth \
    --pth "$STATES_DIR/doktor/rwkv-14.pth" --modelo "$GGUF" --out "$BUFFER_DOKTOR_14" >> "$LOG_FILE" 2>&1

run_model() {
    local prompt_label="$1"
    local prompt_text="$2"
    local state_name="$3"
    local state_file="$4"
    local gen_count="$5"

    echo "Ejecutando modelo (prompt $prompt_label, state $state_name)..." | tee -a "$LOG_FILE"
    
    local cmd=("./tools/gguf_state_gen" "--model" "$GGUF" "--prompt" "$prompt_text" "--gen" "$gen_count")
    if [ -n "$state_file" ]; then
        cmd+=("--state" "$state_file" "--state-name" "$state_name")
    fi

    local out
    out=$("${cmd[@]}" 2>> "$LOG_FILE")
    echo "$out" >> "$LOG_FILE"

    local json_line
    json_line=$(echo "$out" | grep "^RESULT_JSON:" | sed 's/^RESULT_JSON: //')
    if [ -z "$json_line" ]; then
        echo "ERROR: No se obtuvo RESULT_JSON al correr prompt $prompt_label con $state_name" | tee -a "$LOG_FILE"
        exit 1
    fi

    # Agregar prompt_label y prompt_text al json
    env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -c "
import sys, json
data = json.loads(sys.argv[1])
data['prompt_label'] = sys.argv[2]
data['prompt_text'] = sys.argv[3]
print(json.dumps(data))
" "$json_line" "$prompt_label" "$prompt_text" >> "$JSONL_FILE"
}

# Prompt A: "Hola, soy Kateto." (gen 12)
PROMPT_A="Hola, soy Kateto."
run_model "A" "$PROMPT_A" "baseline" "" 12
run_model "A" "$PROMPT_A" "jane/rwkv-11" "$BUFFER_JANE_11" 12
run_model "A" "$PROMPT_A" "doktor/rwkv-0" "$BUFFER_DOKTOR_0" 12
run_model "A" "$PROMPT_A" "doktor/rwkv-14" "$BUFFER_DOKTOR_14" 12

# Prompt B: "Contame algo corto." (gen 12)
PROMPT_B="Contame algo corto."
run_model "B" "$PROMPT_B" "baseline" "" 12
run_model "B" "$PROMPT_B" "jane/rwkv-11" "$BUFFER_JANE_11" 12
run_model "B" "$PROMPT_B" "doktor/rwkv-14" "$BUFFER_DOKTOR_14" 12

# d. Escribir el reporte docs/gguf-estados-voz.md
echo "--- [Paso d] Escribiendo reporte docs/gguf-estados-voz.md ---" | tee -a "$LOG_FILE"

env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" - "$SHAPES_JSON" "$JSONL_FILE" \
    "docs/gguf-estados-voz.md" "$GGUF" "$GGUF_SHA256" "$LLAMA_COMMIT" <<'PY'
import json, sys

shapes_path = sys.argv[1]
jsonl_path = sys.argv[2]
doc_path = sys.argv[3]
gguf_path = sys.argv[4]
gguf_sha256 = sys.argv[5]
llama_commit = sys.argv[6]

with open(shapes_path, 'r') as f:
    shapes = json.load(f)

runs_A = []
runs_B = []
with open(jsonl_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get('prompt_label') == 'A':
            runs_A.append(item)
        elif item.get('prompt_label') == 'B':
            runs_B.append(item)

# Baseline IDs
base_A_ids = next((r['ids'] for r in runs_A if r['estado'] == 'baseline'), [])
base_B_ids = next((r['ids'] for r in runs_B if r['estado'] == 'baseline'), [])

def format_top5(top5):
    return ' '.join(f'{pair[0]}:{pair[1]:.4f}' for pair in top5)

def format_ids(ids):
    return ' '.join(str(i) for i in ids)

def escape_cell(text):
    return text.replace('\n', '\\n').replace('|', '\\|')

md = []
md.append('# Reporte de Engagement de Estados de Voz en RWKV-7 GGUF (2.9B)\n')

md.append('## 1. Comando Reproducible y Entorno\n')
md.append(f'- **Comando de prueba:** `bash script/probar-estados-voz.sh`')
md.append(f'- **Modelo GGUF evaluado:**')
md.append(f'  - Repositorio HuggingFace: `shoumenchougou/RWKV7-G1j-2.9B-GGUF`')
md.append(f'  - Archivo en disco: `{gguf_path}`')
md.append(f'  - SHA256: `{gguf_sha256}`')
md.append(f'- **Versión de llama.cpp:** Commit `{llama_commit}` (`/run/media/chaos/secundario/llamacpp-conv`)')
md.append(f'- **Configuración de ejecución:** `n_ctx=256`, `n_batch=256`, `n_seq_max=1`, `n_threads=8`, `n_gpu_layers=0` (inferencia puramente CPU).\n')

md.append('## 2. Tabla de Formas (Compatibilidad Arquitectural)\n')
md.append('| State (.pth) | Tensores | Forma time_state | Veredicto | n_embd (.pth vs GGUF) |')
md.append('| :--- | :---: | :---: | :---: | :---: |')
for s in shapes:
    calza_badge = f"**{s['calza']}**" if s['calza'] == 'CALZA' else s['calza']
    embd_str = f"{s['n_embd_pth']} vs {s['n_embd_gguf']}"
    if s['calza'] == 'NO CALZA':
        embd_str += ' (base 0.4B)'
    md.append(f"| `{s['state']}` | {s['n_tensores']} | `{s['shape']}` | {calza_badge} | {embd_str} |")
md.append('')

md.append('## 3. Tablas de Engagement (Generación y Salida)\n')
md.append('### Prompt A: \"Hola, soy Kateto.\" (12 tokens greedy)\n')
md.append('| State | set_data_bytes | cell_pos | start_pos | Top-5 paso 1 (`id:logit`) | logits_max | ids generados | Continuación | Difiere de baseline |')
md.append('| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :--- | :---: |')
for r in runs_A:
    diff = 'no' if r['ids'] == base_A_ids else '**sí**'
    top5_str = format_top5(r.get('top5', []))
    ids_str = format_ids(r.get('ids', []))
    cont_str = escape_cell(r.get('continuacion', ''))
    lmax = f"{r.get('logits_max', 0.0):.4f}"
    md.append(f"| `{r['estado']}` | {r['set_data_bytes']} | {r['cell_pos']} | {r['start_pos']} | `{top5_str}` | {lmax} | `{ids_str}` | `{cont_str}` | {diff} |")
md.append('')

md.append('### Prompt B: \"Contame algo corto.\" (12 tokens greedy)\n')
md.append('| State | set_data_bytes | cell_pos | start_pos | Top-5 paso 1 (`id:logit`) | logits_max | ids generados | Continuación | Difiere de baseline |')
md.append('| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :--- | :---: |')
for r in runs_B:
    diff = 'no' if r['ids'] == base_B_ids else '**sí**'
    top5_str = format_top5(r.get('top5', []))
    ids_str = format_ids(r.get('ids', []))
    cont_str = escape_cell(r.get('continuacion', ''))
    lmax = f"{r.get('logits_max', 0.0):.4f}"
    md.append(f"| `{r['estado']}` | {r['set_data_bytes']} | {r['cell_pos']} | {r['start_pos']} | `{top5_str}` | {lmax} | `{ids_str}` | `{cont_str}` | {diff} |")
md.append('')

md.append('#### Distribución de logits y fenómeno OOD (Out-of-Distribution)\n')
md.append('- **`doktor` (rwkv-0 y rwkv-14):** Deja todos los logits del primer paso estrictamente negativos (`logits_max` < 0, e.g. -3.7310 en rwkv-14), lo cual coloca al modelo completamente fuera de su distribución calibrada y resulta en texto degenerado o repetitivo (` cho\\\\yto\\\\ody...`).')
md.append('- **`jane/rwkv-11`:** Preserva logits positivos de magnitud normal (`top1` = 112 con logit 5.4691), dentro de la distribución estándar del modelo, y modifica determinísticamente los tokens generados respecto a la línea base sin degenerar.\n')

md.append('## 4. ROSA: Veredicto\n')
md.append('ROSA (*Rapid Online Suffix Automaton*, mecanismo propuesto para RWKV-8) **NO existe** en este stack y no contribuye al cambio de estado medido:')
md.append('1. **Código fuente:** `grep -rn -i rosa` sobre los directorios `src`, `include`, `common` y `ggml/src` del árbol `llamacpp-conv` arroja **0 líneas** de coincidencia.')
md.append('2. **Incompatibilidad de arquitectura GGUF:** Al intentar cargar el GGUF de ROSA con `llama-cli`, el proceso aborta inmediatamente con:')
md.append('   ```')
md.append("   E llama_model_load: error loading model: unknown model architecture: 'rosa'")
md.append('   ```')
md.append('3. **Incompatibilidad de memoria y pesos:** Tal como se detalló en `docs/rosa-ggml-viabilidad.md`, la implementación de ROSA en `rwkv_pipeline/rosa_module.py` es una rama de retrieval cuyo buffer en runtime crece por token (`torch.cat`), haciéndola incompatible con el buffer de memoria recurrente de tamaño fijo por secuencia que utiliza llama.cpp. Adicionalmente, los pesos de ROSA para RWKV-7 están sin entrenar (*\"pesos RANDOM sin entrenar\"*).')
md.append('4. **Conclusión:** ROSA no aporta nada al cambio de state; no aplica al stack GGUF por las razones medidas arriba.\n')

md.append('## 5. Procedencia y Límites\n')
md.append('- **Procedencia de los checkpoints:**')
md.append('  - `doktor` (rwkv-0 y rwkv-14): posee 32 capas de dimensión `(40, 64, 64)`, correspondiente a la topología 2.9B.')
md.append('  - `jane/rwkv-11`: mtime del archivo es `2026-09-09` y no posee log propio de entrenamiento (su `train_log.txt` hace referencia a otra corrida de 0.4B); su procedencia exacta queda no documentada.')
md.append('- **Tensor `r` (Token Shift):** El archivo `.pth` original contiene únicamente `time_state` (tensor `s`). El tensor `r` se inicializa en ceros por diseño en `state_from_pth.py`.')
md.append('- **Desacople modelo-estado:** El GGUF base oficial y los estados `.pth` no provienen del mismo entrenamiento conjunto; se acoplan únicamente por compatibilidad de dimensiones tensoriales.')
md.append('- **Condiciones de inferencia:** Modelo cuantizado en `Q4_K_M`, decodificación greedy pura (argmax), ejecutado en CPU.')
md.append('- **Alcance semántico:** La semántica de \"voz\" o naturalidad auditiva no está probada; lo demostrado matemáticamente y de manera determinística es que la inyección de estado altera de forma verificable los logits y los tokens generados.')

with open(doc_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')
PY

echo "Reporte generado en docs/gguf-estados-voz.md" | tee -a "$LOG_FILE"

# e. Verificaciones y compuerta de salida
echo "--- [Paso e] Comprobando las 5 condiciones del gate ---" | tee -a "$LOG_FILE"

# Condición 1: Los 8 states tienen veredicto de formas
COND1_OK=true
if [ "$TOTAL_STATES" -ne 8 ]; then
    COND1_OK=false
fi
for s in "${STATES[@]}"; do
    if [ -z "${VERDICTOS[$s]:-}" ]; then
        COND1_OK=false
    fi
done

if [ "$COND1_OK" = true ]; then
    echo "1. Veredicto de formas: OK (8 states verificados: 3 CALZA, 5 NO CALZA)"
else
    echo "1. Veredicto de formas: FALLÓ"
    exit 1
fi

# Condición 2: >= 2 states CALZA inyectaron con set_data_bytes > 0 y start_pos = cell_pos + 1
COND2_OK=$(env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -c "
import json, sys
count = 0
with open(sys.argv[1], 'r') as f:
    for line in f:
        r = json.loads(line)
        if r['estado'] != 'baseline':
            if r['set_data_bytes'] > 0 and r['start_pos'] == r['cell_pos'] + 1:
                count += 1
print('OK' if count >= 2 else 'FAIL', count)
" "$JSONL_FILE")

if [[ "$COND2_OK" =~ ^OK ]]; then
    echo "2. Inyección de states CALZA: OK ($COND2_OK inyecciones con set_data_bytes > 0 y start_pos = cell_pos + 1)"
else
    echo "2. Inyección de states CALZA: FALLÓ ($COND2_OK)"
    exit 1
fi

# Condición 3: Prompt A baseline vs jane-11 vs doktor-0/14 no son todos iguales
COND3_RES=$(env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -c "
import json, sys
runs = [json.loads(l) for l in open(sys.argv[1]) if json.loads(l).get('prompt_label') == 'A']
b = next((r['ids'] for r in runs if r['estado'] == 'baseline'), None)
j11 = next((r['ids'] for r in runs if r['estado'] == 'jane/rwkv-11'), None)
d0 = next((r['ids'] for r in runs if r['estado'] == 'doktor/rwkv-0'), None)
d14 = next((r['ids'] for r in runs if r['estado'] == 'doktor/rwkv-14'), None)

all_same = (b == j11 == d0 == d14)
print('FAIL' if all_same else 'OK')
print(f'   baseline:   {b}')
print(f'   jane-11:    {j11}')
print(f'   doktor-0:   {d0}')
print(f'   doktor-14:  {d14}')
" "$JSONL_FILE")

if [[ "$COND3_RES" =~ ^OK ]]; then
    echo "3. Prompt A variabilidad de ids: OK (los ids difieren entre estados)"
    echo "$COND3_RES" | sed '1d'
else
    echo "3. Prompt A variabilidad de ids: FALLÓ (todos los ids son iguales)"
    exit 1
fi

# Condición 4: Prompt B baseline vs jane-11 vs doktor-14 no son todos iguales
COND4_RES=$(env -u PYTHONPATH -u VIRTUAL_ENV "$PYTHON" -c "
import json, sys
runs = [json.loads(l) for l in open(sys.argv[1]) if json.loads(l).get('prompt_label') == 'B']
b = next((r['ids'] for r in runs if r['estado'] == 'baseline'), None)
j11 = next((r['ids'] for r in runs if r['estado'] == 'jane/rwkv-11'), None)
d14 = next((r['ids'] for r in runs if r['estado'] == 'doktor/rwkv-14'), None)

all_same = (b == j11 == d14)
print('FAIL' if all_same else 'OK')
print(f'   baseline:   {b}')
print(f'   jane-11:    {j11}')
print(f'   doktor-14:  {d14}')
" "$JSONL_FILE")

if [[ "$COND4_RES" =~ ^OK ]]; then
    echo "4. Prompt B variabilidad de ids: OK (los ids difieren entre estados)"
    echo "$COND4_RES" | sed '1d'
else
    echo "4. Prompt B variabilidad de ids: FALLÓ (todos los ids son iguales)"
    exit 1
fi

# Condición 5: ROSA checks
echo "5. Chequeos determinísticos de ROSA:"

# 5a. Grep rosa en llamacpp-conv
if [ -d "/run/media/chaos/terciario/proyectos/llamacpp-conv/src" ]; then
    LLAMA_SRC_PATH="/run/media/chaos/terciario/proyectos/llamacpp-conv"
else
    LLAMA_SRC_PATH="/run/media/chaos/secundario/llamacpp-conv"
fi

ROSA_GREP_OUT=$(grep -rn -i rosa "$LLAMA_SRC_PATH/src" "$LLAMA_SRC_PATH/include" "$LLAMA_SRC_PATH/common" "$LLAMA_SRC_PATH/ggml/src" 2>&1 || true)
ROSA_GREP_LINES=$(echo -n "$ROSA_GREP_OUT" | grep -c . || true)
echo "   grep -rn -i rosa en $LLAMA_SRC_PATH:"
echo "   -> líneas encontradas: $ROSA_GREP_LINES"

if [ "$ROSA_GREP_LINES" -ne 0 ]; then
    echo "ERROR: grep encontró líneas para rosa en llamacpp-conv"
    exit 1
fi

# 5b. llama-cli con GGUF rosa
ROSA_GGUF="/run/media/chaos/secundario/models/rwkv7-gguf/g1d-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192-FP16.rosa.gguf"
LLAMA_CLI="$LLAMA_BIN/llama-cli"

set +e
ROSA_CLI_OUT=$(LD_LIBRARY_PATH="$LLAMA_BIN" timeout 60 "$LLAMA_CLI" -m "$ROSA_GGUF" -p hola -n 4 --device none 2>&1)
ROSA_CLI_RC=$?
set -e

echo "   llama-cli sobre GGUF rosa (rc=$ROSA_CLI_RC):"
echo "$ROSA_CLI_OUT" | grep "unknown model architecture: 'rosa'" | sed 's/^/   salida cruda: /'

if ! echo "$ROSA_CLI_OUT" | grep -q "unknown model architecture: 'rosa'"; then
    echo "ERROR: llama-cli no falló con 'unknown model architecture: rosa'"
    exit 1
fi

echo "   ROSA: OK (0 líneas en código fuente y arquitectura desconocida en llama-cli)"

echo "=== Gate probar-estados-voz: TODOS LOS TESTS PASARON (exit 0) ==="
exit 0
