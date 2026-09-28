# kateto-train → GitHub: plan de publicación (SOLO PLAN, no ejecutar)

> Repo local: `/run/media/chaos/terciario/proyectos/kateto-train` · rama `main` · 17 commits · 94 archivos versionados · sin remoto (`git remote -v` vacío, verificado).
> `.git`: 3,8 GB — `git count-objects -vH`: 3 packs, 3.62 GiB, 726 objetos (números del brief, no recalculados).
> Directorio total 134 GB. `gh` autenticado como `Gonanf`, protocolo HTTPS (verificado con `gh auth status`).
> Estado: **este documento es el plan. No se crea el repo, no se reescribe historia, no se commitea ni se pushea.**
> Comandos citados como `medido con:` fueron ejecutados de solo lectura. Lo demás son números del brief.

## 1. Resumen y recomendación en 5 líneas

1. Crear un **repo nuevo con historia limpia** (un commit inicial curado desde `HEAD`), no reescribir historia: con 12 GB libres en `terciario` y un entrenamiento en curso, `filter-repo` es riesgo sin beneficio.
2. El `.git` de 3,8 GB está compuesto en ~5,2 GB lógicos por **3 volcados `gpucore.*`** + ~0,5 GB de datasets históricos; el `HEAD` actual pesa solo **~34,9 MB** (medido con `git ls-files -s | … xargs git cat-file -s`), así que el repo nuevo nace en **~1–5 MB de `.git`**.
3. Sube **código + docs curadas + datasets de síntesis chicos (<10 MB)**; datasets medianos (19–42 MB) a **Release/Hugging Face**, nada a Git LFS; pesos, `out/`, `venv*/`, `base/`, `gpucore.*` quedan **afuera**.
4. Antes de publicar:barrer **secretos** en 3 notebooks versionados + `kaggle_*/notebooks/config/.prompts/` no versionados (hoy no hay `.env`/`kaggle.json` versionados — verificado con `git ls-files | grep -i -E '\.env|kaggle\.json|\.pem|credentials'` → vacío).
5. Nombre `kateto-train`, **privado al crear y público al curar**; backup del `.git` (3,8 GB) al secundario (71 GB libres) antes de cualquier paso irreversible; el usuario decide 6 puntos de la sección 8.

## 2. Qué se sube y qué no (tabla: ruta · tamaño · decisión · razón)

Leyenda decisión: **SÍ** = al repo nuevo · **REL** = a GitHub Release / Hugging Face, con link desde README · **NO** = afuera.

### 2a. Código versionado en HEAD (94 archivos, ~34,9 MB en total — medido con `git ls-files -s | cut -d' ' -f2 | xargs git cat-file -s | awk suma`)

| Ruta | Tamaño | Decisión | Razón |
|---|---|---|---|
| `scripts/` (28 archivos) | cada uno <100 KB (3er blob mayor del HEAD = 852 KB; el resto código chico) | **SÍ** | Pipeline reproducible: builds de dataset, evals, voz, fixes 118–126 |
| `rwkv_pipeline/` (3 archivos: `infer_kateto.py`, `_stop_helpers.py`, `test_format_prompt.py`) | KB | **SÍ** | Inferencia y formato del modelo |
| Raíz `*.py`/`*.sh`/`*.txt` (`train_qwen_2stage.py`, `merge_*.py`, `patch_*.py`, `fill_toolcalling.py`, `to_sft.py`, `export_gguf.sh`, `run_kateto.sh`, `test_prompt.txt`) | KB cada uno | **SÍ** | Entrenamiento Qwen 2-stage, merges LoRA, parches GGUF |
| `kateto-medicion/` (2 archivos) | KB | **SÍ** | Medición del bit anclado, parte del método |
| `colab/` (6 archivos versionados: 3 `.ipynb` + 2 `.py` + README) | KB–1 MB c/u (estimación: notebooks sin outputs pesados; **verificar con §5 antes**) | **SÍ, tras limpiar outputs y revisar secretos** | Entrada Colab/Kaggle del proyecto |
| `docs/knowledge-rwkv-kateto-rosa-gguf-rx6500xt.md`, `docs/qwen_unsloth_lora_merge_issue.md` | KB | **SÍ** | Documentación técnica del método |
| `README.md`, `README.orig.md`, `SETUP_LOG_UNSLOTH.md` | KB | **SÍ (curar README)** | Presentación; reescribir con links a releases |
| `data/toolcalling_v2/` (15 `.jsonl` + `.manifest.json`) | chicos (el mayor de este grupo <1 MB por descarte: solo 2 blobs >1 MB en HEAD) | **SÍ** | Dataset de síntesis propio, versionable en git |
| `data/splits/` (3), `data/sft_out/` (3), `data/comedians/*.txt`+`.md` (4), `data/toolcalling_*.jsonl` chicos, `data/kateto_qa.jsonl`, `data/*.md` | KB–cientos de KB c/u | **SÍ** | Datos de síntesis chicos, reproducibilidad |
| `data/kateto_qa.v2-3k.jsonl` | **3,27 MB** (2.º blob del HEAD, medido) | **SÍ** | Bajo el límite con holgura |
| `data/rwkv_kateto_base_train_plus.jsonl` (@HEAD) | **29,8 MB** (blob mayor del HEAD, medido); working tree modificado: **42 MB** (brief) | **Decisión del usuario (default: REL)** | 29,8 MB < límite GitHub 100 MB pero > aviso 50 MB; hay ~10 versiones de 28–42 MB en historia. Recomiendo REL + documentar regeneración |
| `.gitignore` (modificado en working tree — `M .gitignore` en `git status`) | — | **SÍ, con los agregados del §7** | Sumar `gpucore.*`, `data/hf/`, `*.jsonl` grandes, `kaggle_*/`, `notebooks/` |

### 2b. Solo en historia (no están en HEAD, pero inflan el `.git` a 3,8 GB — números del brief)

| Ruta | Tamaño | Decisión | Razón |
|---|---|---|---|
| `gpucore.2486405` / `gpucore.2384226` / `gpucore.2350244` | **2751,8 + 1345,6 + 1091,3 MB** | **NO** (ni en repo nuevo ni en releases) | Volcados de crash GPU, irreproducibles como artefacto útil, 0 valor de portfolio |
| `data/hf/sycophancy-reduction-dpo-100k/*.jsonl` | **221,7 MB** | **NO** (re-descargable) | Dataset de terceros; lo cubre `scripts/download_hf_datasets.py` |
| `data/hf/function-calling-sharegpt…parquet` | **99,2 MB** | **NO** | Idem |
| `data/hermes_qa_7k.jsonl` | **20,7 MB** | **REL** (hoy no versionado: `?? data/hermes_qa_7k.jsonl`) | Derivado útil, supera lo cómodo para git junto a otros |
| `data/hf/iberbench_all/…parquet` | **20,2 MB** | **NO** (re-descargable) | Terceros |
| `data/hf/cordeba/data/CordeBA_00*.mp3` | **18–34 MB c/u** | **NO** | Audio de terceros |
| `data/v2/rwkv_kateto_v2_train.jsonl` | **34,9 MB** | **REL** | Síntesis propia mediana |
| `data/kateto_v2_combined_train.jsonl` | **19 MB** | **REL** | Síntesis propia mediana (hoy `??` no versionado) |
| Versiones intermedias `rwkv_kateto_base_train_plus` | **~10 blobs de 28–42 MB** | **NO** (solo la final va a REL) | Historial intermedio sin valor publicado |

### 2c. Working tree no versionado (126 rutas sucias — `git status --porcelain`, verificado; no tocar ni commitear)

| Ruta | Tamaño (brief) | Decisión | Razón |
|---|---|---|---|
| `out/` | **74 GB** | **NO** (ya ignorado) | Checkpoints y salidas de entrenamiento |
| `venv-unsloth-qwen/` | **27 GB** | **NO** (cubre `venv-*/`) | Entorno recreable desde `SETUP_LOG_UNSLOTH.md` |
| `data/` (total working) | **8 GB** | **Parcial**: solo §2a en SÍ; `data/hf/`, `data/v2/`, `data/hermes_qa_7k.jsonl`, `data/externos/`, `youtube_*`, `orpo/`, `real/`, `research/`, `largo/` → **NO o REL** | La mayoría es cache/descargas o trabajo en curso de la ola de dataset |
| `base/` | **5,5 GB** | **NO** (ya ignorado) | Pesos base |
| `tmp_download/` | **4,8 GB** | **NO** (ya ignorado) | Descargas temporales |
| `models/` | **3,8 GB** | **NO** (agregar a `.gitignore`: `models/`) | Pesos: van a Release/HF, nunca a git |
| `venv/` | **2,5 GB** | **NO** (ya ignorado) | Entorno |
| `rwkv-mobile/` | **244 MB** | **NO** (ya ignorado) | Subrepo externo clonado |
| `gpucore.2350244`, `gpucore.2384226`, `gpucore.2486405` (working tree, sin versionar) | **~5,2 GB** | **NO, y candidato a borrar (decide el usuario)** | Volcados; único borrado que libera espacio útil |
| `kaggle_kernel_probe/`, `kaggle_kernel_v2/`, `kaggle_kernel_v3/`, `kaggle_upload/`, `kaggle_v2/`, `kateto_train_kaggle_2xt4.ipynb`, `notebooks/`, `colab/Kateto_State_Tuning_Colab.ipynb`, `config/`, `.prompts/`, `kateto/`, `notes/`, `HANDOFF.md`, `SPEC.md`, `CONVERSACIONES-LARGAS.md`, `start.sh`, `kateto-play.sh` | no medidos (untracked, no se midieron por Regla 0) | **NO por default; SÍ solo lo curado tras §5** | Trabajo en curso + alto riesgo de secretos (tokens Kaggle/HF, rutas con usuario) |
| `docs/` no versionados (20 archivos: eval-briefs, kaggle-checklist, kernel-gate, orpo-spike, etc.) | KB c/u (estimación) | **SÍ selectivo** | Curar: los eval-briefs y decisiones aportan; `kaggle-dryrun`/`checklist` solo si no tienen secretos |

## 3. Historia: cómo dejarla presentable (opciones, costo, recomendación)

Contexto (verificado): 17 commits, mensajes coherentes (fixes 118–126, datasets, merges LoRA). La historia es legible; el problema es solo el peso: 726 objetos / 3,8 GB por los blobs del §2b.

### Opción A — Repo nuevo con historia limpia (RECOMENDADA)

- Cómo: `git archive HEAD` (o `git checkout-index`) a un directorio staging en el secundario → curar (gitignore, secretos, datasets) → `git init` + 1 commit inicial → `gh repo create` → push.
- Costo: **minutos**; espacio: **~50–100 MB** en el secundario (HEAD 35 MB + curaduría); en `terciario` **0 MB extra** (todo se arma fuera del repo en uso).
- `.git` resultante: **~1–5 MB** (estimación: HEAD 34,9 MB medidos comprimen a pocos MB; sin blobs históricos).
- Pros: cero riesgo sobre el repo en uso, no toca el entrenamiento ni la ola de dataset, no necesita borrar nada.
- Contras: se pierden los 17 mensajes (se citan en el README/changelog como referencia al repo local).

### Opción B — Reescribir con `git filter-repo --strip-blobs-bigger-than 10M`

- Cómo: backup del `.git` → `git filter-repo --strip-blobs-bigger-than 10M --force` → `gc` → push a repo nuevo vacío.
- Costo (estimación, decirlo): reescritura de 726 objetos / 17 commits: **decenas de minutos**; espacio necesario: **~8–12 GB libres en el mismo filesystem** (copia de seguridad del `.git` 3,8 GB + objetos reescritos + refs originales en `refs/original/`), que `terciario` **no tiene garantizados** (12 GB libres pero con el entrenamiento escribiendo `out/` y la ola de dataset en el mismo disco: margen inseguro).
- `.git` resultante: **<100 MB** (estimación: HEAD 35 MB + historia de código chica; los blobs >10 MB —todos los del §2b más el `jsonl` de 29,8 MB— saldrían).
- Contras: **irreversible sin backup**, invalida los hashes de los 17 commits, puede corromperse si el disco se llena a medias, y compite por I/O con el entrenamiento en curso. Aporta poco sobre la opción A porque casi todo lo pesado ya no está en HEAD.

### Recomendación

**Opción A.** Justificación en una línea: con HEAD medido en 34,9 MB y el peso concentrado en blobs históricos que ya no están en el checkout, el repo nuevo logra el mismo `.git` mínimo con costo y riesgo un orden de magnitud menores, sin tocar el repo que está entrenando.

## 4. Datos y pesos: git / LFS / release-HF / afuera

Conteos (medidos con `git ls-files`, verificado): 94 archivos = ~57 código/docs/scripts + **37 datos** (12 `data/` + 15 `data/toolcalling_v2/` + 4 `data/comedians/` + 3 `data/splits/` + 3 `data/sft_out/`).

| Grupo | Archivos | Destino | Detalle |
|---|---|---|---|
| Síntesis chicos (<10 MB, todos en HEAD) | **36** (todos menos el `base_train_plus`) | **git** | Incluye `toolcalling_v2/` (15), `splits/`, `sft_out/`, comedians, `toolcalling_*.jsonl` chicos, `kateto_qa*.jsonl` (3,27 MB el mayor) |
| Síntesis medianos (10–100 MB) | **4**: `rwkv_kateto_base_train_plus.jsonl` (42 MB worktree), `v2_train` (34,9 MB), `hermes_qa_7k` (20,7 MB), `kateto_v2_combined_train` (19 MB) | **Release de GitHub o dataset de Hugging Face (REL), no LFS** | GitHub avisa >50 MB y bloquea >100 MB por archivo; LFS tiene cuota (1 GB gratis) que 4 archivos agotan; Release/HF es gratis e ilimitado para este orden |
| Terceros re-descargables | **5+**: sycophancy 221 MB, sharegpt-parquet 99 MB, iberbench 20 MB, mp3s 18–34 MB c/u | **afuera** | Los cubre `scripts/download_hf_datasets.py`; fijar revisiones/commit en el script |
| Pesos y cuantizaciones (`*.pth/.pt/.bin/.safetensors/.gguf/.onnx`, `models/`, `base/`, `out/`) | ya ignorados en HEAD; `models/` hay que agregarlo | **afuera de git → Release/HF** | Solo el GGUF final + imatrix; el resto son intermedios |
| Volcados `gpucore.*` | 3 en historia + 3 en working tree | **afuera total** | Ni git, ni LFS, ni releases |
| Trabajo en curso (`data/v2/`, `externos/`, `youtube_*`, `orpo/`, `real/`, `research/`, `largo/`, `_smoke_capa1`, `debate_speech`) | ~15 rutas `??` | **afuera hasta que la ola termine** | Publicar datasets a medias confunde; segunda tanda después |

**LFS: no usar.** Razón: con solo 4 archivos medianos, LFS suma fricción (cada clon necesita `git-lfs`, cuota de 1 GB) frente a Release/HF que resuelve lo mismo sin costo.

## 5. Secretos: qué revisar y cómo

Verificado hoy (solo lectura): **ningún `.env`/`kaggle.json`/`.pem`/`credentials` versionado** (`git ls-files | grep -i -E '…'` → vacío) y `git grep` en tracked solo muestra referencias seguras (`HF_TOKEN` vía `os.environ`/`kaggle_secrets.UserSecretsClient`, sin valores). El riesgo está en lo **no versionado** y en **outputs de notebooks**. Todo lo de abajo es **público por default salvo que contenga un valor**: los nombres de variables (`HF_TOKEN`, `KAGGLE_*`) son públicos; los **valores** de tokens/claves son privados y nunca se suben.

Revisar antes de publicar, en este orden:

1. **3 notebooks versionados** (`colab/Kateto_Colab_Main.ipynb`, `Kateto_Colab_RWKV8_Small.ipynb`, `Kateto_Finetune.ipynb`): buscar celdas con tokens pegados y **outputs con `gho_`/`hf_` impresos**. Patrón seguro esperado: `UserSecretsClient().get_secret('HF_TOKEN')` / `os.environ.get('HF_TOKEN')` (ya presente en `Kateto_Colab_Main.ipynb` — bien).
2. **No versionados**: `kaggle_*/`, `kateto_train_kaggle_2xt4.ipynb`, `notebooks/`, `colab/Kateto_State_Tuning_Colab.ipynb`, `config/`, `.prompts/`, `scripts/download_hf_datasets.py`, `scripts/*teacher*`, `*probe*`, `HANDOFF.md`, `SPEC.md`, `notes/`.
3. **Rutas con usuario**: `/home/chaos`, `/run/media/chaos`, `/home/study` hardcodeadas en scripts/notebooks (son públicas como texto pero delatan estructura; preferible `$HOME`/rutas relativas).
4. **`.gitignore` de trabajo modificado** (`M .gitignore`): antes de publicar, difuminar qué cambió vs HEAD (`git diff .gitignore`) para no arrastrar una exclusión rota.

Comandos de verificación (todos de solo lectura salvo el staging, que se hace en copia):

```bash
git grep -n -i -E 'kaggle|api_key|apikey|secret|token|password|passwd|aws_|hf_token|huggingface|openai|anthropic' -- . | grep -v -E 'get_secret|os.environ.get|ADD-ONS|Secrets >|settings/tokens|huggingface_hub import|from_pretrained' 
git ls-files | grep -i -E '\.env|kaggle\.json|\.pem|\.key$|credentials|secrets'
git ls-files '*.ipynb' | while read f; do git show HEAD:"$f" | grep -o -E '(gho_|github_pat_|hf_|sk-ant-|xox[bap]-|AKIA)[A-Za-z0-9_]{8,}' | head -5; done
grep -rn -E '(gho_|github_pat_|hf_|sk-ant-|xox[bap]-|AKIA)[A-Za-z0-9_]{8,}|kaggle\.json|"key"\s*:\s*"[A-Za-z0-9]' <STAGING> --include='*.py' --include='*.ipynb' --include='*.md' --include='*.json' --include='*.sh' | head -30
grep -rn -E '/home/[a-z_]+|/run/media|/home/study' <STAGING> --include='*.py' --include='*.sh' --include='*.ipynb' | head -20
```

Si aparece cualquier valor: rotarlo (Kaggle: regenerar token; HF: revocar en `huggingface.co/settings/tokens`) **antes** del push. Paso de higiene para notebooks (lo ejecuta el usuario en la copia staging): re-ejecutar con `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace` solo los 3–4 notebooks a publicar.

## 6. Repo: nombre, visibilidad y descripción

- **Dueño**: `Gonanf` (ya verificado en `gh auth status`).
- **Nombre**: `kateto-train` — coherente con el existente `Kateto` (público) y con el path local; sin renombres creativos.
- **Visibilidad**: **crear PRIVADO y pasar a PÚBLICO tras la curaduría del §7** (una línea de justificación: hay 126 rutas de trabajo en curso y una ola de dataset corriendo, así que publicar directo expone borradores; privado-primero permite verificar secretos y datasets sin exponer nada).
- **Descripción propuesta**: `Entrenamiento y datasets de Kateto (voz ES, tool-calling, turn-taking): pipeline RWKV-7 + Qwen 2-stage con Unsloth, scripts reproducibles y GGUF Q4_K_M para RX 6500 XT.`
- **Extras al crear**: `README.md` curado con links a releases, `LICENSE` (decisión §8.6), topics `rwkv`, `qwen`, `lora`, `dataset`, `tts`, `spanish`.

## 7. Pasos exactos para ejecutar el plan (con espacio en disco por paso)

Condiciones previas: no ejecutar con el entrenamiento escribiendo a toda máquina si se puede evitar; todo el staging ocurre en el **secundario** (`/dev/sdb`, 71 GB libres — números del brief). **Irreversibles marcados con ⚠️**.

```bash
# 0. Backup del .git original (3,8 GB) al secundario. Espacio: 3,8 GB en /dev/sdb (71 GB libres → sobra).
#    Lectura del repo en uso: solo copia, no lo modifica.
cp -a /run/media/chaos/terciario/proyectos/kateto-train/.git /dev/sdb/backup-kateto-train-git-2026-09-28/
sha256sum /dev/sdb/backup-kateto-train-git-2026-09-28/HEAD

# 1. Staging curado FUERA del repo (HEAD + untracked elegidos, sin .git pesado). Espacio: ~50-100 MB en /dev/sdb.
mkdir -p /dev/sdb/kateto-publish && cd /run/media/chaos/terciario/proyectos/kateto-train
git archive HEAD | tar -x -C /dev/sdb/kateto-publish
cp --parents docs/dataset-spec-v1.md docs/eval-brief-A-prompts.md docs/eval-brief-B-scoring.md docs/eval-brief-C-generation.md docs/eval-behavior-brief.md docs/decision-gguf-vs-nativo.md /dev/sdb/kateto-publish/  # solo los curados tras §5
cp colab/Kateto_State_Tuning_Colab.ipynb /dev/sdb/kateto-publish/colab/  # solo si pasa §5

# 2. Endurecer .gitignore en el staging (edición local del staging, reversible).
cat >> /dev/sdb/kateto-publish/.gitignore <<'EOF'
gpucore.*
models/
data/hf/
data/v2/
data/externos/
data/youtube_transcripts/
data/real/
data/research/
data/largo/
data/orpo/
kaggle_*/
kaggle_upload/
notebooks/
config/
.prompts/
kateto/
*.ipynb_checkpoints/
EOF

# 3. Limpiar outputs de notebooks en el staging (escribe solo la copia).
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace /dev/sdb/kateto-publish/colab/*.ipynb

# 4. Barrido de secretos sobre el staging (§5). Solo lectura; si hay hit, rotar tokens y repetir.
# 5. Commit inicial en el staging (repo nuevo, sin historia pesada).
cd /dev/sdb/kateto-publish && git init -b main && git add -A && git status --short | head -120 && git commit -m "Initial commit: Kateto training pipeline, datasets v2, voice turn-taking, Colab/Kaggle setup"

# 6. Crear repo PRIVADO y pushear. Espacio: ~35-60 MB de subida. ⚠️ expone el contenido al remoto (privado; reversible con gh repo delete).
gh repo create Gonanf/kateto-train --private --source=/dev/sdb/kateto-publish --push --description "Entrenamiento y datasets de Kateto (voz ES, tool-calling, turn-taking): pipeline RWKV-7 + Qwen 2-stage con Unsloth, scripts reproducibles y GGUF Q4_K_M para RX 6500 XT."

# 7. Subir datasets medianos como Release v0.1-data (4 archivos, ~115 MB) + pesos finales. ⚠️ publicar release es visible según visibilidad del repo.
gh release create v0.1-data --repo Gonanf/kateto-train --title "Datasets v0.1" --notes "base_train_plus 42MB, v2_train 34.9MB, hermes_qa_7k 20.7MB, combined_train 19MB" <4 archivos>

# 8. Pasar a público solo tras verificar secretos + releases. ⚠️ hacer público un repo es irreversible en la práctica (cualquiera pudo clonar/forkear).
gh repo edit Gonanf/kateto-train --visibility public
```

Si el usuario elige Opción B en vez de A (no recomendado): `git clone --mirror` + `filter-repo` necesitan **~8–12 GB libres contiguos en terciario** (estimación §3) → antes habría que borrar los `gpucore.*` del working tree (5,2 GB, decisión §8.4) y parar el entrenamiento; con 12 GB libres actuales el margen es inseguro.

## 8. Riesgos, reversión y decisiones del usuario

### Riesgos (qué puede salir mal)

1. Pushear `gpucore.*` (5,2 GB) o `out/` (74 GB) por un `git add -A` descuidado → push rechazado o repo inusable; mitigación: staging curado + `.gitignore` del paso 2 **antes** del `add`.
2. `filter-repo` (si se elige B) quedando a medias con 12 GB libres en `terciario` → `.git` corrupto; mitigación: no elegir B; si se elige, backup del paso 0 verificado con `sha256sum`.
3. Pisar las 126 rutas de trabajo en curso (ola de dataset + entrenamiento 0.4B) con un checkout/limpieza dentro del repo → pérdida de trabajo; mitigación: **todo se hace en `/dev/sdb/kateto-publish`**, nunca dentro del repo.
4. Exponer token Kaggle/HF pegado en un notebook o en `kaggle_*/config/` → abuso de cuota; mitigación: barrido §5 + rotación **antes** del push.
5. Publicar datasets a medias de la ola en curso → confusión de versiones; mitigación: solo los 4 archivos cerrados van a `v0.1-data`.
6. GitHub bloquea archivos >100 MB y avisa >50 MB → el `jsonl` de 42 MB pasa pero al límite; mitigación: va a Release, no a git.

### Reversión

- **Backup**: el `.git` copiado en el paso 0 (`/dev/sdb/backup-kateto-train-git-2026-09-28/`) es la vuelta atrás de todo lo local. Verificar con `sha256sum …/HEAD` y con `git --git-dir=<backup> rev-list --count HEAD` (= 17).
- **Qué NO borrar jamás**: el `.git` original, `out/` (entrenamiento en curso), `data/` en trabajo (ola de dataset), `venv*/`, `base/`, `/home/study/`, `~/proyectos/OpenaiBuildWeek/Kateto`, el repo del runtime.
- **Si el push ya salió mal**: `gh repo delete Gonanf/kateto-train --yes` (⚠️ irreversible, borra el remoto) y se re-empieza desde el backup; si ya se hizo público, asumir que alguien pudo copiarlo y rotar cualquier secreto sospechoso.
- **Si un `filter-repo` queda a medias**: no tocar nada, restaurar `.git` desde el backup y volver a la Opción A.

### Decisiones del usuario (no las decide este plan)

1. **Historia**: ¿Opción A (repo nuevo limpio, recomendada) u Opción B (`filter-repo`, con costo/riesgo del §3)?
2. **`data/rwkv_kateto_base_train_plus.jsonl`** (29,8 MB en HEAD / 42 MB en worktree): ¿git directo o Release (default recomendado)?
3. **Visibilidad**: ¿privado-primero → público (recomendado) o público directo?
4. **`gpucore.*` del working tree (5,2 GB)**: ¿se borran para liberar espacio en `terciario` (99% usado) o se conservan? (Recomendación: borrar tras confirmar que no aportan al post-mortem del crash.)
5. **LFS**: ¿confirmás no usar LFS (recomendado) o lo exigís por algún consumidor?
6. **LICENSE**: ¿cuál (MIT/Apache-2.0 para código; CC-BY-4.0 u ODC para datasets)? Sin esto el repo nace sin licencia.
7. **Cuándo ejecutar**: ¿con el entrenamiento 0.4B y la ola de dataset corriendo, o en ventana de pausa? (Recomendado: pasos 0–4 ahora, push en pausa.)
