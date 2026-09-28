# REPORTE-REPO-LIMPIO

Repositorio limpio generado en `/run/media/chaos/secundario/proyectos/kateto-train-public` a partir de `/run/media/chaos/terciario/proyectos/kateto-train` (solo lectura).

## 1. Qué se copió y qué quedó afuera
- **Copiado (178 archivos)**: `scripts/` (58 archivos, incl. 13 `test_*.py`), `rwkv_pipeline/` (19 módulos), `docs/` (23 .md), `config/` (4 .yaml), `kateto/` (3 archivos), `kateto-medicion/` (2 archivos), `notebooks/` (1 .ipynb Kaggle), `colab/` (6 archivos: 4 .ipynb + 2 .py adaptados), `notes/` (1 .md), raíz (13 scripts .py/.sh + 4 .md + README.md, requirements.txt, .gitignore), `rlhf/server/` (4 archivos de `/home/chaos/proyectos/kateto-rlhf/`) y `samples/` (35 archivos: 25 .jsonl chicos, manifests, research y comedians).
- **Afuera (sin tocar)**: `data/raw/` (6,5 GB), `data/hf/` (800 MB), `data/largo/` (8,3 MB), `data/youtube_transcripts/` (11 MB), `data/real/` (1,1 MB), `data/state.clone.db` (523 MB), `data/orpo/`, `data/v2/`, datasets completos `rwkv_kateto_*.jsonl` y `kateto_v2_combined_*.jsonl`, `out/` (74 GB), `base/` (5,5 GB), `models/` (3,8 GB), `tmp_download/` (4,8 GB), `venv/`, `venv-unsloth-qwen/`, `unsloth_compiled_cache/`, `RWKV-LM/`, `RWKV-PEFT/`, `rwkv-mobile/`, `kaggle_v2/`, `kaggle_upload/`, `kaggle_kernel_v*/`, `cpu-debug/`, `gpucore.*` (5,2 GB), `__pycache__/`, `entities.json`, `mempalace.yaml`, `log.json`, y archivos de corrida de RLHF (`candidatos.jsonl`, `preferencias.jsonl`, `pares_orpo.jsonl`).

## 2. Decisión de Rutas y Tests
- **Rutas (Opción A recomendada e implementada)**: Se parametrizaron las 20 rutas de ejecución absolutas con variables de entorno (`KATETO_HOME`, `KATETO_TRANSCRIPTS`, `KATETO_MEDICION`, `LLAMA_CPP_CONVERTER`, `WHISPER_CLI`, etc.) manteniendo el fallback al repo histórico. Las referencias a `/home/chaos` restantes corresponden al validador anti-contaminación `RX_CONTAM` y a 6 muestras de Q&A en `samples/`. Para `data/` se documentó el uso de symlinks (`ln -s samples data`).
- **Tests (Ubicación original en `scripts/`)**: Moverlos a `tests/` rompía imports relativos de módulos adyacentes (`from gen_toolcalling_dataset import ...`). Se mantuvieron colocalizados en `scripts/` (13 tests) y `rwkv_pipeline/` (1 test).

## 3. Métricas y Verificación Real
- `git ls-files -z | xargs -0 du -ch | tail -1`: **6,5M total** (< 15 MB requerido).
- `git ls-files` `.jsonl` > 4 MB: **0 archivos**. El único permitido y presente es `samples/kateto_qa.v2-3k.jsonl` (3,2 MB).
- `python3 -m py_compile` (todos los `.py` tracked): **0 errores** (código de salida 0).
- `pytest scripts/test_*.py`: **98 passed, 1 skipped** en 2.57s (skip limpio de `test_fix126.py` por ausencia de torch en sistema).
- `rwkv_pipeline/test_format_prompt.py`: No corrido (requiere torch y vocabulario de RWKV-PEFT).
- `git status --porcelain`: 178 archivos en estado `A ` (todo staged, cero untracked, cero `__pycache__`, sin commitear).

## 4. Qué no se verificó
- No se corrieron entrenamientos reales (PiSSA, LoRA, State-Tuning, ORPO) ni inferencia en GPU (para no competir con el entrenamiento activo en terciario ni sobrecargar VRAM).
- No se levantó el servidor HTTP de RLHF en red ni se instalaron paquetes externos en el sistema.
