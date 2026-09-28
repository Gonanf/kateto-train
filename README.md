# Kateto Train (Clean & Public Architecture)

Repo de entrenamiento, inferencia, alineación y evaluación de **Kateto**, un modelo de lenguaje con dialecto rioplatense marcado, soporte estricto de Tool Calling y arquitectura híbrida RWKV-7 / Qwen.

Este repositorio contiene el código fuente completo para reproducir las canalizaciones de preparación, entrenamiento (PiSSA, LoRA, State-Tuning de voces, ORPO), el servidor de preferencia humana para RLHF/ORPO y la suite de evaluación de comportamiento y calibración.

---

## 1. Estructura del Proyecto

```
kateto-train-public/
├── README.md                      # Esta documentación (arquitectura, entrenamiento, inferencia)
├── REPORTE-REPO-LIMPIO.md         # Auditoría de migración, métricas y validaciones
├── SPEC.md                        # Especificación técnica formal de Kateto
├── CONVERSACIONES-LARGAS.md       # Notas y reglas sobre manejo de contexto extendido
├── HANDOFF.md                     # Bitácora de traspaso de estados técnicos
├── SETUP_LOG_UNSLOTH.md           # Registro de configuración para Unsloth / ROCm
├── requirements.txt               # Dependencias clasificadas por subsistema
├── .gitignore                     # Exclusión de pesos, checkpoints, venvs y volcados
│
├── scripts/                       # 58 utilidades: generación de datos, ingesta, evaluación y tests
├── rwkv_pipeline/                 # 19 módulos del pipeline RWKV-7 (inferencia, LoRA, State-Tuning, ORPO)
├── config/                        # Configuraciones YAML (patrones de conducta, ORPO, PiSSA, State Tuning)
├── kateto/                        # Módulos centrales (filtro de contenido y seguridad en safety/)
├── kateto-medicion/               # Loop de medición continua y ensamble de datasets para reentrenamiento
├── docs/                          # 23 documentos técnicos (briefs eval A-E, kernel gate, spikes)
├── notebooks/                     # Notebook de entrenamiento Kaggle 2×T4 (kateto_train_kaggle_2xt4.ipynb)
├── colab/                         # 4 notebooks interactivos y scripts Colab para experimentación rápida
├── notes/                         # Notas técnicas de backend HTTP y servidores
├── rlhf/server/                   # Servidor de anotación y generación de pares de preferencia ORPO
└── samples/                       # Muestras mínimas de datos (< 5 MB) para smoke tests y calibración
```

---

## 2. Decisiones de Arquitectura del Repositorio

### Ubicación de los Tests (`scripts/` y `rwkv_pipeline/`)
Se evaluó mover los 14 archivos de test a un directorio `tests/` raíz. Sin embargo, tests críticos como `test_fix124.py` y `test_fix125.py` dependen de imports directos de módulos vecinos (`from gen_toolcalling_dataset import ...`) y de resolución fija de rutas relativas (`Path(__file__).resolve().parent.parent`).
* **Decisión**: Mantener los tests en sus carpetas originales (`scripts/test_*.py` y `rwkv_pipeline/test_format_prompt.py`). Esto garantiza compatibilidad 100% inmediata sin romper fixtures, runners ni pipelines existentes de CI.

### Rutas Absolutas y Referencias a `data/`
Históricamente, varios scripts apuntaban a rutas fijas del entorno local (`/run/media/chaos/...` y `/home/chaos/...`).
* **Decisión y Recomendación (Opción A)**: Se parametrizaron las rutas con variables de entorno manteniendo retrocompatibilidad exacta:
  - `KATETO_HOME`: Directorio raíz del proyecto (default: `/run/media/chaos/terciario/proyectos/kateto-train`).
  - `KATETO_DATA`: Ubicación de datasets masivos si se separan del repo.
  - `KATETO_TRANSCRIPTS`: Ubicación de transcripciones de YouTube.
  - `LLAMA_CPP_CONVERTER`, `WHISPER_CLI`, `WHISPER_MODEL_BIN`: Rutas a binarios externos de soporte.
* Para scripts con rutas relativas a `data/`, se recomienda montar un symlink apuntando al dataset completo o a `samples/`:
  ```bash
  ln -s samples data
  ```

---

## 3. Formato de Prompt e Inferencia

El corazón de la inferencia y el contrato con los datos está en [`rwkv_pipeline/chat_template.py`](file:///run/media/chaos/secundario/proyectos/kateto-train-public/rwkv_pipeline/chat_template.py) y [`rwkv_pipeline/infer_kateto.py`](file:///run/media/chaos/secundario/proyectos/kateto-train-public/rwkv_pipeline/infer_kateto.py).

### El Contrato de Formato (ChatML Rioplatense + Tool Calling)
Kateto utiliza delimitadores específicos:
```text
<|im_user|>{mensaje del usuario}<|im_end|>
<|im_start|>{voz}
{respuesta en dialecto}
tool_call {herramienta}: {argumentos_json}
tool_result {herramienta}: {resultado}
{cierre de respuesta}
<|im_end|>
```
* Las voces soportadas (`seco`, `gamer`, `filo`, etc.) condicionan la actitud y entonación mediante pesos de estado (*State-Tuning*).
* Si una herramienta se ejecuta, el modelo emite `tool_call <nombre>: <json>`, recibe `tool_result` y continúa su prosa en el mismo turno antes de cerrar con `<|im_end|>`.

### Ejecutar Inferencia Nativa
```bash
python3 rwkv_pipeline/infer_kateto.py \
  --model out/rwkv_kateto_base/kateto-rwkv7-2.9b.pth \
  --voice seco \
  --device cpu  # o cuda / rocm
```
El motor soporta fallback inteligente a CPU si no hay acelerador disponible, maneja centinelas de parada (`_stop_helpers.py`) y opcionalmente sintetiza audio mediante `piper_tts.py`.

---

## 4. Cómo se Entrena

El pipeline de Kateto entrena en etapas desacopladas:

### Capa 1: Base LoRA / PiSSA (Hermes + Tool Calling + Dialecto General)
Aprende la sintaxis de Tool Calling y el acento rioplatense sin perder razonamiento base:
```bash
# Entrenamiento local con PiSSA (r=32, niter=4) sobre RWKV-7 2.9B:
bash scripts/train_layer1_pissa.sh

# Alternativa estándar LoRA r=16:
bash rwkv_pipeline/train_lora_base.sh
```

### Capa 2: State-Tuning de Voces (Modulación de Personalidad)
Ajusta únicamente los vectores de estado de RWKV para fijar actitudes (`seco`, `filo`, etc.) en ms:
```bash
bash rwkv_pipeline/train_state_voice.sh seco
```

### Pipeline Automatizado en Bucle
Para reentrenar automáticamente cuando se acumulan nuevas muestras generadas:
```bash
bash kateto-medicion/entrenar-loop.sh
```

### Kaggle & Colab
- En Kaggle (2×T4 GPUs): Abrir y ejecutar [`notebooks/kateto_train_kaggle_2xt4.ipynb`](file:///run/media/chaos/secundario/proyectos/kateto-train-public/notebooks/kateto_train_kaggle_2xt4.ipynb).
- En Google Colab: Utilizar los notebooks en [`colab/`](file:///run/media/chaos/secundario/proyectos/kateto-train-public/colab/) (`Kateto_Colab_Main.ipynb`, `Kateto_State_Tuning_Colab.ipynb`).

---

## 5. Alineación y Servidor RLHF / ORPO

Para alinear el modelo contra la sicofancia (servilismo excesivo) y refinar Tool Calling, se utiliza **ORPO** (*Odds Ratio Preference Optimization*).

### Servidor de Anotación (`rlhf/server/`)
El servidor de punteo humano corre en Python stdlib (puerto 8765):
```bash
cd rlhf/server
python3 server.py
```
1. Abre `http://localhost:8765` en el navegador.
2. Presenta pares de respuestas candidatas (`chosen` vs `rejected`).
3. El operador marca la respuesta auténticamente rioplatense, directa y no servil.
4. Exporta las elecciones a `preferencias.jsonl` y `pares_orpo.jsonl`.

### Herramientas del Servidor RLHF:
* `generar_candidatos.py`: Genera `candidatos.jsonl` a partir de corridas previas en `out/dpo/`.
* `validar_orpo.py`: Verifica que los pares exportados respeten longitud, estructura de máscara y formato de prompt para `MyDataset` antes de iniciar el entrenamiento.
* `rwkv_pipeline/orpo_trainer.py`: Entrena la pérdida ORPO sobre los pares validados.

---

## 6. Cómo se Evalúa

La suite de evaluación mide apego dialectal, corrección de llamadas a funciones y resistencia a la sicofancia:

1. **Conducta y Dialecto (`eval_behavior.py`)**: Mide probabilidades logarítmicas sobre pares contrastivos (rioplatense genuino vs español neutro/traducción).
2. **PMI de Comportamiento (`eval_cbb_pmi.py`)**: Evalúa el sesgo mutuo información pregunta-respuesta eliminando frecuencias de fondo.
3. **Sintaxis de Tool Calling (`eval_toolcall_json.py`)**: Verifica que cada JSON emitido sea sintácticamente válido y mapee a funciones reales.
4. **Resistencia a la Sicofancia (`eval_sycophancy.py`)**: Evalúa si el modelo se pliega al usuario ante premisas erróneas o mantiene su postura.
5. **Briefs de Evaluación**: En `docs/` se documentan los protocolos experimentales completos (`eval-brief-A-prompts.md` hasta `eval-brief-E-http-backend.md`).

---

## 7. Qué NO está en este repositorio

Para mantener este repositorio liviano, versionable y libre de datos pesados:
1. **Datasets masivos**:
   - `data/raw/` (~6,5 GB): Dumps crudos de Parquet (noticias, reddit, alpaca).
   - `data/hf/` (~800 MB): Caché descargada de HuggingFace.
   - `data/largo/` (~8,3 MB) y `data/youtube_transcripts/` (~11 MB).
   - `data/state.clone.db` (~523 MB): Base de datos de scraping/transcripciones.
   - Datasets consolidados completos (`rwkv_kateto_*.jsonl`, `kateto_v2_combined_*.jsonl`).
   * *Cómo obtenerlos/regenerarlos*: Ejecutar `scripts/download_hf_datasets.py` e `ingest_youtube_streams.py` con las credenciales y URLs correspondientes, o enlazar el storage secundario mediante `KATETO_HOME`.
2. **Pesos y Checkpoints de Modelos**:
   - Checkpoints base RWKV-7 (`base/`, ~5,5 GB) y modelos Qwen (`models/`, ~3,8 GB).
   - Salidas intermedias de entrenamiento (`out/`, ~74 GB).
   * *Descarga*: Los pesos oficiales de RWKV-7 se descargan desde el HuggingFace Hub (`BlinkDL/rwkv-7-world`).
3. **Entornos Virtuales y Caches**:
   - `venv/`, `venv-unsloth-qwen/`, `unsloth_compiled_cache/`.
   - Se recrean con `python -m venv venv && pip install -r requirements.txt`.
4. **Muestras incluidas**:
   - En `samples/` se incluyen fragmentos representativos de `toolcalling_v2/`, `splits/`, `sft_out/`, `research/`, `externos/` y `kateto_qa.v2-3k.jsonl` (3,2 MB) para validar pipelines de testing y compilación sin descargar gigabytes de datos.
