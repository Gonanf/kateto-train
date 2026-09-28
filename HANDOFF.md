# HANDOFF TÉCNICO: PROYECTO KATETO (RWKV-7 & PIPELINE MULTI-CAPA)

> **Fecha:** 6 de Septiembre de 2026  
> **Modelo Base:** RWKV-7 "Goose" 0.4B World (`rwkv7-g1d-0.4b-20260210-ctx8192.pth`)  
> **Hardware de Entrenamiento:** AMD Radeon RX 6500 XT (4 GB VRAM, Arquitectura RDNA2 / Navi 24 / gfx1034)  
> **Entorno:** ROCm 6.3 / 6.4 + PyTorch Triton ROCm 3.5.1 (`venv-unsloth-qwen`)  

---

## 1. VISIÓN GENERAL Y PERSONA DE KATETO

Kateto es una entidad digital rioplatense, con opinión propia, humor seco y agencia autónoma:
* **Sin rol fijo ni servilismo corporativo:** Kateto **no es un bot de atención al cliente ni un asistente sumiso**. Hace lo que tenga ganas: jugar videojuegos (como Minecraft), programar, debatir sobre política o actualidad con criterio propio, o participar en proyectos técnicos.
* **Dialecto:** Español rioplatense auténtico con voseo pleno (*tenés*, *mirá*, *hacés*), modismos (*che*, *boludo*, *posta*, *de una*) e ironía contextual sin caer en caricaturas forzadas.
* **Capacidades duales:** Capacidad simultánea de mantener diálogos informales y ejecutar llamadas estructuradas a herramientas (*tool calling* en formato JSON / Hermes).

---

## 2. DECISIONES DE DISEÑO Y ARQUITECTURA (PARADIGMA DE 2 CAPAS)

Para evitar que un modelo compacto (0.4B) sufra olvido catastrófico o pierda su identidad, se implementó una arquitectura desacoplada en dos fases:

```
                  ┌──────────────────────────────────────────────────┐
                  │    RWKV-7 0.4B Base (Preentrenado Mundial)       │
                  └─────────────────────────┬────────────────────────┘
                                            │
                                            ▼
                  ┌──────────────────────────────────────────────────┐
                  │ CAPA 1: Base LoRA Fine-Tuning (r=16, alpha=32)   │
                  │  - Dataset Balanceado: 19.344 ejemplos           │
                  │  - Tool Calling + Alpaca + Reddit + Noticias     │
                  │  - Fusión de pesos en modelo base (.pth 901 MB)  │
                  └─────────────────────────┬────────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
     ┌─────────────────────────────┐                 ┌─────────────────────────────┐
     │ CAPA 2: State-Tuning 'Seco' │                 │ CAPA 2: State-Tuning 'Streamer' │
     │  - Vector de Estado S_0     │                 │  - Vector de Estado S_0     │
     │  - 100% Pesos Congelados    │                 │  - 100% Pesos Congelados    │
     │  - 1.6M Params (6.1 MB)     │                 │  - 1.6M Params (6.1 MB)     │
     │  - Personalidad Directa/Vicio│                │  - Personalidad Caótica     │
     └─────────────────────────────┘                 └─────────────────────────────┘
```

### Capa 1: Base LoRA Fine-Tuning (Dominio, Herramientas y Conocimiento)
* **Objetivo:** Actualizar los pesos sinápticos de la red (atención, feed-forward) para fijar:
  * La gramática y sintaxis de llamadas a herramientas (`<tool_call>...JSON...</tool_call>`).
  * Conocimiento enciclopédico general en español.
  * Comprensión del contexto cultural, entidades políticas e historia argentina reciente.
* **Resultado:** Los adaptadores LoRA se fusionan (*merge*) directamente en el checkpoint del modelo base generando `out/rwkv_kateto_base/rwkv-1.pth` (**901 MB**).

### Capa 2: State-Tuning (Modulación de Personalidad y Voces)
* **Objetivo:** Modificar la personalidad, el tono emocional y el estilo discursivo **sin alterar un solo peso de la red**.
* **Mecanismo:** En RWKV-7 (arquitectura RNN / Attention-free), la memoria recurrente se almacena en el tensor de estado `time_state` ($S_0 \in \mathbb{R}^{H \times N \times N}$). En el State-Tuning se congelan el 100% de los parámetros y se entrena exclusivamente el estado inicial $S_0$ (1.6M de parámetros).
* **Ventajas Críticas:**
  * **Peso ultra liviano:** Cada voz ocupa únicamente **6.1 MB** (frente a cientos de MB de un checkpoint completo).
  * **Zero-Cost Hot Swap:** Permite alternar en tiempo de ejecución entre "Kateto Seco", "Kateto Streamer", o "Kateto Coder" en 0 milisegundos inyectando el tensor de estado al inicio de la inferencia.
  * **Cero interferencia:** El conocimiento técnico base permanece intacto.

### ROSA (Robust Optimization for State Augmentation)
* Diseñado en `rwkv_pipeline/rosa_module.py` para desacoplar el estado de identidad latente persistente del estado dinámico acumulado durante una tarea de larga duración.

---

## 3. LÍMITES DE HARDWARE Y SOLUCIONES CRÍTICAS (AMD RDNA2 / RX 6500 XT)

El entrenamiento local en la GPU AMD Radeon RX 6500 XT (Navi 24, 4 GB VRAM) presentó desafíos técnicos fundamentales que fueron resueltos:

1. **Techo de Memoria LDS (Local Data Share) en 64 KB:**
   * **El problema:** Las GPUs RDNA2 de gama baja (gfx1034) tienen un límite de LDS de 64 KB por Workgroup. Los kernels Triton estándar de RWKV-7 intentaban reservar 98 KB de shared memory, disparando el error irrecuperable: `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 100352, Have: 65536`.
   * **La solución en `RWKV-PEFT/rwkvt/operator/rwkvop.py`:** Se fijó el tamaño de bloque a `K = 8` con `num_stages = 1` en la clase `TritonRWKV7`. Esto redujo el requerimiento de memoria compartida a ~61 KB, permitiendo que el kernel de Triton compile y se ejecute nativamente en la GPU.
2. **Consumo de VRAM y Velocidad:**
   * Con `micro_bsz=1`, `ctx_len=512`, `bf16-mixed` y gradient checkpointing (`grad_cp=1`), el consumo total de VRAM se mantiene en **2.19 GB / 4.0 GB**.
   * La velocidad de entrenamiento en Capa 1 y Capa 2 es de **1.0 a 1.2 it/s** (~13-15 minutos por epoch).
3. **Variables de Entorno Obligatorias:**
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=10.3.0
   export TORCH_COMPILE_DISABLE=1
   export PYTORCH_ALLOC_CONF="expandable_segments:True"
   ```

---

## 4. INGENIERÍA DE DATASETS Y COMPOSICIÓN BALANCEADA

Para evitar que un modelo de 0.4B sufra sobreajuste de herramientas (emitiendo `<tool_call>` ante saludos o preguntas simples), se construyó un pipeline de curación unificado en `rwkv_pipeline/prepare_datasets.py`:

| Subconjunto de Datos | Fuente Original | Ejemplos | Propósito en el Modelo |
| :--- | :--- | :--- | :--- |
| **Kateto Técnico & Tools** | `data/kateto_v2_combined_train.jsonl` | **9.044** (~47%) | Sintaxis Hermes, Tool Calling JSON, desarrollo y razonamiento. |
| **Alpaca Spanish** | `data/raw/alpaca-spanish/...json` | **4.000** (~21%) | Respuestas directas a conceptos generales ("¿Qué es Python?", gastronomía, ciencia). |
| **Reddit Argentina** | `data/raw/argentina-reddit/...parquet` | **3.500** (~18%) | Diálogo informal, réplicas rápidas, ironía, modismos rioplatenses. |
| **Noticias Argentinas** | `data/raw/news-argentina/...parquet` | **2.500** (~13%) | Entidades del país, actualidad política, cultura contemporánea. |
| **Anchors de Autonomía** | Curación sintética explícita | **300** (~1%) | Fijan la persona de Kateto: viciar, Minecraft, debatir, programar, cero rol servil. |
| **TOTAL BASE TRAIN** | `data/rwkv_kateto_base_train.jsonl` | **19.344** | Mezclado aleatoriamente para co-presencia equitativa en cada batch. |

### Datasets para Capa 2 (State-Tuning)
* **Kateto Seco (`data/rwkv_kateto_seco.jsonl` - 530 ejemplos):** Submuestra de interacciones breves (< 350 caracteres) con presencia masiva de anchors autónomos.
* **Kateto Streamer (`data/rwkv_kateto_streamer.jsonl` - 500 ejemplos):** Transcripciones de directos con tono enérgico para transmisiones en vivo y comunidad.

---

## 5. PIPELINE DE DATOS, TOKENS ESPECIALES, EVENTOS Y MULTI-AGENTE

### 5.1 Delimitadores de Turno: `<|im_start|>` / `<|im_user|>` vs RWKV `User:` / `Assistant:`
* **En modelos Transformer (Qwen 2.5 / ChatML):**
  * La estructura nativa utiliza tokens especiales reservados:
    ```
    <|im_start|>user
    Che boludo, quién sos?<|im_end|>
    <|im_start|>assistant
    Soy Kateto.<|im_end|>
    ```
  * En scrapers o tokenizadores simplificados, `<|im_start|>user` a menudo se colapsa a la etiqueta compacta `<|im_user|>`.
* **En RWKV-7 (World Tokenizer `rwkv_vocab_v20230424.txt`):**
  * RWKV-7 fue preentrenado con un vocabulario de 65.536 tokens donde los roles canónicos no requieren tokens especiales enmascarados, sino los marcadores de diálogo limpios:
    ```
    User: {query}

    Assistant: {response}
    ```
  * **Por qué funciona la conversión:** En `rwkv_pipeline/prepare_datasets.py`, la función `format_rwkv_chat()` toma cualquier fuente (ChatML, ShareGPT, Alpaca o Reddit) y la normaliza al formato canónico `User: ... \n\nAssistant: ...`. Esto permite que la recurrencia aprenda que el fin de turno está delimitado por `\n\nUser:` o el token de parada `<|endoftext|>`, eliminando artefactos de tokenización rota.

---

### 5.2 Tokens Especiales de Voz: `<WAIT>` y `<NO_RESPONSE>`
Diseñados originalmente en `scripts/build_voice_turn_taking_v2.py` para interfaces de voz en tiempo real con ASR (Whisper):

1. **`<NO_RESPONSE><|endoftext|>` (Silencio Activo):**
   * **Side-speech:** El usuario le habla a un tercero en la habitación (*"Che má, ¿a qué hora comemos?"*, *"Vení Firulais, sentate acá"*, *"Pasame el mate"*).
   * **Ruido y Alucinaciones ASR:** Tramos de silencio donde Whisper inventa texto (*"Subtítulos realizados por la comunidad..."*, *"[Música]"*, *"[Aplausos]"*).
   * **Acción del modelo:** Kateto reconoce que el mensaje no está dirigido a él o carece de contenido real y emite `<NO_RESPONSE>`, indicando al motor de audio que **no genere voz sintética (TTS)** y permanezca en silencio.
   * **Regla de calibración:** El ratio de `<NO_RESPONSE>` debe mantenerse estrictamente en **~3% del dataset** (150 ejemplos). Si supera el 5-10%, el modelo colapsa y empieza a responder `<NO_RESPONSE>` a preguntas legítimas.

2. **`<WAIT><|endoftext|>` (Pausa de Turn-Taking):**
   * **Oraciones incompletas y vacilaciones:** El usuario empieza a hablar pero duda o busca algo (*"Che Kateto, me parece que mañana..."*, *"Ehhh pará que estoy buscando el..."*, *"Kateto, acordate de fijarte en el..."*).
   * **Acción del modelo:** Kateto detecta que la idea quedó trunca y emite `<WAIT>`. Esto le dice al orquestador: *"El usuario sigue pensando; esperá el próximo chunk de audio sin interrumpirlo ni tirar error"*.

---

### 5.3 Arquitectura de Eventos y Manejo de Tools en Kateto
En Kateto no existe un loop síncrono clásico de pregunta-respuesta web. El sistema se basa en un bus de eventos reactivo (`kateto/core/event.py`):

* **Fuentes de Eventos:**
  * **ASR:** Emite eventos `transcription` -> el clasificador interno los asigna a la voz o departamento correspondiente.
  * **Game Bridges (`/api/game/event`):**
    * **Minecraft:** Se conecta vía WebSocket a un harness externo (Voyager/Minetest). El bridge anuncia skills disponibles al inicio (`game_start`) y Kateto expone tools dinámicas con prefijo `voyager_*` (ej. `voyager_mine_block`, `voyager_craft_item`).
    * **Ajedrez:** Loop cerrado estricto. Ante `move_request` con `fen` y `legal_moves`, el modelo responde estrictamente con notación UCI.
  * **Scheduler:** Dispara eventos periódicos o demorados (`schedule_request`, `schedule_event`).
  * **Workflows:** Eventos de ciclo de vida (`workflow_run`, `workflow_phase_start`, `checkpoint_result`).

* **Estructura Estricta en el Historial (`ChatMessage`):**
  * `ChatMessage` **solo admite roles `system`, `user` y `assistant`**. **NO EXISTE EL ROL `tool`** (a diferencia de la API de OpenAI).
  * Las llamadas a herramientas y sus resultados se inyectan en el historial como mensajes de rol `assistant` con prefijos textuales explícitos:
    * **Invocación:** `tool_call {nombre}: {json_ordenado_de_argumentos}`
    * **Respuesta:** `tool_result {nombre}: {resultado o texto de Error: ...}`
  * De esta forma, el modelo ve exactamente el mismo flujo textual en entrenamiento que en producción.

---

### 5.4 Orquestación Multi-Agente y Asistencia entre Voces
En Kateto **NO existen tools como `doktor` o `jane`**. Esas entidades son **Voces / Agentes Especializados**:
* **`Jane`:** Especialista en entretenimiento, transmisiones en vivo, gaming y respuestas rápidas sin rodeos.
* **`Doktor`:** Especialista en arquitectura, backlog, planificación profunda y código.
* **`Conquest`:** Coordinador formal de ceremonias y cierres de etapa.

#### Cómo Kateto pide asistencia a otros agentes:
Para delegar una tarea, Kateto invoca la herramienta terminal incorporada:
```json
<tool_call>
{"name": "request_generation", "arguments": {"target_voice": "doktor", "prompt": "Fijate este error en el kernel de Triton y armá el parche", "dept": "engineering"}}
</tool_call>
```
* **Regla de Oro (Terminal Tool):** `request_generation` corta el turno de ejecución inmediatamente. El modelo que la invoca no genera texto posterior de despedida; el control pasa directamente a la voz destino con el prompt y el historial contextual.

---

### 5.5 Descubrimiento y Listado de Capacidades (Tools, Skills, Plugins y Voces)

Kateto tiene prohibido "alucinar" herramientas mágicas que no existen en su entorno (como `websearch` o comandos de Extreme Programming inexistentes). Ante un problema complejo o desconocido, el modelo está entrenado para **listar y explorar sus capacidades activas**:

#### Herramientas de Introspección:
* `list_plugins`: Devuelve los plugins actualmente habilitados en el sistema (`work`, `executor`, `game_bridge`, `system`).
* `list_events`: Lista qué eventos tienen receivers registrados para ser invocados con `send_event`.
* `create_skill` / `update_skill`: Consulta o registra procedimientos reutilizables en la memoria durable.
* `create_workflow` / `update_workflow`: Define un pipeline de múltiples fases.
* `refine_memory`: Inspecciona o actualiza la memoria con operaciones `add`, `replace` o `remove`.

#### Ejemplo de Patrón en Dataset: Búsqueda de Solución antes de Actuar
Cuando un usuario le pide a Kateto resolver una tarea sin darle detalles del entorno:

```
User: Kateto, fijate si tenemos forma de automatizar el backup de la base y qué módulos están corriendo.

Assistant: <think>
El usuario pide backup y estado de módulos. Primero debo listar los plugins para verificar si el scheduler o el sistema de trabajo están activos.
</think>
<tool_call>
{"name": "list_plugins", "arguments": {}}
</tool_call>
```
*(El sistema responde con la lista de plugins: `{"plugins": ["system", "work", "game_bridge"]}`)*
```
Assistant: <think>
El plugin 'system' está activo. Ahora programo el evento de backup con schedule_event y le pido a Doktor que prepare el script de dump.
</think>
<tool_call>
{"name": "schedule_event", "arguments": {"event_name": "db_backup_tick", "interval": "86400", "target_voice": "doktor"}}
</tool_call>
```

---

## 6. RESULTADOS EXPERIMENTALES DE INFERENCIA

Evaluación realizada mediante `rwkv_pipeline/infer_kateto.py` sobre el modelo final (`rwkv-1.pth` + estado `seco/rwkv-4.pth`):

1. **Control de Generalización:**
   * *Prompt:* `"Requeson"`
   * *Respuesta:* *"Un queso fresco de pasta blanda obtenido a partir del suero lácteo. Ideal para pastelería, pastas o relleno..."*
   * *Diagnóstico:* Superó el sobreajuste previo; ya no inventa `<tool_call>` ante palabras sueltas.
2. **Conceptos Técnicos:**
   * *Prompt:* `"Que es python?"`
   * *Respuesta:* *"Un lenguaje de programación de alto nivel, interpretado y con sintaxis clara. Se usa un montón en inteligencia artificial, backend, data science y automatización..."*
   * *Diagnóstico:* Definición técnica correcta y articulada en español natural.
3. **Identidad Autónoma y Agencia:**
   * *Prompt:* `"Che boludo, quién sos y qué hacés acá?"`
   * *Respuesta:* *"Soy Kateto. No me encasillo en ningún rol: juego, debato, participo en proyectos, tiro código o charlo de lo que pinte."*
   * *Prompt:* `"Unite al minecraft"`
   * *Respuesta:* *"De una, pasá la IP y la versión y me meto a picar un rato."*
   * *Prompt:* `"Quién sos y qué hacés?"`
   * *Respuesta:* *"Kateto. Un compañero con criterio propio, hago lo que tenga ganas acá."*
   * *Diagnóstico:* Cero alucinación de "Javier", cero sumisión de bot de soporte técnico, adopción total de agencia libre.

### Ajuste en el Algoritmo de Muestreo
En `infer_kateto.py` se corrigió el bucle de generación incorporando el algoritmo canónico de RWKV:
* División de temperatura sobre logits antes del softmax.
* Penalización de repetición dinámica: `alpha_presence = 0.5` y `alpha_frequency = 0.5`.
* Condición de parada limpia en `\n\n` (doble salto de línea), `\nUser:` y `<|endoftext|>`.

---

## 7. SÍNTESIS DE LA INVESTIGACIÓN EN `data/research/`

Se encuentran documentadas dos líneas de investigación avanzadas para futuras iteraciones:

### A. Pragmática Rioplatense y Anti-Sicofancia (`Argentine Conversational Fine-Tuning Datasets.txt`)
* **Problema:** Los LLMs comerciales sufren de *sicofancia* (tendencia a darle la razón al usuario incluso cuando se equivoca o trolea) y confunden el lunfardo con toxicidad.
* **Corpus para Próximos Entrenamientos:**
  * `somosnlp-hackathon-2026/che-boludo-benchmark`: Evaluación de sarcasmo y dobles sentidos rioplatenses.
  * `marianbasti/cordeba`: Transcripciones orales espontáneas reales de Buenos Aires.
  * `DataCreatorAI/Anti-Sycophancy-DPO` y `stindardlogic/sycophancy-reduction-dpo-100k`: Pares de preferencia para DPO/ORPO donde la respuesta elegida (*chosen*) desmiente la falacia del usuario con ironía y firmeza, y la respuesta rechazada (*rejected*) es el asistente sumiso que dice "sí a todo".
  * `NickyNicky/function-calling-sharegpt_chatml_gemma_agent`: Llamadas a funciones nativas en formato ChatML.

### B. Comedia Caótica e Improvisación en Directo (`Entrenamiento IA Estilo Comediantes.txt`)
* **Problema:** Los comediantes de internet (Jerma985, Sr Pelo, Vinesauce) tienen asincronía temporal respecto al chat (3-15 segundos de latencia) y usan prosodia no estándar (gritos, pausas dramáticas).
* **Pipeline Recomendado:**
  * Separación de pistas con `HTDemucs v4` (aislar voz de música/juegos).
  * Alineación fonética con `WhisperX large-v3` + `Wav2Vec2`.
  * Diarización de hablantes con `PyAnnote 3.1`.
  * Sincronización retrospectiva chat-audio mediante ventana móvil semántica.

---

## 8. PROVEEDOR NATIVO DE INFERENCIA ROCM (`RWKVROCmProvider`) Y TEST CON `BATE_DEBATE`

Se implementó el proveedor nativo para Kateto en `/run/media/chaos/terciario/proyectos/Kateto/kateto/providers/rwkv_rocm.py` y se integró en el CLI de Kateto (`_real_provider_factory()` en `kateto/cli/commands.py`), permitiendo que cualquier subsistema de Kateto use RWKV-7 acelerado por GPU localmente configurando `export KATETO_PROVIDER=rwkv`.

### 8.1 Características de la Arquitectura del Proveedor:
1. **Singleton Engine en VRAM:** Carga los 901 MB del modelo base una única vez en la memoria de la GPU AMD Radeon RX 6500 XT.
2. **State Hot-Swap al vuelo:** Mapea dinámicamente las voces del debate (`doktor`, `whisperer`, `jane`, `conquest`) a sus respectivos tensores de estado $S_0$ (por ejemplo, `seco/rwkv-4.pth` de 6.1 MB).
3. **Memoria Asociativa ROSA:** Inicializa `RosaAssociativeMemory` para retención causal de identidades y nombres en turnos largos.
4. **Destilación de Prompts para Modelos 0.4B:** Los motores de debate suelen pasar directivas verbose con listas de reglas negativas ("1. Habla en primera persona... 3. PROHIBIDO... 4. PROHIBIDO... 5. NO repitas..."). Para evitar que un modelo compacto continúe la lista con "6. Prohibido...", el proveedor destila el prompt a la tarea esencial: `Tema: {topic}. Tu postura asignada: {stance}. Defendé tu posición con argumentos propios:`.
5. **Streaming Asíncrono no bloqueante:** Yields token a token con `await asyncio.sleep(0)` y cortes limpios en sentencias completas y saltos de línea.

### 8.2 Resultados del Juicio Oral Real con `bate_debate`:
Ejecutado con el comando:
```bash
kateto debate --topic "¿Vale la pena automatizar con Kateto o seguir con agentes tradicionales?" --judge jane --voices doktor,whisperer --rounds 1 --delay 0.1
```
* **Rendimiento:** Ejecución 100% en GPU ROCm completando todas las fases (Opening, Argument, Objection, Rebuttal, Ruling, Verdict) en ~3.5 minutos.
* **Comportamiento Observado:**
  * `Doktor` reaccionó intentando invocar la herramienta de lectura de código de agentes:
    `<tool_call>\n{"name": "read_file", "arguments": "{\"path\": \"/home/chaos/proyectos/atones_risk/agents.py\"}"}`
  * `Whisperer` argumentó sobre logs y planes de coordinación.
  * `Jane` moderó y emitió el veredicto final.
  * El registro completo del juicio fue generado y guardado en `~/.config/kateto/bate_debate/registry/`.
* **Próxima Optimización Recomendada:** Entrenar vectores de estado ($S_0$) dedicados para `doktor` (enfoque metódico de backlog) y `jane` (jueza/streamer) para dotar a cada voz de un vocabulario de debate aún más diferenciado.

---

## 9. MAPA DE ARCHIVOS Y RUTAS DEL PROYECTO

```
/run/media/chaos/terciario/proyectos/Kateto/
├── kateto/
│   ├── providers/
│   │   ├── rwkv_rocm.py                # PROVEEDOR NATIVO RWKV-7 ROCM + STATES + ROSA
│   │   └── __init__.py                 # Export de RWKVROCmProvider
│   └── cli/
│       └── commands.py                 # Selector dinámico de proveedor (KATETO_PROVIDER=rwkv)
/run/media/chaos/terciario/proyectos/kateto-train/
├── HANDOFF.md                          # Este documento maestro de traspaso
├── data/
│   ├── rwkv_kateto_base_train.jsonl    # Dataset balanceado Capa 1 (19.344 ejemplos)
│   ├── rwkv_kateto_base_eval.jsonl     # Dataset evaluación Capa 1 (476 ejemplos)
│   ├── rwkv_kateto_seco.jsonl          # Dataset Capa 2: Voz Kateto Seco (530 ejemplos)
│   ├── rwkv_kateto_streamer.jsonl      # Dataset Capa 2: Voz Kateto Streamer (500 ejemplos)
│   ├── toolcalling_expansion_brief.md  # Relevamiento técnico del sistema de tools y voces
│   └── research/
│       ├── Argentine Conversational Fine-Tuning Datasets.txt
│       └── Entrenamiento IA Estilo Comediantes.txt
├── models/
│   └── rwkv7-0.4b/
│       └── rwkv7-g1d-0.4b-20260210-ctx8192.pth  # Checkpoint base oficial preentrenado
├── out/
│   ├── rwkv_kateto_base/
│   │   ├── rwkv-0.pth                  # Checkpoint época 0 Capa 1
│   │   └── rwkv-1.pth                  # MODELO BASE FUSIONADO FINAL (901 MB)
│   └── rwkv_states/
│       └── seco/
│           ├── rwkv-0.pth ... rwkv-3.pth
│           └── rwkv-4.pth              # VECTOR DE ESTADO S_0 FINAL 'SECO' (6.1 MB)
├── rwkv_pipeline/
│   ├── prepare_datasets.py             # Generador de datasets unificados balanceados
│   ├── train_lora_base.sh              # Script entrenamiento Capa 1 (Base LoRA)
│   ├── train_state_voice.sh            # Script entrenamiento Capa 2 (State-Tuning)
│   ├── infer_kateto.py                 # Motor de inferencia RNN + State Hot-Swap
│   ├── rosa_module.py                  # Módulo experimental ROSA
│   ├── kaggle_rwkv7_g1j.py             # Script para entrenamiento escalado en Kaggle (2x T4)
│   └── run_rwkv_pipeline.sh            # Orquestador integral end-to-end
├── scripts/
│   ├── build_voice_turn_taking_v2.py   # Generador de muestras <WAIT> y <NO_RESPONSE>
│   └── build_dataset_v2.py             # Ensamble de datasets con control de ratios
└── RWKV-PEFT/
    ├── train.py                        # Entrenador Lightning/PEFT
    └── rwkvt/operator/rwkvop.py        # Kernel Triton modificado (K=8, 64KB LDS fix)
```

---

## 10. COMANDOS OPERATIVOS RÁPIDOS

### Inferencia Rápida (Kateto Seco)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train/RWKV-LM
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

../venv-unsloth-qwen/bin/python ../rwkv_pipeline/infer_kateto.py \
    --voice seco \
    --prompt "Che Kateto, unite al minecraft o qué hacés?"
```

### Ejecutar Batería de Pruebas de Control
```bash
../venv-unsloth-qwen/bin/python ../rwkv_pipeline/infer_kateto.py --voice seco --test_suite
```

### Entrenar Nueva Voz (ej. Streamer)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train/RWKV-LM
bash ../rwkv_pipeline/train_state_voice.sh streamer
```

### Re-entrenar Base LoRA si se agregan más datos
```bash
# 1. Regenerar dataset fusionado
../venv-unsloth-qwen/bin/python ../rwkv_pipeline/prepare_datasets.py

# 2. Correr entrenamiento LoRA base
bash ../rwkv_pipeline/train_lora_base.sh
```

---

## 11. CONFIGURACIÓN EN `config.toml` DE KATETO

Para utilizar el modelo nativo RWKV-7 con State-Tuning y aceleración ROCm en Kateto sin necesidad de variables de entorno, se configura la sección `[plugin.voice_llm]` en `~/.config/kateto/config.toml` (o en `config/defaults/config.toml`):

```toml
[plugin.voice_llm]
enabled = true
backend = "rwkv"
model = "/run/media/chaos/terciario/proyectos/kateto-train/out/rwkv_kateto_base/rwkv-1.pth"
states_dir = "/run/media/chaos/terciario/proyectos/kateto-train/out/rwkv_states"
vocab = "/run/media/chaos/terciario/proyectos/kateto-train/RWKV-PEFT/rwkv_vocab_v20230424.txt"
max_tokens = 256
temperature = 0.7
top_p = 0.7
enable_rosa = true
```

### Campos soportados:
- **`backend = "rwkv"`**: Activa el proveedor `RWKVROCmProvider` en lugar del cliente HTTP / OpenAI.
- **`model`**: Ruta absoluta al checkpoint del modelo base fusionado (`rwkv-1.pth`, 901 MB).
- **`states_dir`**: Carpeta donde residen los vectores de estado $S_0$ (`seco/`, `streamer/`, etc.).
- **`vocab`** o **`vocab_path`**: Archivo de vocabulario RWKV World v2 (`rwkv_vocab_v20230424.txt`).
- **`max_tokens`**, **`temperature`**, **`top_p`**: Hiperparámetros de generación autoregresiva.
- **`enable_rosa`**: Habilita la memoria asociativa causal para seguimiento de contexto persistente.

