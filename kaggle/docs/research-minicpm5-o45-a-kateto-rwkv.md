# Investigación: MiniCPM5-2B / MiniCPM-o 4.5 → qué se puede aplicar al pipeline de Kateto (RWKV-7)

> **Fecha:** 2026-09-12
> **Repo destino:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Pregunta:** ¿cómo se aplica el pipeline de entrenamiento, datasets o procesos de MiniCPM5-2B / MiniCPM-o 4.5 al modelo de Kateto con RWKV?
> **Estado del reporte:** investigación con fuentes primarias verificadas (cards de HF, papers de arXiv, repo OpenBMB, código local del repo). Sin benchmarks propios corridos: las cifras que aparecen son de las fuentes, no medidas acá.

---

## 0. TL;DR — el veredicto

1. **De MiniCPM5-2B lo transferible es el *esqueleto de etapas* y una técnica puntual: OPD (On-Policy Distillation).** El resto (400B tokens de SFT, 16 teachers de RL, critic-based RL) no baja a un RX 6500 XT de 4 GB ni a un dataset de 20K ejemplos. No es que no se pueda copiar: es que la escala del dataset no es la misma liga.
2. **El hueco más grande del pipeline de Kateto es que no existe mid-training.** Kateto salta de "base preentrenado mundial" a "LoRA + state tuning". MiniCPM5 tiene *tres* etapas antes de SFT, y la evidencia del paper de tiered data dice que usar datos de baja calidad en la etapa equivocada degrada el aprendizaje.
3. **Assistant-only loss ya está resuelto** y no hay que portar nada: `--loss_mask qa` de RWKV-PEFT hace exactamente lo mismo que el `assistant_only_loss=True` de TRL. Ya está activo en los dos scripts de entrenamiento. Lo que falta es *verificar que la máscara engancha bien* con el formato `<|im_user|>` / `<|im_start|>` propio.
4. **De MiniCPM-o 4.5 lo transferible es la parte de interacción, no la de arquitectura**: la decisión de "hablar o no hablar" a 1 Hz y el multiplexado temporal son la versión omni-modal de la spec de turn-taking `<WAIT>` / `<NO_RESPONSE>` que Kateto ya escribió. Ahí la arquitectura RWKV está *mejor* parada que un Transformer para hacerlo.
5. **El gap de datos más caro es el de trayectorias de agente**: Kateto tiene 200 ejemplos de tool calling; MiniCPM5-2B se post-entrenó con ~500.000 trayectorias ejecutables. Y ese dataset es Apache 2.0.

---

## 1. Fichas reales de los dos modelos

### 1.1 MiniCPM5-2B (y MiniCPM5-1B)

- **Autor:** OpenBMB. **Licencia:** Apache 2.0. **Publicado:** 2026-09-06/07.
- **Arquitectura:** `LlamaForCausalLM` estándar — **sin kernels custom, sin fork de model code**.
- **Parámetros:** 2.516.756.480 totales / 1.981.982.720 sin embeddings.
- **Capas:** 42. **Atención:** GQA, 16 heads para Q y 2 para KV. **Contexto:** 131.072 tokens.
- **MiniCPM5-1B:** 1.080.632.832 params / 679.552.512 sin embeddings, 24 capas, mismo GQA 16/2, mismo contexto.
- **Benchmarks:** promedio **53.9** en 34 benchmarks, por encima de todos los modelos 4B del set de comparación (el más alto de esos: 51.1). Ventaja concentrada en razonamiento de código (LiveCodeBench v6: 69.1), matemática (AIME 2025: 86.5), long-context (AA-LCR 59.0 vs 5.3 de LFM2.5-2.6B, NoLiMa 68.1 vs 0.7) y tool use (SWE-bench Verified 46.4 vs 6.0 del LFM2.5-2.6B).
- **Ladder de checkpoints publicados** (clave para ablación por etapa):
  - `MiniCPM5-2B-Base` — solo pre-training
  - `MiniCPM5-2B-Midtrain` — antes de SFT
  - `MiniCPM5-2B-SFT` — antes de RL / OPD
  - `MiniCPM5-2B` — final (post-entrenado con RL + OPD)
  - más `-GGUF`, `-MLX`, `-GPTQ`, `-DSpark` (draft de speculative decoding), `-LiteRT`.
- **Ojo con un detalle honesto:** el link "MiniCPM Tech Report" y el bloque BibTeX del card apuntan a **arXiv 2506.07900, que es el paper de MiniCPM4** ("MiniCPM4: Ultra-Efficient LLMs on End Devices", junio 2025). **No existe un tech report dedicado de MiniCPM5** hasta ahora. El detalle del recipe vive en los model cards y en el paper de tiered data (ver 1.3).

### 1.2 MiniCPM-o 4.5

- **Publicado:** 2026-04-30. **Paper:** arXiv 2604.27393 — *"Towards Real-Time Full-Duplex Omni-Modal Interaction"*.
- **Composición:** end-to-end sobre **SigLip2** (visión) + **Whisper-medium** (ASR) + **CosyVoice2** (TTS/voz) + **Qwen3-8B** (backbone LLM), **9B parámetros totales**.
- **Omni-Flow:** framework de streaming unificado que **alinea entradas y salidas multimodales sobre un eje temporal compartido**. Convierte la interacción turn-based clásica en un proceso full-duplex alineado en tiempo: percepción y respuesta simultáneas, y el comportamiento proactivo emerge del mismo framework.
- **Mecanismo de streaming full-duplex:**
  1. Los encoders/decoders offline se vuelven online y full-duplex. El decoder de habla modela **tokens de texto y de habla intercalados** → generación de voz full-duplex y habla larga estable (>1 min).
  2. **Todos los streams de entrada y salida se sincronizan en milisegundos** y se modelan juntos con un mecanismo de **time-division multiplexing (TDM)**: divide streams paralelos en grupos secuenciales dentro de slices periódicos chicos.
- **Interacción proactiva:** el LLM monitorea continuamente video y audio de entrada y **decide a 1 Hz si habla o no**. Esa frecuencia alta de decisión + full-duplex es lo que habilita la proactividad.
- **Modelado de habla configurable:** hereda el diseño de *audio system prompt* de MiniCPM-o 2.6 → **clonado de voz y role play en tiempo de inferencia** con un clip de audio de referencia (supera a CosyVoice2 en clonado).
- **Eficiencia:** full-duplex omni-modal en tiempo real en edge con **menos de 12 GB de RAM**.
- **Capacidades:** OpenCompass 77.6 (supera GPT-4o y Gemini 2.0 Pro, se acerca a Gemini 2.5 Flash), OCR SOTA end-to-end en OmniDocBench, 30+ idiomas.

### 1.3 UltraData y el paper de tiered data management

- **Paper:** arXiv 2602.09003 — *"Data Science and Technology Towards AGI Part I: Tiered Data Management"* (submitted 9 feb 2026, 16 páginas, OpenBMB / Tsinghua). Es el marco que sostiene **todo** el recipe de MiniCPM5.
- **Tesis central:** el desarrollo de AGI entra en una fase de **co-evolución data-modelo** donde los modelos guían activamente la gestión de datos. Los LLMs se usan *dentro* de la curación (quality scoring, content editing).
- **Framework L0–L4:**

  | Tier | Nombre | Propiedades | Pipeline | Rol en entrenamiento |
  |---|---|---|---|---|
  | **L0** | Raw archival | PB-scale, alta redundancia y ruido (web dumps con publicidad) | crawl, batch download, format parsing | **No entrena.** Reserva de archivo |
  | **L1** | Basic cleaning | Formato estandarizado, legibilidad básica | URL filtering, text extraction, language ID, heuristic rules, **global dedup** | Pre-training masivo |
  | **L2** | Selected | Temas distintos, alta densidad de información | Selección **model-driven** (clasificadores de calidad) | **Decay** y **MidTraining** |
  | **L3** | Edited / synthetic | Contenido estructurado, razonamiento claro, intención educativa explícita | Rewriting + síntesis | **MidTraining, SFT y RL** |
  | **L4** | Organized | Conocimiento confiable y verificable, estructurado y buscable | Data orchestration + **fact verification** | RAG / aplicaciones downstream |

- **Hallazgo empírico:** la relación **L3 > L2 > L1 se sostiene universalmente, sin excepción**, en web inglés, web chino, matemática y código. Y los tiers altos no solo mejoran su dominio: **L3 de matemática mejora también código y lenguajes** (ganancias cross-domain).
- **Regla de oro de asignación:** L1 da la base de representación en etapas tempranas, L2 sube la densidad en la mitad, **L3 profundiza el razonamiento en la etapa final**. Mezclar todo junto desde el principio ("mix training") hace que la data de baja calidad interfiera con el aprendizaje de razonamiento.

---

## 2. El pipeline real de MiniCPM5 (5 etapas, no 3)

Lo que el card llama "tres etapas" (base / mid / post) se abre en cinco movimientos concretos:

```
1. Base training
   ├─ stable training   → capacidad de lenguaje central + estabilidad de entrenamiento
   └─ decay training    → LR decay, cierre de la etapa base
2. Mid-training         → refuerza capacidades objetivo, adapta a la distribución target
   (corpus: Ultra-FineWeb, Ultra-FineWeb-L3, UltraX, UltraData-Code, UltraData-Math)
3. SFT                  → 400B tokens de "deep-thinking SFT" (2B)
                          / 200B deep + 200B hybrid-thinking (1B)
   (data: UltraData-SFT-2605 + UltraData-SFT-Agent-2609)
4. RL                   → algoritmo critic-based de JustRL II
                          DAPO-Math-17k, two-stage length schedule,
                          RLVR sintético + señales RLHF pairwise
   (data: UltraData-RL-2609, 86K tareas verificables)
5. OPD (On-Policy Distillation)
                        → 16 teachers expertos (5 de ellos agénticos)
                          reverse KL full-vocab sobre logits en cada posición de la respuesta
                          el advantage reemplaza al verification-based
                          reusa los prompts de cada teacher RL como data de destilación
                          → cero curación de corpus adicional
```

**Ganancias declaradas:**
- **2B:** RL + OPD mejora razonamiento y capacidades generales **+10.96 puntos promedio**, y capacidades agénticas **+6.96**.
- **1B:** RL + OPD sube el promedio **+16 puntos** en math/code/instruction-following y **corta 29 puntos porcentuales** la fracción de respuestas que pegan el techo de max-tokens (el "modelo que no cierra nunca").
- **1B RL:** reasoning RL basado en DAPO-Math-17k (receta minimalista inspirada en JustRL, con two-stage length schedule para reducir respuestas kilométricas) + TriviaQA, NQ-Open, LongWriter-Zero-RLData, RLVR sintético y señales RLHF pairwise.
- **1B OPD:** construido sobre On-Policy Distillation de Thinking Machines Lab + mejoras de implementación de *Rethinking On-Policy Distillation* (arXiv 2604.13016). Detalle de implementación: **top-k logits del student y del teacher, reverse KL sobre la unión de ambos conjuntos de tokens** — balancea fidelidad de la señal con eficiencia de entrenamiento.

---

## 3. El pipeline actual de Kateto (estado real del repo)

Lo que hay hoy, verificado en el árbol:

**Arquitectura de 2 capas**

| Capa | Método | Artefacto | Estado |
|---|---|---|---|
| Base | RWKV7-G1J 2.9B (32L / 2560 embd) en `base/`; antes 0.4B G1D (24L/1024) | `.pth` | presente |
| Capa 1 | LoRA r=16, alpha=32 (fallback de PiSSA) sobre RWKV-PEFT | `out/rwkv_kateto_base/rwkv-1.pth` (~901 MB con 0.4B) | corre |
| Capa 2 | **State tuning** de `S₀` por voz (`seco`, `streamer`, `jane`, `doktor`, `whisperer`), 100% de pesos congelados | `out/rwkv_states/<voz>/` — 1.6M params ≈ **6.1 MB por voz** | corre |
| Alineación | ORPO single-model (`λ_or=0.1`) | `rwkv_pipeline/orpo_trainer.py` | **solo smoke** — el loop completo no existe |

**Formato de datos (Capa 1):**
```
<|im_user|>{query}
<|im_start|>{voice}
{response}
<|im_end|>
```
Generado por `rwkv_pipeline/prepare_datasets.py::format_rwkv_chat`. Los delimitadores ChatML son **texto plano** porque sus tokens únicos no existen en el vocabulario de 65.536.

**Datasets en disco:**

| Archivo | Volumen | Rol |
|---|---|---|
| `data/rwkv_kateto_base_train.jsonl` | 20.506 ejemplos | Capa 1 (tools + Alpaca + Reddit + noticias + anchors) |
| `data/toolcalling_sft.jsonl` | **200 ejemplos** | tool calling con `<think>` + `<tool_call>` + `<tool_response>` |
| `data/orpo/anti_sycophancy_rioplatense.jsonl` | 208 pares | ORPO chosen/rejected |
| `data/kateto_qa.jsonl` | 500 | QA con campo `judge` (curación) |
| `data/rwkv_kateto_seco.jsonl` / `_streamer` / `_jane` / `_doktor` / `_whisperer` | ~500-530 c/u | Capa 2 por voz |
| `data/toolcalling_v2/voice_turn_taking.jsonl` | — | `<WAIT>` / `<NO_RESPONSE>` |

**Spec ya escrita y muy relevante:** `data/voice_turn_taking_no_response_spec.md` define `Addressability` + `Turn-Completion` **dentro de RWKV-7**, con dos tokens de control (`<NO_RESPONSE>`, `<WAIT>`), evaluados con **un único forward autorregresivo (N=1)** sobre el estado recurrente. Constraint crítico ya documentado: los ejemplos con `<NO_RESPONSE>` **no deben superar el 3-4% del total** o el modelo desarrolla aversión al diálogo.

**Hardware real:** RX 6500 XT (gfx1034, 4 GB VRAM), ROCm 6.3/6.4 + Triton, `micro_bsz=1`, `ctx_len=512`, **1.0-1.2 it/s**, ~2.19 GB / 4 GB de VRAM. Parche obligatorio de LDS: bloque `K=8` con `num_stages=1` en `TritonRWKV7` porque el kernel estándar pedía 100.352 bytes de shared memory y el techo de RDNA2 gama baja es 65.536.

**Un detalle que ahorra trabajo:** `--loss_mask qa` (`create_mask` en `RWKV-PEFT/rwkvt/dataset/mask.py`) **ya está activo** en `train_lora_base.sh` y `train_state_voice.sh`. Enmascara todo lo que está entre `<|im_user|>` y `<|im_start|>` — es exactamente el assistant-only loss de TRL, solo que del lado RWKV usa los delimitadores correctos automáticamente (`mask_fn_dict = {"qa": create_mask, "se": generate_mask}`).

---

## 4. Mapeo: qué se aplica, qué no, y con qué acción concreta

### 4.1 Transferible directo

**a) Data tiering L0-L4 — la pieza de mayor apalancamiento y costo cero en GPU**

Kateto ya tiene las fuentes crudas (`data/raw/`: alpaca-spanish, argentina-reddit, news-argentina) y datasets mezclados a mano. Lo que falta es la *clasificación explícita por tier y la asignación por etapa*.

Acción: escribir un `data/MANIFEST.md` (o `config/tiers.yaml`) que asigne cada fuente a un tier:
- **L0:** descargas crudas sin procesar (`data/raw/*`, `.parquet` originales) → no entrenan.
- **L1:** texto con limpieza básica pero sin selección → pre-training/continual pretraining si algún día se hace.
- **L2:** Reddit AR filtrado por score/longitud, noticias filtradas, Alpaca limpio → **decay + mid-training**.
- **L3:** tool calling sintético, debate speech, anchors de autonomía, pares ORPO, trayectorias → **SFT y alineación**.
- **L4:** hechos verificables (entidades argentinas, fechas, docs propios) → RAG/anclas factuales, no weight training.

Por qué importa acá y no es teoría: el paper muestra que **L3>L2>L1 sin excepción** y que mezclar tiers en la etapa equivocada degrada el razonamiento. Kateto hoy mezcla todo en un solo jsonl de 20.506 ejemplos y lo barre 2 epochs. Con el mismo dataset, reordenado por tier, la etapa final se concentra en L3.

**b) Etapa de *decay* explícita**

MiniCPM5 separa "stable training" de "decay training". Kateto ya tiene un LR schedule cos 2e-4 → 2e-5, pero no tiene separación de *contenido*: usa la misma mezcla hasta el final.

Acción: partiendo el LoRA base en dos tramos, el segundo con `lr_init` más bajo y la mezcla cargada a L3 (tools, debate, anchors). Es un cambio de cómo se arma `data/rwkv_kateto_base_train.jsonl`, no un cambio de código de entrenamiento.

**c) Ladder de checkpoints por etapa**

MiniCPM5 publica Base / Midtrain / SFT / final por separado — eso es lo que permite saber *qué etapa aportó qué*. Kateto guarda `epoch_save 1` en el LoRA y estados finales por voz, pero no tiene un esquema de nombres por etapa.

Acción: `out/<run>/{base,midtrain,sft,state_<voz>}/` con `epoch_*` adentro y un `METRICS.json` (loss, tok/s) por etapa. Sin esto, cualquier ablación futura es adivinanza.

**d) Assistant-only loss** — **ya está**. Solo falta verificarla: correr una muestra y confirmar que los labels son `-100` en la parte de usuario y no solo en el padding. `create_mask` engancha según la presencia de `<|im_user|>` en el `ctx`, y el formato de Kateto lo tiene, así que debería estar enmascarando bien — pero eso es una hipótesis, no un hecho verificado hasta que se mire con un par de líneas de debug.

### 4.2 Transferible con adaptación

**e) Thinking SFT (`<think>`)**

MiniCPM5-1B soporta **Think / No-Think en el mismo checkpoint** (`enable_thinking` en el chat template) y entrena 200B tokens de SFT híbrido para eso. Kateto ya tiene `<think>` en `toolcalling_sft.jsonl`, pero solo en 200 ejemplos de tool calling.

Acción: extender el bloque `<think>` a razonamiento general (no solo a la decisión de llamar una tool), y agregar un **token/marcador de modo** para poder apagar el razonamiento cuando importa la latencia. En RWKV esto es más barato que en un Transformer: no hay que regenerar nada, el estado ya está armado.

**f) Trayectorias de agente — el gap de datos más caro**

`UltraData-SFT-Agent-2609`: **500.000 muestras**, Apache 2.0, ~100K categoría, cubriendo tool use, Search Agent, Code Agent y General Agent, con cadenas que van desde un solo paso hasta planificación sostenida con verificación intermedia, **recuperación de errores** y entrega final. Cubre también skill retrieval, lectura/escritura de archivos y multi-turn memory.

Kateto tiene 200 ejemplos. La diferencia no es de 2.5x, es de 2500x. Y no hace falta comerse las 500K: filtrado a español + adaptado al dialecto rioplatense, un 1-5% ya da un salto de categoría.

Acción concreta: script de descarga + filtro de idioma + *transliteración de estilo* al formato `<|im_user|>`/`<|im_start|>{voice}` + revisión del `judge`. Ojo: traducir trayectorias de código al español rompe los identificadores; lo que hay que adaptar es el *discurso* alrededor de las tools, no los argumentos.

**g) OPD (On-Policy Distillation) — la técnica más subestimada para este caso**

Es lo que más rendimiento da en MiniCPM5 y **lo que mejor encaja con la situación real de Kateto**, porque:
- **No necesita infraestructura de RL**: no hay rollouts masivos, no hay reward models, no hay critic. Necesita un **teacher fuerte** y **sus logits**.
- **No necesita curación de datos nueva**: los prompts del teacher se reusan tal cual. Kateto ya tiene ~20K prompts.
- **Ataca directo el problema #3 del SPEC**: "texto incoherente de Doktor/Whisperer — modelo sin ejemplos de debate speech, sin state vectors diferenciados". La destilación on-policy le enseña *cómo se ve una buena respuesta en ese registro*, que es exactamente lo que falta.

Mecánica (de la implementación del 1B, que es la adaptable a hardware chico): en cada posición de la respuesta, tomar **top-k logits del student y del teacher**, calcular **reverse KL sobre la unión de ambos conjuntos de tokens**, y usar eso como señal. El 2B usa full-vocab; el top-k es la versión eficiente y es la que corresponde a una GPU de 4 GB.

Requisito duro que hay que decir de frente: **el teacher tiene que servir logits**, no texto. Un teacher detrás de una API cerrada (aunque sea local vía gateway) devuelve tokens, no la distribución. Para OPD hace falta correr el teacher en local (llama.cpp con `--logits`, o transformers) y acceder a la distribución completa. Eso es un requisito de cómputo propio que hoy no está resuelto en el repo.

**h) Multi-teacher / especialización por voz**

MiniCPM5 destila **16 teachers** (5 agénticos) en un solo modelo. Kateto ya tiene una estructura *mejor* para esto: los state vectors por voz son un mecanismo de especialización de 6 MB con hot swap a costo cero. La conexión natural: **cada voz es un teacher especializado** y OPD es cómo se le enseña a la capa base a no perder las capacidades que la voz necesita. No hay que portar los 16 teachers; con 2-3 alcanza para el problema real.

### 4.3 De MiniCPM-o 4.5: interacción, no arquitectura

Acá hay que ser claro: **la arquitectura omni-modal de 9B no es transferible** — Kateto es texto + TTS externo, no un modelo end-to-end con SigLip2/Whisper/CosyVoice. Lo que sí mapea:

- **La decisión de hablar o no, a 1 Hz** ↔ la spec de turn-taking de Kateto. `<WAIT>` y `<NO_RESPONSE>` son la versión RWKV de la misma decisión, con una ventaja: la spec ya notó que hace falta **un solo forward (N=1)** por chunk para resolverla, porque el estado es O(1). Eso es *más barato* que lo que hace un Transformer con KV-cache creciente. La arquitectura RWKV está mejor parada acá que la de MiniCPM-o, no peor.
- **TDM (time-division multiplexing)** ↔ el interleaving de tokens de texto con tokens de control en el mismo stream, en slices temporales chicos y periódicos. Es una formalización de lo que la spec describe como "purga o archiva el fragmento como ruido" vs "retiene lo escuchado".
- **Audio system prompt para voz configurable** ↔ las voces de Kateto. MiniCPM-o resuelve cambio de voz en **inferencia** con un prompt de audio; Kateto lo resuelve con **state tuning** (6 MB por voz, hot swap). Son dos soluciones al mismo problema; la de Kateto es más liviana en almacenamiento, la de MiniCPM-o más flexible en runtime. Vale la pena tener las dos: state tuning para las voces fijas, audio prompt para personajes ad-hoc.
- **Modelar texto y habla intercalados** ↔ la spec de Kateto ya apunta a esto; MiniCPM-o confirma que es el camino para habla larga estable (>1 min).

### 4.4 Lo que NO se transfiere

| Cosa | Por qué no |
|---|---|
| **400B tokens de deep-thinking SFT** (2B) / 200B+200B (1B) | Kateto tiene ~20K ejemplos. No es un factor de 2x, es de 7 órdenes de magnitud. Copiar el *recipe* sí; copiar el *volumen* no |
| **RL con critic (JustRL II) + 86K tareas verificables** | Requiere rollouts masivos + reward model + critic. Inviable en 4 GB de VRAM, y RWKV-PEFT **no tiene ningún camino de RL** (no hay dpo/orpo/rl en el repo; grep sobre `*.py` da vacío) |
| **16 teachers + OPD full-vocab** | La versión full-vocab del 2B necesita servir 16 modelos. La versión top-k con 1-2 teachers es la única viable acá |
| **Tier L4 (verificación factual a escala)** | Necesita orchestration + fact verification contra knowledge bases. Para Kateto eso vive mejor en RAG que en pesos |
| **131K de contexto "nativo"** | MiniCPM5 lo logra por arquitectura Transformer + entrenamiento largo. RWKV-7 llega al mismo lugar por otra vía — estado constante y **infctx training** (que RWKV-PEFT ya soporta y es la razón por la que la spec de turn-taking puede plantear streaming continuo) |
| **"Sin kernels custom"** | Eso es propiedad de ser un Llama. RWKV sí necesita el parche de LDS de RDNA2 (`K=8`, `num_stages=1`) o el kernel no compila en gfx1034 |

---

## 5. Plan concreto por fases

### P0 — sin GPU nueva, esta semana

1. **Verificar assistant-only loss de verdad.** No alcanza con que el flag esté: hay que dumpear `(x, y)` de un batch real y confirmar que los `-100` cubren el turno del usuario y no solo el padding. `create_mask` engancha por la presencia de `<|im_user|>` en el contexto — confirmarlo con evidencia, no por lectura del flag.
2. **Escribir `data/MANIFEST.md` con los tiers L0-L4** de cada fuente de `data/` y la etapa a la que va cada una. Una tabla, cero código.
3. **Esquema de checkpoints por etapa**: `out/<run>/{base,midtrain,sft,state_<voz>}/` + `METRICS.json` por etapa.

### P1 — cambio de composición de datos, no de código

4. **Partir el LoRA base en stable + decay**, con el tramo de decay cargado a L3 (tools, debate, anchors) y LR más bajo. Mismo `train.py`, distinta mezcla.
5. **Escalar tool calling de 200 → 5.000+**, mezclando generación propia con `UltraData-SFT-Agent-2609` filtrado a español. Prioridad alta: es el gap de 2500x y el dataset es Apache 2.0.
6. **Extender `<think>` a razonamiento general** con un marcador de modo on/off. El 1B demuestra que un mismo checkpoint puede servir para modo rápido y modo deliberado.

### P2 — experimentos de mayor riesgo

7. **Prototipo de OPD con 1 teacher local.** Requisito bloqueante: servir el teacher en local con acceso a logits (no API). Empezar con ~500-2000 prompts de `data/kateto_qa.jsonl` + `toolcalling_sft.jsonl`. Métrica: reverse KL promedio y, sobre todo, si las voces dejan de emitir `<tool_call>` en contextos de debate (el síntoma #2 del SPEC).
8. **ORPO completo.** Hoy `orpo_trainer.py` es smoke-only: hace el forward de pares, imprime `logP/odds/L_OR/L_total` y prueba el reset de estado, pero el loop de entrenamiento no existe (por diseño: "todo 17 owns it"). Es más barato y más seguro que intentar RL.
9. **Mid-training explícita** entre base y state tuning, sobre L2. Es el hueco estructural más grande del pipeline actual.

---

## 6. Fuentes

| # | Fuente | Uso |
|---|---|---|
| 1 | https://huggingface.co/openbmb/MiniCPM5-2B (+ `/raw/main/README.md`) | Ficha técnica, benchmarks, training recipe, RL+OPD |
| 2 | https://huggingface.co/openbmb/MiniCPM5-1B (+ `/raw/main/README.md`) | Recipe 1B, detalle de RL two-stage y mejora de respuesta kilométrica |
| 3 | https://huggingface.co/openbmb/MiniCPM5-2B-Midtrain | Confirmación del ladder Base→Midtrain→SFT→final |
| 4 | arXiv 2602.09003 — *Data Science and Technology Towards AGI Part I: Tiered Data Management* (abs + HTML) | Framework L0-L4, propiedades por tier, asignación por etapa, hallazgo L3>L2>L1 |
| 5 | https://huggingface.co/datasets/openbmb/UltraData-SFT-Agent-2609 | 500K trayectorias de agente, tipos de tarea, Apache 2.0 |
| 6 | https://huggingface.co/datasets/openbmb/UltraData-RL-2609 | 86K tareas verificables, JustRL II, dominios |
| 7 | https://huggingface.co/datasets/openbmb/UltraData-Math | Tiers L1/L2/L3 (170.5B / 33.7B / 88B tokens) y motivación de cada uno |
| 8 | arXiv 2604.27393 — *MiniCPM-o 4.5: Towards Real-Time Full-Duplex Omni-Modal Interaction* | Omni-Flow, TDM, proactive a 1 Hz, <12GB RAM |
| 9 | https://huggingface.co/openbmb/MiniCPM-o-4_5 (+ `/raw/main/README.md`) | Composición de 9B, clonado de voz, audio system prompt |
| 10 | https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/skills/minicpm5-finetune-trl/SKILL.md y `docs/finetune/trl.md` | Patrón de assistant-only loss vía `{% generation %}` |
| 11 | https://github.com/OpenBMB/MiniCPM | Repo oficial, cookbooks de deploy y finetune |
| 12 | Código local `kateto-train`: `rwkv_pipeline/*.sh`, `rwkv_pipeline/prepare_datasets.py`, `rwkv_pipeline/orpo_trainer.py`, `RWKV-PEFT/rwkvt/dataset/mask.py`, `RWKV-PEFT/train.py`, `config/*.yaml`, `SPEC.md`, `HANDOFF.md`, `data/voice_turn_taking_no_response_spec.md` | Estado real del pipeline de Kateto y capacidades de RWKV-PEFT |

**Nota de verificación:** el bloque BibTeX del card de MiniCPM5-2B y su link "MiniCPM Tech Report" apuntan a arXiv 2506.07900, que es el paper de **MiniCPM4**, no de MiniCPM5. No hay tech report dedicado de MiniCPM5 publicado al 2026-09-12.

---

## Anexo — Caso de estudio: Qwen3.8-27B-Humanlike-Chat (2026-09-11)

> Agregado después de la investigación inicial. Es **el precedente más directamente aplicable a Kateto** de todo lo relevado: mismo objetivo declarado (dejar de sonar a asistente), pipeline *más simple* que el de MiniCPM5, y métricas de comportamiento en vez de benchmarks.

### A.1 Qué es

- **Post:** r/LocalLLaMA, u/kvyb, 2026-09-11 — *"A model I tuned to imitate realistic human-to-human conversation"*.
- **Artefacto:** `LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF` (Apache-2.0, ~9.8K descargas).
- **Lineage:** `Qwen/Qwen3.8-27B` → `huihui-ai/Huihui-Qwen3.8-27B-abliterated` → **LoRA rank-256 step-863** → BF16 merged text-only → GGUF.
- **Motivación textual del autor:** *"I was getting genuinely annoyed at trying to have a normal conversation with LLMs... too helpful, polished, verbose, using words we never use in conversation. The goal wasn't to make Qwen smarter or improve benchmark scores. I was trying to change its conversational habits: stop turning every reply into an explanation, agreeing with everything, and writing stuff just to keep the conversation going."*

### A.2 Ficha técnica

| Item | Valor |
|---|---|
| Adaptación | **rank 256, alpha 32**, 496 módulos de lenguaje, 992 tensores LoRA |
| Parámetros entrenables | **1.867.644.928** (~6.9% del modelo) |
| Contexto de entrenamiento | 4.096 tokens (el base soporta 262.144 nativo) |
| Datos | **139.845 mensajes reales / 1.396 sesiones** 1-a-1 → **7.006 ejemplos de train + 194 de validación** |
| Tokens supervisados | **158.085** tokens de respuesta |
| V3 (rerun) | mismo base/formato/optimizer/schedule + **2.000 de 9.201 filas con reasoning nativo en el canal ` thinking` de Qwen, 7.201 con reasoning OFF** |
| Loss | **solo el turno del hablante siguiente. Prompt e historial NO contribuyen al loss** |
| LoRA publicada | FP32, 7.47 GB (GGUF para `llama.cpp --lora`, aplicable solo a base sin adaptar) |
| Endpoint | `https://api.lesssthanthreeai.com/v1`, gratis, scale-to-zero, ~2-3 min en frío |

### A.3 Métricas de comportamiento (590 turnos reproducidos)

| Medida | Resultado |
|---|---|
| Slop trigrams detectados | **0** |
| Construcciones "not X but Y" | **0** |
| Respuestas en formato lista | **0** |
| Longitud mediana de respuesta vs base | **427 → 32 caracteres** |
| Preguntas reflexivas ("¿querés que...?") | **27/29 → 3/29 turnos** |
| Turnos pareados donde dio la respuesta más corta | **300** |

### A.4 Por qué esto le pega directo a Kateto

**1. Es la misma tesis, ya ejecutada — y sin RL, sin OPD, sin 16 teachers.** Un LoRA de comportamiento sobre datos conversacionales reales + assistant-only loss alcanza para lo que el SOUL de Kateto declara como objetivo central ("no sos un asistente genérico", "nada de '¿en qué puedo ayudarte hoy?'") y para el síntoma #5 del SPEC. El pipeline caro de MiniCPM5 no es el requisito para *esto*.

**2. El volumen no es el problema de Kateto.** 7.006 ejemplos y 158.085 tokens supervisados cambiaron el comportamiento conversacional de un 27B. Kateto tiene **20.506 ejemplos** en `rwkv_kateto_base_train.jsonl`. Esto **corrige** la conclusión de la §4.2.f: el gap de 2500x es real para *trayectorias de agente*, pero para **comportamiento conversacional Kateto ya pasó la masa crítica**.

**3. La métrica es lo que falta, no los datos.** No midieron benchmark: midieron *comportamiento* contra un replay del base. Slop trigrams, "not X but Y", formato lista, longitud mediana, preguntas reflexivas. Es un eval suite de comportamiento, es barato, y **Kateto no lo tiene**. Es la respuesta concreta a "¿cómo sé si Kateto dejó de sonar a asistente?".

**4. Tercera confirmación del assistant-only loss** (TRL → MiniCPM5 → este caso). En RWKV ya está cubierto por `--loss_mask qa`.

**5. Think/no-think mixto en el mismo run** (2.000 filas con reasoning, 7.201 sin). Confirmación independiente del patrón de MiniCPM5-1B y de la recomendación **P1.6** de este reporte.

**6. Abliterated por diseño.** El autor lo justifica explícitamente: *"refusal-related directions can affect more than which topics a model will discuss. They can also influence tone, phrasing, response structure, conversational choices, and how readily a character follows the natural direction of an interaction."* Es una posición de diseño que Kateto **no** toma (el SOUL pone límites explícitos en el prompt/datos, no ablitea el modelo). Vale documentarlo como decisión consciente y no como ausencia de decisión.

### A.5 El riesgo, que es exactamente el riesgo del SOUL de Kateto

El autor lo admite sin maquillaje: *"it's really more 'less assistant-like'. The model is still going to inherit the style of the conversations in the dataset, and that's a relatively small sample size. I definitely wouldn't claim this is how everyone talks."*

Y el thread lo dice crudo:

- *"That's more you-like than human-like. Everyone I text with writes more like the base model."*
- *"The one on the left sounds like an asshole. The one on the right seems overly engaged. There needs to be a middle ground."*
- *"It's an age thing... I'm mid 40s and the example on the left is painful to look at — it doesn't feel 'more real', it just feels young."*
- *"To me it mostly reads like someone who just doesn't feel like talking."*
- *"Short messages like that generally give me the vibe of 'I don't really pay much attention to you'."*

**Traducción a Kateto:** el SOUL pide "una palabra" como modo de respuesta, "emoji solo", "alguna letra comida, tildes opcionales", "insultos como puntuación". Si entrenás eso tal cual, el resultado es un personaje que **solo sabe ser seco** — exactamente la crítica del thread. Tres contramedidas concretas:

1. **La métrica correcta es la varianza, no la mediana.** "300 turnos donde dio la respuesta más corta" mide *dirección*. Un personaje con una sola longitud es tan falso como un asistente verboso. Hay que medir distribución de longitudes (y de registros), no promedio.
2. **El modo atípico debe quedar cuantificado y medido.** El SOUL dice ~80/20. Si el dataset no respeta esa proporción, el 20% se vuelve 100%.
3. **El turn-taking de Kateto ya tiene el guardarraíl que este caso no tenía**: el constraint de que `<NO_RESPONSE>` no supere el 3-4% del total, documentado en `voice_turn_taking_no_response_spec.md`, por el mismo motivo (aversión al diálogo). Es el mismo mecanismo que acá hace falta para las respuestas ultra-cortas.

### A.6 Lo que este caso NO resolvió y Kateto sí puede aportar

- El autor **no midió IFEval en el checkpoint liberado** (una iteración anterior había dado **5 puntos menos** en instruction following, y lo publica honestamente). Para una voz con tool calling, esa degradación es un riesgo real: si el modelo se vuelve más "humano", puede fallar más el formato de `<tool_call>`. Kateto ya tiene `scripts/eval_toolcall_json.py` — hay que correrlo antes y después de cualquier cambio de comportamiento.
- El modelo es de **un solo registro** (chat casual). Kateto tiene 5 voces con state tuning + 6 MB por voz. La estructura de Kateto es más ambiciosa y mejor adaptada a ese problema.

### A.7 Fuentes del anexo

| # | Fuente |
|---|---|
| 13 | r/LocalLLaMA — https://www.reddit.com/r/LocalLLaMA/comments/1wdl2qa/qwen3827bhumanlikechat_a_model_i_tuned_to_imitate/ (u/kvyb, 2026-09-11) |
| 14 | https://huggingface.co/LessThanThreeAI/Qwen3.8-27B-Humanlike-Chat-GGUF (+ `/raw/main/README.md`) |

**Nota de coherencia numérica:** el post dice **125.217 mensajes obfuscados**; el card dice **139.845 mensajes reales** y luego **9.201 filas de entrenamiento** en el rerun V3. Son cifras de momentos distintos del proyecto (el post describe una versión anterior). Las cifras del card son las del artefacto publicado.
