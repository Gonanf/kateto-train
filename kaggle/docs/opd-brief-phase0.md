# Brief de implementación — OPD (On-Policy Distillation) para Kateto RWKV-7

> **Repo:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Creado:** 2026-09-12 · **Fase:** 0 (smoke + prototipo)
> **Ejecutor:** harness externo (opencode / agy / cline) — el orquestador no escribe el código
> **Referencia:** `docs/research-minicpm5-o45-a-kateto-rwkv.md` §4.2.g (OPD)

---

## 0. Objetivo

Prototipo funcional de **On-Policy Distillation** con la receta del MiniCPM5-1B (top-k logits del student y del teacher, reverse KL sobre la unión):
- **Student:** RWKV-7 G1j **1.5B** (`models/rwkv7-1.5b/rwkv7-g1j-1.5b-20260831-ctx16384.pth`, ctx 16384).
- **Teacher:** modelo local fuerte, servido localmente, con acceso a **distribución** (no solo texto).
- **Salida:** `out/opd/<run>/` con checkpoints, `metrics.jsonl` y un reporte con el KL medido.

**Criterio de éxito del prototipo:** el reverse KL promedio **baja** de forma monótona-ish en ≥30 pasos sobre un set de prompts fijo, con las métricas de cobertura reportadas. No se busca calidad; se busca **que el mecanismo funcione y que sus límites queden medidos**.

---

## 1. Hechos verificados en el entorno (2026-09-12, no re-investigar)

| Hecho | Evidencia |
|---|---|
| El router llama.cpp está en `http://127.0.0.1:11434` con `--models-max 1` | `ss -ltnp`, `GET /v1/models` |
| El router ya tiene presets para `MiniCPM5-1B`, `MiniCPM5-2B-GGUF:Q6_K`, `Qwen3.8-2B`, `Nanbeige-4.2-3B`, `Qwen3.8-27B`, `Muse-Glimmer-30B`, `RWKV-2.9B`, `RWKV7-G1j-7.2B`, `RWKV7-Kateto-Retrain`, `Kateto` | `~/.config/llama/config.ini` |
| **llama-server SÍ devuelve `logprobs` + `top_logprobs`** en `/v1/chat/completions` (top-k real por posición) | probado: `tok=' che' lp=-0.0023 top=[(' che',-0.002),...]` |
| **llama-server NO puede scorear una continuación arbitraria**: `echo=true`, `max_tokens=0` y `/completion` con `n_probs` **ignoran el prompt** y devuelven solo el token generado | probado en `:42665` con LFM2.5-VL-3B — los tres devolvieron 1 sola entrada |
| `venv-unsloth-qwen`: torch 2.9.1+rocm6.3, transformers **5.5.0**, peft 0.20.0, datasets 4.3.0. **Sin** llama_cpp, **sin** vllm | introspección del venv |
| `venv/`: torch 2.14.0+cpu, transformers 4.57.1 | idem |
| RAM: 31 GB totales, **20 GB disponibles** | `free -g` |
| VRAM: RX 6500 XT, **4 GB** (gfx1034) | `HANDOFF.md`, y el guard de `infer_kateto.py` |
| Checkpoint 1.5B: **3.055 GB** en HF (`BlinkDL/rwkv7-g1`) | `GET /api/models/BlinkDL/rwkv7-g1?blobs=true` |
| `rwkv_pipeline/infer_kateto.py::RWKV_RNN.forward` está decorado `@torch.jit.script_method` **y** `@torch.no_grad()` → **NO sirve para entrenar** | lectura del archivo, línea ~180 |
| El student entrenable es `RWKV-PEFT/rwkvt/rwkv7/model.py::RWKV7.forward_normal(input_ids,...)` → devuelve logits `(B,T,vocab)` con autograd | lectura del archivo |
| `RWKV-PEFT/rwkvt/dataset/mask.py::mask_fn_dict = {"qa": create_mask, "se": generate_mask}` — assistant-only loss ya activa en los dos scripts de entrenamiento | lectura del archivo |

### 1.2 Fase 0.1 EJECUTADA — resultado verificado (2026-09-12)

Introspección real de los tres checkpoints (con `torch.load`, no asumido):

```
[0.4B G1d-20260210] n_layer=24 n_embd=1024 n_head=16 head_size=64 vocab=65536 tensores=798
[2.9B G1j-20260831] n_layer=32 n_embd=2560 n_head=40 head_size=64 vocab=65536 tensores=1062
[1.5B G1j-20260831] pendiente de descarga al momento de esta medición
```

Generación local con el **0.4B base**, `device=cuda` (ROCm, `torch.cuda.is_available()=True`), GPU `AMD Radeon RX 6500 XT 3.98 GB`, **~12-13 tok/s**, **VRAM pico 1.01 GB / 3.98 GB**:

| Formato de prompt | Resultado |
|---|---|
| `User: Hello! Who are you?\n\nAssistant:` | ✅ coherente — *"I'm an AI assistant created by OpenAI, specifically designed to provide helpful and informative responses..."* |
| `User: Che, quién sos y qué hacés?\n\nAssistant:` | ✅ coherente — *"Soy un modelo de lenguaje, y estoy aquí para ayudarte con tus necesidades. ¿Qué puedo hacer para ti?"* |
| `User: Explain what a neural network is.\n\nAssistant:` | ✅ coherente, 60 tokens de explicación real |
| `<|im_user|>Che, quién sos?<|im_start|>seco\n` (ChatML de Kateto) | ❌ **bucle degenerado**: `\|Moi\|Nou?<\|im_start\|seco` repetido |

**⚠️ CORRECCIÓN AL BRIEF, y es importante.** Los checkpoints **base** (`.4B G1d`, `1.5B G1j`) **no hablan el ChatML de Kateto**. Hablan el **formato RWKV World**: `User: {query}\n\nAssistant: {respuesta}`. Alimentarlos con `<|im_user|>` / `<|im_start|>` produce basura degenerada (verificado). La §3 Fase 0.3 de este brief decía usar `format_rwkv_chat` (ChatML) — **eso es válido solo para los checkpoints YA fine-tuneados de Kateto** (`out/rwkv_kateto_base/rwkv-1.pth`, `out/kateto-rwkv29-merged.pth`), no para el 1.5B base.

**Regla operativa:** el student de OPD se elige así —
- **base 1.5B** → prompt `User: {q}\n\nAssistant:` y loss sobre solo los tokens de la respuesta.
- **checkpoint Kateto** → prompt `<|im_user|>{q}\n<|im_start|>{voice}\n{resp}<|im_end|>`.

Y no mezclar: un rollout con el formato equivocado no da un KL alto, da basura.

**Dato de contexto que refuerza el objetivo:** el base 0.4B se auto-presenta como *"an AI assistant created by OpenAI... designed to provide helpful and informative responses"*. Es exactamente el vicio de asistente que el SOUL de Kateto quiere extirpar — o sea que el punto de partida del student está medido, no supuesto.

### 1.3 Fase 0.1 EJECUTADA — el **1.5B** (medición completa, 2026-09-12)

Descarga verificada byte-exacta: **3.055.444.605 bytes** == tamaño reportado por HF (`aria2c -x8`, `MATCH`).

```
rwkv7-g1j-1.5b-20260831-ctx16384.pth
  n_layer=24  n_embd=2048  n_head=32  head_size=64  vocab=65536
```

**Corre en `device=cuda` (ROCm), sin fallback a CPU:**

| Métrica | Valor medido |
|---|---|
| Carga + primera generación | 17.7 s |
| Velocidad (1.5B, bf16) | **4.4 – 4.6 tok/s** |
| Velocidad (0.4B, bf16, referencia) | 12 – 13 tok/s |
| **VRAM pico 1.5B** | **3.15 GB / 3.98 GB** |
| Headroom restante | **~0.83 GB** |

Salidas reales (formato World):

- `User: Explain in one sentence what recursion is.\n\nAssistant:` → *"Recursion is when a function calls itself to solve a problem by breaking it down into smaller, similar subproblems until it reaches a base case that can be solved directly."* ✅
- `User: Che, quién sos y qué hacés?\n\nAssistant:` → *"`<think>`El usuario está hablando en español y pide una traducción de la frase "Che, quién sos y qué hacés?" al español. La frase es informal y tiene un tono ligeramente irónico..."* ⚠️

**Dos hallazgos nuevos que cambian el diseño:**

1. **El 1.5B G1j tiene canal de `<think>` y se dispara con input en español**, derivando en meta-análisis del prompt en vez de responder. El 0.4B G1d (generación anterior) **no** lo tiene. Para el rollout on-policy de OPD esto hay que **controlarlo explícitamente** — si no, buena parte de los tokens de la respuesta son razonamiento interno y el KL se mide sobre lo que no queremos. Es el mismo problema que MiniCPM5 resuelve con `enable_thinking`, y es una decisión de diseño que el harness **no** debe tomar solo: implementar el switch y **reportar la fracción de tokens de `<think>` por rollout**.

2. **El teacher NO puede vivir en la misma GPU.** El student solo ya come 3.15 de 3.98 GB. El teacher va **obligatoriamente en CPU** (RAM: 20 GB disponibles), o por el router llama.cpp. Cualquier intento de cargar teacher y student en `cuda` va a fallar por VRAM — que nadie lo intente y lo reporte como bug.

### 1.1 Estado de freellmapi / Infisical (leer antes de tocar nada)

- **freellmapi NO está corriendo**: no hay contenedor docker ni puerto 3001. El repo está en `~/proyectos/freellmapi` con `docker-compose.yml`.
- La unified key existe en disco: `~/Documentos/importante/freellm.key` (`freellmapi-…`, 61 bytes). **No hace falta Infisical para leerla.**
- **El CLI de `infisical` NO está instalado** (`which infisical` → vacío). El server sí corre en docker (`http://localhost:80`). Sin CLI ni sesión no se puede extraer el secreto de ahí.
- **Y lo importante:** un relay OpenAI-compatible gratuito **no expone top-k logprobs de forma confiable**. OPD necesita la **distribución del teacher**, no su texto. Por eso el teacher va **local**, no por freellmapi. freellmapi queda como backend secundario para tareas de solo texto (ej. generar respuestas de referencia), no para el KL.

---

## 2. La restricción técnica que define el diseño

OPD necesita, en cada posición de la respuesta generada por el student:

```
P_teacher(v | prefijo del student)   para cada v en el soporte
```

`llama-server` **no puede** dar eso: solo devuelve el top-k **de los tokens que él mismo genera**, y no acepta scorear una continuación arbitraria. Tres caminos, en orden de fidelidad:

| Camino | Fidelidad | Costo | Disponibilidad |
|---|---|---|---|
| **A. Teacher en `transformers`/`llama-cpp-python` con logits completos** | Exacta (full-vocab, como el 2B de MiniCPM5) | 1 forward del teacher por respuesta; el teacher tiene que convivir con el student en RAM/VRAM | ✅ transformers 5.5.0 ya está instalado |
| **B. `llama-server` con `top_logprobs`, 1 request por posición del student** | Aproximada (soporte top-k, sin masa fuera del top-k) | N requests por respuesta (con prompt caching) | ✅ verificado que anda |
| **C. `vllm` con `prompt_logprobs`** | Exacta y eficiente | vLLM no está instalado; en gfx1034 4 GB es dudoso | ❌ |

### Mapeo de tokenizadores (el otro problema duro, y es real)

El tokenizer del teacher (BPE estilo Llama/Qwen) ≠ el de RWKV (vocab de 65536, `rwkv_vocab_v20230424.txt`). El reverse KL necesita un soporte común. MiniCPM5 no tiene este problema porque student y teacher comparten familia de tokenizer; **Kateto sí**. Reglas para este prototipo:

1. El soporte se define sobre **byte-strings**, no sobre ids: un candidato del student y uno del teacher son "el mismo símbolo" si `decode()` da el mismo byte-string.
2. Masas no representables se **descartan y renormalizan**, y la fracción descartada se loguea como **`coverage`** por paso. `coverage` es un entregable, no un detalle.
3. Si `coverage` promedio < 20% sobre 30 pasos, **el camino B queda descalificado** y el reporte debe decirlo: se pasa al camino A.

---

## 3. Plan de trabajo

### Fase 0.1 — Verificación de generación local ✅ **YA EJECUTADA, NO REPETIR**

Resultado completo en **§1.2 (0.4B) y §1.3 (1.5B)**. Resumen: los dos checkpoints se cargan y generan localmente en `device=cuda`; el 1.5B mide `n_layer=24 n_embd=2048 n_head=32 head_size=64`, corre a 4.4-4.6 tok/s con 3.15 GB de VRAM pico. **No volver a correr esta fase** — arrancar por la Fase 0.2. Lo único que queda de esta fase es el artefacto `out/opd/phase01_localgen.md`: escribirlo **copiando** los resultados de §1.2 y §1.3, sin re-ejecutar nada.

### Fase 0.2 — Teacher client

Crear `rwkv_pipeline/teacher_client.py`:

- **Modo A (primario):** carga el teacher con `transformers` (`AutoModelForCausalLM`), `torch_dtype` bf16 o fp32 según device, `device_map` CPU. Expone:
  ```python
  logits_at(prompt_text: str, continuation_text: str) -> torch.Tensor  # (T_cont, vocab_teacher)
  ```
  Un solo forward sobre `prompt+continuation`; devuelve los logits alineados a los tokens de la continuación.
- **Modo B (fallback):** cliente HTTP al router `:11434`, `top_logprobs=K`, 1 request por posición. Expone la misma interfaz pero devuelve `list[(token_bytes, logprob)]` en vez de logits densos.
- Selección por flag `--teacher-backend {hf,router}`, default `hf`.
- **Teacher candidates (elegir uno, chico y ya conocido):** `MiniCPM5-1B` (requiere `transformers>=5.6`, hay 5.5.0 → **si se elige este, hay que actualizar transformers**, y eso puede romper el venv de entrenamiento: **no tocar `venv-unsloth-qwen` como parte de este brief**), `Qwen3.8-2B`, `Nanbeige-4.2-3B`. **Empezar por el más chico que cargue con el transformers instalado.**
- **Gate A:** `logits_at` sobre un prompt de prueba devuelve `(T_cont, vocab_teacher)` con `T_cont == len(teacher_tokenizer(continuation))` y un `log_softmax` que suma ~1.0 (dentro de 1e-3).

### Fase 0.3 — Rollout on-policy del student

Crear `rwkv_pipeline/opd_rollout.py`:

- Carga `RWKV-PEFT/rwkvt/rwkv7/model.py::RWKV7` con el checkpoint 1.5B (**con autograd**, no el `RWKV_RNN` de inferencia).
- Genera la respuesta del student **on-policy** (sampling) desde un prompt del set, guardando por posición: `token_id`, `logits` (o al menos los top-K con sus logprobs), y el **texto del prefijo** hasta esa posición.
- **Formato de prompt — leer §1.2 antes de escribir esta parte.** El student **base 1.5B** usa el formato **RWKV World**: `User: {query}\n\nAssistant: {respuesta}`, con la loss solo sobre los tokens de la respuesta. El ChatML de Kateto (`<|im_user|>` / `<|im_start|>{voice}` / `<|im_end|>`, ver `rwkv_pipeline/prepare_datasets.py::format_rwkv_chat`) es **solo para checkpoints ya fine-tuneados de Kateto**, y alimentarlo al base produce un bucle degenerado (verificado, §1.2). Los delimitadores de ambos formatos son **texto plano** (sus ids no están en el vocab de 65536).
- **Gate B:** para 8 prompts de `data/kateto_qa.jsonl`, el rollout corre y guarda las estructuras. Reportar longitud media de respuesta y tokens/s.

### Fase 0.4 — Mapeo + reverse KL + loop

Crear `rwkv_pipeline/opd_train.py`:

1. **Mapeo byte-string** entre el top-K del student y la distribución del teacher (reglas de §2).
2. **Loss por posición:**
   ```
   L = Σ_t  Σ_{v∈U_t} P_s(v) · [ log P_s(v) − log P_t(v) ]
   ```
   con `P_s` por softmax sobre `U_t` (con gradiente) y `P_t` **detached**.
3. **Optimizador:** AdamW, `lr` bajo (empezar 1e-5), 30-60 pasos, grad clipping.
4. **Métricas por paso → `out/opd/<run>/metrics.jsonl`:** `step`, `loss_rkl`, `coverage`, `teacher_entropy`, `student_entropy`, `response_tokens`, `wall_s`.
5. **Checkpoints** cada N pasos y al final.
6. **Baseline obligatorio:** correr el KL **sin actualizar pesos** (lr=0) sobre el mismo set y guardar la curva. Sin ese baseline, "el KL bajó" no significa nada.

**Gate C (decisión de diseño):** mirar `coverage` de los primeros 5 pasos.
- `coverage ≥ 0.5` → seguir con el camino elegido.
- `0.2 ≤ coverage < 0.5` → seguir, pero documentar el techo.
- `coverage < 0.2` → **parar**, cambiar de backend de teacher (B→A o A con teacher más chico), y reportar. No maquillar el número.

### Fase 0.5 — Reporte

`out/opd/REPORT.md` con: curvas de KL (baseline vs entrenado), coverage, entropías, tok/s, device real de student y teacher, y una sección **"Qué NO funcionó"** explícita.

---

## 4. Restricciones duras (violarlas invalida el trabajo)

1. **No tocar `venv-unsloth-qwen`** — es el venv de entrenamiento con torch+rocm. Si un teacher necesita otra versión de transformers, crear un venv **aparte** (`venv-opd-teacher`) y documentarlo.
2. **No hacer push, no deployar, no tocar `base/`.** Todo el output va a `out/opd/`.
3. **No inventar métricas.** Si algo no se puede medir, escribir "no medido" y por qué.
4. **Separar hecho de hipótesis** en cada reporte: lo que se verificó con output real vs lo que se infiere.
5. **Ningún número de KL se reporta sin el baseline de lr=0.**
6. El student de **inferencia** (`infer_kateto.RWKV_RNN`) tiene `no_grad` y torchscript: **no intentar entrenar con esa clase**. Si se necesita baseline de generación, usar esa; si se necesita gradiente, usar `RWKV-PEFT/rwkvt/rwkv7/model.py`.
7. Correr todo con `PYTORCH_ALLOC_CONF=expandable_segments:True`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `TORCH_COMPILE_DISABLE=1` exportados **antes** de importar torch.
8. **Student en `cuda`, teacher en CPU.** Medido: el student 1.5B solo ya ocupa **3.15 GB de 3.98 GB** de VRAM (§1.3). Cargar los dos en `cuda` **no entra**. No intentarlo ni reportarlo como bug: es una restricción física del hardware.
9. **El canal `<think>` del 1.5B hay que controlarlo y medirlo.** El G1j se dispara a pensar con input en español (§1.3). Implementar el switch think/no-think y **reportar la fracción de tokens de `<think>` por rollout**. No elegir un default silenciosamente: dejar el flag y reportar ambos números.
10. **Un solo proceso tocando la GPU a la vez.** El router llama.cpp (`:11434`, `--models-max 1`) también usa la GPU: si el harness carga un modelo ahí mientras el student entrena, se pelean por VRAM. Serializar, o usar el teacher en CPU por `transformers`.

---

## 5. Entregables

| # | Artefacto | Ubicación |
|---|---|---|
| 1 | Verificación de generación local 1.5B vs 0.4B | `out/opd/phase01_localgen.md` |
| 2 | Teacher client con los dos backends | `rwkv_pipeline/teacher_client.py` |
| 3 | Rollout on-policy con logits por posición | `rwkv_pipeline/opd_rollout.py` |
| 4 | Loop de OPD con mapeo + RKL + métricas | `rwkv_pipeline/opd_train.py` |
| 5 | Curva de KL (baseline lr=0 + entrenado) + coverage | `out/opd/<run>/metrics.jsonl` |
| 6 | Reporte honesto con "Qué NO funcionó" | `out/opd/REPORT.md` |

---

## 6. Lo que este brief NO pide

- No pide un modelo mejor. Pide **que el mecanismo de OPD funcione y que sus límites queden medidos**.
- No pide usar freellmapi para el teacher (ver §1.1: no expone logprobs de forma confiable, y no está corriendo).
- No pide tocar los scripts de entrenamiento existentes ni la composición de datasets.
- No pide bajar el 2.9B ni el 7.2B.
