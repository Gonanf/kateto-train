# Arquitecturas alternativas a RWKV-7 para Kateto — reporte de investigación

**Fecha:** 2026-10-06 · **Tipo:** investigación, no implementa nada.
**Árboles:** `kateto-train` @ `48c0ba7`; `llama.cpp-master` (ggerganov/llama.cpp) @ `ce8caa6`.
**Binario instalado:** `llama-server 0.1.2-dev (build 10502, commit 8ff87cc)`.
**Fuentes:** código del repo + código de llama.cpp + web (papers, PRs, forks).

> **Titular.** La premisa "los voice states no son serializables en GGUF" es **falsa a nivel de API**
> y ya está medida en este repo: el round-trip de estado sobre GGUF dio **PASS** (`max_abs_diff: 0`).
> Lo que no se puede es *hornear* el estado en los pesos (ni `.pth` ni GGUF) — eso sí es cierto y está
> explicado. La consecuencia práctica cambia: **no hay que migrar de arquitectura para recuperar las voces**.

---

## 1. Tabla comparativa de arquitecturas

Leyenda de "llama.cpp": soporte nativo en upstream (`src/llama-arch.cpp`), es decir cargable desde GGUF.
"Estado" = KV cache vs estado recurrente de tamaño fijo.

| Arquitectura | Familia | llama.cpp nativo | Inferencia CPU | Estado por secuencia (2.9B) | Contexto largo | Encaje con Kateto |
|---|---|---|---|---|---|---|
| **RWKV-7 "Goose"** (actual) | RNN lineal / generalized delta rule | **SÍ** (`rwkv7`) | **SÍ** (`wkv7` en ggml-cpu, Vulkan, CUDA, Metal, HIP, SYCL) | **recurrente, ~20,6 MB F32**, **inyectable** | degrada >28K passkey | **Base: único con estado inyectable + kernels CPU de este host** |
| RWKV-6 | linear attention | SÍ (`rwkv6`) | SÍ | recurrente | más débil que v7 | fallback, menos expresivo |
| **RWKV-X** | **híbrido RWKV-7 + top-k chunk sparse attention** | **NO** (research `howard-hou/RWKV-X`) | bajo los kernels de v7 + softmax | recurrente **+ KV paginado** | **64K passkey casi perfecto, decode a 1M cte.** | **el híbrido correcto si el dolor es coherencia larga** |
| Mamba / Mamba-2 | SSM | SÍ (`mamba`, `mamba2`) | SÍ | recurrente | degrada en largo (misma familia) | sin chat multilingüe a escala voz |
| Jamba | hybrid transformer+Mamba (MoE) | SÍ (`jamba`) | SÍ | **híbrido (KV + recurrente)** | bueno | 12B-398B, no entra en 4 GB |
| Falcon-H1 | hybrid mamba+attn | SÍ (`falcon-h1`) | SÍ | híbrido | fuerte | 34B+, no entra |
| Granite-Hybrid | hybrid | SÍ (`granitehybrid`) | SÍ | híbrido | bueno | variantes chicas, sin es-AR |
| Qwen3-Next (Gated DeltaNet) | linear attn + MoE híbrido | SÍ (`qwen3next`) | SÍ | híbrido | fuerte | 80B tot / 3B activos, sin es-AR |
| Kimi Linear (KDA) | linear attn | SÍ (`kimi-linear`) | parcial | híbrido | fuerte | enorme |
| RetNet | linear attention | **NO** | — | — | — | sin runtime |
| Hawk (RG-LRU puro) | linear RNN | **NO** | — | — | — | sin modelos chat, sin runtime |
| Griffin (RG-LRU + SWA) | hybrid | **NO** (solo RecurrentGemma 2B por HF) | HF/TF | fijo + SWA | fuerte (7-13B) | sin chat a escala voz |
| xLSTM | linear RNN | **NO** | HF | recurrente | fuerte | sin runtime GGUF |
| Zamba2 | Mamba2-hybrid | **NO** en llama.cpp (runtime ZINC/1BP) | — | híbrido | bueno | solo ApUs |
| Hymba / Samba / MiniMax-01 | híbridos | no / parcial | — | híbrido | fuerte | demasiado grandes |

**Lectura de la tabla.** De toda la familia sub-cuadrática, **las únicas con runtime CPU real y estado
inyectable son RWKV (v6/v7) y Mamba/2**, y de esos solo RWKV-7 trae base chat multilingüe del tamaño que
entra en el hardware. Los híbridos "de fábrica" (Jamba, Granite-Hybrid, Qwen3-Next, Falcon-H1) **conservan
un KV cache** → para el caso de N voces sobre un base cargado son un retroceso, no una mejora.
El único híbrido que **no** reintroduce KV cuadrático es **RWKV-X**.

---

## 2. Veredicto por pregunta

### Q2 — ¿Hay fork de llama.cpp o convertidor GGUF con inyección de estado RWKV? → **SÍ, confirmado y medido**

**No hace falta ningún fork para inyectar estado: es API de llama.cpp upstream.**

| Evidencia | Detalle |
|---|---|
| `include/llama.h:892` | `LLAMA_API size_t llama_state_seq_set_data(...)` — inyecta estado por secuencia |
| `src/llama-context.cpp:3127` / `:4223` | implementación; acepta un buffer externo |
| `src/llama-memory-recurrent.cpp:766` / `:847` / `:897` | `state_write`/`state_read`/`state_write_data` — formato **byte-stream crudo por capa** |
| `src/llama-context.cpp:3093` | `io_magic = 0xaf143cd8` |

**Medición cruda en este host** (`docs/gguf-estado-roundtrip.md`, `docs/gguf-estado-python.md`):

```
llama.cpp: 0.1.2-dev (build 10502, commit 8ff87cc)   modelo: out/kateto-rwkv29-Q4_K_M.gguf (RWKV-7 2.9B)
size_bytes: 21627676
test1_identidad: OK (21627668 payload bytes identical after 8 header bytes)
max_abs_diff: 0   argmax_igual: si
test3_continuacion: OK
RESULTADO: PASS        (docs/ggml-rwkv7-rosa-gap.md: PASS re-corrido)
```

**El productor ya está escrito en Python** (`rwkv_pipeline/state_bytes.py`, `state_from_pth.py`):
mapea `blocks.{i}.att.time_state (n_head, head_size, head_size)` → tensor `s` de la capa
(`n_embd_s = n_embd * wkv_head_size`, reshape gratis head-major a `(n_embd, head_size)`).
`tests/test_state_bytes.py` y `tests/test_state_from_pth.py`: **8 passed**.

**El fork existe pero para OTRA cosa:** upstream `convert_hf_to_gguf.py` tiene **0** ramas RWKV
(`grep -nic rwkv` = 0). La conversión la hace **`MollySophia/rwkv-mobile`** (vendoreado en el repo,
`export_gguf.sh:35-37`). Ese fork además **trae** `load_initial_state()` (`src/c_api.cpp:527`) y
`serialize_runtime_state`/`deserialize_runtime_state` por backend (`src/backends/*/`) — o sea el
runtime móvil también serializa estado. **Es el mecanismo que "funcionó antes".**

**Caveats medidos (no ocultar):**
- **Formas:** el estado `seco` que hay es **0.4B / 16 cabezas**; el GGUF servido es **2.9B / 40 cabezas**
  → `SHAPES_NO_CALZAN` (`docs/gguf-estado-pth.md`). Fabricar el buffer para el base servido exige
  reentrenar la Capa 2 sobre ESE base.
- **`r` (token shift) va en ceros:** el `.pth` no lo trae. El estado inyectado no es una captura nativa.
- **Semántica NO medida:** que con el estado inyectado el modelo conteste como el camino nativo no está probado.
- **Formato sin versión:** solo el `magic` de sesión → frágil ante upgrades de llama.cpp; mitigar con round-trip test.
- **Sin flag CLI `--state-load`:** `grep -rn "state-save\|state-load" common/ examples/` viene vacío → hay que agregarlo o inyectar por API.
- **Server `/slots/{id}?action=save|restore`**: persiste estado vía `llama_state_seq_save_file`, pero
  issue **#25913**: en modelos recurrentes/híbridos el restore on-disk **pierde los context checkpoints**
  (re-procesa todo). PR **#20955** agrega sidecar de checkpoints.

### Q3 — ¿ROSA (indexado de K / ruteo de Q / re-inyección de V) se puede serializar en checkpoints? → **PARCIAL**

Hay **dos cosas distintas llamadas ROSA**:

**(a) La del usuario** — `rwkv_pipeline/rosa_module.py` = *Retrieval On Soft Attention*: rama
head-paralela `softmax(QKᵀ·máscara)V` con `proj_q/k/v`, `out_head`, `gate` (8,78 M params; `out_head` = 95%).

| Parte | ¿Serializable? | Evidencia |
|---|---|---|
| Pesos entrenables (`proj_*`, `out_head`, `gate`) | **SÍ, ya lo es** | `infer_kateto.py:337` carga `rosa_head` con `state_dict` + `retrieval_dim` + `step` |
| Memoria de runtime `(keys, values, pos)` | **SÍ en principio** — es una tupla de tensores + int (`rosa_module.py:75-83`) | `step_infer` devuelve `(rosa_logits, gate_weight, memory)` |
| Guardarla hoy | **NO** — no hay save/load del buffer | `infer_kateto.py` arrastra `rosa_memory` pero no lo serializa |

**Blockers reales antes de un checkpoint portable:**
- `step_infer` crece el buffer con `torch.cat` (`rosa_module.py:67-68`) → **O(T) por token, O(T²) por contexto**
  (medido: ratio 2.02 al duplicar T). Hay que preasignar con índice de escritura primero.
- El estado recurrente de llama.cpp tiene **celdas de tamaño fijo** (`cell_count = 1`,
  `llama-memory-recurrent.h:109`). ROSA necesita **memoria que crece por token** → no entra en el layout
  de `state_bytes.py` sin romper B1; su lugar natural es el **KV cache paginado** (`src/llama-kv-cache.cpp`).
- `attach_rosa` y el wiring existen (`infer_kateto.py:322`, `opd_train.py:224`), pero los **pesos del A/B
  `rosa-on` son RANDOM sin entrenar** (`infer_kateto.py:325`): +9,4 pts de ruido, no de recall
  (`docs/rosa-ggml-viabilidad.md`).

**(b) La de BlinkDL** — *RWKV Online Suffix Automation* (`zyaaa-ux/ROSA-Tuning`, origen
`BlinkDL/RWKV-LM/tree/main/RWKV-v8`): autómata de sufijos **no neuronal**, en CPU. "ROSA only needs to
cache the `rosa_token_id` … O(1) per step". Es, por diseño, **un cache de ids serializable** —
convergencia interesante con la conclusión de (a).

**Veredicto:** serializable nativo = **sí** (son tensores); serializable en el **formato de estado GGUF
actual = no** sin un mecanismo de memoria que crece (KV cache paginado) y extensión del formato.
El orden correcto es **cablear + entrenar ROSA nativo primero**, y recién después portar (si gana al ruido).

### Q4 — ¿Híbridos que bajen KV e igual mejoren coherencia larga? → **RWKV-X**

- **RWKV-X** (arXiv 2504.21463, `howard-hou/RWKV-X`): bloques RWKV-7 + **Top-k Chunk Sparse Attention**
  insertados periódicamente. **O(N) en train, O(1) en decode**, **64K passkey casi perfecto** donde
  RWKV-7 degrada >28K, decode estable a **1M** con memoria constante. Se construye por **block-expansion
  sobre RWKV-7** (zero-init, compatible con el base existente). **NO está en llama.cpp** → port propio.
- Los híbridos de fábrica (Jamba, Granite-Hybrid, Qwen3-Next/GDN, Falcon-H1) **sí están en llama.cpp**
  pero **retienen KV cuadrático** → no ayudan al objetivo de N voces en 4 GB.
- **Trampa:** los híbridos SSM→GGUF pierden ~7 pp porque llama.cpp emite **F16 fall-through** para los
  tensores SSM (quirks de conversión), no por cuantización.

**Recomendación Q4:** si el dolor real es coherencia >28K, **RWKV-X es el único híbrido que cumple el
contrato** (KV constante + recall largo). Es un proyecto de port, no un swap de GGUF.

### Q5 — Voces pre-entrenadas en rioplatense usables como base o referencia → **existen, pero son TTS (salida), no base del LLM de voz**

| Modelo / recurso | Qué es | Licencia / nota |
|---|---|---|
| **`marianbasti/` y `surus-ai/Llama-3.2-3B-Orpheus-Rioplatense-1795`** | Orpheus TTS (Speech-LLM Llama-3.2-3B) afinado desde `canopylabs/3b-es_it-ft-research_release` con `ylacombe/google-argentinian-spanish`; acento **rioplatense**, clonación zero-shot, ~200 ms streaming | **el candidato rioplatense medido** del proyecto (ya bajado/verificado por el usuario) |
| `ylacombe/google-argentinian-spanish` | **Corpus** de referencia del acento (usado para el fine-tune de arriba) | dataset HF |
| `audio8-asr-0.1b-es-rioplatense` | ASR rioplatense (Arkasr, Qwen3-ASR) | **CC-BY-NC 4.0 (no comercial)** — solo proyecto personal |
| `Qwen/Qwen3-TTS-*` | TTS Apache-2.0, clonación 3 s, 10 idiomas (incl. español) | Apache 2.0, corre en ~4 GB |
| Piper `es_AR-daniela-high` | TTS rioplatense CPU (el que sirve hoy Kateto) | voz argentina del catálogo |

**Nota importante de encuadre:** Kateto es un **LLM de texto** que hace de voz (persona rioplatense);
el acento lo pone el **TTS aguas abajo** (Piper/Orpheus), no el base RWKV. Por eso "voice model base"
en sentido estricto = **TTS/ASR rioplatense** (tabla de arriba). No hay (ni hace falta) un RWKV-7
1.5B "rioplatense" pre-entrenado: el acento se clona en el TTS, y el LLM se afina con datos argentinos.

---

## 3. Recomendación de arquitectura

**No migrar de RWKV-7.** Es la única opción que satisface simultáneamente: (1) runtime CPU/Vulkan real en
este host, (2) estado recurrente **chico, serializable e inyectable** → N voces sobre **un base cargado**
sin un KV cache por voz (clave para 31 GB RAM / 4 GB VRAM), (3) base chat multilingüe del tamaño
que entra (1.5B/2.9B), (4) converter GGUF probado. Cambiar a Mamba/Griffin/RetNet/xLSTM **no arregla un
problema y agrega dos**: la mayoría no tiene runtime GGUF, y los que sí (Mamba/2) reintroducen KV por voz.

**Orden de trabajo (mayor valor por esfuerzo):**

1. **Cerrar la inyección de estado en el base servido** (no reasignar arquitectura). El blocker es
   **shapes**: reentrenar la Capa 2 sobre el base que se sirve (2.9B/40 cabezas) y volver a correr B3
   para pasar `SHAPES_NO_CALZAN` → `PASS`. El formato ya está probado (B1/B2 PASS).
2. **Coherencia larga**: si >28K es el cuello, **RWKV-X** es la respuesta arquitectónica correcta
   (KV constante + 64K recall). Presupuestar como port propio, no como GGUF drop-in.
3. **ROSA nativo primero**: arreglar el `torch.cat` O(T²), arrastrar `(keys,values,pos)` en el `state`
   (serializable — son tensores), entrenar los pesos. **Recién si gana al piso de 32,8 %** se abre el port.
4. **TTS rioplatense**: evaluar Orpheus Rioplatense 3B como salida; mantener Piper `es_AR-daniela-high`.

**No hacer:** cazar RetNet/Hawk/Griffin/xLSTM (sin runtime), ni híbridos con KV cuadrático (Jamba,
Qwen3-Next) para el objetivo multi-voz, ni hornear estados en pesos (`merge_state.py` produce 24 tensores
muertos — está marcado NO SIRVE en el propio archivo).

---

## 4. Repos y refs fijados (pinned)

| Ref | commit / versión | Rol |
|---|---|---|
| `kateto-train` (local) | `48c0ba7b9836c8815cadee981350c0a6c450786c` | repo del proyecto |
| `ggerganov/llama.cpp` (local `llama.cpp-master`) | `ce8caa6e60a03093351d6016a818720e0d46f0fb` (upstream PR #29152) | fuente del formato de estado auditado |
| llama.cpp instalado | `0.1.2-dev build 10502 commit 8ff87cc` | binario del round-trip PASS |
| `MollySophia/rwkv-mobile` | vendoreado (`export_gguf.sh:35`) | **conversión** pth→GGUF + `load_initial_state` + serialize/deserialize runtime state |
| `zyaaa-ux/ROSA-Tuning` + `BlinkDL/RWKV-LM` (RWKV-v8) | — | ROSA "Online Suffix Automation" (origen) |
| `howard-hou/RWKV-X` | arXiv 2504.21463 | híbrido RWKV-7 + top-k sparse attn (checkpoints públicos) |
| llama.cpp PR **#11452** (MollySophia) | merged | soporte nativo RWKV v7 |
| llama.cpp PR **#27523** | — | fuse CUDA RWKV7 recurrent input+state |
| llama.cpp issue **#25913** / PR **#20955** | — | slot save/restore pierde checkpoints en recurrentes / sidecar |
| vLLM PR **#48686** | — | serving nativo RWKV7 con plomería de state-cache de Mamba |
| `fla-hub/RWKV7-G1j-2.9B-20260831`, `RWKV/RWKV7-G1j-1.5B-20260831` | — | bases oficiales |
| `marianbasti` / `surus-ai` `Llama-3.2-3B-Orpheus-Rioplatense-1795` | — | TTS rioplatense |

**Evidencia local reutilizable:** `docs/gguf-estado-roundtrip.md` (B1, PASS),
`docs/gguf-estado-python.md` (B2, PASS), `docs/gguf-estado-pth.md` (B3, SHAPES_NO_CALZAN),
`docs/ggml-rwkv7-rosa-gap.md` (auditoría de kernels/hooks), `docs/rosa-diagnostico.md` (ROSA línea por línea),
`docs/rosa-ggml-viabilidad.md` (gate DESPUÉS), `docs/decision-gguf-vs-nativo.md` (corregido).
Artefactos: `rwkv_pipeline/{state_bytes,state_from_pth,rosa_module,infer_kateto}.py`;
`tests/{test_state_bytes,test_state_from_pth,test_rosa_*}.py`.

---

## 5. Lo NO verificado

- Que el estado `.pth` inyectado produzca la **misma semántica** que el camino nativo (solo se probó layout + logits del propio dump).
- Compilación/ejecución de `wkv7.comp` (Vulkan) en `gfx1030`: **no verificado** (no se compiló).
- VRAM real del 2.9B Q4_K_M + estado F32 en 4 GB: **no verificado**.
- Velocidad de swap de estado por GGUF (los 48-135 ms son del camino nativo, no de GGUF).
- Que ROSA **mejore** `tasa_utilizable` con pesos entrenados (hoy: +9,4 pts de ruido con pesos random).
- Soporte de un `.pth` de shapes correctas: no existe todavía (el `seco` es de otro modelo).
- `rwkv-mobile` vs `master` actual de llama.cpp: el fork es de septiembre, el master posterior → compatibilidad **no verificada**.
