# Gap de GGML/llama.cpp para RWKV-7 + ROSA + estados (Etapa 4.0)

**Fecha:** 2026-10-01 · **Tipo:** auditoría de solo lectura, no implementa nada.
**Árbol auditado:** `/run/media/chaos/terciario/proyectos/llama.cpp-master`
**Commit:** `ce8caa6` (rama `master`, upstream #29152).
**Hardware de referencia:** AMD Radeon RX 6500 XT (Navi 24, `gfx1030`), 31 GB RAM, i3-10105F.

> **Corrección de ruta.** El brief pedía auditar `chaos-cache/llama.cpp`:
> **ese directorio existe pero está vacío**. Se auditó el árbol real más completo
> de la máquina, `proyectos/llama.cpp-master` (único con `ggml-vulkan`, `ggml-hip` y
> RWKV-7 completo). Rutas relativas a esa raíz.
>
> **Evidencia.** Cada afirmación lleva `ruta:línea`; los greps negativos citan el
> comando. Lo no verificado se dice **"no verificado"**.

---

## 1. Inventario de lo que YA existe

### 1.1 Arquitectura RWKV en llama.cpp

Sí existe RWKV-7, y es una arquitectura de primera clase (no experimental).

| Evidencia | Línea cruda |
|---|---|
| `ggml/include/ggml.h:579` | `GGML_OP_RWKV_WKV6,` |
| `ggml/include/ggml.h:581` | `GGML_OP_RWKV_WKV7,` |
| `ggml/include/ggml.h:2608` | `GGML_API struct ggml_tensor * ggml_rwkv_wkv7(` |
| `src/llama-arch.cpp:101` | `{ LLM_ARCH_RWKV7,            "rwkv7"            },` |

Grafo, todo en `src/models/`: `rwkv7.cpp` (211 líneas, `llama_model_rwkv7`),
`rwkv7-base.cpp` (137, `build_rwkv7_time_mix` y `..._channel_mix`),
`arwkv7.cpp` (202, la variante RWKV-7 **attention** con `softmax`), y además
`rwkv6.cpp`, `rwkv6-base.cpp`, `rwkv6qwen2.cpp`: RWKV-6 también está.

**Ops que usa el grafo de RWKV-7.** No es una arquitectura de transformer: es un `time_mix` puramente
lineal/recurrente. Las ops que aparecen en `build_rwkv7_time_mix`
(`src/models/rwkv7-base.cpp:30-137`): `ggml_sub`, `ggml_repeat`, `ggml_add`, `ggml_mul`,
`ggml_sqr`, `ggml_relu`, `ggml_sigmoid`, `ggml_tanh`, `ggml_exp`, `ggml_scale`,
`ggml_mul_mat`, `ggml_l2_norm`, `ggml_reshape_2d/3d`, `ggml_view_1d/2d`,
`ggml_norm`, `ggml_sum_rows`, `ggml_cpy`, y **una sola** op exclusiva:
`ggml_rwkv_wkv7`. No hay `softmax` ni KV-cache: el estado recurrente queda absorbido dentro de esa op.

**El punto de enganche exacto** — este es el número que importa para ROSA:

Evidencia del enganche: `src/models/rwkv7-base.cpp:107` →
`ggml_tensor * wkv_output = ggml_rwkv_wkv7(ctx0, r, w, k, v, ggml_neg(ctx0, kk), ggml_mul(ctx0, kk, a), wkv_state);`

Todo lo que ROSA quiserá tocar (r, w, k, v, a) se materializa en tensores
**antes** de esa línea y se puede interceptar ahí. `g1j`/`g1d` no aparecen
como arquitecturas distintas: el gating es genérico vía `has_gating`
(`src/models/rwkv7-base.cpp:49`), así que el 2.9B G1J del repo entra por el mismo grafo.

### 1.2 Matriz de backends de los kernels `wkv` — **una línea por backend**

Este es el hallazgo que decide si el camino es viable en esta máquina.

| Backend | ¿WKV6? | ¿WKV7? | Evidencia |
|---|---|---|---|
| **CPU** | Sí | Sí | `ggml/src/ggml-cpu/ggml-cpu.c:2076` → `case GGML_OP_RWKV_WKV6:` ; `:2084` → `case GGML_OP_RWKV_WKV7:` |
| **CUDA** | Sí | Sí | `ggml/src/ggml-cuda/ggml-cuda.cu:2376` (WKV6), `:2394` (WKV7); impl en `ggml/src/ggml-cuda/wkv.cu` |
| **Vulkan** | Sí | **Sí** | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:12261` y `:12266` (dispatch), `:8705`/`:8710` (pipelines) |
| **Metal** | Sí | Sí | `ggml/src/ggml-metal/ggml-metal-ops.cpp:1882` — `const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? ...` |
| **HIP** (AMD) | Sí | Sí | indirecto: `ggml/src/ggml-hip/CMakeLists.txt:63` → `file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")` |
| SYCL | Sí | Sí | `ggml/src/ggml-sycl/ggml-sycl.cpp:5698`, `:5701` (fuera de alcance, se anota por completitud) |

No es un `case` vacío: la implementación real está en
`ggml/src/ggml-cpu/ops.cpp:10413` (`ggml_compute_forward_rwkv_wkv6_f32`) y
`:11422` (`...wkv7_f32`), y en Vulkan los shaders existen como archivos
(`ggml/src/ggml-vulkan/vulkan-shaders/wkv6.comp`, `wkv7.comp`), con pipeline
`ggml/src/ggml-vulkan/ggml-vulkan.cpp:8705`/`:8710` y `supports_op` que devuelve
`true` sin condiciones en `:15391` (`return true; // all inputs are contiguous`).

**Para esta máquina.** La RX 6500 XT no tiene CUDA ni Metal: quedan **Vulkan**
(shaders ya escritos) e **HIP** (herencia del glob de CUDA). Los dos tienen WKV7, así
que no hace falta escribir un kernel para que RWKV-7 ande acá.

Salvedad honesta: que el shader exista y que `supports_op` devuelva `true` **no**
prueba que el shader compile o corra en `gfx1030`. Eso exige compilar, y la regla
de esta etapa prohíbe compilar. **No verificado.**

### 1.3 Cuantización

No hay nada especial para RWKV, y no hace falta: la cuantización standard se
aplica a las matrices lineales (el `wkv` op entra por `mul_mat`, que ya está
cuantizado en todos los backends). Lo importante es el otro lado:

| Evidencia | Línea cruda |
|---|---|
| `ggml/src/ggml.c:5993` | `result->op = GGML_OP_RWKV_WKV7;` |
| `ggml/src/ggml.c:5992` | `struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);` |

**El tensor de estado y la salida del `wkv` son F32 duro**, y las assertions de
`ggml.c:5966-5983` exigen contigüidad de `r,w,k,v,a,b,state`. O sea: se puede
cuantizar el modelo a `Q4_K_M` (el `export_gguf.sh` del repo ya lo hace), pero el
estado recurrente **no** se comprime. 6-20 MB de estado, como dice el doc previo.

### 1.4 Estado: guardar, restaurar e **inyectar**

Acá hay que corregir un belief que ya estaba escrito en el repo.
`docs/decision-gguf-vs-nativo.md:20` afirma que "llama.cpp no tiene bandera ni API
para inyectar un estado inicial externo". **Eso es incorrecto a nivel API.**
Auditando la *header*, no los flags del CLI, el camino existe:

Y la implementación real del guardado es **byte-stream crudo por capa**, no un
blob opaco. Eso es lo que hace el camino barato:

| Evidencia | Línea cruda |
|---|---|
| `include/llama.h:892` | `LLAMA_API size_t llama_state_seq_set_data(` |
| `src/llama-context.cpp:4223` | `size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {` |
| `src/llama-context.cpp:3127` | `size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {` |
| `src/llama-memory-recurrent.cpp:766` | `void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {` |
| `src/llama-memory-recurrent.cpp:847` | `void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {` |
| `src/llama-memory-recurrent.cpp:897` | `void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) const {` |
| `src/llama-memory-recurrent.cpp:923` | `io.write_tensor(r_l[il], range.first * r_size_row, buf_size);` |
| `src/llama-context.cpp:3093` | `static constexpr uint32_t io_magic = 0xaf143cd8;` |

**Conclusión de estado: la inyección es posible sin C++ nuevo.** El formato es
determinista y trivial de construir desde Python: `uint32 io_magic (0xaf143cd8)`,
`int32 seq_id`, `uint32 cell_count`, los rangos de celdas, y por capa los rows F32
de `r_l` (más `p_l` si hay conv PLE, más `s_l`). El `export_gguf.sh` ya produce un
GGUF F16 y un Q4_K_M que se cargan; el paso que falta es un `.pth` → byte-stream.

Dos salvedades: (a) `state_write` depende de `cell_ranges`, o sea del historial
de celdas que el runtime acumuló — para un estado suelto hay que escribir una sola
celda; (b) el formato no tiene versión propia (solo el `magic` de sesión), así que
es **frágil ante upgrades de llama.cpp**. Eso se mitiga con un test de round-trip.

### 1.5 Conversión

**Upstream no convierte RWKV.** El grep lo demuestra:

```
$ grep -nic rwkv convert_hf_to_gguf.py
0
```

No hay rama RWKV ahí, no hay `convert_rwkv_to_gguf.py`, y `grep -rln -i rwkv *.py`
en la raíz no devuelve nada. La conversión real viene de un **fork externo**,
`MollySophia/rwkv-mobile`, que el repo ya clona y ya usa en `export_gguf.sh:28`:

| Evidencia | Línea cruda |
|---|---|
| `rwkv-mobile/converter/convert_rwkv_pth_to_gguf.py:539` | `model_arch = gguf.MODEL_ARCH.RWKV7` |
| `rwkv-mobile/converter/convert_rwkv_pth_to_gguf.py:467` | `model_arch = gguf.MODEL_ARCH.RWKV6` |
| `export_gguf.sh:28` | `python "$TDIR/rwkv-mobile/converter/convert_rwkv_pth_to_gguf.py" "$MERGED" ... --outtype f16` |

Checkpoint esperado: **`.pth` de PyTorch de RWKV-7** (el mismo formato que produce
`merge_lora.py`), más el vocabulario `rwkv_vocab_v20230424.txt`. Hay soporte
adicional para `safetensors` (`convert_rwkv_to_safetensors.py`).

---

## 2. Qué es ROSA y por qué complica todo

`rwkv_pipeline/rosa_module.py`, 78 líneas. Es una **capa de atención softmax
paralela** pegada al hidden state, no una modificación del `wkv`:

1. Proyecta `hidden_states` a un espacio chico de `retrieval_dim` (128) con
   `proj_q`, `proj_k`, `proj_v` (`rosa_module.py:22-24`).
2. `forward_train` (:40) hace `scores = einsum("bid,bjd->bij", q, k) * scale`,
   máscara causal con `torch.tril` (:43) y `softmax` (:45).
3. `retrieved = einsum("bij,bjd->bid", attn, v)` (:48), pasa por `out_head` a
   vocab (:49) y se devuelve **junto a un gate** `sigmoid(gate(h))` (:50) que el
   llamador debería mezclar con los logits reales.

Dónde se engancha: **en ningún lado todavía.** `grep -rn "RosaAssociativeMemory"`
en todo el repo no encuentra ningún `import` ni instanciación fuera del propio
módulo y sus copias en `kaggle_v2/` y `kaggle_upload/`. `infer_kateto.py` no lo
importa. ROSA hoy es un módulo experimental, diseñado pero no cableado. La
afirmación de `HANDOFF.md:274` ("Inicializa `RosaAssociativeMemory`") no se
sostiene contra el código actual.

Qué necesitaría el backend GGML: ROSA es `softmax(QKᵀ + máscara causal)V` sobre
el stream — es decir, **es `ggml_soft_max` con máscara**, que ya existe y está
cuantizado en todos los backends. Lo que falta no es un kernel de math: es
(1) un lugar donde enganchar el `hidden` de la capa (el `cur` justo antes de
`build_rwkv7_channel_mix`, `src/models/rwkv7.cpp:190`), (2) los 4 tensores de ROSA
persistidos en el GGUF, y (3) el `gate` mezclado con los logits en el sampler.
Son **pesos + glue**, no una op nueva.

Ojo con un detalle del módulo: `step_infer` (:67) hace
`memory_k = torch.cat([memory_k, k], dim=1)`. Eso es **O(T) por token**, es decir
O(T²) por contexto, y contradice su propio docstring "O(1)" (:56). Para GGML esto importa: el equivalente natural es el mismo KV-cache
paginado que ya existe para attention, no un buffer que crece. **Es un defecto del
módulo PyTorch, y hay que arreglarlo antes de portarlo**, no después.

---

## 3. La lista de lo que FALTA

### 3.1 Kernels (ops nuevas en ggml + backends)

| Componente | Qué falta | Dónde iría | Tamaño |
|---|---|---|---|
| Kernels `wkv6`/`wkv7` | **Nada.** Ya están en CPU/CUDA/Vulkan/Metal/HIP/SYCL | — | **0** |
| Op de ROSA | **Nada.** Es `soft_max` + `mul_mat` existentes; el `wkv` no se toca | — | **0** |
| Hook de ROSA en el grafo | Un tensor extra por capa + 4 pesos, con `ggml_*` existentes | `src/models/rwkv7-base.cpp` | **días** |
| `wkv7` en Vulkan para `gfx1030` | Verificar que `wkv7.comp` compila/corre en Navi 24 (no verificado) | sin código, solo compilar | **1 día** |

Las tres primeras filas no necesitan C++. Pese al nombre de la sección, el trabajo
pesado de kernels ya está hecho upstream.

### 3.2 Serialización de estado (lo que hoy se hace con `.pth`)

| Componente | Qué falta | Dónde iría | Tamaño |
|---|---|---|---|
| Inyección de estado completo | **Ya existe** vía `llama_state_seq_set_data`; falta el *productor* | cliente, no llama.cpp | **días** |
| `.pth` → byte-stream | No existe. Hay que leer `blocks.{i}.att.time_*` y emitir `magic/seq_id/cell_count/ranges/rows F32` en el orden de `state_write_data` | `scripts/` del repo de entrega | **1 semana** |
| Test de round-trip | Falta. Sin él, cualquier upgrade de llama.cpp rompe silenciosamente | test en el repo de entrega | **2-3 días** |
| Flag de CLI para cargar estado | No existe: `grep -rn "state-save\|state-load" common/ examples/` viene **vacío**. Hay que agregar `--state-load` | `common/arg.cpp` | **1 semana** |
| Estado multi-agente con el mismo modelo | Falta la ergonomía (un slot por voz). `llama_memory_seq_cp` (`include/llama.h:765`) ya existe como base | `examples/` | **semana** |

Razón de las cuatro filas: ninguna es un kernel. Es leer un `.pth` y escribir bytes
en un orden que ya está fijado por `state_write_data`. El trabajo real es
**convencernos de que el orden no cambia**, y eso lo resuelve el round-trip test,
no más C++.

### 3.3 Glue (carga de pesos, conversión)

| Componente | Qué falta | Dónde iría | Tamaño |
|---|---|---|---|
| Conversión base RWKV-7 → GGUF | **Ya existe** en el fork `rwkv-mobile`, ya usada por `export_gguf.sh` | — | **0** |
| Nombres de tensor de ROSA en el GGUF | Hay que agregar `rosa.proj_q/k/v`, `rosa.out_head`, `rosa.gate` al esquema y al conversor | `gguf-py/gguf.py` + conversor | **semana** |
| Mapeo de esas capas en el grafo | `load_arch_tensors` de RWKV-7 no las conoce | `src/models/rwkv7-base.cpp` | **días** |
| Gate de ROSA sobre los logits | El `gate_weight` no se aplica en ningún lado del sampler de llama.cpp | `src/llama-sampling.cpp` o equivalente | **semana** |
| LoRA merge → GGUF con ROSA | `merge_lora.py` funde LoRA; hay que extenderlo a los tensores de ROSA | repo de entrega | **días** |

---

## 4. Recomendación

### Camino A — GGUF "plano", sin ROSA (el de hoy). Costo: 0. Riesgo: 0.
Es lo que `export_gguf.sh` ya hace: inference y evals. No sirve para state-tuning.
Sirve de base y de control.

### Camino B — GGUF + estado inyectable, sin ROSA. Costo: ~3 semanas. Riesgo: **bajo**.
El único blocker del proyecto (inyectar `S_0` entrenado) **ya está resuelto por
API**; falta el adaptador `.pth` → bytes y un flag. No se toca un solo kernel.
Es el que da valor por unidad de esfuerzo: convierte el state-tuning —la Capa 2,
que es lo que kateto-train existe para hacer— en algo que corre en Vulkan en esta
máquina. **Es el que conviene primero.**

### Camino C — GGUF + estado + ROSA completo. Costo: ~2 meses. Riesgo: medio-alto.
No por los kernels (no hay kernels nuevos) sino porque ROSA **no está cableada
ni en PyTorch hoy**. Portar a GGML algo que todavía no funcionaba natively es
construir sobre arena. Además arrastra el defecto O(T²) de `step_infer`, que hay
que arreglar antes.

**Orden: A (ya está) → B → y recién ahí mirar C.**

Y una advertencia de honestidad sobre el supuesto que organiza todo esto: el
documento previo (`decision-gguf-vs-nativo.md`) concludes que GGUF es inviable
porque "no hay API para inyectar estados". Auditando la header, esa API existe
(`include/llama.h:892`). Antes de decidir con números hay que decidir con el
número correcto, y ese era incorrecto.

---

## 5. El experimento mínimo (lo más importante del documento)

**Se puede verificar casi todo lo anterior sin escribir una línea de C++ de
ggml, y sin compilar kernels.**

El experimento es un programa chico que linkea `libllama` y hace un **round-trip
de identidad sobre el estado**:

1. Cargar el GGUF Q4_K_M que `export_gguf.sh` ya produce.
2. Decodificar N tokens (`llama_decode`).
3. `llama_state_seq_get_data` → buffer A.
4. `llama_state_seq_set_data` con el mismo buffer A en otra `seq_id`.
5. Decodificar el mismo token desde el estado inyectado y comparar logits contra
   la corrida original.

Si el paso 5 reproduce los logits, quedan probadas **tres** cosas de una: que
RWKV-7 corre en Vulkan sobre `gfx1030` (o HIP); que el estado es byte-stream
determinista y por lo tanto **construible desde Python**; y que el state-tuning es
viable sobre GGUF.

Eso son 80-120 líneas de C++ **de cliente, no de ggml**, contra un binario que se
compila sin tocar backends. Si el round-trip falla, el culpable queda acotado a dos
lugares (orden del byte stream, o `cell_ranges`), y ambos se debuggean en el cliente
sin tocar un kernel.

Lo que **no** se puede verificar sin escribir C++ de verdad: que el shader
`wkv7.comp` funcione en gfx1030 (se verifica compilando y corriendo, no leyendo),
y que ROSA quantizada sea tolerable (se verifica después del Camino B, y solo si
ROSA se cablea en PyTorch primero).

---

### Registro de lo no verificado

- Compilación y ejecución de `wkv7.comp` en `gfx1030`: **no verificado** (prohibido
  compilar en esta etapa).
- VRAM real ocupada por el 2.9B Q4_K_M + estado F32 en 4 GB: **no verificado**.
- Que `rwkv-mobile` siga siendo compatible con el `master` actual de llama.cpp:
  **no verificado** (el fork es de septiembre, el master es posterior).
- Corrección o no de los logits en el round-trip: **es justamente lo que propone
  el experimento de arriba**, todavía no corrido.