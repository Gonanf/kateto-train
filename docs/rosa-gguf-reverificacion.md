# Reverificación: ROSA en GGUF contra el llama.cpp de HOY

**Fecha:** 2026-10-23 · **Tipo:** reverificación de solo lectura. No modifica `llama.cpp`,
no entrena, no implementa.
**Qué re-verifica:** `docs/ggml-rwkv7-rosa-gap.md` (2026-10-01, árbol @ `ce8caa6`).
**Card:** `t_2750e4cf` — «ROSA en GGUF: revisar si se puede cablear a llama.cpp (Camino C)».

Regla de este documento: lo que **medí yo** va marcado *(medido hoy)*; lo que viene de un doc
del repo va con su ruta; lo que no verifiqué está en la sección 8.

---

## 0. Veredicto en tres líneas

1. **ROSA sigue afuera de llama.cpp hoy**, y no por falta de buscar: no está en el `master`
   de upstream (`0ae0e19`), ni en los dos árboles locales. El GGUF de **V7 es Camino A** (plano,
   sin ROSA) y no hay forma de meter ROSA en un `.gguf` sin escribir C++ en el grafo.
2. **RWKV-8 no es Camino A ni Camino B: no existe como arquitectura.** llama.cpp soporta
   `rwkv6`, `rwkv6qwen2`, `rwkv7`, `arwkv7` — y nada más. Un GGUF de V8 exige portar una
   arquitectura nueva (con su conversor), no convertir.
3. El **Camino B ya no es «arrancable»: está hecho y lo volví a correr hoy** — round-trip de
   estado **PASS** sobre un llama.cpp **más nuevo** que el del doc (ver §4). Lo que falta del
   B no es código de inyección: es **semántica** (el estado no es captura del camino nativo) y
   **ergonomía** (falta el flag de CLI).

Y la corrección incómoda: la premisa central del audit —«ROSA no está cableada»— **es falsa
desde el 1/10 a las 13:12**. Está cableada en PyTorch, opt-in. Eso **no** abre el Camino C,
porque lo que lo traba cambió de lugar: ya no es el enganche, son los **pesos** (§3) y la
**memoria** (§5).

---

## 1. llama.cpp de hoy: ROSA sigue ausente

### 1.1 Qué árbol es «hoy»

| Árbol | Commit | Fecha | ROSA |
|---|---|---|---|
| `/run/media/chaos/terciario/proyectos/llama.cpp-master` (el del audit) | `ce8caa6` | 2026-09-20 | no |
| `/run/media/chaos/secundario/llamacpp-conv` (el que usa `llama-cli` de `~/.local/bin`) | `f2cc728` | 2026-10-11 | no |
| `ggml-org/llama.cpp` `master` upstream | `0ae0e19` | hoy | no |

*(medido hoy)* El árbol del audit **no se movió** desde el 1/10: sigue en `ce8caa6`
(`git log -1`). O sea que el audit no estaba desactualizado por su árbol, sino por el código
del repo `kateto-train`. Por eso la reverificación se hizo contra upstream, no contra el árbol
local: `git ls-remote origin HEAD` → `0ae0e19`.

### 1.2 Evidencia de ausencia, upstream (`0ae0e19`)

| Prueba | Comando / fuente | Resultado |
|---|---|---|
| Archs RWKV soportadas | `src/llama-arch.cpp` | `:105 rwkv6`, `:106 rwkv6qwen2`, `:107 rwkv7`, `:108 arwkv7`. **No hay rwkv8 ni rosa.** |
| Archivos de modelo | `contents/src/models` (API) | 168 archivos; `rwkv6.cpp`, `rwkv6-base.cpp`, `rwkv6qwen2.cpp`, `rwkv7.cpp`, `rwkv7-base.cpp`, `arwkv7.cpp`. **Ningún `rosa*.cpp`.** |
| Ops de ggml | `ggml/include/ggml.h` | `:579 GGML_OP_RWKV_WKV6`, `:581 GGML_OP_RWKV_WKV7`. Mismas dos líneas que en `ce8caa6`: no se agregó ninguna op. |
| Esquema GGUF | `gguf-py/gguf/constants.py` | `:1435 MODEL_ARCH.RWKV7: "rwkv7"` y `:3797` su lista de tensores. **Ninguna clave `rosa.*`.** |
| Búsqueda de código | GitHub code search `rosa repo:ggml-org/llama.cpp` | **1 hit**: `tools/mtmd/mtmd-audio.cpp` — es «Slaney», falso positivo. |
| Historial | GitHub commit search `rwkv repo:ggml-org/llama.cpp` | 21 commits; el de la arquitectura es de 2025-03-18, el más nuevo («fix RWKV ops thread assignment») de 2026-04-01. **Ninguno de ROSA.** |
| Árbol local nuevo | `grep -rn -i rosa src ggml/src include` en `llamacpp-conv` | 0 líneas *(medido hoy)* |

### 1.3 Consecuencia para el pedido del dueño

El pedido era «convertir V7/V8 en GGUF completos con ROSA». Desarmado en las tres cosas que
realmente pide:

- **V7 → GGUF**: sí, hoy, con el conversor `rwkv-mobile` (`MODEL_ARCH.RWKV7` en
  `rwkv-mobile/converter/convert_rwkv_pth_to_gguf.py:539`) y los GGUF ya construidos
  (`out/kateto-rwkv29-Q4_K_M.gguf`, `out/kateto-rwkv29-Q8_0.gguf` del 23/10 16:35). **Camino A.**
- **V8 → GGUF**: no. No hay arch, no hay conversor (`rwkv-mobile` sólo conoce `RWKV6` en `:467` y
  `RWKV7` en `:539`), no hay pesos preentrenados de chat (`BlinkDL/rwkv-8-pile` es inglés-only y
  «early testing», ver `docs/knowledge-rwkv-kateto-rosa-gguf-rx6500xt.md:30`). Portar la
  arquitectura es un trabajo de otra naturaleza y otro tamaño.
- **ROSA dentro del GGUF**: no, y por lo medido en §5 tampoco sería barato si mañana estuviera.

---

## 2. Lo que el audit del 1/10 dice y ya no es cierto

El audit cierra la sección 2 con: «Dónde se engancha: **en ningún lado todavía**»
(`docs/ggml-rwkv7-rosa-gap.md:168`). **Hoy es falso.** *(medido hoy sobre el árbol actual)*

| Afirmación del audit | Estado verificado hoy |
|---|---|
| `grep RosaAssociativeMemory` no encuentra enganchador | **Falso.** `rwkv_pipeline/infer_kateto.py:324` lo importa, `:327` lo instancia; `:334-359` carga `--rosa-head` con guardas de forma (`n_embd`/`vocab_size`/`retrieval_dim`) que abortan con `SystemExit` si el head es de otro modelo. Además `tests/test_rosa_wiring.py` y `tests/test_rosa_equivalence.py` existen. |
| ROSA «pesos + glue, no una op nueva» | **Mitad falso.** Glue/pesos sí; la **memoria** no (§5). Lo corrige `docs/rosa-ggml-viabilidad.md:255`. |
| El gate de ROSA va en el sampler | **Falso.** `docs/rosa-ggml-viabilidad.md` §3: `sigmoid(gate(h))` es función pura del hidden; va en el grafo (`src/models/rwkv7.cpp:205`, sobre `res->t_logits` en `:208`), no en `src/llama-sampler.cpp`. |
| No hay pesos de ROSA para RWKV-7 | **Sigue cierto, pero cambió de forma:** hay heads entrenados (`out/rosa_head_corto/`, `out/rosa_head_largo/rosa_head.pth` del 7/10) y **ninguno está medido con el A/B multiturno salvo el de 300 steps**, que dio delta 0 (abajo). |

Cómo se desincronizó: el audit es de las 11:19 y el cableado de ROSA entró en `infer_kateto.py`
entre las 13:12 y las 13:49 del mismo día. `docs/rosa-ggml-viabilidad.md` (14:53) ya lo apunta.

---

## 3. ROSA: lo que se midió, y por qué «no ayuda» no es lo que dicen los números

Del A/B con head entrenado (`docs/rosa-medicion-entrenada.md`, `R9`):

| Brazo | utilizables | textos idénticos |
|---|---|---|
| `--rosa --rosa-head out/rosa_head_corto/rosa_head.pth` (step 300) | **15/64** | **64/64** contra OFF |
| sin ROSA | **15/64** | — |

Delta exactamente 0, con textos idénticos fila por fila: la rama entrenada **no cambió un solo
token** en 64 turnos con seed 1337. Y el `train.log` de ese head explica por qué: `rosa_grad_norm`
en 0 desde ~step 93, norma de ROSA +1.1%. Un A/B contra una rama que no aporta mide el ruido de
la rama, no el recall de ROSA — igual que pasó con pesos random (donde «ON» dio 21/64 vs 15/64 =
32.8% vs 23.4%, y el propio instrumento aclara en `scripts/eval_multiturn.py:359-364` que eso es
el ruido de una capa no entrenada).

**Lo que sí hay nuevo y no está medido: `out/rosa_head_largo/rosa_head.pth` (7/10, 300 steps).**
*(medido hoy sobre sus propios logs, no con el A/B)*

| Señal | valor |
|---|---|
| `metrics.jsonl` | 300 filas |
| loss, media del primer cuartil → último | **5.82 → 5.00** (‑14%, con `label_tokens` variable por step: 5-27) |
| `rosa_grad_norm`, primer → último cuartil | **52.1 → 68.8** (gradiente vivo, a diferencia del head corto) |
| norma de ROSA, inicio → fin (de `train.log`) | **11.33 → 62.81** (+51.5, **×5.5**) |

Lectura honesta: este head **sí entrenó** (el gradiente no murió) pero la norma del módulo se
quintuplicó con una loss que apenas bajó, que es la firma de un paso inestable, no de un
aprendizaje limpio. **No hay A/B de este head contra OFF.** Sin eso no se puede decir que ROSA
sirva ni que no sirva: es el único experimento pendiente que vale la pena, y no necesita GGML.

---

## 4. Camino B: estado inyectable — hecho, y re-corrido hoy

### 4.1 Lo que ya estaba (octubre)

| Paso | Artefacto | Veredicto |
|---|---|---|
| B1 round-trip de identidad sobre GGUF | `docs/gguf-estado-roundtrip.md` | **PASS**: 21.627.676 bytes, `max_abs_diff: 0`, misma continuación |
| B2 el buffer se fabrica en Python | `docs/gguf-estado-python.md`, `rwkv_pipeline/state_bytes.py` | **PASS**: re-serialización con 0 diferencias |
| B3 `.pth` nativo → byte-stream | `docs/gguf-estado-pth.md`, `rwkv_pipeline/state_from_pth.py` | **`SHAPES_NO_CALZAN`** con el `.pth` de voz (0.4B, 24 capas) contra el GGUF 2.9B (32 capas). El productor se **niega** con `ValueError` en vez de inventar tensores: es el comportamiento correcto |
| Estados 2.9B reales sobre GGUF 2.9B | `docs/gguf-estados-voz.md` (14/10) | `jane/rwkv-11` y `doktor/*` **calzan** (32 capas, `(40,64,64)`) y la inyección **cambia los logits de forma determinista**; `jane` queda in-distribution, `doktor` cae OOD (todos los logits negativos, texto degenerado) |

### 4.2 Lo que re-corrí hoy *(medido hoy)*

El gate B1, contra el árbol **`llamacpp-conv` @ `f2cc728` (11/10)**, más nuevo que el que el doc
usó (`8ff87cc`):

```
backend: cpu
model: out/kateto-rwkv29-Q4_K_M.gguf
prompt: Hola, soy Kateto.
prompt_tokens: 7
size_bytes: 21627676
state_seq_set_data_bytes: 21627676
header_seq_id_0: 0
header_seq_id_1: 1
test1_identidad: OK (21627668 payload bytes identical after 8 header bytes)
max_abs_diff: 0
argmax_0: 261
argmax_1: 261
argmax_igual: si
test2_logits: OK
continuacion_igual: si
test3_continuacion: OK
RESULTADO: PASS
```

Dos cosas más que verifiqué para que ese PASS signifique algo **contra el master de hoy** y no
sólo contra el árbol que compilé:

- **La API sigue existiendo:** `include/llama.h:890 llama_state_seq_get_data`, `:900
  llama_state_seq_set_data`, más las nuevas variantes `_ext` en `:936-948`.
- **El formato del byte-stream no cambió:** `diff` del `src/llama-memory-recurrent.cpp` de
  `ce8caa6` contra el `master` de hoy da **7 hunks / 127 líneas**, y **ninguno toca el formato**:
  son rollback de snapshots, `is_empty()`, y el `state_clear()` de error. `s_trans` sigue
  hardcodeado en `0` (upstream `:920`, local `:899`), el orden de campos es idéntico
  (`s_trans`, `n_layer`, por capa `r_type`, `r_size_row`, filas; después lo mismo para `s`), y el
  `io_magic` sigue siendo `0xaf143cd8` (`src/llama-context.cpp:3416` upstream).
  → `state_bytes.py` y `state_from_pth.py` **siguen válidos** contra el master de hoy, no sólo
  contra `ce8caa6`.

### 4.3 Lo que falta del B (y no es "3 semanas de código")

1. **Semántica, no layout.** El `.pth` trae sólo `time_state` (tensor `s`); el `r` (token shift)
   va en ceros por diseño (`state_from_pth.py`, y lo repite `docs/gguf-estados-voz.md` §5). Con
   `r` en cero un estado **no es una captura** del estado nativo, aunque calce. Falta la
   comparación token a token entre el camino nativo (`.pth`) y el GGUF con el estado inyectado,
   sobre un estado del **mismo** modelo.
2. **El `.pth` de estado y el GGUF tienen que ser del mismo modelo.** Hoy los estados de 2.9B que
   calzan (`jane`, `doktor`) son de otro entrenamiento que el GGUF base, y los de Kateto (0.4B)
   no calzan con el GGUF de 2.9B. Es producción de artefactos, no C++.
3. **Ergonomía: sigue sin haber flag de CLI.** *(medido hoy)* `grep 'state-save\|state-load'` en
   el `common/arg.cpp` de upstream sólo encuentra el nombre del ejemplo `llama-save-load-state`
   (`:1114`); `--prompt-cache` en `:1880`. El flag `--state-load` que el audit pedía **no existe
   todavía**: hoy el estado se carga con un cliente que linkea `libllama`
   (`tools/gguf_state_roundtrip.cpp`), que es lo que el repo ya tiene.
4. **El script del gate ya no corre tal cual.** *(medido hoy)* `bash script/probar-gguf-estado.sh`
   falla en el primer paso acá:

   ```
   tools/gguf_state_roundtrip.cpp:4:10: fatal error: llama.h: No existe el fichero o el directorio
   BUILD FAILED (exit 1)
   ```

   El script compila con `g++ ... -lllama -lggml -lggml-base` asumiendo un llama.cpp instalado en
   el sistema, y en esta máquina **no hay** `llama.h` ni `libllama` en `/usr{,/local}/include|lib`
   (`ldconfig -p | grep llama` → 0). Para que vuelva a correr hay que pasarle los `-I`/`-L` de un
   árbol concreto. Lo corrí así y dio PASS:

   ```bash
   T=/run/media/chaos/secundario/llamacpp-conv
   g++ -O2 -std=c++17 tools/gguf_state_roundtrip.cpp -o /tmp/b1/gguf_state_roundtrip \
     -I"$T/include" -I"$T/ggml/include" -L"$T/build/bin" -Wl,-rpath,"$T/build/bin" \
     -lllama -lggml -lggml-base
   LD_LIBRARY_PATH="$T/build/bin" /tmp/b1/gguf_state_roundtrip \
     --model out/kateto-rwkv29-Q4_K_M.gguf --prompt "Hola, soy Kateto." --backend cpu
   ```

---

## 5. Camino C: sigue cerrado, y por dos razones distintas a las del audit

**(a) Los pesos.** ROSA entrenada no demostró nada todavía: el único A/B con pesos entrenados dio
delta 0 con textos idénticos (`docs/rosa-medicion-entrenada.md`), y el head «largo» —que sí tiene
gradiente vivo— no está medido (§3). Portar a GGML algo que no demostró ganar es construir sobre
arena, igual que en el audit, pero ahora la arena tiene nombre: **el A/B pendiente**, no el
«cableado faltante».

**(b) La memoria.** Es el costo real y no es glue: el `ggml_rwkv_wkv7` es una op monolítica y la
cache recurrente de llama.cpp guarda **celdas de tamaño fijo** por secuencia, mientras ROSA
necesita memoria que **crece por token** (`rosa_module.py` `step_infer` con `torch.cat`, O(T) por
token). Meterla en las celdas recurrentes **cambia el formato de estado** y rompe B1/B2/B3; meterla
en el KV cache paginado es más C++ todavía (`docs/rosa-ggml-viabilidad.md` §4).

Orden que se sostiene, con los números de hoy: **A (hecho) → B (hecho salvo semántica) → cerrar el
A/B de ROSA nativo → y sólo si gana, estimar C**. Estimar C antes de ese A/B es ponerle precio a
algo que no se sabe si sirve.

---

## 6. Guardas del pedido, aplicadas

- **Multi-semilla:** el A/B de ROSA del repo es **una sola seed (1337)**. Por eso este documento
  **no afirma** que ROSA ayude ni que no ayude; sólo reporta que el delta medido es 0 y que un
  delta 0 en una seed no distingue «no aporta» de «el head no aprendió». El A/B pendiente del head
  `largo` debería ir con ≥3 seeds para poder decir algo.
- **El criterio de éxito no puede ser «siempre responde».** El modelo está entrenado para no
  hablar: en el A/B de 8×8 el modo de falla dominante es el **silencio** (texto real 29,
  `<|no_response|>` 14, `<|wait|>` 4, vacía 17, `rep8 > 0` en **0/64**). Un banco que cuente
  respuestas vacías como fallo mide la política del modelo, no la calidad. Coherente con esto:
  `rep8 = 0.000` en 128/128 filas de los dos brazos.
- **Anti-hack (velocidad/calidad no puede subir degradando coherencia):** el único «gain» que
  existió —32.8% vs 23.4% de utilizables— venía de **pesos random** y desapareció cuando el head
  tuvo pesos entrenados (15/64 vs 15/64). Es exactamente el caso que la guarda quiere atrapar: se
  lee como mejora y es ruido de una capa que perturba los logits.

---

## 7. Qué hacer, en orden y con costo

1. **Cerrar el A/B de ROSA con el head `largo`, nativo, 3 seeds** (2 brazos × 3 seeds × ~16 min de
   GPU ≈ 1.5 h). Criterio de decisión escrito antes de correr: el brazo ON tiene que superar el
   piso de ruido **32.8%** de utilizables **y** no empeorar `rep8`/eco. Si no lo supera, ROSA se
   cancela para GGUF (Camino C muerto) y el pedido del dueño queda respondido con números.
2. **Camino B: cerrar la semántica**, no el formato. Un `.pth` de estado del **mismo** modelo que
   el GGUF + comparación token a token nativo vs GGUF. Sin tocar C++.
3. **Arreglar la reproducibilidad del gate**: que `probar-gguf-estado.sh` tome el árbol por
   variable (`LLAMA_TREE=...`) y no asuma `llama.h` instalado. Hoy el gate no corre de cero.
4. **No** empezar el Camino C. Y decirlo así en el reporte al dueño: el GGUF de V7 que ya existe
   **es** el Camino A, y ROSA no viaja adentro.

---

## 8. Registro de lo NO verificado

- **Que el shader `wkv7` de Vulkan compile y corra en `gfx1030`**: no se compiló nada. El round-trip
  que corrí fue **CPU**.
- **Que el head `largo` de ROSA mejore algo**: no hay A/B. Sólo leí sus logs.
- **Semántica del estado inyectado**: sigue sin medirse la comparación nativo vs GGUF token a token.
- **Que upstream no agregue ROSA mañana**: la ausencia es de hoy (`0ae0e19`), no una promesa.
- **El pedido textual del dueño sobre V8**: no lo leí; lo que verifico es el estado de la técnica
  (no hay arch `rwkv8` ni conversor), no la intención.
