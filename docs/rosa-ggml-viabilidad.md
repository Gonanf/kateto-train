# R6 — Gate de viabilidad: port de ROSA a GGML

**Fecha:** 2026-10-01 · **Tipo:** decision de gate. No escribe C++, no toca `llama.cpp`.
**Arbol:** `/run/media/chaos/terciario/proyectos/llama.cpp-master` @ `ce8caa6` (solo lectura).
**Gate:** `docs/gguf-estado-roundtrip.md` + su salida cruda, re-corrida hoy.
**Veredicto: DESPUES.** El gate dio PASS, asi que la inyeccion de estado esta abierta y el port es
planificable. Pero PASS no es "portar ahora": el port se traba en otra cosa, y esa cosa se mide
en semanas sin escribir una linea de C++.

## 1. El gate, re-corrido hoy (PASS)

`bash script/probar-gguf-estado.sh` (CPU; la GPU estaba ocupada, 1.7/4 GB used). Salida cruda,
recortada solo en el volcado de dispositivos Vulkan (2 lineas largas, que el binario imprime
dos veces) y en los saltos de linea del token generado. La salida completa esta en el reporte
de la corrida.

```
=== build ===
build ok
=== llama.cpp build info ===
version: 0.1.2-dev (build 10502, commit 8ff87cc), GNU 16.2.1, x86_64
ggml_vulkan: Found 2 Vulkan devices:   [+2 lineas de device, repetido al inicializar]

=== round-trip (backend=cpu) ===
backend: cpu
model: out/kateto-rwkv29-Q4_K_M.gguf
prompt: Hola, soy Kateto.   (7 tokens)
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
continuacion_seq0: \n\n[IMPORTANTE]\n\n[     (seq1 identica)
continuacion_igual: si
test3_continuacion: OK
RESULTADO: PASS
script_exit: 0
```

Confirma el doc: 21.627.676 bytes, `max_abs_diff: 0`, misma continuacion. **La barrera de estado NO es el obstáculo.**

## 2. Por que "despues" y no "ahora"

**(a) No hay pesos de ROSA que portar para RWKV-7.** Es lo unico que bloquea de verdad.
`infer_kateto.py:325` instancia la rama y avisa en stderr: *"pesos RANDOM sin entrenar"*. Los
unicos checkpoints ROSA del arbol son **RWKV-8** (`RWKV-LM/RWKV-v8/*.pth`) y este repo es
RWKV-7: portar pesos aleatorios no porta nada.

**(b) El A/B "ROSA gana" no prueba que ROSA sirva.** `out/multiturno/rosa-{on,off}.json`, mismo
modelo, misma politica (`carry`), misma seed: OFF 15/64 = 23.4% utilizables, ON 21/64 =
**32.8%**; `tasa_vacia` 2.33 -> 1.24, `len_median` 63 -> 43. Pero `scripts/eval_multiturn.py:359-364`
aclara que ese A/B *"mide el ruido que mete una capa no entrenada, NO el recall de ROSA con pesos
buenos"*. Son +9.4 puntos de **ruido**, no de valor.

**(c) La auditoria previa quedo desactualizada.** Verificar, no repetir:
- `docs/ggml-rwkv7-rosa-gap.md:168` dice que ROSA no esta cableado. **Falso hoy**:
  `infer_kateto.py:322` lo instancia (opt-in, `rosa=False` por default) y `opd_train.py:224` lo
  engancha con `attach_rosa` como entrenable. El grep del brief quedo viejo por lo mismo: la
  auditoria es de 11:19 y el cableado de 13:12-13:49. Ese doc llama ademas a ROSA *"pesos +
  glue, no una op nueva"*: cierto para los pesos, **falso** para la memoria (seccion 4).

## 3. Puntos de hook (verificados en `ce8caa6`)

ROSA es head-paralelo (`RosaHead` envuelve `model.head`), asi que engancha en el hidden final, no
por capa. Ojo: `rwkv7.cpp:190` (channel mix) seria el hook de una variante *por capa*, y el
modulo tal como esta no lo usa.

| Que | Donde |
| Hidden de entrada | `src/models/rwkv7.cpp:200` (`build_norm` -> `result_norm` en `:202`): el `hidden_states` de `forward_train` |
| Mezcla de logits | `src/models/rwkv7.cpp:205` (`build_lora_mm(model.output, ...)`): sumar `gate * rosa_logits`; `res->t_logits` en `:208` |
| 4 tensores nuevos | `src/models/rwkv7.cpp:58` (junto a `output`); enum en `src/llama-arch.cpp` (~`:630`, precedente `LLM_TENSOR_TIME_MIX_*`) y `LLM_TENSOR_MAP` en `:827` |
| Softmax con mascara | `ggml/include/ggml.h:1813` `ggml_soft_max_ext` — ya existe, cuantizado en todos los backends |
| Memoria (costo real) | `src/llama-memory-recurrent.h:109` `std::vector<mem_cell> cells` — seccion 4 |
| Formato de estado | `src/llama-context.cpp:4215`/`:4223`; `src/llama-memory-recurrent.cpp:766`/`:847`/`:897` |

**Correccion al gap doc, item (3):** la mezcla del gate **no va en el sampler**
(`src/llama-sampler.cpp:382`). El gate es `sigmoid(gate(h))`, funcion pura del hidden: va en el
grafo (`:205`), si no se pierde cuantizacion y coherencia con el resto del modelo.

## 4. Lo que NO es glue: la memoria

- `ggml_rwkv_wkv7` (`ggml/include/ggml.h:2608`) es una op **monolitica**
  (`r,w,k,v,a,b,state -> out`): no hay `soft_max` sobre historial que copiar. A diferencia de
  `arwkv7`, aqui la "atencion" es modulacion de decay (`a`,`b`), no softmax sobre celdas: **no
  hay precedente en el camino RWKV de llama.cpp.**
- La cache recurrente (`llama-memory-recurrent.h:109`) guarda celdas de tamano fijo, no un KV
  que crece. Por eso `cell_count` es 1 (`docs/gguf-estado-python.md:52-53`).
- ROSA necesita memoria que **crece por token**. (i) En las celdas recurrentes cambia el formato
  de estado y rompe B1 y `rwkv_pipeline/state_bytes.py`. (ii) En el KV cache paginado
  (`src/llama-kv-cache.cpp`) no toca ese formato, pero es mas C++: hay que darle una vista.
- Antes de portar, arreglar el defecto del modulo: `rosa_module.py:78-80` crece el buffer con
  `torch.cat`, O(T) por token, contradiciendo su docstring (`:59-63`).

## 5. Experimento minimo siguiente (~1-2 semanas, cero C++)

El blocker son los pesos, asi que el proximo paso es nativo, no GGML:

1. Entrenar la rama ROSA sobre RWKV-7 con el camino que ya existe (`opd_train.py:223-224` ya
   cuenta sus params como entrenables). Sin tocar GGML.
2. Re-correr el A/B con pesos entrenados (`--tag rosa-on-entrenado --state-policy carry --rosa`)
   contra el `rosa-off` de la misma seed.
3. **Criterio:** el gain tiene que superar el ruido del brazo random. El piso a batir es
   **32.8%**, no 23.4%. Si `tests/test_rosa_equivalence.py` no separa los brazos, ROSA no
   aporta y el port se cancela para siempre.

Si el gain es real, recien ahi se abre el port: ~2-4 semanas de C++ (la op es glue, la memoria
no), en este orden: (1) fix del `torch.cat`, (2) 4 tensores + subgraph, (3) memoria en el KV
cache paginado, (4) extender `state_bytes.py`, (5) round-trip con ROSA encendida.

## 6. Que NO prueba este doc

- No midio velocidad de swap con ROSA (los 21 MB de B1 son sin ROSA), ni nada del camino nativo
  mas alla del A/B de 8x8 turnos.
- No verifico de donde salen los pesos del A/B `rosa-on` mas alla del aviso "random sin
  entrenar": no se abrio el `.pth` para confirmar que no trae rama ROSA.
