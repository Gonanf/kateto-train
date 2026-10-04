# Eval multiturno (`scripts/eval_multiturn.py`)

Mide la degradacion por turnos del checkpoint. No arregla nada: es el instrumento
para que un arreglo no sea a ojo.

El problema que mide: *"el 1.5B responde bien con 1 mensaje pero se degrada
rapido si hay mas de uno"*. Hoy la unica evaluacion de comportamiento es de un
turno, asi que ese fenomeno no estaba medido.

## Correr

`systemd-run ... MemoryMax=infinity` es **obligatorio**, no cosmetico: los procesos
de background de este host corren en un cgroup capado a 4 GiB y un checkpoint
cargado ahi muere con exit 137.

```bash
systemd-run --user --scope -p MemoryMax=infinity --unit=kateto-mt \
  ./venv-unsloth-qwen/bin/python -u scripts/eval_multiturn.py \
  --model out/rwkv_kateto_base_plus_0.4b/hist/rwkv-1-20260928-1739.pth \
  --conversaciones 1 --turns 3 --device cuda --tag smoke
```

El barrido completo (8x8) y el 1.5B los corre el orquestador.

## Parametros

| param | default | que hace |
|---|---|---|
| `--model` | (obligatorio) | checkpoint `.pth` merged (base + fine-tune). El estado NO se carga aparte: la voz es informativo |
| `--turns` | 8 | turnos por conversacion |
| `--conversaciones` | todas | cuantas conversaciones del set usar |
| `--prompt-set` | `data/multiturno_set.jsonl` | JSONL de conversaciones |
| `--out-dir` | `out/multiturno` | salida |
| `--tag` | `mt` | prefijo de `<tag>.jsonl` y `<tag>.json` |
| `--device` | `cuda` | |
| `--thinking` | `off` | `off` descarta el bloque `<think>` y mide la respuesta visible |
| `--temperature` | `0.80` | ver abajo |
| `--top-p` | `0.70` | |
| `--seed` | `1337` | `PROBE_SEED` del repo |
| `--state-policy` | `carry` | `carry` arrastra el estado entre turnos; `reset` lo pone en cero antes de cada turno |
| `--ctx-truncate` | `512` | cuantos tokens de historial se le pasan |
| `--max-tokens` | `64` | tope de tokens generados por turno |
| `--selfcheck` | | asserts de las metricas puras, sin GPU |

### Sampling: por que 0.80 / 0.70 y no greedy

**Se eligio temp 0.80 / top_p 0.70**, el mismo contrato que ya usan
`scripts/eval_generation.py` y `probe_format_generation()`. Es lo que permite
comparar estos numeros con cualquier otra medicion del repo.

Greedy (`--temperature 0`) esta **implementado pero NO es el default porque en el
0.4B se midio patologico**: sobre el primer prompt del set, el argmax elige el
token `<` (logit 6.625, primer byte de todos los marcadores) y ahi se cae en
`<|no me cajajajajajajaja`. O sea, greedy no mide el modelo, mide el artefacto de
argmax. Con sampling la corrida es reproducible igual por `--seed`.

Trampa de `sample_logits`: **`sample_logits(temperature=0)` NO es greedy.** Con
`temperature=0` el `if temperature > 0` no divide pero igual cae en
`torch.multinomial(probs)`, o sea muestreo aleatorio sobre la distribucion
completa. El script pide `argmax` a mano en esa rama.

## Los dos modos: como se distingue "estado" de "contexto"

En **los dos** modos se pasa el mismo historial de texto. La unica variable es el
estado:

- `--state-policy carry` (default): el estado del RNN se arrastra de turno a turno.
- `--state-policy reset`: el estado vuelve a cero antes de cada turno.

- Si con `reset` **no** se degrada y con `carry` **si** -> la causa esta en el
  estado que se arrastra.
- Si degrada en los dos -> la causa es el contexto que se pasa de ctx512.
- `--ctx-truncate 256`: si con 256 no se degrada, la causa es el largo del prompt.

## Salida

`out/multiturno/<tag>.jsonl` — una linea por turno:

| columna | que es |
|---|---|
| `conversacion` | id del set |
| `turno` | 1..N |
| `len_chars` | largo de la respuesta |
| `n_tokens_gen` | tokens generados (incluye lo que se corto despues) |
| `prompt_tokens` | tokens del prompt DESPUES de `--ctx-truncate` |
| `rep8` | fraccion de 8-gramas de palabra repetidos. `1 - unicos/totales` |
| `vacia` | respuesta vacia, whitespace, o centinela de silencio |
| `eco_del_prompt` | >70% de las palabras de la respuesta ya estaban en el prompt |
| `degen` | `rep8>0.5` o `vacia` o `eco`. Define "utilizable" |
| `state_hash` | sha256 del estado completo al TERMINAR el turno |
| `state_cambio` | el hash difiere del turno anterior |
| `state_norm` | norma L2 del estado (float32, CPU) |
| `stop_reason` | `eos` / `id_end` / `marker` / `repetition` / `max_tokens` |
| `policy`, `ctx_truncate` | config de la corrida, por linea |
| `t_s` | segundos del turno |
| `texto` | primeros 400 chars de la respuesta |

`out/multiturno/<tag>.json` — resumen por modelo:

| clave | que es |
|---|---|
| `turno_max_sano` | primer turno en el que **alguna** conversacion se degrada. `null` = ninguna se degrado en N turnos |
| `turno_max_sano_por_conversacion` | lo mismo por conversacion |
| `conversaciones_sanas` | cuantas no se degradaron nunca |
| `rep8_por_turno` | `rep8` medio del turno, **solo sobre utilizables**. `null` = ese turno no dejo ningun utilizable |
| `utilizables_por_turno` | denominador de esa media |
| `state_norm_por_turno` | norma media del estado por turno |
| `len_median`, `len_iqr` | largo de respuesta sobre utilizables |
| `utilizables` / `total` | cuantas respuestas de las N*8 no cayeron en vacia/eco/degenerada |
| `tasa_*` | todas sobre `utilizables`, nunca sobre el total |

## Como se lee la tabla

1. **`turno_max_sano` es la fila de cabecera.** Dice en que turno deja de poder
   hablar el modelo. Si es 1, no hay vida multiturna: hay que mirar el
   `stop_reason` (`vacia` con `<|no_response|>` = el modelo esta emitiendo el
   centinela de silencio, no contestando).
2. **`state_norm_por_turno` es la columna de la hipotesis del bug.** Si con
   ctx512 el estado crece turno a turno sin techo, ahi esta. Es monotonamente
   creciente y termina rompiendo.
3. **`rep8_por_turno` subiendo** = la respuesta se esta volviendo mas repetitiva.
   Junto con `state_norm` subiendo, es el cuadro clasico de estado que se arrastra.
4. **`null` en `rep8_por_turno` no es "0 repeticiones"**: es que no quedo ninguna
   respuesta utilizable en ese turno. Para verlo, mirar `utilizables_por_turno` y
   las columnas `vacia`/`eco` del jsonl.

Ojo con elordnung: `state_norm` **sube siempre** un poco porque el estado acumula
contexto real; lo que busca es que suba *desproporcionado* frente a
`state_norm_por_turno[1]`.

## Dos decisiones que conviene conocer antes de comparar

**El vacio no se rellena.** `infer_kateto.generate()` sustituye una respuesta vacia
por `"Che, no te entendí bien, ¿me repetís?"`. Este script NO lo hace: el
instrumento tiene que poder distinguir "no dijo nada" de "dijo algo". Por eso
`vacia` es observable en vez de estar tapado por un texto de fallback.

**La respuesta vacia no entra al historial.** Si un turno sale vacio, no se
agrega como turno cerrado del prompt siguiente. Es lo que hace `chat_loop()` del
repo, y evita que un centinela de silencio contamine el contexto de los turnos
que siguen.

## La trampa de triton, y por que NO aplica aca

La trampa del repo (chunk `K=8`, padear el forward a multiplo de 64, leer
`logits[0, len(ids)-1]` y nunca `logits[0, -1]`) es de la via **vectorizada**:
`RWKV-PEFT/rwkvt/rwkv7/model.py` `forward_normal` con `WKV=triton`, la que usa
`rwkv_pipeline/opd_rollout.py` (`_pad_len` / `_forward_padded` / `_next_logits`).

Este script usa la via de **inferencia** de `rwkv_pipeline/infer_kateto.py`, cuyo
`RWKV_RNN.forward(token, state) -> logits` es un RNN **por token**: no tiene
dimension `T`, no tiene chunks, y sus logits son un vector 1-D. No hay
`logits[0, len(ids)-1]` que leer. Por eso no se aplica padding. Se deja
documentado para que nadie lo porte por error a la otra via.

## Prompt set

`data/multiturno_set.jsonl`: 8 conversaciones x 8 turnos, congelado, en el dominio
del dataset (platica rioplatense de casa, boliche, laburo, economically, mascota,
deporte, reflexiones). Solo mensajes del usuario: las respuestas las genera el
modelo. `id`, `tema`, `turnos`.

## Selfcheck

```bash
./venv-unsloth-qwen/bin/python scripts/eval_multiturn.py --selfcheck
```

Asserts sobre `rep8`, `eco_del_prompt`, `es_vacia` y `limpiar_think`. Corre sin
GPU ni modelo. Es el chequeo minimo que falla si la logica de repeticion/eco se
rompe.

## Que NO hace

No arregla el modelo, no reentrena, no toca `models/`, `RWKV-PEFT/` ni
`venv-unsloth-qwen/`.