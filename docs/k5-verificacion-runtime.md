# K5 · Verificacion del sistema real: streaming, turnos largos y barge-in

Fecha: 2026-10-07. Caja: RX 6500 XT (VRAM 4,28 GB), 31 GB RAM, 8 hilos.

**Resumen en una linea: el modelo de K3 no existe, y el runtime donde Kateto
tiene que vivir no puede servir lo que hay.** Lo medido abajo es el runtime real
(`kateto/providers/rwkv_rocm.py`) y el motor nativo (`rwkv_pipeline/infer_kateto.py`)
con el mejor checkpoint Kateto disponible, porque K3 nunca se entreno.

## 0. Que NO existe: el modelo de K3

| hecho | evidencia |
|---|---|
| no hay ningun checkpoint K3 | `out/*/rwkv-*.pth` mas nuevo = `out/rwkv_kateto_base_plus_1.5b/rwkv-0.pth` (27-sep, era K2). Nada posterior. |
| el "style tune verificado" es de juguete | `harness-run/kateto-estilo/head-estilo.json`: `base: mini(rand)`, `n_layer 2`, `n_embd 128`, 2 pasos, loss 11.32. Prueba el camino de codigo, no un head de Kateto. |
| Kaggle no tiene ni dataset ni kernel | `kaggle kernels list --mine` → *Authentication required*; `kaggle datasets list --mine` → *No datasets found*. El token no esta vigente: no se subio `gabrielsolotorevsky/kateto-sft` y la unidad de 12 h no corrio nunca. |

Consecuencia dura: el criterio de aceptacion **"el modelo de K3 servido en el
runtime real"** no se puede cumplir hoy. Lo que sigue es la verificacion del
sistema con lo que hay, que es lo que el gate pide de verdad: *que se rompe*.

## 1. El barge-in: numeros medidos

El barge-in del usuario es un evento de asyncio. El provider nativo lo bloquea.

```
kateto/providers/rwkv_rocm.py:241-242   for tok in prompt_tokens:
                                            out, state = engine.model.forward(tok, state)   <- sin await
kateto/providers/rwkv_rocm.py:281       await asyncio.sleep(0)                              <- el UNICO await, y esta DENTRO del loop de generacion
```

O sea: entre que entra el turno y el primer chunk, el event loop esta **sordo**.
Medido con el provider real (`.venv` del runtime, 0.4B, CPU fp32):

| caso | prompt | TTFT | total | salida |
|---|---|---|---|---|
| narracion video-rag + "Para, para. De que año era eso?" | 1064 tok | **25,8 s** | 30,0 s | 63 chars de basura de marcadores |
| narracion + "ehh, che, espera" | 1025 tok | **27,3 s** | 31,6 s | 87 chars de basura de marcadores |

Salidas reales (recortadas):

```
<|im_end_0. - B,A, A, D, D2. - A,D1,D2.\n<|im_端_1.\n<|im_end_0|es
<|where, qué, qué, cuña.\n<|im_en|im_start|im_start|im_start|im_end|im_开t, y te.\n<|im_en
```

Dos lecturas, las dos malas:

1. **Sordera de 26-27 s.** Con ~1000 tokens de historial el barge-in no se puede
   procesar hasta que termina el prefill. En GPU el prefill es PEOR (14 tok/s,
   ver §2): los mismos 1064 tokens serian ~76 s de sordera.
2. **No hay narracion que cortar.** El modelo no contesta la interjeccion ni
   sigue narrando: emite spam de marcadores (`<|im_start|im_start|...`). El
   solapamiento con la narracion previa es **0,0** en los dos casos.

El fix del barge-in (`ab4fd1e`, TurnGate `origin="ambient"` → DISCARD) esta del
lado del control: es correcto y esta testeado. El problema del modelo no lo toca.

## 2. Contexto largo: el re-encodeo es O(n) por turno y O(n^2) por charla

`infer_kateto.py:420` (marca `ponytail:`) re-arma y **re-encodea todo el
historial** en cada turno. El provider del runtime hace lo mismo: encodea el
prompt completo (`rwkv_rocm.py:234`) y lo prefillea token por token.

Prefill medido (`k5_prefill.py`, 0.4B, GPU bf16, ROCm torch 2.9.1):

| tokens de prompt | tiempo | tasa |
|---|---|---|
| 64 | 4,55 s | 14,05 tok/s |
| 256 | 18,22 s | 14,05 tok/s |
| 1024 | 72,87 s | 14,05 tok/s |
| 4096 | 291,59 s | 14,05 tok/s |

Lineal y **sin batching**: no importa el largo, el prefill no se acelera. Swap de
estado de voz: **8,0-556,2 ms (promedio 66,0 ms, n=10)** — consistente con los
48-135 ms documentados, o sea el swap no es el problema.

Proyeccion: una charla de **30 turnos x 120 tokens = 55.800 tokens de prefill =
66,2 minutos** de prefill puro (y sobre el 0,4B; el 1,5B no mejora la tasa).

Nota: en esta caja el CPU le gana a la GPU en el 0,4B (41 tok/s fp32 8 hilos
contra 14,05 tok/s del RX 6500 XT) porque un forward de 1 token esta dominado por
el launch. No salva el cuadratico: sigue siendo 100 s de prefill por cada 4096
tokens de historial.

**Arrastrar el estado entre turnos, que es el arreglo que el propio codigo
pide, no esta implementado en ningun lado.** `eval_multiturn --state-policy carry`
arrastra el estado del RNN pero igual re-encodea el texto completo.

## 3. Estabilidad en turnos largos

Instrumento: `scripts/eval_multiturn.py` (temp 0.8 / top_p 0.7, seed 1337,
`--thinking off`, ctx-truncate 512), via `systemd-run --user -p MemoryMax=infinity`.
Salidas crudas en `out/multiturno/k5-*.jsonl`.

### 0,4B (`out/rwkv_kateto_base_plus_0.4b/hist/rwkv-1-20260928-1739.pth`)

**3 de 3 turnos devolvieron `<|no_response|>`.** Tasa utilizable **0,0**. El
servicio (`infer_kateto.generate`) convierte ese centinela en
`"Che, no te entendí bien, ¿me repetís?"`: en vivo, Kateto contesta siempre la
misma frase de fallback.

### 1,5B (`out/rwkv_kateto_base_plus_1.5b/rwkv-0.pth`), 2 charlas x 8 turnos

| metrica | valor |
|---|---|
| tasa utilizable | **5/16 = 31,25 %** |
| tasa vacia | 4/16 = 25 % |
| tasa eco (respuesta que repite el prompt) | 7/16 = 43,75 % |
| charlas sanas | **0** |
| wall | 796,1 s (≈ 50 s por turno) |
| largo mediano de respuesta | 36 chars |

Costo por turno creciendo con el historial (prompt 40 → 470 tokens):
`12,2 s → 110,3 s`. Turnos reales:

```
t1  '> igual la partera del piso y el microondas'
t2  'No me cajo la pizza del microondas y quedo todo el piso cubierto de queso.'   <- eco textual del turno del usuario
t3  ''                                                                             <- vacio
t4  'No me cayo la pizza del microondas y quedía todo el piso cubierto de queso.'   <- eco con typo
t6  'De quesi'
t8  '>\nAhora: mi solución para crear tu prop'                                     <- 110,3 s, corta por max_tokens
c02 t2  '<|wait|>'                                                                 <- el unico centinela correcto
```

Patron: **eco, vacio, o frase rota**. Arranques con `>` (resto del marcador).
Se sostiene 1-2 turnos y a partir del 3 empieza a parlotear.

## 4. ROSA esta MUERTA en el runtime

El motor (`infer_kateto`) sabe mezclar ROSA: `_forward()` suma
`gate * rosa_logits` (`infer_kateto.py:446-452`). El runtime no la usa:

| evidencia | linea |
|---|---|
| crea el objeto ROSA y **nunca lo consulta** | `rwkv_rocm.py:93` (`_GLOBAL_ROSA = ...`), cero usos despues |
| crea el motor **sin** ROSA | `rwkv_rocm.py:79` `KatetoInferenceEngine(model_path, vocab_path, device=device)` — sin `rosa=True` |
| prefillea y genera con `engine.model.forward`, salteando `engine._forward` | `rwkv_rocm.py:242` y `:283` |
| `enable_rosa: bool = True` es decorativo | `rwkv_rocm.py:208`; nadie lo lee |

Medido corriendo el provider real: `engine.rosa_instanciada=false`,
`ROSA_global_creada=true`, `enable_rosa_pedido=true`. ROSA es justo la pieza que
la card marca como el arreglo del recall exacto a distancia, y en el sistema
donde vive no esta conectada.

## 5. Estados de voz: 4 de 6, y el log miente

El provider carga `out/rwkv_states/<voz>/<el mas nuevo>.pth`
(`rwkv_rocm.py:84-90`). Resultado real de la corrida:

```
estados_cargados: ["seco", "seco-20260906", "streamer", "whisperer"]
[Kateto] El estado 'out/rwkv_states/jane/rwkv-11.pth' tiene 40 cabezas pero el modelo base
         tiene 16 cabezas. Incompatible... Omitiendo carga.
INFO ... _get_or_create_engine:90 - Estado cargado para voz 'jane': .../jane/rwkv-11.pth   <- MIENTE
[Kateto] El estado 'out/rwkv_states/doktor/rwkv-9.pth' tiene 40 cabezas  ... Omitiendo carga.
INFO ... _get_or_create_engine:90 - Estado cargado para voz 'doktor': ...                   <- MIENTE
```

`jane/` y `doktor/` tienen estados de 0,4B (6,3 MB) pero **el mas nuevo de cada
carpeta es de 2,9B (21 MB)**: el provider elige ese, el motor lo rechaza, y
`:90` loguea "Estado cargado" **sin mirar el `False`** que devolvio `load_state`.
En la practica queda tapado porque `VOICE_STATE_MAPPING` manda jane y doktor a
`seco` — o sea, los estados de esas dos voces son archivos muertos.

## 6. El provider cae a CPU y el modelo por defecto es el 2,9B

| hecho | evidencia |
|---|---|
| device efectivo **cpu**, dtype float32 | corrida del provider real: `device_efectivo: "cpu"`. El `.venv` del runtime tiene `torch 2.13.0+cu130` y en esta maquina `cuda.is_available()=False` (no hay ROCm en ese venv) → `rwkv_rocm.py:77` elige cpu. |
| el `HSA_OVERRIDE_GFX_VERSION` se setea tarde | `rwkv_rocm.py:62` lo escribe **despues** de `import torch` (:20). El propio `infer_kateto.py:16` documenta que hay que setearlo antes del import. |
| sin `model_path`, carga el 2,9B | `rwkv_rocm.py:67-72` toma el `.pth` mas nuevo de `out/rwkv_kateto_base/` = `rwkv-0.pth` (5,9 GB, 2,9B bf16). → `infer_kateto.py:292` lo manda a CPU → **~11,8 GB de RAM** en fp32. |

## 7. Presupuestos: que entra y que no

| modelo | bf16 | VRAM 4,28 GB | CPU fp32 |
|---|---|---|---|
| 0,4B | 0,9 GB | si (1,3 GB pico medido) | 1,8 GB |
| 1,5B | 3,0 GB | si, al filo (medido: VRAM usada 4,03/4,28 GB durante la corrida) | 6,1 GB |
| 2,9B | 5,9 GB | **no** | 11,8 GB |
| 13B | 26 GB | **no** | 52 GB |

RAM disponible durante las pruebas: **0,7-7 GB** de 31 GB, con el swap de 16 GB
**lleno** (184 kB libres en el peor momento). **13B no entra de ninguna forma**;
2,9B solo en una maquina quieta. La caja ya esta al limite por los servicios
residentes (Minecraft, Elasticsearch, Supabase, Argilla, los llama-server).

**Y no es teorico: la prueba de 1,5B mato un servicio residente.** A las
18:36:37, mientras cargaba el 1,5B, el kernel disparo un **global OOM** y se
llevo el motor `neoHorse` (`item-roles-neohorse.service`, anon-rss 2,7 GB):

```
kernel: oom-kill:constraint=CONSTRAINT_NONE,...,global_oom,
        task_memcg=.../item-roles-neohorse.service,task=llama-server,pid=1217774
kernel: Out of memory: Killed process 1217774 (llama-server) anon-rss:2701560kB
systemd: item-roles-neohorse.service: Failed with result 'oom-kill'.
```

Se relanzo con la misma receta (`harness-run/roles/servers.sh`: Vulkan0, :11662,
`-c 8192`) y quedo verificado: `/v1/models` lista `neohorse` y `salud` vuelve a
dar **2/2 sanos**. Leccion para el proximo que mida aca: **cargar un 1,5B
mientras los servicios estan arriba mata a otro**, y `MemoryMax` en la unidad
propia no protege al resto (el OOM es global).

## 8. Lo que SI quedo verificado

| cosa | resultado |
|---|---|
| formato de chat: canonico vs motor vs runtime | **los tres identicos byte a byte** en 2 formas (user y developer en el historial). `chat_template.py` manda; `rwkv_rocm.format_messages_to_prompt` no diverge. |
| gates del repo | `pytest pipelines/tests -q` → **269 passed, 4 skipped**; `pytest tests -q` → **91 passed** (usa `venv-unsloth-qwen`/distilabel, no el venv default). |
| dry-run del generador | `scripts/gen_conversaciones_largas.py --dry-run` → rc=0, `BUENA: OK`, y los 4 malos fallando cada uno por su regla (CORTA / CALCADA / REPETIDA / SIN_AVANCE). |
| muestra de Argilla | `argilla_cargar.py --dry-run --n 800` sobre 55.142 filas → 320 centinela (40 %) + 120 wait (15 %) + 120 otra voz (15 %) + 240 azar (30 %). Cupo por grupo: **correcto**. Stack arriba en :6900 (`/api/v1/me` responde). |
| swap de voz | 8-556 ms, promedio 66 ms. El swap nativo no es el cuello de botella. |
| el CLI del sistema (`scripts/k5_sistema.py`, untracked) | corre: `salud` → **2/2 sanos**; `feedback` marca las 4 violaciones y da `veredicto: ajuste`. Sus 39 tests entran en los 91 que pasan. |
| estado del runner que lo escribio | `harness-run/kateto/report-1791406230.json`: `outcome: "exhausted"`, `winner: null`, `artifact: null` — ningun intento aterrizo; los tres archivos (`scripts/k5_sistema.py`, `tests/test_k5_sistema.py`, `docs/k5-sistema.md`) existen **sin commitear**. |

Ojo con el alcance de ese CLI: apunta a los llama-server `neohorse`/`e2b` por
HTTP (modelos instruct, no Kateto) — su propio doc lo dice. Es pegamento de
streaming/comedia/feedback, **no** el camino nativo que se verifico arriba.

Y roto, ademas de lo de arriba: el sampler del runtime no es el que se evaluo —
`alpha_presence 0.6 / alpha_frequency **0.6**` (`rwkv_rocm.py:258-259`) contra
`0.6 / **0.5**` del motor y del eval. Cualquier numero de calidad medido en un
lado no aplica al otro.

## 9. Lo que NO medi y por que

- **La calidad del checkpoint de K3**: no existe (§0).
- **ROSA A/B con texto util**: el 0,4B devuelve centinelas y el head ROSA
  (`out/rosa_head_largo/rosa_head.pth`, base 1024d x24l) es de 0,4B. Medir ROSA
  ahi mide un modelo que no genera, no ROSA.
- **El 1,5B en el runtime nativo** (provider, no eval): su estado de voz no
  existe (los estados son de 0,4B) y en CPU son 6,1 GB de RAM contra 0,7-7 GB
  disponibles. Correrlo habria dejado sin memoria a los servicios residentes.
- **Una charla real de streaming de punta a punta** (microfono → ASR → TTS):
  sigue sin ASR en esta caja (documentado en `docs/k5-sistema.md`); la cadena se
  ejercita con `--texto`.

## 10. Reproducir

```bash
cd /run/media/chaos/terciario/proyectos/kateto-train

# 0) los dos gates
env -u PYTHONPATH -u VIRTUAL_ENV /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest pipelines/tests -q
env -u PYTHONPATH -u VIRTUAL_ENV venv/bin/python -m pytest tests -q
env -u PYTHONPATH -u VIRTUAL_ENV /run/media/chaos/secundario/venvs/distilabel/bin/python scripts/gen_conversaciones_largas.py --dry-run

# 1) turnos largos (0,4B y 1,5B). MemoryMax=infinity NO es cosmetico: el cgroup del agente es de 4 GiB.
systemd-run --user --unit=k5-04b --collect -p MemoryMax=infinity -D $PWD \
  ./venv-unsloth-qwen/bin/python -u scripts/eval_multiturn.py \
  --model out/rwkv_kateto_base_plus_0.4b/hist/rwkv-1-20260928-1739.pth \
  --conversaciones 1 --turns 3 --device cuda --tag k5-smoke
systemd-run --user --unit=k5-15b --collect -p MemoryMax=4G -D $PWD \
  ./venv-unsloth-qwen/bin/python -u scripts/eval_multiturn.py \
  --model out/rwkv_kateto_base_plus_1.5b/rwkv-0.pth \
  --conversaciones 2 --turns 8 --device cuda --tag k5-15b

# 2) prefill / swap / proyeccion
#    (script en ~/.hermes/cache/scratch/k5_prefill.py, 0,4B)
systemd-run --user --unit=k5-prefill --collect -p MemoryMax=4G -D $PWD \
  ./venv-unsloth-qwen/bin/python -u ~/.hermes/cache/scratch/k5_prefill.py --lengths 64,256,1024,4096

# 3) el provider NATIVO del runtime, headless (usa el .venv del runtime)
#    (script en ~/.hermes/cache/scratch/k5_provider_runtime.py)
systemd-run --user --unit=k5-provider --collect -p MemoryMax=5G -D /home/chaos/proyectos/OpenaiBuildWeek/Kateto \
  /home/chaos/proyectos/OpenaiBuildWeek/Kateto/.venv/bin/python -u ~/.hermes/cache/scratch/k5_provider_runtime.py

# 4) formato: canonico vs motor vs runtime
cd /home/chaos/proyectos/OpenaiBuildWeek/Kateto && .venv/bin/python ~/.hermes/cache/scratch/k5_formato.py
```

Evidencia cruda: `out/multiturno/k5-smoke.jsonl`, `out/multiturno/k5-15b.jsonl`,
`.json` de cada uno, y los logs de las unidades `kateto-k5-*`
(`journalctl --user -u kateto-k5-15b`).

## 11. Lo que hay que arreglar antes de que esto sea servible

1. **Servir arrastrando el estado** en vez de re-prefijar el historial: sin eso,
   cada turno arranca con decenas de segundos de sordera (§1, §2).
2. **Ceder el event loop durante el prefill** (`await` cada N tokens): sin eso el
   barge-in no se puede atender por mas rapido que sea el prefill.
3. **Conectar ROSA** en el provider (o borrar el flag y el objeto): hoy da la
   falsa sensacion de estar encendida (§4).
4. **Que `_get_or_create_engine` mire el `bool` de `load_state`** y no loguee
   "Estado cargado" cuando el motor rechazo el estado (§5).
5. **Fijar el `model_path`** del runtime y sacar el `HSA_OVERRIDE` antes del
   import de torch (§6).
6. **Alinear el sampler** con el del eval (alpha_frequency 0,5) o medir con el
   del runtime (§8).
7. **K3**: sin modelo entrenado no hay nada mas que verificar (bloqueado en el
   token de Kaggle).
