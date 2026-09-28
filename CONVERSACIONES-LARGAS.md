# Conversaciones de contexto largo (1M) con conectores

## Qué cambió en `scripts/build_long_docs.py`

Dos cambios, como pedía el brief. Filtro de contenido (workmatch/sherut/podcast/cliente)
intacto, solo en `load_fuente_b`, ni se tocó.

**1. Agrupar hasta el target, mezclando elementos.**
Los elementos (sesión hermes / video youtube / transcript videorag / video youtube_txt)
ya no salen 1:1 como documentos: se juntan de a varios, de fuentes y temas distintos,
hasta `--target-tokens`. Defaults nuevos: `--target-tokens 1000000`,
`--max-tokens 1000000`, `--min-tokens 1000`. Se eliminó `split_max` y el truncado de
fuente C: si un elemento solo supera el target sale en su propio documento, nunca se
corta un transcript a la mitad. Orden de llenado: elementos más grandes primero
(menos saltos por documento).

**2. Conector generado por el modelo en cada salto.**
Entre elemento y elemento se inserta un turno corto con el formato chat del script
(`<|im_start|>seco\n...<|im_end|>`), voz Kateto, 1–2 frases que cierran el tema viejo
y abren el nuevo, sin contenido técnico inventado. Gateway y key: igual que
`build_qa_dataset.py` (flag/env `--openai-base-url`, default `http://127.0.0.1:3001/v1`;
key explícita > `OPENAI_API_KEY` > `FREELLMAPI_KEY` > `opencode.json`; llamada con
stdlib `urllib`, sin deps nuevas). Cache en JSON (default
`data/largo/conectores_cache.json`, clave = par de elementos + voz): los hits no
cuentan contra el cupo y la corrida completa es reanudable. Filtro anti-meta
(`_RX_META`): si el modelo explica la tarea en vez de dar el turno, se descarta y el
salto queda sin conector (contado como omitido). Flags: `--no-conectores`,
`--max-documentos N`, `--max-conectores N` (solo llamadas reales al gateway).

Bugs encontrados y corregidos durante la verificación: `emitir()` no reseteaba el
acumulador y duplicaba contenido entre documentos; `--max-documentos` dejaba escapar
un documento de cola. Ambos fixed, ids únicos verificados.

## Corrida chica (verificación obligatoria)

Comando (17 s, entra cómoda en el harness):

```
python3 scripts/build_long_docs.py --stats --max-documentos 1 --max-conectores 4 \
  --openai-model gemini-3.5-flash-lite \
  --out /tmp/largo_gw/train.jsonl --out-youtube /tmp/largo_gw/yt.jsonl \
  --stats-json /tmp/largo_gw/est.json --conectores-cache /tmp/largo_gw/cache.json
```

Resultado: **1 documento, 984.279 tokens est. (24 elementos, 3 fuentes), 4 conectores
generados** (`gateway=4 cache=0 omitidos=19`: el cupo corta el resto, como diseñado).
Nota: `--openai-model gemini-3.5-flash-lite` en vez del `auto` default — el router
`auto` devolvía meta-respuestas en inglés ("First, I need to write a short transition
turn…", 3 de 4 descartadas por `_RX_META`); el flash-lite obedece el formato a 0,8 s
por llamada. Mismo gateway, misma key; solo cambia el modelo, igual que
`build_qa_dataset.py` permite vía `--openai-model`.

Conectores generados (textual, los 4):

1. `Bueno, eso ya pasó, saquemosle el cierre a esto. Che, agarremos el chat y arranquemos con este otro correo que nos mandaron.`
2. `Listo, dejemos el tier list ahí que ya nos fuimos al pasto. Che, escuchen una cosa, ahora sí, arranquemos bien este antro con los pibes.`
3. `Dejalo ahí, che. Che, pasemos a otra cosa, escuchen esto.`
4. `Bueno, dejemos eso de lado y retomemos. Che, enderecemos la nave y arrancamos de una vez con esto.`

Suenan a transición de stream (voseo, seco, sin asistente). El 3 es el más genérico;
aceptable para costura, no enseña contenido.

`--no-conectores` (sin gateway, solo CPU): corre limpio, 4 documentos / 3.788.860
tokens / max 994.776 en la pasada completa local.

`--stats` sobre la misma corrida chica (re-ejecución; los conectores salen de cache):

```
documentos largos: 1 (elementos<1000tok descartados: 217, colas: 0)
total tokens est: 984440 | max doc: 984440 | >=target(1e+06): 0 | mixtos: 1
conectores: gateway=4 cache=4 omitidos=15
histograma (tokens est):
  0-8000: 0 | 8000-16000: 0 | 16000-32000: 0 | 32000-64000: 0 | 64000-128000: 0
  128000-256000: 0 | 256000-512000: 0 | 512000-1048576: 1 | >=1048576: 0
```

(Ojo: `--stats` re-deriva de las fuentes, no lee el jsonl; es determinista dado el
mismo cache. En la re-ejecución el cupo de 4 generó los 4 pares siguientes:
`gateway=4 cache=4`. Ningún doc llega exacto a 1M porque el empaquetado para antes
de exceder el target sin cortar elementos; el más largo queda en ~985–995k.)

## Elementos recuperados al bajar `--min-tokens`

Misma corrida con `--min-tokens 8000`: **581 descartados**, 3 docs, 2.586.568 tokens
(581 coincide exacto con la corrida del orquestador). Con `--min-tokens 1000`:
**217 descartados**, 4 docs, 3.788.860 tokens.
**Recuperados: 364 elementos, +1.202.292 tokens est.** El filtro de contenido no cambió: fuente B sigue excluyendo 44 archivos
(workmatch/sherut/cliente/vacíos), listados en `estadisticas.json`.

## Anti-429: delay + backoff + modo paciente (2026-09-27)

Tres flags nuevos, nada más (empaquetado y filtro intactos):

- `--conectores-delay-s` (default **2.0**): pausa antes de cada llamada real al
  gateway. Los hits de cache no duermen.
- `--conectores-retries` (default **4**): intentos ante 429/5xx con backoff
  creciente **5s, 15s, 45s, ...** (`5*3**i`). Si el 429 trae `Retry-After` se
  respeta.
- `--conectores-espera-max-s` (default **120**): tope por espera del backoff.
  Agotados los reintentos, el salto queda sin conector (como antes) y se cuenta
  aparte: `conectores.omitidos_por_429`, distinto de `conectores.omitidos`
  (cupo + filtro anti-meta). Nuevo contador `conectores.respuestas_429` = total
  de 429/5xx vistos.
- `gateway_chat` nunca levanta: cualquier corte del gateway devuelve `("", ...)`
  y la corrida **sigue sin conector hasta el resumen final** (confirmado en la
  corrida de abajo: 100% de fallos y el doc igual se escribió, exit 0).
- Bug encontrado al verificar: `--max-conectores` contaba solo éxitos
  (`generados`), así que con todo-429 el cupo nunca cortaba y cada salto
  quemaba el backoff completo. Ahora cuenta **llamadas reales** (`llamadas`:
  éxitos + 429; cache no cuenta), como ya decía su `--help`. Sin esto, la
  corrida de 30 con backoff paciente no termina nunca bajo 429 duro.
- Reanudable: lo generado queda en cache; los `omitidos_por_429` NO se cachean,
  así que una re-ejecución del mismo comando reintenta exactamente los que
  faltan.

## Verificación real contra el gateway (2026-09-27, 14:30 -03)

Comando (salida a /tmp para no pisar `data/largo/`):

```
python3 scripts/build_long_docs.py --stats --max-documentos 1 --max-conectores 3 \
  --openai-model gemini-3.5-flash-lite --conectores-delay-s 2 \
  --conectores-espera-max-s 10 \
  --out /tmp/largo_429/train2.jsonl --out-youtube /tmp/largo_429/yt2.jsonl \
  --stats-json /tmp/largo_429/est2.json --conectores-cache /tmp/largo_429/cache2.json
```

Resultado (`conectores: gateway=0 cache=0 omitidos=20 omitidos_por_429=3
respuestas_429=12`): **0 generados / 0 cache / 3 omitidos por 429 / 20
omitidos por cupo** (el doc de 1 doc tiene 23 saltos; el cupo de 3 llamadas
reales corta el resto, como diseñado). **429 vistos: 12** (3 conectores × 4
intentos). El log muestra el ciclo paciente funcionando:

```
conector: gateway 429, reintento 1/4 en 10s
...
conector: reintentos agotados (HTTP Error 429: Too Many Requests), sigo sin conector
```

y al final `escritos 1 docs -> ... + est2.json` — no aborta.

**Por qué no hay "corrida limpia" ni `--max-conectores 30` esta vez** (dato
medido, no suposición): el gateway responde al modelo pedido con

```json
{"error":{"message":"All models exhausted: 1 route checked (1 rate-limited or on cooldown). Add more API keys or wait for rate limits to reset. Soonest reset ~3h.","type":"rate_limit_exceeded","code":"rate_limit_exceeded","retryAtMs":1790539477135}}
```

con `Retry-After: 9814` (~2,7 h). Es **cupo upstream agotado**, no pacing:
ningún delay lo limpia hasta el reset. El patrón del orquestador (19 OK y
después todo 429) cuadra con lo mismo. El cap se verificó igual: el log dice
`en 10s` en vez de 9814s (cap `--conectores-espera-max-s 10` usado solo para
que la prueba termine; el default sigue en 120). Lógica del backoff además
chequeada sin gateway: sin header → 5s; `Retry-After: 3` → 3s;
secuencia con cap 120 → 5/15/45/120.

**Delay que queda: 2.0 s** (el default; con el cupo sano las llamadas salían a
~0,8 s, así que 2 s da margen sin alargar de más: ~485 pares × ~3 s ≈ 25 min
estimados de conectores en la corrida completa).

## Comando para la corrida completa (orquestador, proceso largo fuera del harness)

```
nohup python3 -u scripts/build_long_docs.py --stats \
  --openai-model gemini-3.5-flash-lite \
  --conectores-delay-s 2 \
  > data/largo/full_run.log 2>&1 &
```

Defaults: target 1M, min 1000, sin topes, retries 4, espera-max 120 s. Escribe
a `data/largo/` (train + youtube + `estadisticas.json` + `conectores_cache.json`).
Lanzar **después del reset del cupo** (el gateway decía ~3 h desde las 14:21
-03 del 2026-09-27; si al arrancar el log muestra `reintento ... en 120s` en
cada conector, el cupo sigue agotado: matar y esperar). Requiere la key de
freellmapi (ya está en `~/.config/opencode/opencode.json`, el script la levanta
solo). GPU: nada, CPU + gateway. No pisa nada existente hasta el final
(escribe los jsonl de una sola vez al terminar). Si se corta a la mitad,
re-lanzar el mismo comando: lo cacheado sale como `hits_cache` y solo se
reintentan los `omitidos_por_429`.

## Qué quedó sin verificar

- La corrida completa con conectores (la lanza el orquestador; lo chico sí verificado).
- Calidad de ~485 conectores a escala: el filtro `_RX_META` puede necesitar más
  patrones si el modelo cambia de estilo de meta-respuesta (los omitidos quedan
  visibles en `estadisticas.json` → `conectores.omitidos`).
- `out-youtube` ahora solo incluye documentos 100% `youtube_txt`; con mezcla total
  sale 0 archivos (pasó en las corridas: todos los docs son mixtos). Si ese split
  sigue haciendo falta, definir criterio (p. ej. mayoría youtube) — por ahora se
  deja como está.
- `youtube_txt:rayos419` sigue entrando por fuente C (nunca tuvo filtro por nombre;
  no se tocó por prohibición de cambiar filtros).
