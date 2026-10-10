# K5 · Sistema donde vive Kateto

Streaming, comedia generativa y verificacion punta a punta. Todo vive en un solo
archivo: `scripts/k5_sistema.py`, con cuatro subcomandos.

```
python scripts/k5_sistema.py salud
python scripts/k5_sistema.py comedia --voz jane "me trabé con el deploy"
python scripts/k5_sistema.py stream --turnos 1 --texto "hola" --sonar
python scripts/k5_sistema.py feedback --prompt "hola" --respuesta "..." --juez
```

## Que NO esta aca

K5 es pegamento, no una reimplementacion. Cada pieza que hace el trabajo real ya
existe en el repo y K5 la llama:

| pieza | dueño |
|---|---|
| inferencia HTTP | `scripts/eval_http_backend.py::generate_http` |
| TTS | `rwkv_pipeline/piper_tts.py::PiperTTSProvider` |
| filtro pre-TTS | `kateto/safety/content_filter.py::ContentFilter` |
| registro | `prompts/prompt-compartido.md` (VERBATIM) |
| reglas de estilo | `prompts/registro-kateto.md` (R*), `prompts/comedia.md` (M*) |

Stdlib puro, a proposito: los scripts de produccion del repo ya corren con el
python del sistema. `k5_sistema.py` no importa nada fuera de la stdlib mas los
tres modulos de arriba.

## Los dos servidores

| nombre | url | checkpoint | n_ctx |
|---|---|---|---|
| `neohorse` | 127.0.0.1:11662 | NeoHorse-1-4B.Q4_K_M.gguf | 8192 |
| `e2b` | 127.0.0.1:11661 | gemma-4-E2B_q4_0-it.gguf | 8192 |

`salud` reporta las TRES capas por separado porque mienten distinto: `/health`
dice que el proceso responde, `/v1/models` que el modelo esta cargado, y la
completion que la inferencia anda de verdad. Un servidor caido es un dato, no un
error: el comando nunca levanta.

```
$ python scripts/k5_sistema.py salud
servidor   url                        health  modelo   infer       s
neohorse   http://127.0.0.1:11662     True    True     True     0.28
e2b        http://127.0.0.1:11661     True    True     True     0.13

2/2 sanos
```

## Prompt: por que crudo y no `/v1/chat/completions`

Kateto se entrena con los markers de RWKV, no con el chat template de NeoHorse ni
el de Gemma. Un prompt crudo con el registro arriba es el unico formato que las
dos partes comparten. Si se cambia el modelo servido, `prompt_turno()` es el unico
punto a tocar.

Las tres voces (`jane`, `doktor`, `conquest`) salen de `prompts/tics-por-voz.md`;
cada una tiene su `max_len` porque una linea seca y una toma larga no se cortan
en el mismo token.

## El turno mudo (medido 2026-10-07)

**2 de 6 llamadas con el mismo prompt y la misma seed devolvieron 1 token y
nada** (`n_tokens: 1`, `stopping_word: ''` — EOS inmediato, no un stop). En vivo
es un turno mudo.

La seed no lo arregla: con `cache_prompt: False` mas batching continuo, el
muestreo de llama.cpp no es bit-determinista — la misma seed dio 6 salidas
distintas en 6 llamadas. Por eso `hablar()` reintenta **cambiando la seed**
(`seed + reintentos`), hasta 3 intentos.

Despues del guard: **0/12 mudos**, con 2 reintentos absorbidos.

El guard va en `hablar()` y no en los callers porque `comedia`, `stream` y
`feedback --desde-comedia` pasan todos por ahi. Si tras el ultimo intento sigue
mudo devuelve `""` con `reintentos` contado — un stream no puede colgarse.

## Streaming: captura → ASR → inferencia → emision

El filtro corre **por oracion y antes de sintetizar**, que es el contrato que
`ContentFilter` deja escrito en su docstring. HARD devuelve texto vacio, asi que
la frase entera se calla; SOFT emite `[BLEEP]` y sigue.

```
$ python scripts/k5_sistema.py stream --turnos 1 --texto "me trabé con el deploy"
[k5] turno 1/1
  transcripcion (inyectada): me trabé con el deploy tres horas
  kateto: no me trabé con el deploy tres horas
  wav out/k5-stream/turno_000/turno_000.wav (7.92s)
```

Verificado con `volumedetect`: `mean_volume: -14.5 dB`, `max_volume: -0.0 dB` —
audio real, no silencio. 22050 Hz mono 16 bit (piper).

### Lo que falta en esta caja

- **ASR**: no hay whisper ni whisper-cli. `--asr-cmd` deja el binario como
  decision del operador; `--texto` ejercita toda la cadena sin microfono.
- **Microfono**: el ffmpeg de `/home/chaos/.hermes/tools/ffmpeg-9.0.1` no trae
  `alsa` (`Unknown input format: 'alsa'`). `capturar()` necesita un ffmpeg con
  ALSA, o `--dispositivo` apuntando a algo que ffmpeg sepa leer.

Ambas son del entorno, no del codigo: la cadena se ejercita igual con `--texto`.

## Verificacion

`feedback` tiene dos capas y el veredicto sale de la primera.

**Deterministica** — regex contra las reglas que ya existen en
`prompts/registro-kateto.md` (R*) y `prompts/comedia.md` (M*). Sin red, siempre
corre. `grave` tira el veredicto a `ajuste`; `aviso` solo baja el puntaje.

```
$ python scripts/k5_sistema.py feedback --prompt hola \
    --respuesta '<think>...</think> Hola! ¿En que puedo ayudarte? (risa)'
{
  "deterministico": {
    "veredicto": "ajuste",
    "violaciones": [
      {"regla": "R4/M6",   "cita": "(risa)",  "gravedad": "grave"},
      {"regla": "R8",      "cita": "!",       "gravedad": "aviso"},
      {"regla": "R21",     "cita": "En que puedo ayudar", "gravedad": "grave"},
      {"regla": "think_leak", "cita": "<think>", "gravedad": "grave"}
    ],
    "puntajes": {"limites": 0, "limpieza": 2}
  },
  "acciones": ["R4/M6: sacar '(risa)'", "..."],
  "veredicto": "ajuste"
}
```

Exit code: `0` si `ok`, `1` si `ajuste`. Sirve para gatear en shell.

### El juez va forzado por gramatica

Pedir JSON en el prompt no funcionaba: neohorse gastaba los 180 tokens en un
`<think>` y no emitia nada parseable (0/1). Ahora la salida pasa por
`json_schema` de llama.cpp, asi que el shape sale garantizado.

Dos detalles que hubo que medir:

- `notas` lleva `maxItems`/`maxLength` porque sin tope el modelo encadenaba
  espacios hasta agotar el presupuesto y devolvia JSON truncado (4/6 → 6/6).
- `JUEZ_MAX_TOKENS = 600` es un **techo, no un costo**: el modelo corta en EOS
  igual, pero con 180 truncaba. Bajarlo reintroduce el bug.

Resultado: **6/6 JSON parseables en los dos servidores**, ~2 s por juicio.

El juez es best-effort por diseno: si el servidor cae, `juez.ok` es `false` con
el error, y el veredicto **sigue saliendo del deterministico**. Un juez caido
nunca es un "approved".

## Hallazgo de fondo: los modelos servidos no son Kateto

La mecanica anda punta a punta, pero los checkpoints que sirven hoy no son Kateto:

```
<think>We need to respond as Kateto.        ← think_leak
Hola! ¿En que puedo ayudarte hoy? (risa)    ← M9/R19 + R8 + R21 + R4/M6
Tu turno termina con una pregunta o un      ← el modelo HABLA del turno
hablar\natencion al cliente\nservicio       ← word salad
```

Es el mismo sintoma en dos lugares, por la misma causa: `/completion` crudo sobre
instruct models que esperan su propio chat template. La solucion es servir el
checkpoint de Kateto (o usar el template nativo de cada served model), no
parchear el prompt.

Lo que este doc deja es el circuito cerrado: los fallos **se detectan solos**.
`feedback` es el que dice "esto no es Kateto", y con `stream` + `--sonar` se
escucha.

## Tests

```
venv/bin/python -m pytest tests/test_k5_sistema.py -q   # 39
venv/bin/python -m pytest tests/ -q                     # 91
```

Los 39 tests son stdlib + pytest, sin red. Cubren lo que no se ve leyendo: que
cada regla matchee su ejemplo **y no matchee un turno limpio** (el filtro no
puede ser tan ruidoso que marque todo), el guard de turno mudo, el schema del
juez, y que los ids de las reglas sigan apuntando a las que existen.

Verificados por mutacion: revertir R21 al orden viejo rompe
`test_r21_acepta_los_dos_ordenes`; desactivar el guard de mudo rompe los dos
tests de reintento.

Lo que necesita servidor no va en los tests — se verifica corriendo el CLI,
como esta arriba.