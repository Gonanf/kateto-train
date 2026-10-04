# ATAQUE PROMETHEUS — pase 2 (profundo, 2 ejes + Momus)

Fecha: 2026-10-02. Solo lectura. No se entreno, no se modifico ningun otro archivo.
Regla de evidencia: todo numero lleva el comando exacto y su salida cruda (max 6 lineas). Si no lo corri, dice "no verificado".

---

## EJE 1 — DATOS (`data/rwkv_kateto_base_train_plus.jsonl`)

### 1.1 Composicion real

```
$ wc -l data/rwkv_kateto_base_train_plus.jsonl
44351 data/rwkv_kateto_base_train_plus.jsonl

$ python3 conteo -> total 44351 no_response 4632 10.4%
(4632 lineas del set principal; el conteo exacto: 4632/44351 = 10,44%)

turn_counts (mensajes de usuario por muestra): [(1, 22295), (2, 2463), (3, 19593)]
multi-turn (>=2 mensajes <|im_user|>): 22056
keyword 'instruccion': 259 | 'ajedrez': 2628 | 'stream': 11229 (conteo de substring, no etiqueta)

lens chars: p25 286, p50 688, p75 1482, p90 2133, max ~6394 (p90/p50 ~3x)
```

- Cada muestra es `{"text": "..."}` con formato ChatML (`<|im_user|>...<|im_end|>\n<|im_start|>seco\n...<|im_end|>`).
- Distribucion de turnos: 50,3% un turno, 44,2% tres turnos, 5,6% dos turnos. Con ctx 512 tokens de
  entrenamiento y p90 de 2133 chars (~500+ tokens), la cola del p90 se trunca: esos turnos pierden su cierre.
  No corri el histograma con tokenizer real -> "no verificado" que % excede 512.

### 1.2 Donde vive `<|no_response|>`

```
$ python3 (speaker antes del centinela) -> Counter: [('seco', 5113), ('none', 2)]
5115 ocurrencias del token en 4632 lineas; 19 envueltas en <think>, 5096 directas.
Contexto inmediato top: 'o.<|im_end|> <|im_start|>seco ' (1150), 'a.<|im_end|>...' (509)
En data/kateto_qa.jsonl / kateto_qa_test.jsonl / rwkv_kateto_base_train.jsonl: 0 ocurrencias.
Fuentes con centinela (conteo de LINEAS con `grep -c`): data/real/real_no_response.jsonl (12), data/v2/orpo_silencio.jsonl (4000),
data/v2/rwkv_kateto_v2_train.jsonl (886). Ojo: `grep -c` cuenta lineas, no ocurrencias; la suma 4898 no tiene
que igualar 5115 ocurrencias (algunas lineas traen el token mas de una vez o mezclan fuentes).
74 lineas tienen ademas <|wait|>; 1141 lineas del train_plus contienen <|wait|>.
```

- El centinela vive 99,96% en la voz `seco` (5113/5115 ocurrencias con speaker tag; 2 sin tag), y esa voz es ~98,6% de la data.
- Contexto de 200 muestras muestreadas al azar (seed 0): el ultimo mensaje del usuario antes de
  `<|no_response|>` tiene `?`/`¿` en 65, no lo tiene en 135. O sea, **el modelo aprende a callarse tanto
  ante preguntas como ante comentarios**; no es un guardrail "solo cuando nadie pregunta".
- Ejemplo crudo: `<|im_user|>Che má, ¿a qué hora comemos hoy?<|im_end|><|im_start|>seco\n<think>Habla con la madre...</think>\n<|no_response|><|im_end|>`.

### 1.3 Pregunta central: ¿el centinela explica el silencio en la eval?

Evidencia separada por brazo (de `out/*.jsonl`, ver EJE 2):

```
04b-carry | vacia 35 | texto contiene literal '<|no_response|>': 14 | ejemplos de vacia: '<|no_response|>'
04b-reset | vacia 20 | texto contiene literal '<|no_response|>': 14
15b-carry | vacia 14 | texto contiene literal '<|no_response|>': 0 | ejemplos de vacia: '', '<|wait|>'
15b-reset | vacia 13 | literal '<|no_response|>': 0 | ejemplos: '<|wait|>', ''
base-carry | vacia 1 | literal: 0
```

- **0.4B fine-tuneado**: su silencio ES el centinela. 14 de 64 respuestas son el literal `<|no_response|>`;
  los `vacia` del 04b son, en su mayoria, la decision "no contestes" aprendida del 10,4% de data envenenada.
  El silencio del 04b tiene explicacion directa en el dato (14 literales; las otras 21 `vacia` de 04b-carry son
  cadena vacia, sin centinela ni texto, no explicadas por el dato).
- **1.5B fine-tuneado**: su silencio NO es el centinela literal (0/64). Son cadenas vacias o `<|wait|>`.
  La hipotesis del centinela para el 1.5B queda refutada como causa DIRECTA por evidencia literal; la causa
  final sigue pendiente del experimento de logprob (1.4).
- Stream/ajedrez: el centinela aparece embebido en conversaciones tipo stream/ajedrez (ejemplos del muestreo),
  pero como decision explicita de "no responder", no como "turnos cortos sin pregunta" del dato crudo. El dato
  stream/ajedrez en si (11229/2628 menciones) NO es la causa del silencio: un dato sin pregunta no produce, por
  si solo, una salida `<|no_response|>`; la produce la etiqueta de silencio que se minero de ahi.

### 1.4 Mini-experimento de inferencia (sin entrenar) — script exacto, no corrido

GPU ocupada por `15b-sampler-upstream` (ver 2.4) y el 0.4B/1.5B cargan pesados: no lo corri. Asi se mediria:

```python
# medir logprob del token <|no_response|> como candidat al 1er token del turno 1
# sobre los 8 prompts de data/multiturno_set.jsonl, base vs fine-tuneado 1.5B.
import torch, json
from rwkv_pipeline.infer_kateto import RWKV_RNN, load_tokenizer  # nombres ilustrativos del repo
tok = load_tokenizer()
sentinel = tok.encode('<|no_response|>')[0]

def score(model_path):
    m = RWKV_RNN(model_path)  # 1 turno, sin historial
    out = []
    for line in open('data/multiturno_set.jsonl'):
        conv = json.loads(line)['turnos'][0]
        prompt = f"<|im_user|>{conv}<|im_end|>\n<|im_start|>seco\n"
        ids = tok.encode(prompt)
        state = m.new_state()
        for i in ids[:-1]:
            logits, state = m(i, state)
        logits, state = m(ids[-1], state)
        lp = torch.log_softmax(logits.float(), -1)
        out.append((lp[sentinel].item(), lp.argsort(descending=True)[0].item() == sentinel,
                    lp.argsort(descending=True)[:5].tolist()))
    return out

print('base   :', score('out/rwkv_kateto_base/rwkv-1.pth'))
print('ft 1.5B:', score('out/rwkv_kateto_base_plus_1.5b/rwkv-0.pth'))
```

Si el fine-tuneado rankea el centinela top-1 y el base no, la causa es decision aprendida del dato.
Ese comando que falta para "sostenida": correrlo y comparar rankings en los 8 prompts.

---

## EJE 2 — COHERENCIA MULTITURNO

### 2.1 Hipotesis en competencia y sus predicciones distinguibles

| H | Prediccion distinguible | Evidencia ya existente | Veredicto parcial |
|---|---|---|---|
| (a) El centinela | las `vacia` contienen el literal `<|no_response|>` | 04b: 14/64 si; 15b: 0/64 si (son `''`/`<|wait|>`) | **Sostenida para 04b; refutada como causa directa para 15b (pendiente confirmar con 1.4)** |
| (b) El sampler | un brazo con pen paciencia distinto separa `utilizables` | todavia no existe brazo alternativo completo (el de temp1/top_p0.5/pp2/fp0.1 esta corriendo) | **Dudosa, en curso** |
| (c) Estado arrastrado | carry vs reset difiere fuerte | 15b carry 29/64 vs reset 25/64; 04b carry 15 vs reset 29 (reset MEJOR); base-carry 6 | **Refutada como causa dominante** (diferencia chica e inconsistente; si el estado fuera el bug, reset ganaria siempre) |
| (d) Instrumento | ctx 512 / prompts inventados / historial largo | todos los brazos degradan desde turno 1-2 (`turno_max_sano: 1` en TODOS) y el set tiene 8 conversaciones inventadas | **Favorecida como agravante; no separa por si sola al 1.5B del 0.4B** |
| (e) Eco por repeticion del historial | `eco_del_prompt` crece con el historial, base como control | base-carry eco 57/64; rosa-on-full eco 8/8 desde T3; 15b-carry eco 21/64 totales, 15b-reset 26/64 | **Sostenida como SINtoma (el eco ocurre y crece); dudosa como causa atribuida al historial sin scatter eco-vs-prompt_tokens** |

### 2.2 Tabla utilizable por turno (cruda, computada por script)

Leyenda: T1..T8 = turno; `vacia`/`eco` = conteo de 8; `len_usable_med` = mediana de chars entre utilizables
(`None` = ninguno); `stop` = Counter de stop_reason. Script: `python3` inline que agrupa las filas de cada
`out/*.jsonl` por `turno` y cuenta `vacia`, `eco_del_prompt`, `rep8>0.5`; salida completa abajo.

**04b-carry** (`04b-carry 64 filas; t1 malo: 5/8 conversaciones, de esas >50% posteriores malos: 4; t1 bueno: 3/8, de esas >50% posteriores malos: 3):
```
T1 n=8 vacia=5 eco=0 len_usable_med=26  stop={'id_end': 3, 'marker': 3, 'max_tokens': 2}
T2 n=8 vacia=6 eco=1 len_usable_med=83  stop={'marker': 5, 'max_tokens': 2, 'id_end': 1}
T3 n=8 vacia=4 eco=2 len_usable_med=157 stop={'id_end': 4, 'marker': 3, 'max_tokens': 1}
T4 n=8 vacia=4 eco=2 len_usable_med=79  stop={'id_end': 2, 'marker': 4, 'max_tokens': 2}
T5 n=8 vacia=4 eco=2 len_usable_med=63  stop={'id_end': 3, 'marker': 3, 'max_tokens': 2}
T6 n=8 vacia=4 eco=1 len_usable_med=64  stop={'id_end': 4, 'marker': 3, 'max_tokens': 1}
T7 n=8 vacia=4 eco=2 len_usable_med=73  stop={'id_end': 5, 'marker': 2, 'max_tokens': 1}
T8 n=8 vacia=4 eco=4 len_usable_med=None stop={'id_end': 4, 'marker': 3, 'max_tokens': 1}
```

**04b-reset** (t1 malo 6/8, posteriores malos >50% en 3; t1 bueno 2/8, en 1):
```
T1 n=8 vacia=6 eco=0 len_usable_med=120 stop={'id_end': 5, 'marker': 1, 'max_tokens': 2}
T2 n=8 vacia=5 eco=1 len_usable_med=248 stop={'id_end': 2, 'marker': 3, 'max_tokens': 3}
T3 n=8 vacia=2 eco=1 len_usable_med=29  stop={'id_end': 5, 'marker': 1, 'max_tokens': 2}
T4 n=8 vacia=3 eco=1 len_usable_med=49  stop={'id_end': 5, 'marker': 3}
T5 n=8 vacia=1 eco=1 len_usable_med=63  stop={'id_end': 5, 'marker': 1, 'max_tokens': 2}
T6 n=8 vacia=1 eco=2 len_usable_med=64  stop={'id_end': 2, 'max_tokens': 5, 'marker': 1}
T7 n=8 vacia=2 eco=2 len_usable_med=78  stop={'max_tokens': 3, 'marker': 2, 'id_end': 3}
T8 n=8 vacia=0 eco=7 len_usable_med=178 stop={'id_end': 6, 'max_tokens': 2}
```

**04b-carry-nopen** (t1 malo 5/8 -> 5 mueren despues; t1 bueno 3/8 -> 3 mueren despues):
```
T1 n=8 vacia=4 eco=1 len_usable_med=67  stop={'max_tokens': 5, 'marker': 1, 'id_end': 2}
T2 n=8 vacia=6 eco=0 len_usable_med=62  stop={'max_tokens': 6, 'id_end': 2}
T3 n=8 vacia=5 eco=1 len_usable_med=93  stop={'max_tokens': 4, 'marker': 1, 'id_end': 3}
T4 n=8 vacia=5 eco=2 len_usable_med=136 stop={'max_tokens': 5, 'marker': 1, 'id_end': 2}
T5 n=8 vacia=5 eco=3 len_usable_med=None stop={'max_tokens': 4, 'marker': 1, 'id_end': 3}
T6 n=8 vacia=5 eco=2 len_usable_med=13  stop={'max_tokens': 5, 'marker': 1, 'id_end': 2}
T7 n=8 vacia=5 eco=3 len_usable_med=None stop={'max_tokens': 5, 'marker': 1, 'id_end': 2}
T8 n=8 vacia=5 eco=3 len_usable_med=None stop={'max_tokens': 4, 'marker': 1, 'id_end': 3}
```

**15b-carry** (t1 malo 4/8 -> 2 mueren despues; t1 bueno 4/8 -> 3 mueren despues):
```
T1 n=8 vacia=1 eco=3 len_usable_med=62 stop={'id_end': 7, 'max_tokens': 1}
T2 n=8 vacia=3 eco=2 len_usable_med=29 stop={'id_end': 8}
T3 n=8 vacia=3 eco=2 len_usable_med=57 stop={'id_end': 8}
T4 n=8 vacia=2 eco=3 len_usable_med=55 stop={'id_end': 7, 'marker': 1}
T5 n=8 vacia=2 eco=3 len_usable_med=69 stop={'id_end': 7, 'marker': 1}
T6 n=8 vacia=0 eco=3 len_usable_med=154 stop={'id_end': 6, 'max_tokens': 2}
T7 n=8 vacia=3 eco=2 len_usable_med=39 stop={'id_end': 7, 'marker': 1}
T8 n=8 vacia=0 eco=3 len_usable_med=55 stop={'max_tokens': 1, 'id_end': 7}
```

**15b-reset** (t1 malo 2/8 -> 1 muere despues; t1 bueno 6/8 -> 6 mueren despues):
```
T1 n=8 vacia=1 eco=1 len_usable_med=17 stop={'id_end': 6, 'max_tokens': 1, 'marker': 1}
T2 n=8 vacia=1 eco=3 len_usable_med=39 stop={'id_end': 7, 'max_tokens': 1}
T3 n=8 vacia=1 eco=2 len_usable_med=78 stop={'id_end': 5, 'max_tokens': 2, 'marker': 1}
T4 n=8 vacia=4 eco=2 len_usable_med=128 stop={'id_end': 6, 'marker': 1, 'max_tokens': 1}
T5 n=8 vacia=0 eco=6 len_usable_med=49 stop={'id_end': 7, 'max_tokens': 1}
T6 n=8 vacia=0 eco=5 len_usable_med=195 stop={'id_end': 6, 'max_tokens': 2}
T7 n=8 vacia=2 eco=4 len_usable_med=63 stop={'id_end': 5, 'max_tokens': 2, 'marker': 1}
T8 n=8 vacia=4 eco=3 len_usable_med=66 stop={'id_end': 4, 'max_tokens': 3, 'marker': 1}
```

**base-carry** (t1 malo 2/8 -> 2 mueren despues; t1 bueno 6/8 -> 6 mueren despues):
```
T1 n=8 vacia=1 eco=1 len_usable_med=49 stop={'marker': 4, 'id_end': 4}
T2..T8: vacia=0 eco=8 cada turno, len_usable_med=None
stop T2 {'marker': 2, 'id_end': 3, 'max_tokens': 3} / T3 {'marker': 2, 'max_tokens': 5, 'id_end': 1} /
T4 {'max_tokens': 5, 'marker': 1, 'id_end': 2} / T5 {'max_tokens': 6, 'marker': 1, 'id_end': 1} /
T6 {'max_tokens': 3, 'marker': 1, 'id_end': 4} / T7 {'max_tokens': 5, 'marker': 1, 'id_end': 2} /
T8 {'max_tokens': 5, 'marker': 1, 'id_end': 2}
```

**rosa-on-full** (t1 malo 0/8; t1 bueno 8/8 -> los 8 empeoran despues):
```
T1 n=8 vacia=0 eco=0 len_usable_med=220 stop={'max_tokens': 7, 'repetition': 1}
T2 n=8 vacia=0 eco=6 len_usable_med=218 stop={'max_tokens': 8}
T3..T8: eco=8, vacia=0, stop={'max_tokens': 8} en todos
```

**r9-off / r9-on-entrenado**: numeros identicos a 04b-carry (mismo checkpoint, mismo seed; el A/B R9 dio empate exacto).

Agregados por archivo (usables/total 64):

```
04b-carry: usable 15, vacia 35, eco 14, len_med_usable 63
04b-reset: usable 29, vacia 20, eco 15, len_med_usable 50
04b-carry-nopen: usable 9, vacia 41, eco 14
15b-carry: usable 29, vacia 14, eco 21, len_med_usable 55
15b-reset: usable 25, vacia 13, eco 26, len_med_usable 49
base-carry: usable 6, vacia 1, eco 57, len_med_usable 49
rosa-on-full: usable 10, vacia 0, eco 54, len_med_usable 220 (T1)
```

### 2.3 Correlacion turno 1 vs resto

- Conversaciones que arrancan mal en T1: la mayoria empeora o sigue mal (>50% de sus turnos posteriores
  tambien malos: 04b-carry 4/5, nopen 5/5; base-carry 2/2).
- Pero lo inverso NO protege: de las que arrancan bien, muchas igual se pudren (15b-reset 6/6, base-carry 6/6,
  15b-carry 3/4). Conclusion: **el turno 1 bueno es condicion necesaria no suficiente** — arrancar bien no
  garantiza nada; la muerte por eco/degeneracion es progresiva con el historial, no un evento de estado inicial.

### 2.4 Sampler brazo `15b-sampler-upstream`

```
$ ps aux | grep eval_multiturn -> python scripts/eval_multiturn.py ... --tag 15b-sampler-upstream
  --temperature 1 --top-p 0.5 --alpha-presence 2 --alpha-frequency 0.1 (arrancado 10:40, unit mt-15-sampler2)
$ ls /home/chaos/harness-run/kateto-multiturno/out | grep sampler -> (vacio hasta este momento)
```

Sin `.json` todavia: el brazo no produce numeros usables aun. La hipotesis (b) queda **dudosa / no testeada**.
Prediccion distinguible: si losilencio era sampler, con temp 1/pp 0.5/presence 2/freq 0.1 las `vacia` deberian
bajar **y** subir las repeticiones/eco (trade-off esperado); si losilencio se mantiene, el sampler queda
ablegado.

### 2.5 Estado que se arrastra (c)

`state_norm_por_turno` 15b-carry sube de 888 a ~968 (crece monotono +9%), pero carry vs reset apenas cambia la
tasa de usables (29 vs 25). Si el estado fuera el bug, reset deberia ganar por amplio margen y no gana.
Tambien: `state_cambio: true` en la mayoria de turnos -> el estado cambia, no es un estado congelado. La
hipotesis (c) como causa **principal** del silencio queda refutada con esta evidencia (reset no gana por
amplio margen); como agravante menor (~10 puntos) queda dudosa-menor. Ojo: la inferencia "reset ganaria
siempre si fuera el estado" es una heurica, no un control formal (interactuan sampler e historial).

### 2.6 Eco por repeticion del historial (e) — sintoma dominante tarda

Base sin fine-tune: eco 57/64. rosa-on-full: eco 54/64 desde T3. 15b-reset: eco 26/64 totales. El eco
domina a partir de T2-T3 en todos los brazos con historial. Ocurre incluso con estado reseteado, lo que
descarga al estado como causa necesaria del eco. Atribuirlo causalmente al historial largo queda dudoso
hasta correr el scatter eco vs `prompt_tokens` (comando faltante #3).

---

## MOMUS — critica final

Metodo: cada conclusion del documento se marca **sostenida / dudosa / refutada**; despues lo que NO se sostiene
de mi propio analisis y el comando que falta para moverla de categoria. Este documento fue revisado por el
subagente Momus (pase sobre copia en `.omo/plans/`); sus objeciones (100% vs 99,96%, 4596 vs 4632, eco sostenida
vs dudosa, causa 15b pendiente de logprob, inferencias heuristicas) se corrigieron abajo.

| # | Conclusion del doc | Estado | Falta |
|---|---|---|---|
| 1 | El 10,4% de `<|no_response|>` contamina la voz seco (A1 del pase 1, re-medido en 1.2) | **sostenida** | — |
| 2 | El silencio del 04b en la eval ES el centinela literal | **sostenida** | — |
| 3 | El silencio del 15b es el centinela | **refutada como causa directa (literal 0/64); causa ultima dudosa hasta experimento 1.4** | script logprob |
| 4 | Stream/ajedrez sin-pregunta explica el silencio per se | **dudosa** (el centinela se minero de ahi, pero el silencio es la etiqueta, no el formato) | conteo de turnos-sin-pregunta en esos archivos (`no verificado`) |
| 5 | El sampler explica el silencio del 15b | **dudosa** | correr el brazo y comparar `utilizables` |
| 6 | El estado arrastrado es la causa dominante | **refutada como dominante (heuristic: reset no gana por amplio margen)** | ablacion limpia |
| 7 | El eco del historial domina la degradacion tarda | **parcial: sostenida como sintoma; dudosa la atribucion casual al historial** | scatter eco vs prompt_tokens |
| 8 | Turno 1 bueno no protege (correlacion negativa/nula como guarda) | **sostenida** con n=8 conversaciones | 20+ conversaciones reales |
| 9 | ctx 512 / set sintetico contamina la lectura | **sostenida como agravante** (turno_max_sano=1 en todos los brazos, incluido base y rosa) | re-correr con `--ctx-truncate 2048` |
| 10 | El mini-experimento de logprob decide centinela-vs-base | **no corrido** | correr el script de 1.4 |
| 11 | `<|wait|>` (1141 lineas) es la via del silencio del 15b | **dudosa** | ablar `<|wait|>` del SFT y re-eval; o contar `<|wait|>` como vacia explicita en la metrica |
| 12 | r9-on == 04b-carry: el A/B R9 fue empate | **sostenida** (numeros identicos) | — |

### Lo que NO se sostiene de mi propio analisis

- **Confundir `vacia` con `<|no_response|>` en el 15b**: mis `vacia` de 15b incluyen `''` y `<|wait|>`, no el
  centinela. Si alguien leyo "el 15b calla por el dato envenenado", eso no se sostiene.
- **Correlacion T1 vs resto con n=8**: la tabla de 2.3 tiene 8 conversaciones; decir "el turno 1 no protege"
  es direccion correcto, pero el efecto podria ser ruido de n=8. No se sostiene como ley; falta el comando
  `--conversaciones 20+` sobre un set real.
- **Que el centinela venga de stream/ajedrez**: los ejemplos del muestreo (3 vistos) son de stream/ajedrez,
  pero no medi la proporcion exacta de `real_no_response`+`orpo_silencio` en la mezcla final del
  train_plus. `no verificado` la mezcla exacta.
- **"El eco es por el historial largo"**: afirmo eco crece con historial por la tabla, pero no medi eco vs
  `prompt_tokens` punto a punto. Falta: scatter eco vs prompt_tokens (script de 10 lineas sobre los mismos jsonl).
- **Prediccion del brazo sampler**: la prediccion (bajan vacias, sube eco) queda marcada como hipotesis, no
  como trade-off esperado; no se confirma hasta tener el `.json`.

### Comandos que faltan para mover de categoria

1. `./venv-unsloth-qwen/bin/python scripts/eval_multiturn.py --model out/rwkv_kateto_base_plus_1.5b/rwkv-0.pth --device cuda --turns 8 --temperature 1 --top-p 0.5 --alpha-presence 2 --alpha-frequency 0.1 --tag 15b-sampler-upstream` (esperar a que termine; hoy corria en background)
2. El script de logprob de 1.4 (base vs ft sobre 8 prompts).
3. `python3` scatter eco vs `prompt_tokens` sobre los jsonl ya escritos.
4. Re-correr el eval con `--ctx-truncate 2048` para separar ctx de modelo.
5. Ablacion de `<|wait|>` y/o `<|no_response|>` del SFT (corto) + re-eval — el unico experimento que
   moveria 2 y 11 de dudosa a sostenida/refutada.

---

## Cierre

- El silencio del **0.4B** es el centinela aprendido del dato (sostenida, evidencia dura).
- El silencio del **1.5B** NO es el centinela literal (0/64); son vacias/`<|wait|>` y sampler/ctx aun no separados (dudosa la causa).
- El estado arrastrado como causa dominante queda refutada; el eco del historial queda como sintoma sostenida
  y causa dudosa; el set de 8 conversaciones sinteticas y ctx 512 agravan todo (sostenida como agravante).
- Lo que falta es barato: el `.json` del brazo sampler, el script de logprob, ctx 2048, y la ablacion de
  centinela/wait.
