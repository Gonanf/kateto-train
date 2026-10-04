# ROSA medido con el instrumento multiturno (R5)

Pregunta: la rama ROSA del head, encendida y apagada, cambia la salud de la
respuesta en un dialogo de 8 turnos? Con numeros, no con intuicion.

## Veredicto

**No se puede decidir todavia.** Los pesos de ROSA estan RANDOM y sin entrenar
(lo advierte el propio instrumento), asi que el delta medido es el ruido de una
capa no entrenada perturbando los logits, no el recall de ROSA; y con n=64 de un
solo seed la diferencia (21/64 vs 15/64, z=1.18) no se separa del ruido.

Lo que si quedo medido y es accionable: **el modo de falla no es la repeticion,
es la no-respuesta.** `rep8` da 0.000 en 64/64 filas de los dos brazos; lo que
degenera es que el modelo emita el centinela `<|no_response|>` o cadena vacia.

## Configuracion de la corrida

Los dos `.jsonl` son la misma corrida con un solo flag distinto. Todo lo demas
identico, verificado fila por fila:

| | rosa-on | rosa-off |
|---|---|---|
| checkpoint | `out/rwkv_kateto_base_plus_0.4b/hist/rwkv-1-20260928-1739.pth` | el mismo |
| `--state-policy` | `carry` | `carry` |
| `--ctx-truncate` | 512 | 512 |
| `--turns` | 8 | 8 |
| conversaciones | 8 | 8 |
| sampling | `temp=0.8 top_p=0.7 seed=1337` | el mismo |
| `--rosa` | si | no |
| filas | **64** | **64** |
| wall | 863.6 s | 898.9 s |

Se uso el **0.4B**: el 0.4B es el checkpoint que produce la linea base del
instrumento, asi que el brazo OFF tiene contra que compararse. El 1.5B no se
intento (la GPU tiene 4 GB).

## Gate: misma cantidad de filas

```
$ wc -l out/multiturno/rosa-on.jsonl out/multiturno/rosa-off.jsonl
  64 out/multiturno/rosa-on.jsonl
  64 out/multiturno/rosa-off.jsonl
128 total
```

64 filas cada uno = 8 conversaciones x 8 turnos. La grilla `(conversacion, turno)`
es identica y en el mismo orden en los dos archivos, y las claves por fila son
las mismas. El flag quedo registrado por fila: `rosa=true` en las 64 de
`rosa-on`, `rosa=false` en las 64 de `rosa-off`.

## Tabla por turno

`util` = filas no degeneradas de las 8 de ese turno. `rep8` = promedio sobre las
utilizables de ese turno (misma convencion que el instrumento: `rep8` no tiene
8-gramas en textos cortos, promedia solo lo que hay). `eco` = filas con eco del
prompt, contadas sobre las **8** filas del turno (incluye las degeneradas, que es
donde se Concentra el eco).

| turno | util ON | util OFF | rep8 ON | rep8 OFF | eco ON | eco OFF |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 3 | 0.000 | 0.000 | 0/8 | 0/8 |
| 2 | 0 | 1 | n/a | 0.000 | 4/8 | 1/8 |
| 3 | 2 | 2 | 0.000 | 0.000 | 2/8 | 2/8 |
| 4 | 3 | 2 | 0.000 | 0.000 | 1/8 | 2/8 |
| 5 | 3 | 2 | 0.000 | 0.000 | 3/8 | 2/8 |
| 6 | 3 | 3 | 0.000 | 0.000 | 3/8 | 1/8 |
| 7 | 2 | 2 | 0.000 | 0.000 | 3/8 | 2/8 |
| 8 | 4 | 0 | 0.000 | n/a | 1/8 | 4/8 |
| **total** | **21/64** | **15/64** | 0.000 | 0.000 | 17/64 | 14/64 |

Totales: tasa de utilizables **0.328** (ON) vs **0.234** (OFF). Longitud mediana de
la respuesta: 23.5 chars (ON) vs 15.0 (OFF).

`n/a` = el turno no tiene ninguna fila utilizable, asi que no hay promedio.

## Que modo de falla es, en realidad

| | rosa-off | rosa-on |
|---|---:|---:|
| texto real | 29 | 38 |
| `<\|no_response\|>` | 14 | 3 |
| `<\|wait\|>` | 4 | 4 |
| cadena vacia | 17 | 19 |
| filas con `rep8 > 0` | **0/64** | **0/64** |

El delta de 6 filas utilizables viene casi entero del centinela
`<|no_response|>`: 14 -> 3. No viene de repeticion, porque no hay repeticion en
ningun turno de ninguna de las dos corridas.

## Por que igual no se puede decidir

1. **Los pesos no existen.** `infer_kateto.py:332` lo dice al construir la rama:
   random, sin entrenar. Cualquier delta de un A/B asi es el efecto de una
   proyeccion aleatoria que suma `gate * rosa_logits` sobre los logits, no una
   capacidad de recuperacion. El numero "+9 puntos de utilizables" no se puede
   atribuir a ROSA: se puede atribuir a "ROSA cambia los logits", que es otra
   cosa.
2. **n = 64, un seed.** 21/64 vs 15/64: diferencia +0.094, error estandar 0.080,
   z = 1.18. Wilson 95%: ON [0.226, 0.450], OFF [0.147, 0.351]. Los intervalos se
   pisan. Con este n, dos manos que difieren en 6 filas son indistinguibles del
   azar de muestreo.
3. **Los brazos se divergen en la longitud del prompt.** No es un bug del setup,
   es una consecuencia: una respuesta vacia no se agrega al historial, y como el
   brazo ON responde mas, su prompt crece. Prompt del turno 8 (mediana sobre las
   8 conversaciones): 103 tokens en OFF, 382 en ON. Ninguno llega a tocar
   `ctx_truncate=512`, asi que no hay truncamiento, pero los dos brazos ya no
   estan leyendo la misma cantidad de contexto en los turnos altos. Cualquier
   comparacion turno a turno desde el 4 en adelante mezcla "ROSA" con "mas
   historial".

## El brazo OFF es la linea base, bit a bit

`rosa-off` da 15/64 = 0.2344, identico a la corrida externa
`/home/chaos/harness-run/kateto-multiturno/out/04b-carry.jsonl` (mismo
checkpoint, mismo `policy`, mismo `ctx_truncate`, mismo sampling, mismo seed).
Como `engine._forward` con `rosa=None` devuelve `engine.model.forward` sin tocar
nada (`infer_kateto.py:412-414`), eso era esperable, y que salga igual confirma
que el brazo OFF no toco el modelo.

## Un bug del instrumento, encontrado y corregido

`construir_resumen` reportaba `tasa_vacia` y `tasa_eco` dividiendo por
`len(utilizables)`, pero `vacia` y `eco` fuerzan `degen`, o sea que esas filas
estan **fuera** de `utilizables`: el numerador contaba filas que el denominador
descartaba. Daba tasas imposibles, medidas: `tasa_vacia` 1.2381 (ON) y 2.3333
(OFF), `tasa_eco` 0.8095 y 0.9333.

Corregido en `scripts/eval_multiturn.py`: esas dos tasas ahora van sobre el
total. `tasa_utilizable` y `tasa_rep8_alto` no se tocaron (ya usaban
denominadores coherentes). Los dos `.json` de este documento fueron recalculados
desde los `.jsonl` **sin re-correr el modelo**; los `.jsonl` no se tocaron
(md5 `57ff75aef594af45042b727a54ec3a17` ON, `a1e1d2379b646008277db73b26c05b36`
OFF). Valores nuevos: `tasa_vacia` 0.4062 ON / 0.5469 OFF, `tasa_eco` 0.2656 ON /
0.2188 OFF.

La tabla por turno de arriba no dependia de esas tasas: usa
`utilizables_por_turno` y `rep8_por_turno`, que ya tenian denominadores
coherentes.

## Correccion al contexto del brief

El brief daba por hecho que ROSA era codigo huerfano. No lo es, y esto cambia la
pregunta: la rama ya estaba enganchada y solo faltaba medirla.

- `rwkv_pipeline/infer_kateto.py:322-325` instancia `RosaAssociativeMemory`
  cuando se pasa `rosa=True`; `_forward` (`:407-419`) mezcla
  `gate * rosa_logits` sobre los logits del head.
- `rwkv_pipeline/opd_train.py:223-224` usa `attach_rosa` para el entrenamiento.
- `tests/test_rosa_wiring.py`, `tests/test_rosa_causal.py` y
  `tests/test_rosa_equivalence.py` la cubren.
- `scripts/eval_multiturn.py` ya tenia el flag `--rosa` y la politica de arrastre
  de `engine.rosa_memory`.

Ademas el A/B ya estaba corrido cuando empezo este item (los `.jsonl` son de las
14:32 y 14:47, secuenciales, con 15 min de diferencia, no en paralelo). No se
volvio a correr: con seed fijo los dos brazos son reproducibles, y el brazo OFF
ya demostro reproducir la linea base externa al bit.

## Donde queda la medicion de ROSA, entonces

Falta lo unico que podria contestar la pregunta: pesos de ROSA entrenados. Sin
ellos cualquier A/B mide ruido de capa. Con ellos, el mismo comando de este
documento alcanza:
`--rosa` con el head entrenado, y el contraste con estos dos `.jsonl` es directo.
