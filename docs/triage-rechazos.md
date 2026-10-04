# Triage de los rechazos del brazo v2 (R11)

Que rechazo es defecto del teacher, que es regla nuestra de mas y que es prompt nuestro.
**Este doc es diagnostico: no se toco ninguna regla, validador ni prompt.** Los cambios van en
un brief aparte.

- Datos (solo lectura): `pipelines/smoke/teacher-pool-v2/{rechazos,muestras}.jsonl`,
  `muestras.stats.json`, `comparacion-simulador.json`, del repo principal.
- Reporte mecanico, una entrada por rechazo: `pipelines/triage-rechazos.json`.
- Validadores usados para reproducir: `pipelines/kateto_gen.py::validar` +
  `pipelines/kateto_purge.py::evaluar`, sin modificar.

## Como se reconstruyeron los 12

Los 12 rechazos se recalcularon con los validadores del repo y **las listas de errores coinciden
una por una con las guardadas**, sin una sola diferencia. No hay ninguno que haya quedado sin
explicar: `sin_evidencia_suficiente` = 0.

Las citas del JSON no estan escritas a mano: se extraen con la misma regex que dispara cada
error y el generador verifica que cada una sea subcadena del texto guardado. No hay parafraseo
posible.

## Los numeros

12 rechazos, **17 instancias de error**, 8 motivos distintos. Un rechazo puede tener varios
motivos, asi que hay dos formas de contar y las dos importan:

**Por rechazo** (la clase del primer error de la lista; el detalle por error esta en
`errores[].clase` del JSON):

| clase | rechazos |
|---|---|
| `defecto_del_teacher` | 9 |
| `regla_nuestra_de_mas` | 3 |
| `prompt_nuestro` | 0 |
| `sin_evidencia_suficiente` | 0 |

**Por instancia de error** (17, que es el numero accionable):

| clase | instancias |
|---|---|
| `defecto_del_teacher` | 9 |
| `regla_nuestra_de_mas` | 7 |
| `prompt_nuestro` | 1 |

**Por motivo:**

| motivo | inst. | `defecto_del_teacher` | `regla_nuestra_de_mas` | `prompt_nuestro` |
|---|---|---|---|---|
| `purga:simil_repetido` | 8 | 5 | 3 | – |
| `caracteres_no_latinos` | 2 | 2 | – | – |
| `pocos_turnos` | 2 | 1 | 1 | – |
| `spanglish(differente)` | 1 | – | 1 | – |
| `tuteo:mira` | 1 | – | – | 1 |
| `pregunta_en_todos_los_turnos` | 1 | 1 | – | – |
| `purga:placeholder_degenerado` | 1 | – | 1 | – |
| `purga:charla_de_asistente` | 1 | – | 1 | – |

## Los 12, uno por uno

Cita = fragmento exacto del texto guardado. La clase es la del motivo; cuando un rechazo tiene
varias clases, se aclara en la columna.

| # | escenario | motivo | clase | cita | recomendacion |
|---|---|---|---|---|---|
| 0 | `solo_charla` | `purga:simil_repetido` | teacher | `...me levanté de la cama como un zombie, fijate...` | falta la regla de clausula repetida; `simil_repetido` describe mal el lazo |
| 1 | `solo_charla` | `purga:simil_repetido` | teacher | `...como el chofer de ese camión, cargado hasta los lími` | idem #0 |
| 2 | `solo_charla` | `caracteres_no_latinos` | teacher | `es el默认.` | no tocar la regla: CJK del teacher |
| 2 | " | `tuteo:mira` | **prompt** | `Che, mira, si el sol es mentira y igual sale, eso es c` | la linea 372 del prompt ensena la palabra que la regla prohibe |
| 2 | " | `purga:simil_repetido` | regla | `...como cuando el laburo te pide disculpas pero ya te cobró` | la rama `<=3 turnos` no deberia contar similes sueltos |
| 3 | `solo_charla` | `spanglish(differente)` | regla (conf. media) | `...te entiendo, pero differente, porque vos tenés mérito` | medir `differente` en el train set antes de sacarlo |
| 3 | " | `purga:placeholder_degenerado` | regla | `<\|im_user\|>che kateto hoy me hice el picante...` | mirar solo los turnos de la voz, como las otras dos reglas |
| 4 | `solo_charla` | `pregunta_en_todos_los_turnos` | teacher | `¿Ya hiciste la pausa o segís transmittiendo sin descanso?` | no tocar la regla: el prompt ya lo prohibia |
| 4 | " | `purga:simil_repetido` | regla | `...o seguí apretando como si el mundo se fuera a acabar?` | con 3 turnos, la rama `<=3` es la unica que puede disparar |
| 5 | `solo_debate` | `caracteres_no_latinos` | teacher | `...es训练 a los pibes como si les importara` | no tocar la regla: CJK del teacher |
| 6 | `solo_debate` | `purga:simil_repetido` | regla | `...que corre como si tuviera las botas pegadas al piso` | contar repeticiones del mismo simil, no ocurrencias del patron |
| 7 | `solo_debate` | `purga:simil_repetido` | teacher | `...dímelo con el pecho para atrás, no como si estuvieras pidiendo` | idem #0 |
| 7 | " | `purga:charla_de_asistente` | regla | `...lo que se dice cuando algo esta mal: dímelo con el pecho para atr` | `"esta mal"` como substring suelto matchea argumentacion normal |
| 8 | `no_tool...` | `purga:simil_repetido` | teacher | `...mirando la pantalla como si el stream te debiera un abono.` | falta la regla de turno duplicado entre turnos de voz |
| 9 | `no_tool...` | `purga:simil_repetido` | teacher (conf. media) | `Andás preguntando de boludos, che, como si no tuviera nada mej` | es el contraejemplo de subir el umbral: #9 tiene clausula repetida |
| 10 | `no_tool...` | `pocos_turnos(1<2)` | teacher (conf. media) | la voz es copia **verbatim** del usuario | el motivo deberia ser el eco, no la forma: falta persistir `fallo` |
| 11 | `no_tool...` | `pocos_turnos(1<2)` | regla (conf. media) | el contenido que quedo esta sano | `MAX_ECOS = 1` es lo que corta la charla |

## El hallazgo de fondo: `simil_repetido`

8 de los 12 rechazos. Y las tres cosas que hay que saber de esa regla:

**1. La rama calibrada es inalcanzable en este brazo.** `simil_repetido`
(`pipelines/kateto_purge.py:191`) tiene dos ramas:

```python
if len(conteos) <= 3:
    return True                      # 2+ similes en cualquier parte
return any(conteos[i] >= 1 and conteos[i+1] >= 1 ...)   # turnos adyacentes
```

Los tres escenarios del brazo tienen `n_ex = 3` (`kateto_gen.py:750`), o sea **exactamente 3
turnos de voz**. Medido sobre las 18 muestras del brazo: **0 tienen mas de 3 turnos de voz**. La
ultima linea —la que exige adyacencia y fue la que alguien calibro— no se ejecuta nunca. La que
manda es `len(conteos) <= 3`, que no mira la relacion entre turnos: dos similes en turnos
distintos y no contiguos ya alcanzan.

**2. La ladera de aceptacion es literalmente "1 simil".** Conteos de `como (cuando|si|el|la|un|una)`
por muestra:

| | conteos |
|---|---|
| las 6 aceptadas | 0, 0, 0, 1, 1, 0 |
| las 8 rechazadas por esta regla | 2, 2, 3, 3, 3, 3, 3, 8 |

El docstring dice "Un simil suelto se cuenta pero no descarta" (`kateto_purge.py:195`) y eso es
literalmente el borde: 1 pasa, 2 no.

**3. La regla no dice lo que cree decir, y de paso es lo unico que hay.** Se llama
`simil_repetido` pero cuenta ocurrencias de un patron, no repeticiones. Medido:

| # | subcadena comun mas larga entre turnos de voz | veredicto |
|---|---|---|
| 8 | **468 chars = 100%** (voz[1] y voz[2] verbatim) | lazo real |
| 1 | **178 chars** | lazo real |
| 0 | **78 chars** (la misma pregunta de cierre en voz[1] y voz[2]) | lazo real |
| 7 | **85-87 chars**, y `user[0] == user[2]` verbatim (127 chars) | lazo real |
| 9 | **71 chars = 95%** del turno corto | repeticion corta |
| 2 · 4 · 6 | ninguna >= 40 chars | **falso positivo** |

O sea: **5 de los 8 hay un defecto real y 3 no**. Y para los 5 la regla acierta por el motivo
equivocado.

Ademas, **no existe ninguna regla de duplicacion**: la tabla `REGLAS` de `kateto_purge.py:227` es
`placeholder_degenerado`, `andamiaje_en_dialogo`, `contrato_en_dialogo`, `exclamacion_prosodia`,
`simil_repetido`, `charla_de_asistente`. Y en el loop, `es_eco` (`kateto_gen.py:568`) solo
compara el turno de **usuario** contra la voz anterior; el turno de la voz no tiene guardia
ninguna (`kateto_gen.py:586-594`). Duplicacion literal entre dos turnos de la voz no la caza
nadie: entra al bote de basura de rebote, por el contador de similes.

## Ranking de palancas (contrafactuales medidos, nada aplicado)

"Recuperan" = cuantas de las 12 pasarian con ese cambio y ningun otro.

| palanca | recupera | costo de calidad |
|---|---|---|
| `simil_repetido` desactivada | **+5** (#0, #1, #6, #8, #9) | **malo**: admite #8 (dos turnos de voz identicos, 468 chars) |
| `simil_repetido` con umbral `>= 4` | +4 (#1, #6, #8, #9) | **malo**: sigue admitiendo #8 |
| `simil_repetido` con umbral `>= 3` | +2 (#6, #9) | mediocre: #9 tiene 71 chars repetidos |
| `simil_repetido` con senal de **clausula repetida** | +1 (#6) | **ninguno**: mantiene #0, #1, #7, #8, #9 rechazados y no rechaza ninguna de las 6 aceptadas |
| `placeholder_degenerado` solo en la voz | 0 sola | ninguno |
| `spanglish` sin `differente` | 0 solo | ninguno (pero `differente` no esta medido en ningun archivo) |
| las dos anteriores juntas | **+2** (#3, #6) | **ninguno** |
| las tres, con umbral `>= 3` en vez de la senal | +3 (#3, #6, #9) | mediocre |

**El rechazo que mas aceptacion devuelve, sin costo de calidad, es el #3**: es el unico que
muere por dos falsos positivos de nuestras reglas —`placeholder_degenerado` porque el usuario
llama por su nombre al personaje, `spanglish` por `differente`— y los dos se arreglan con una
palabra cada uno. Suma **#6** y el brazo pasa de 6/18 a **8/18** sin entrar ni una muestra con
duplicacion literal.

**La causa con mas alcance es `purga:simil_repetido`** (8 de 12), pero es la mas peligrosa de
tocar: toda afloja que recupera muestras admite duplicacion verbatim, porque es la unica que la
esta pescando de rebote. El cambio correcto no es el umbral, es la senal.

## Lo que las reglas NO detectaron en este mismo brazo

Evidencia del mismo contraste que vuelve la regla de spanglish unreliable: hay ingles inequivoco
en turnos de **voz** que la lista de `scripts/gen_toolcalling_dataset.py:503` no cubre.

| # | palabra en turno de voz | en `RX_SPANGLISH`? |
|---|---|---|
| 4 | `being`, `transmittiendo` | no / no |
| 5 | `still`, `after` | no / no |
| 6 | `changes` | no |

`transmittiendo` con doble `t` ademas es un error de tipeo que ninguna regla de estilo ve. Y al
revés: `differente` esta en la lista y si tiene uso rioplatense.

## Lo que no se pudo determinar

- **Que motivo de eco corto #10 y #11.** El unico camino del codigo que devuelve texto no vacio
  y corto es el corte por eco agotado (`kateto_gen.py:574-578`, `stats["corte_eco"] = True`), asi
  que el corte esta probado. Pero el motivo concreto (`trigramas` / `arranque`) **no quedo
  persistido**: `GenerarMuestras` llena la columna `fallo` (`kateto_gen.py:763`) y el sink escribe
  solo `scenario` / `errors` / `text`. Consta el agregado de la corrida
  (`comparacion-simulador.json`): `ecos_rechazados_por_la_guardia: 6`,
  `motivos_eco {trigramas: 4, arranque: 2}`. Por eso esas dos entradas van con `confianza: media`.
- **Si `differente` deberia estar en la lista.** No hay en ningun archivo del repo una
  frecuencia de `differente` sobre el train set. La clasificacion es `regla_nuestra_de_mas` con
  `confianza: media`, apoyada en la politica declarada de la propia lista ("palabras medidas o
  sin uso rioplatense") y en que el mismo brazo produce cinco fugas de ingles que si son
  inequivocas.
- **Que texto daria el loop con mas reintentos de eco.** No es simulable sin rerun del teacher.
  Lo medido es que `MAX_ECOS = 1` da dos intentos y que el agregado de la corrida mato 6 turnos de
  usuario; lo que pasaria despues no se puede afirmar desde estos archivos.

> Nota de concurrencia (no se toco el repo principal). Mientras este triage corria, el
> `pipelines/kateto_gen.py` del repo principal cambio a las 21:10 con una guardia de eco
> **del lado de la voz** ("R10 voz", `es_eco` simetrica con `(voz, usuario)`). Eso es
> exactamente lo que pide el rechazo #10 ("la voz es copia verbatim del usuario y no hay
> guardia de ese lado", `kateto_gen.py:586-594` en este worktree). Todo este doc cita la copia
> del worktree, que es la que correspondia al brief; las lineas de `kateto_gen.py` pueden no
> coincidir con el repo principal ya actualizado.

## Un riesgo latente, medido de yapa

`RX_TUTEO_HOMOGRAFO` (`scripts/gen_toolcalling_dataset.py:144`) es sensible a la normalizacion
Unicode, no solo al acento:

| forma | dispara `tuteo`? |
|---|---|
| `Mirá` con á precompuesta (NFC) | no |
| `Mira` + acento combinante U+0301 | **si** |
| `mira` sin acento | **si** |

O sea: la forma correcta de voseo pasa, pero solo si el texto llega en NFC. Cualquier
normalizacion Unicode en el camino puede empezar a rechazar voseo correcto. No lo toco (es
diagnostico), queda anotado.