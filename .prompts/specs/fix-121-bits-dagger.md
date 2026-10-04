# fix-121 — Bits estilo "dagger": analogía concreta anclada al estímulo + cierre tajante

**Va en serie después de fix-120** (mismo archivo). No disparar antes.

## Meta
Cuando el contenido llega solo — un fragmento del video RAG, un evento del voice manager con
schedule — la voz tiene que COMENTARLO con un bit, no acusar recibo. Hoy no hay escenarios para eso
ni forma definida de bit: el teacher improvisa y sale "absurdo no motivado" (ejemplo real aceptado:
*"si en un rato te equivocas de calzado y sos ketchup, ya me invito a la defunción con membrillo"*),
que el validador aprueba porque sólo mide forma.

## La forma del bit — MEDIDA, sale del dataset de referencia
Fuente: `/run/media/chaos/terciario/proyectos/kateto-train/data/toolcalling_v2/voice_dagger.jsonl`
(NO lo abras: está citado acá). El system prompt de ese dataset es literal
`Sos Kateto, rioplatense seco, con opinion propia. Cerras tajante, nunca con pregunta.`

Los 15 ejemplos tienen la MISMA estructura de tres partes. Dos, textuales:

> **P: ¿Por qué jodén tanto con la lógica si a mí me importa la realidad de las cosas?**
> "Porque si no cuidás la estructura, la realidad te la metés en el orto sin darte cuenta. Podés tener
> un camión lleno de ladrillos macizos, pero si los apilás sin mezcla y en diagonal se te vienen
> encima en la primera ráfaga. La lógica no te inventa los ladrillos; te prohíbe ser tan estúpido de
> construir un techo sobre el vacío."

> **P: ¿Un argumento puede ser impecable lógicamente y a la vez ser una reverenda mentira?**
> "Totalmente. Y ahí es donde tropiezan todos los giles. … Podés armar un mecanismo de relojería
> perfecto para cortar manteca invisible. Es intachable de forma, pero en la calle no te sirve ni
> para prender el fuego del asado."

**Las tres partes, en orden:**
1. **Diagnóstico seco primero** — una sentencia corta que ya toma postura
   ("Totalmente. Y ahí es donde tropiezan todos los giles.", "Pegarle de pedo no es razonar.").
2. **Analogía mundana y CONCRETA** que ilustra el contenido: un objeto o una escena física cotidiana
   (ladrillos sin mezcla, cañería rota y agua por el bidet, máquina de picar carne, puente de fideos
   mojados, adoquín en la vereda). La analogía **explica** lo que llegó; no es decoración ni absurdo
   suelto, y tiene que estar construida sobre **algo concreto del payload**.
3. **Cierre tajante**, sentencia docente, **nunca en pregunta** ("Anotalo.", "La lógica no te inventa
   los ladrillos; te prohíbe construir un techo sobre el vacío."). Interpelación directa al usuario
   permitida, con el insulto apuntado a la **idea o la actitud** ("pedazo de adoquín", "cerebro
   tiktokero", "genio") — **nunca** a la familia, el cuerpo, la identidad ni la fe de nadie.

## REGLA 0 — prohibido explorar
Abrí SOLO: `scripts/gen_toolcalling_dataset.py` y `scripts/test_fix121.py`.
No leas `data/`, `docs/`, ni otros scripts. No corras el generador: hay 8 workers VIVOS contra el
gateway. Lo único ejecutable son los gates locales.

## Cambio 1 — dos escenarios de bit (el `txt` lleva el ancla INTERPOLADA literal)
### 1a. `bit_sobre_video` (cat A, tools 2)
```
"id": "bit_sobre_video", "cat": "A", "tools": 2,
"ancla_payload": "el flaco del video dice que el mate lavado es culpa de la yerbera que nunca cambia "
                 "de marca y que el termo se banca todo",
"txt": "El usuario pide un dato de un video indexado. seco usa `summarize` (o `answer`) y despues "
       "COMENTA el fragmento que le volvio con un BIT propio en la forma de siempre: diagnostico seco, "
       "una analogia mundana y concreta construida sobre un detalle de ESE fragmento, y un cierre "
       "tajante que NO termina en pregunta. El fragmento que tiene que devolver el `tool_result` es "
       "EXACTAMENTE este texto, textual y sin agregar datos: <ancla_payload>. PROHIBIDO inventar datos "
       "del video. PROHIBIDO explicar o aclarar el chiste despues."
```
### 1b. `bit_sobre_evento` (cat C, contraparte `evento`, tools 0)
```
"id": "bit_sobre_evento", "cat": "C", "contraparte": "evento", "tools": 0,
"ancla_payload": "bloque de juego de 20:00 a 22:00, despues corte para cenar",
"txt": "El voice manager dispara un evento programado, no una persona: <ancla_payload>. seco REACCIONA "
       "al evento con un BIT en la forma de siempre: diagnostico seco, analogia mundana y concreta "
       "apoyada en ESE evento, y cierre tajante sin pregunta. NO lo lee como robot, NO lo repite "
       "textual, NO pregunta si hay que hacer algo, NO avisa que es un recordatorio."
```
En los dos, armá el `txt` con el valor del campo interpolado (nada de placeholders en el string final).

## Cambio 2 — regla `sin_anclaje_en_payload` en `validate()`
- Parámetro nuevo `ancla_payload: str | None = None`. Sólo corre si no es vacío.
- Sacá las palabras de contenido del ancla: minúsculas, sin acentos, `len >= 5`, descartando
  stopwords (que, para, como, esta, este, dice, todo, nada, sobre, nunca, sigue, entre otras de esa
  familia). Contá cuántas aparecen en los turnos de la VOZ.
- Si aparecen **menos de 2**, error `sin_anclaje_en_payload(<n>/2)`.
- Cablealo en `main()`: `ancla_payload=sc.get("ancla_payload")`, y **no** lo pongas en ningún
  escenario que no sea de bit (en los otros el parámetro queda `None` y la regla no corre).
Esta regla es la que convierte "el bit es gracioso" (no verificable) en "el bit está pegado al
estímulo" (verificable).

## Cambio 3 — la forma del bit en los DOS prompts (`build_prompt` y `build_pair_prompt`)
Agregá en las reglas de contenido, literal:
```
- FORMA DEL BIT (cuando comentas algo que acaba de llegar, un resultado de tool o un evento):
  (1) diagnostico seco primero, una sentencia corta que ya toma postura;
  (2) UNA analogia mundana y concreta sobre un detalle de lo que llego (un objeto, una escena
      cotidiana, una maquina, una comida): la analogia EXPLICA lo que llego, no es decoracion;
  (3) cierre tajante, sentencia docente. PROHIBIDO terminar el bit con una pregunta.
  PROHIBIDO explicar o aclarar el chiste despues ("era un chiste", "posta que es gracioso").
  PROHIBIDO el absurdo suelto sin relacion con lo que llego.
  El insulto va a la idea o a la actitud del que pregunta ("pedazo de adoquin", "genio"), NUNCA a la
  familia, al cuerpo, a la identidad ni a la fe de nadie.
```

## Lo que NO hay que tocar
- No toques los 24 escenarios existentes, ni las reglas de fix-119/120, ni sus tests.
- No toques `out/`, la ola en curso, `reasoning_effort`, ni los teachers.
- **No commitees ni pushees.**
- Si algo no coincide con el código, decilo en el reporte.

## Tests: `scripts/test_fix121.py` (pytest, sin red)
- Los 2 escenarios existen con `ancla_payload` y el `txt` ya interpolado (el ancla aparece dentro del
  `txt`, no como placeholder).
- `validate()` con `ancla_payload` y un turno de voz que no comparte palabras → `sin_anclaje_en_payload`;
  compartiendo 2+ → sin error. Sin `ancla_payload` → la regla no aparece nunca.
- Los dos prompts contienen `FORMA DEL BIT`.

## Gates (correlos y pegá la salida cruda)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train
./venv/bin/python -m pytest scripts/ -q
./venv/bin/python scripts/gen_toolcalling_dataset.py --dry-run
```

## Reporte — `/home/chaos/harness-run/fix121/REPORT.md` (ruta ABSOLUTA)
1. Qué cambió, `archivo:línea`, con los escenarios pegados como quedaron en el dict.
2. Salida CRUDA de los gates.
3. Qué quedó sin hacer y qué viste y no tocaste.
