# fix-123 — El validador corta el JSON en la primera llave + el tuteo con 2/3 de falsos positivos

## Meta (dos defectos del VALIDADOR, medidos sobre la ola `kateto-ola99`)
1. `args_no_json` (212 rechazos, y **188 bloques** en total) **no es del teacher**: el JSON válido
   está en el texto y lo corta el regex del parser.
2. `tuteo` (334 rechazos, el motivo número uno): **222 de esos 334 son falsos positivos** (66 %).

## REGLA 0 — prohibido explorar
Abrí SOLO `scripts/gen_toolcalling_dataset.py`, `scripts/test_fix123.py` y, si hace falta,
`scripts/test_fix119.py`/`test_fix122.py` para no repetir fixtures. Nada de `data/`, `docs/` ni otros
scripts. **No corras el generador**: hay 12 workers VIVOS contra el gateway. Sólo gates locales.

## Causa 1 — MEDIDA: el regex de args corta en el primer `}`
`scripts/gen_toolcalling_dataset.py:174`:
```python
RX_TOOLCALL = re.compile(r"tool_call\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\{.*?\})\s*(?:\n|$)", re.S)
```
El `\{.*?\}` es **non-greedy**: con un objeto anidado captura hasta el `}` INTERNO y deja el JSON
inválido. Texto REAL de una rechazada (crudo, sin tocar):
```
tool_call send_event: {"event_name": "chat_message", "data": {"message": "sistema chequeado: chat y
follows vivos", "user": "seco"}
, "target": "overlay_plugin"}
```
El teacher escribió JSON perfectamente válido (el `}` exterior está ahí, partido en dos líneas por el
wrap). Lo que hace el validador es capturar `{"event_name": …, "data": {…}` — hasta el `}` del objeto
`data` — y eso no parsea.

**Medido sobre las 230 capturas que fallan en esta ola:** agregándoles el cierre que falta, **189
parsean** (186 son `send_event`; el resto `schedule_event`, `list_videos`, `list_events`). De las 188
capturas de `send_event`, **60 tienen el cierre en la línea siguiente** (la coma quedó al principio de
la línea) y **128 no tienen cierre cerca**: las dos son el mismo corte.
`args_no_json:send_event` por escenario: `events_lookup` 168, `video_summary_share` 14,
`no_se_programar` 6, `scheduled_reminder` 3, `memory_refine` 3, resto 1-2. Es lo que hunde a
`events_lookup` (**2 aceptadas / 103 rechazos**).

`RX_TOOLRES` (línea 175) tiene el mismo problema con `(.*?)(?:\n|$)`: los `tool_result` con arrays u
objetos anidados en varias líneas también se truncan.

### Cambio 1
1. Extraé los bloques **por balance de llaves/corchetes, no por regex non-greedy**. Función nueva:
   `extraer_bloques(txt) -> list[(nombre, inicio, fin)]` que, para cada `tool_call <nombre>:` /
   `tool_result <nombre>:`, avance desde el primer `{` o `[` contando aperturas y cierres (ignorando
   los que están dentro de strings) hasta balancear. Si nunca balancea, el bloque llega hasta el final
   de la línea, como hoy.
2. **Colapsá los saltos de línea internos del bloque** antes de parsear (el wrap medido deja la coma
   al principio de la línea siguiente). Si ya existe `collapse_tool_call_newlines` (línea 181), usalo;
   si no cubre este caso, cubrilo.
3. `validate()` (el `llamadas = RX_TOOLCALL.findall(sample)` de la línea 476 y el `json.loads` de la
   485) tiene que usar la extracción nueva. **Si el bloque extraído por balance parsea, NO es
   `args_no_json`** — el texto ya era válido.
4. Si un bloque sigue sin parsear después de eso (comillas sin escapar, truncado de verdad), ahí sí
   `args_no_json`, como hoy.
5. **No agregues una reparación que escriba llaves en el texto guardado**: el texto del teacher está
   bien; el que corta es el parser. No toques lo que se escribe al dataset.

## Causa 2 — MEDIDA: el tuteo mira donde no debe
`RX_TUTEO` (línea 136) + la regla de la línea 543 hacen `RX_TUTEO.search(sample)`: **la muestra
entera, incluido el turno del usuario y de la contraparte**. Clasificación de las 334 muestras
rechazadas por tuteo en la ola:
- **147: el tuteo está SÓLO en el turno del usuario** ("Che, mira que tiré un par de sticks…"). El
  usuario no es la voz: no tiene que hablar en voseo. Falso positivo.
- **75: está en la voz pero son homógrafos de 3ª persona** — "la heladera te mira con cara de boludo",
  "todos lo ven y nadie lo tapa". Falso positivo.
- **112: tuteo real en la voz** ("primero dime qué querés planificar", "si de verdad quieres algo
  profesional", "tienes", "puedes", "eres", "hazlo"). Estos sí van.
Por palabra: `mira` 283, `ven` 21, `dime` 10, `tienes` 6, `quieres` 4, `puedes` 3, `eres` 3, `hazlo` 2.

### Cambio 2
1. **La regla corre sólo sobre los turnos de la VOZ** (`<|im_start|>{voice} … <|im_end|>`), nunca sobre
   el turno del usuario ni sobre la contraparte. Mismo criterio que la regla `spanglish` de fix-119.
2. Partí `RX_TUTEO` en dos:
   - **Inequívocos** (siempre error en la voz): `tienes|puedes|quieres|eres|dime|hazlo|amigo m[ií]o`.
   - **Homógrafos** (`mira`, `ven`, `haz`): sólo cuentan como error si **no** están precedidos por
     clítico o sujeto. Guard concreto: no matchear si antes del match hay
     `\b(te|me|lo|la|se|nos|les|todos|nadie|ellos)\s+` (medido: "te mira", "todos lo ven").
3. En los DOS prompts, en las reglas de estilo, agregá el bloque literal:
```
- Imperativos SIEMPRE en voseo: mira (no "mira"), decime (no "dime"), veni (no "ven"),
  hacelo (no "hazlo"), escucha (no "escucha"), fijate (no "fijate").
- PROHIBIDO el tuteo: tienes, puedes, quieres, eres, dime, hazlo, ven, amigo mio.
```
(usá las formas con tilde donde el texto final las lleve: mirá, vení, escuchá, fijate; lo que importa
es que la lista de formas voseo y la lista de prohibidas estén las dos.)

## Lo que NO hay que tocar
- No toques `RX_VOSEO`, ni la lista de spanglish, ni las reglas de anclaje/bit, ni `SCENARIOS`.
- No toques `out/`, la ola `kateto-ola99`, ni los teachers.
- **No commitees ni pushees.**
- Si algo no coincide con el código, decilo en el reporte y seguí con el ancla textual.

## Tests: `scripts/test_fix123.py` (pytest, sin red)
- **Balance de bloques**: el texto real de arriba (`send_event` con `data` anidado y el cierre en la
  línea siguiente) → el bloque extraído **parsea** y NO da `args_no_json`. Hoy da error.
- Un `tool_result` con un array de objetos en dos líneas → parsea.
- Un bloque con comillas sin escapar → sigue dando `args_no_json`.
- **Tuteo**: muestra con tuteo SOLO en el turno del usuario → sin error. "la heladera te mira" en la
  voz → sin error. "todos lo ven" en la voz → sin error. "mira, boludo" en la voz → `tuteo`. "dime" /
  "tienes" / "quieres" en la voz → `tuteo`.
- Los tests de fix-119/120/121/122 que dependan de `RX_TUTEO` se actualizan si rompen (la regla, no el
  script, es lo que cambió).

## Gates (correlos y pegá la salida cruda)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train
./venv/bin/python -m pytest scripts/ -q
./venv/bin/python scripts/gen_toolcalling_dataset.py --dry-run
```

## Reporte — `/home/chaos/harness-run/fix123/REPORT.md` (ruta ABSOLUTA)
1. Qué cambió, `archivo:línea`, con los fragmentos pegados.
2. Salida CRUDA de los gates.
3. Qué quedó sin hacer y qué viste y no tocaste.
