# fix-124 — El validador nuevo explota con args que no son objeto (mata al worker)

**Delta sobre fix-123.** Urgente: bloquea cualquier relanzamiento de la ola.

## Qué pasa — MEDIDO por el orquestador, no re-diagnostiques
fix-123 cambió la extracción de bloques por balance, así que ahora también levanta `tool_call` cuyos
argumentos **no son un objeto JSON**. La validación de argumentos asume dict y **revienta**:

`scripts/gen_toolcalling_dataset.py:620`
```python
            for k, v in args.items():
```
→ `AttributeError: 'list' object has no attribute 'items'`.

Replay de los rechazos reales de la ola cerrada: **9 bloques** en 3 escenarios. Casos crudos:
- `no_se_programar`, `tool_call run_command: ["systemctl", "is-active", "docker"]` (array)
- `no_sabe_algo`, `tool_call run_command: "resultado River vs Boca anoche"` (string pelado)
- `voice_ask_help`, `tool_call refine_memory: ["finances", "subtraction", "evidence: upcoming bills", "fact: zero surplus"]` (array)

**Por qué es grave y no cosmético:** `validate()` se llama en `main()` **fuera** del `try/except` que
envuelve la generación. Una excepción ahí mata el proceso del worker. Con 12 workers y miles de
muestras, ese `AttributeError` va matando workers en silencio durante horas. Medido en el replay: el
script que validaba las muestras reales murió en la novena muestra.

## REGLA 0 — prohibido explorar
Abrí SOLO `scripts/gen_toolcalling_dataset.py` y `scripts/test_fix124.py`. No leas `data/`, `docs/` ni
otros scripts. **No corras el generador** (no hay ola corriendo, pero el gateway lo usan otras cosas):
sólo gates locales.

## Cambio 1 — args que no son dict: error, no crash
En el bucle de `llamadas` (líneas 599-623), después del `json.loads`:
- Si el resultado **no es un `dict`**: `errs.append(f"args_no_dict:{nombre}")` y `continue` (no corras
  `faltan_requeridos` / `args_extra` / `args_tipo`: los tres asumen claves, y el esquema de
  `ALL_TOOLS` es todo argumentos con nombre).
- Los tres chequeos existentes quedan como están para el caso dict.
- No toques `args_no_json`: sigue para lo que no parsea.

## Cambio 2 — un bug del validador no puede matar al worker
En `main()`, envolvé la llamada a `validate(...)` (la que hoy está fuera del `try`) para que una
excepción se registre como rechazo en vez de tumbar el proceso:
- `except Exception as e:` → contá el rechazo, imprimí
  `[{i+1}/{n}] {sc['id']}: VALIDADOR EXPLOTO -> {type(e).__name__}: {e}` y
  `record_rejection(fr, sc["id"], [f"exc_validate:{type(e).__name__}"], raw=raw or "", text=txt)`.
- Tiene que quedar en el resumen de motivos como cualquier otro.
No borres ni reordenes el `try/except` que ya existe para la generación: es otro.

## Lo que NO hay que tocar
- No toques `extraer_bloques`/`_fin_balanceado` (la extracción por balance está bien: el bug es que la
  validación asumía dict).
- No toques `RX_TUTEO`, la lista de spanglish, las reglas de anclaje/bit, `SCENARIOS`, ni los prompts.
- No toques `out/`, ni los teachers. **No commitees ni pushees.**
- Si algo no coincide con el código, decilo en el reporte y seguí con el ancla textual.

## Tests: `scripts/test_fix124.py` (pytest, sin red)
- Los tres bloques crudos de arriba, dentro de una muestra válida, **no levantan excepción** y dan
  `args_no_dict:<tool>`.
- Un `tool_call` con args dict sigue dando `faltan_requeridos` / `args_extra` / `args_tipo` como hoy.
- Un `tool_call` con JSON inválido sigue dando `args_no_json`.
- `validate()` sobre una muestra sana sigue devolviendo `[]`.

## Gates (correlos y pegá la salida cruda)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train
./venv/bin/python -m pytest scripts/ -q
./venv/bin/python scripts/gen_toolcalling_dataset.py --dry-run
```
Además, como gate propio del brief: corré el validador sobre **todas** las muestras de
`out/dataset/shards/rechazos-w9*.jsonl` y `parcial-w9*.jsonl` (w99..w110) y pegá cuántas procesó y si
alguna levantó excepción. Es la prueba de que ya no hay crash con datos reales. **No llames al
teacher**: es validación local, no red.

## Reporte — `/home/chaos/harness-run/fix124/REPORT.md` (ruta ABSOLUTA)
1. Qué cambió, `archivo:línea`, con los fragmentos pegados.
2. Salida CRUDA de los gates, incluido el conteo del replay local.
3. Qué quedó sin hacer y qué viste y no tocaste.
