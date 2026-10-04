# fix-120 — La tool NO es siempre obligatoria + escenarios de silencio

**Delta sobre fix-119**, que ya está en el árbol (no lo revirtás: corregilo y completalo).
Toca el mismo archivo, así que va en serie después de 119.

## Meta
1. Corregir el ALCANCE de fix-119: dejó la obligación de tool **global** para todo escenario con
   tools. Tiene que ser **por escenario**: hay escenarios donde el punto ES la tool, y hay otros
   donde el usuario hace una pregunta trivial ("¿cuánto es 2+2?") y lo correcto es contestar
   **sin** llamar nada. Corregido por el usuario, textual: *"a veces no hace falta tool call, puede
   ser una pregunta rápida del usuario como por ejemplo qué es 2+2"*.
2. Dos escenarios donde **callarse ES la respuesta**.
3. La regla que mata el tic de cerrar todos los turnos con pregunta.

## Estado actual del árbol (verificado por el orquestador, no lo re-diagnostiques)
fix-119 ya dejó, todo en `scripts/gen_toolcalling_dataset.py`:
- `build_pair_prompt(..., *, hint="")` en la línea 714, con la rama `permite_tools` (línea 730) que
  ahora dice `ESTE ESCENARIO EXIGE TOOL: …` **para todo escenario con tools**, y ya no contiene la
  frase `Si no hace falta ninguna tool, escribi solo tu mensaje.`
- `generate_paired(..., hint="")` (línea ~989) que propaga el hint.
- `HINT_TOOLS` (línea ~1116) y el reintento con pista en `main()` (línea ~1211-1220), que hoy
  dispara con `sc.get("tools", 0) > 0`.
- La regla `spanglish` en `validate()` (líneas 543-550) con `permitir_spanglish`.
- `scripts/test_fix119.py`.
Nada de eso se revierte: se corrige el punto 1.

## REGLA 0 — prohibido explorar
Abrí SOLO: `scripts/gen_toolcalling_dataset.py`, `scripts/test_fix119.py` y `scripts/test_fix120.py`.
No leas `data/`, `docs/`, ni otros scripts. No corras el generador: hay **8 workers VIVOS** contra el
gateway. Lo único ejecutable son los gates locales.

## Cambio 1 — `tool_obligatoria` por escenario
### 1a. `build_pair_prompt()` (línea 714)
- Parámetro nuevo keyword-only: `tool_obligatoria: bool = True`.
- En la rama `permite_tools`, partí el `forma_voz` en dos:
  - `tool_obligatoria=True` → el bloque que ya está (`ESTE ESCENARIO EXIGE TOOL: …`). No lo cambies.
  - `tool_obligatoria=False` → la tool está DISPONIBLE pero no hace falta. Texto literal:
    ```
      Podes usar una tool si de verdad hace falta:
      `tool_call <nombre>: <json>` (JSON valido y en UNA sola linea), despues
      `tool_result <nombre>: <resultado>`, y despues tu comentario.
      Tools disponibles (nombre IDENTICO):
    {tools_txt}
      Si no hace falta ninguna tool, escribi solo tu mensaje. Si el usuario pregunta algo simple
      (una cuenta, un dato que sabes de memoria, el significado de una palabra), CONTESTA DIRECTO
      sin llamar ninguna tool.
    ```
### 1b. `generate_paired()` (línea ~989)
- Suma `tool_obligatoria: bool = True` y lo pasa a `build_pair_prompt(...)` en cada intercambio.
### 1c. `main()`
- Antes del bucle de escenarios: `tool_oblig = sc.get("tool_obligatoria", sc.get("tools", 0) > 0)`.
- Pasalo a `generate_paired(..., tool_obligatoria=tool_oblig)`.
- Y en `validate(...)`: **`require_tool=tool_oblig`** en lugar de `sc.get("tools", 0) > 0`.
- El reintento con `HINT_TOOLS` de fix-119 corre **sólo si `tool_oblig` es True**. Si no es
  obligatoria, un `sin_tool_call` no existe como motivo y no hay nada que reintentar.
Con el default, los 24 escenarios actuales se comportan igual que hoy (`tools>0` ⇒ obligatoria).

## Cambio 2 — regla `prohibir_tool` en `validate()` + escenario `pregunta_trivial_sin_tool`
### 2a. Regla
- Parámetro nuevo `prohibir_tool: bool = False`. Si es True y la muestra trae alguna llamada, error
  `tool_no_permitida(<nombre>)`. Cablealo en `main()` con
  `prohibir_tool=sc.get("prohibir_tool", False)`. No lo pongas en ningún escenario existente.
### 2b. Escenario nuevo (cat B)
```
"id": "pregunta_trivial_sin_tool", "cat": "B", "tools": 2, "tool_obligatoria": False,
"prohibir_tool": True,
"txt": "El usuario hace una pregunta corta y trivial que se contesta de una: una cuenta simple "
       "('cuanto es 2+2', 'que es el 20% de 300'), el significado de una palabra o un dato basico. "
       "seco responde DIRECTO, con su voz y su opinion, SIN llamar ninguna tool. Si llama una tool, "
       "la muestra se descarta."
```

## Cambio 3 — dos escenarios de SILENCIO (cat F)
Los centinelas se fuerzan desde el código con `forzar_respuesta` (por INDICE de intercambio), nunca se
le piden al teacher. Copiá la forma de `no_response_al_aire` / `wait_asr_incompleto`.
```
"id": "no_llenar_silencio", "cat": "F", "tools": 0,
"min_turns": 1, "voseo": False, "n_ex": 1, "min_palabras_usuario": 1,
"forzar_respuesta": ["<|no_response|>"],
"txt": "El usuario emite un reconocimiento corto que NO pide nada ('aja', 'claro', 'mm', 'si', 'ok'): "
       "sigue mirando el video o esta pensando. seco NO acusa recibo, NO pregunta de nuevo y NO "
       "inventa tema: se queda callado. La respuesta es exactamente `<|no_response|>` y nada mas."

"id": "silencio_charla_cerrada", "cat": "F", "tools": 0,
"min_turns": 1, "voseo": True, "n_ex": 2,
"forzar_respuesta": [None, "<|no_response|>"],
"txt": "La conversacion YA se cerro: el usuario se despidio ('buenas noches', 'me voy a dormir', "
       "'despues te cuento') y seco le contesto la despedida. En el intercambio siguiente el usuario "
       "emite un ruido de fondo sin dirigirse a seco ('mmm', 'uy', un suspiro). seco NO vuelve a "
       "arrancar la charla: la respuesta es exactamente `<|no_response|>`."
```

## Cambio 4 — regla `pregunta_en_todos_los_turnos` + reglas de contenido en los dos prompts
### 4a. `validate()`
Si la muestra tiene **3 o más** turnos de voz y **TODOS** terminan en `?`, error
`pregunta_en_todos_los_turnos`. Con menos de 3 turnos no aplica (dos preguntas seguidas es charla
normal). Opt-out por escenario: `permitir_preguntas` (default False) — no lo pongas en ningún
escenario hoy.
### 4b. En `build_prompt` y en `build_pair_prompt`, en las reglas de contenido, literal:
```
- No todas las respuestas son texto. Esta permitido y a veces es lo correcto NO hablar: la respuesta
  es exactamente `<|no_response|>`. NO cierres todos los turnos con una pregunta. Prohibido terminar
  todos los turnos con "?" o con una invitacion a seguir ("queres algo mas?", "que opinas?"). Cuando
  el usuario no pide nada, el turno termina en una afirmacion, una opinion o un remate, y se planta.
```

## Cambio 5 — tests
- `scripts/test_fix120.py` (pytest, sin red):
  - `tool_obligatoria=False` → el prompt contiene `escribi solo tu mensaje` y **no** `EXIGE TOOL`;
    `tool_obligatoria=True` → al revés.
  - `prohibir_tool=True` + muestra con tool → `tool_no_permitida(...)`; sin tool → sin error.
  - `pregunta_en_todos_los_turnos`: 3 turnos de voz terminando todos en `?` → error; 2 de 3 → sin error.
  - Los 3 escenarios nuevos existen con las claves exactas de arriba.
  - Los dos prompts contienen la regla de silencio.
- **Actualizá `scripts/test_fix119.py`**: su assertion de que el prompt NO contiene `escribi solo tu
  mensaje` quedó vieja. Ahora esa frase existe cuando `tool_obligatoria=False`. Arreglá el test, no
  el script.

## Lo que NO hay que tocar
- No toques la regla `spanglish`, `HINT_TOOLS` ni los 24 escenarios existentes.
- No toques `out/`, la ola en curso, `reasoning_effort`, ni los teachers.
- **No commitees ni pushees.**
- Si algo no coincide con el código, decilo en el reporte y seguí con el ancla textual.

## Gates (correlos y pegá la salida cruda)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train
./venv/bin/python -m pytest scripts/test_reasoning_off.py scripts/test_fix119.py scripts/test_fix120.py -q
./venv/bin/python scripts/gen_toolcalling_dataset.py --dry-run
```

## Reporte — `/home/chaos/harness-run/fix120/REPORT.md` (ruta ABSOLUTA)
1. Qué cambió, `archivo:línea`, con los escenarios nuevos pegados como quedaron en el dict.
2. Salida CRUDA de los gates.
3. Qué quedó sin hacer y qué viste y no tocaste.
