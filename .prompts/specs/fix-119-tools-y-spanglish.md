# fix-119 — El prompt licencia no usar la tool + filtro de spanglish

## Meta (dos cosas, nada más)
1. **`sin_tool_call`** es el 62 % de los rechazos de la ola en curso (286 de 460). La causa está
   medida: el prompt **dice lo contrario de lo que valida el validador**.
2. **Filtro de spanglish** en los turnos de la VOZ (medido: 14 fugas en 506 aceptadas, tipo
   "No te worries, pibe" y "lo recuerdo bien differente").

## Causa raíz 1 — MEDIDA, no la re-diagnostiques
`build_pair_prompt()` línea 696; la rama `permite_tools` (líneas 711-719) termina con:

```python
            f"  Tools disponibles (nombre IDENTICO):\n{tools_txt}\n"
            "  Si no hace falta ninguna tool, escribi solo tu mensaje.")
```

Y `permite_tools` llega como `con_tools = sc.get("tools", 0) > 0` (línea 972), o sea que esa rama
**sólo se entra en escenarios que exigen tool**. Encima, `validate()` línea 534 rechaza con
`sin_tool_call` si no hay ninguna llamada (`require_tool=sc.get("tools", 0) > 0`, línea 1173).
Resultado: el prompt le dice al teacher "escribí solo tu mensaje" y el validador le tira la muestra
por la cabeza. Distribución medida de `sin_tool_call` (ola nueva w63-w70): `giveup_honest` 35,
`voice_refused` 31, `video_summary_share` 30, `no_sabe_algo` 29, `voice_ask_help` 28,
`no_se_programar` 27, `scheduled_reminder` 26, `minecraft_gate_negado` 23, `memory_refine` 20,
`video_rag_chain` 18, `events_lookup` 13. Es parejo: es sistémico, no de un escenario.

## REGLA 0 — prohibido explorar
Abrí SOLO:
- `/run/media/chaos/terciario/proyectos/kateto-train/scripts/gen_toolcalling_dataset.py`
- los dos archivos de test nuevos (abajo)
No leas `data/`, `docs/`, ni otros scripts. No corras el generador. **Hay 8 workers de generación
VIVOS** contra el gateway: no lances nada que hable por red ni toques `out/`.
Lo único ejecutable son los gates locales de abajo.

## Cambio 1 — alinear el prompt con el validador, y un reintento con pista
### 1a. `build_pair_prompt()` (línea 696)
- Sacá la frase `Si no hace falta ninguna tool, escribi solo tu mensaje.` de la rama
  `permite_tools`. Esa rama sólo existe para escenarios con tool: la frase contradice al validador.
- Poné en su lugar, literal:
  `  ESTE ESCENARIO EXIGE TOOL: la linea VOS tiene que incluir al menos un ``tool_call <nombre>: <json>``
   con el nombre EXACTO de la lista, seguido de su ``tool_result``. Una respuesta sin tool_call se
   descarta entera, no se guarda.`
- Nuevo parámetro keyword-only `hint: str = ""`. Si no está vacío, se agrega **al final** del prompt
  como bloque `PISTA OBLIGATORIA:\n{hint}`.

### 1b. `generate_paired()` (línea 964)
- Suma `hint: str = ""` y lo pasa a `build_pair_prompt(...)` en cada intercambio. No cambies nada más
  de su lógica.

### 1c. `main()` (el `if errs:` de la línea ~1174)
- Si la muestra falla y **TODOS** los errores son `sin_tool_call`, y el escenario tiene tools:
  rehacé la muestra **UNA sola vez** con la pista de abajo, revalidá, y si ahora pasa se guarda;
  si vuelve a fallar se registra el rechazo como hoy (con los errores del segundo intento).
  El reintento NO aplica a ningún otro motivo de rechazo.
- Constante nueva, literal, arriba de `main()`:
  `HINT_TOOLS = ("OBLIGATORIO: la linea VOS de ESTE turno tiene que llamar la tool que el escenario "
                 "pide, con la sintaxis `tool_call <nombre>: <json en UNA linea>` y despues su "
                 "`tool_result <nombre>: <resultado>`. Sin tool_call la muestra se descarta.")`
- Dejá registro del reintento en el print: `RECHAZADA -> [...] (retry con pista)`.

## Cambio 2 — regla `spanglish` en `validate()` (línea 383)
- Parámetros nuevos: `permitir_spanglish: bool = False`.
- **Sólo mira los turnos de la VOZ**, nunca el turno del usuario: el usuario puede citar chat o un
  NPC en inglés y eso es legítimo. Extraé cada turno con `<|im_start|>{voice}` … `<|im_end|>`.
- **Eximí los tramos entre comillas** dentro del turno de la voz (`"…"`, `'…'`, `«…»`): el teacher
  cita voces ajenas ahí adentro y eso no es spanglish del personaje.
- Marcador, con este listado EXACTO (nada de inventar la lista: estas son las palabras que se
  midieron fugándose o que no tienen uso rioplatense):
  `RX_SPANGLISH = re.compile(r"\b(worries|worry|differente|okay|okey|whatever|actually|basically|anyway|anyways|dude|buddy|guys|amazing|awesome|by the way|you know|i mean|it's|don't|can't)\b", re.I)`
- Error: `spanglish(<palabra>)` (la palabra tal cual la encontró, en minúscula).
- Si `permitir_spanglish=True`, la regla no corre. Cablealo en `main()` igual que las otras:
  `permitir_spanglish=sc.get("permitir_spanglish", False)`. **No agregues ese flag a ningún
  escenario hoy** — queda para el futuro.
- Ojo: `chequear` y `chance` NO están en la lista y no hay que agregarlas: son uso rioplatense real
  ("dejame chequear", "sin chance"). `look`/`man`/`bro` quedan afuera a propósito por los falsos
  positivos dentro de citas.

## Lo que NO hay que tocar
- **No cambies `require_tool`** ni la lógica de tools del validador: la obligación ya existe, lo que
  falta es que el prompt la diga.
- **No toques** `SCENARIOS`, `forzar_respuesta`, el modo `turnwise`, los teachers `opencode`/`cline`,
  `reasoning_effort`, `out/`, `ola2-freellmapi.sh` ni la ola en curso.
- **No agregues escenarios nuevos** en esta corrida (van en el brief siguiente).
- **No commitees ni pushees.** El orquestador revisa y commitea.
- Si algo del brief no coincide con el código, decilo en el reporte y seguí con el ancla textual.

## Tests: `scripts/test_fix119.py` (pytest, sin red)
- `spanglish` en turno de la VOZ → error. Mismo texto entre comillas dentro del turno de la voz →
  sin error. Mismo texto en el turno del USUARIO → sin error. Turno limpio → sin error.
  Con `permitir_spanglish=True` → sin error.
- `build_pair_prompt(..., permite_tools=True)` contiene la obligación de tool y **no** contiene
  `escribi solo tu mensaje`; con `hint="X"` el prompt incluye `PISTA OBLIGATORIA:`. Con
  `permite_tools=False` no aparece la obligación.
- `validate(SAMPLE_OK)` sigue devolviendo `[]`.
- El reintento de `main()`: monkeypatcheá `generate_paired` para que devuelva la primera vez una
  muestra sin tool y la segunda una con tool, y verificá que se guarda y que el hint llegó
  (`hint` no vacío). Sin red.

## Gates (correlos y pegá la salida cruda)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train
./venv/bin/python -m pytest scripts/test_reasoning_off.py scripts/test_fix119.py -q
./venv/bin/python scripts/gen_toolcalling_dataset.py --dry-run
```
`--dry-run` tiene que seguir dando `BUENA: PASA` / `MALA: RECHAZADA` con sus motivos.

## Reporte — `/home/chaos/harness-run/fix119/REPORT.md` (ruta ABSOLUTA)
1. Qué cambió, `archivo:línea`, con los fragmentos resultantes pegados.
2. Salida CRUDA de los dos gates.
3. Qué quedó sin hacer y qué viste y no tocaste. Si no llegaste a algo, decilo.
