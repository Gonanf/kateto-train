# fix-118 — Apagar `reasoning` en el cliente HTTP del teacher

## Meta (una sola cosa)
En `generate()` del cliente HTTP del teacher, mandar `reasoning_effort: "none"` en el body del request,
configurable por CLI, para que el teacher deje de devolver su **monólogo interno en `content`**.

## Causa raíz — MEDIDA por el orquestador, no la re-diagnostiques
`--teacher http --model auto:smart` sirve primero los modelos más "inteligentes" (el eje `smartest`;
`auto:smart` → `globalAxis=smart` → `orderChain(..., 'smartest')` en
`~/proyectos/freellmapi/server/src/services/router.ts:1020-1029`). Esos modelos son de razonamiento y
vuelcan el thinking **dentro de `content`**.

Medición sobre la ola en curso (workers w55-w62 de `ola2-freellmapi.sh`):
- **315 de 496 rechazos = `exc_RuntimeError`** → `generate_paired()` devolvió `None` tras agotar
  reintentos (`scripts/gen_toolcalling_dataset.py:1133-1136`) y el `raw` guardado es el monólogo.
- Pega justo en los escenarios CON tools: `voice_refused` 0/32, `minecraft_gate_negado` 0/23,
  `minecraft_gate_ok` 0/20, `events_lookup` 0/20, `chess_uci` 1/35.
- Los que sí pasan son los no-tool: `no_response_al_aire` 34/34 (100%), `no_response_otra_voz` 25/26.

**Prueba directa del fix (2 llamadas reales al gateway, escenario `voice_refused`)**
- SIN el parámetro → `content` empieza con `'First, the user wants me to generate a training data
  conversation for a character named Kateto...'` y `validate()` devuelve
  `['no_termina_en_im_end','pocos_turnos(0<2)','sin_voseo']`.
- CON `"reasoning_effort": "none"` → `content` empieza con
  `'<|im_user|> Doktor, querés que te pase un audio de un type que está hablando de la inflación en
  1989...<|im_start|>seco\ntool_call request_generation: {...}'` — o sea **una conversación de Kateto**.
  El monólogo sigue existiendo pero en el campo `reasoning` de la respuesta, NO en `content`.

El filtro que hoy lo caza es `RX_RAZONAMIENTO` + `RX_ANCLA_CONTRATO` (líneas 272-292): si el teacher sólo
devuelve razonamiento, no hay ninguna línea `PERSONA:`/`VOS:` y el recorte al contrato da vacío.
Hoy el reasoning se FILTRA después; lo que se pide es que no llegue a `content`.

## REGLA 0 — prohibido explorar
No recorras el repo. No leas otros scripts, ni `data/`, ni `docs/`, ni tests que no sean el que vas a
escribir. No hagas greps de descubrimiento. Abrí SOLO:
- `/run/media/chaos/terciario/proyectos/kateto-train/scripts/gen_toolcalling_dataset.py`
- el archivo de test nuevo que vas a crear (abajo)

**Contexto de host que tenés que respetar:**
- Hay **8 workers de generación VIVOS** contra el mismo gateway. **NO corras el generador, NO lances
  olas, NO toques `out/`, NO hagas llamadas de red.** La verificación end-to-end la corre el orquestador.
- Lo único que podés ejecutar son los dos gates locales de abajo (no tocan red).

## Cómo está hoy (pegado — no lo busques)
`scripts/gen_toolcalling_dataset.py`, `def generate(...)`, línea 874:

```python
def generate(base_url: str, model: str, prompt: str, key: str | None,
             *, temperature: float = 1.0, max_tokens: int = 1200, timeout: int = 240,
             retries: int = 4) -> str:
    """Pide una muestra. Reintenta en 429/503: los free tiers limitan seguido y
    freellmapi ya maneja cooldowns internos, pero la espera puede superarlos."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
```

Y el parser (línea ~1061) y los call sites en `main()` (línea ~1124):

```python
                elif args.teacher == "http":
                    maestro_base = lambda p: generate(args.base_url, model, p, key)
```

## Cambio pedido (3 items, nada más)

### 1. `reasoning_effort` en el body
- `generate()` suma el parámetro keyword-only `reasoning: str = "none"`.
- El body del request suma la clave **top-level** `"reasoning_effort": reasoning`.
  Es la forma canónica del gateway (`REASONING_EFFORTS = ['none','minimal','low','medium','high']` en
  `~/proyectos/freellmapi/server/src/lib/sampling-params.ts:32`). **No** uses la forma de objeto
  `reasoning: {"effort": ...}`: en el gateway es sólo un fallback.
- En `main()`, un flag nuevo `--reasoning` con `choices=["none","minimal","low","medium","high"]`,
  `default="none"`, que se pasa a `generate()` en el call site de `teacher == "http"`.
- Ojo: el flag `--dry-run` YA existe en el parser del generador. No lo toques ni lo redefinas.

### 2. `max_tokens` del teacher HTTP: 1200 → 2500
Medido: con `reasoning_effort: none` y `max_tokens: 1200` la respuesta se corta **a mitad del JSON de un
`tool_call`** (el validador lo marca `tool_call_sin_resultado`), porque el razonamiento sigue corriendo
— sólo que ahora en el campo `reasoning` de la respuesta — y consume presupuesto. Cambiá el default
`max_tokens: int = 1200` de `generate()` a `2500`. Una sola línea.

### 3. Test nuevo: `scripts/test_reasoning_off.py`
- pytest (el venv del repo lo tiene: `./venv/bin/python -m pytest`), **sin red**: monkeypatcheá
  `urllib.request.urlopen` y capturá el body que arma `generate()`.
- Casos: (a) por default el body incluye `reasoning_effort == "none"`; (b) con `reasoning="high"` el
  body lleva `"high"`; (c) el parser acepta `--reasoning high` y rechaza un valor inválido.
- Un test que sólo mira `sys.argv` o el código fuente no sirve: tiene que inspeccionar el **body real**
  que se manda.

## Lo que NO hay que tocar
- **No toques el validador** (`validate()`, `RX_RAZONAMIENTO`, `RX_ANCLA_CONTRATO`, `SCENARIOS`, las
  reglas por escenario). El filtro se queda como está: la red de seguridad. El pedido es no generar el
  monólogo, no dejar de detectarlo.
- **No agregues** lógica de reintento nueva, ni `reasoning` en los teachers `opencode`/`cline`, ni un
  `--reasoning` para ellos.
- **No toques** `out/`, la ola en curso, ni `ola2-freellmapi.sh`, ni nada fuera de los dos archivos.
- **No commitees y no pushees.** El repo ya estaba sucio antes de esta corrida; el orquestador revisa y
  commitea. Si alguien commiteó, decilo en el reporte.
- Si te parece que algo más debería cambiar, **decilo en el reporte, no lo cambies.**

## Gates (correlos y pegá la salida cruda)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train
./venv/bin/python -m pytest scripts/test_reasoning_off.py -q
./venv/bin/python scripts/gen_toolcalling_dataset.py --dry-run
```
El `--dry-run` tiene que seguir imprimiendo `BUENA: PASA` y `MALA: RECHAZADA` con sus errores: es la
prueba de que no rompiste el validador.

## Reporte (el artefacto de esta corrida)
Escribilo en **`/home/chaos/harness-run/fix118/REPORT.md`** (ruta ABSOLUTA, fuera del repo). Contenido:
1. Qué cambió, `archivo:línea`, con el `body` resultante pegado.
2. La salida CRUDA de los dos gates (comando + resultado, sin resumir).
3. Cualquier cosa del brief que no haya coincidido con el código real, y cualquier cosa que hayas visto
   y NO hayas tocado.
4. Si no llegaste a algo, decilo tal cual en vez de darlo por hecho.
