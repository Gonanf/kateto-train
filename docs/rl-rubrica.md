# Rúbrica y Reward de RL para Kateto

Especificación técnica de la función de recompensa (reward) de RL para Kateto, basada en la reutilización estricta de los criterios de destilación de producción y la rúbrica de persona con juez semántico.

---

## 1. Qué se reusa y de dónde (`archivo:línea`)

Para no duplicar expresiones regulares ni divergir de los gates de producción, la capa determinista importa directamente las fuentes autoritativas:

- `pipelines/kateto_purge.py:282` (`REGLAS` dict con las 6 reglas canónicas) y `pipelines/kateto_purge.py:298` (`evaluar(text) -> list[str]`):
  - `placeholder_degenerado` (dura)
  - `andamiaje_en_dialogo` (dura)
  - `contrato_en_dialogo` (dura)
  - `exclamacion_prosodia` (blanda)
  - `simil_repetido` (blanda)
  - `charla_de_asistente` (dura)
  *Nota sobre la voz:* `kateto_purge.evaluar` sólo analiza turnos de la voz (`seco`). La entrada se envuelve en `wrap_turno()` simulando `<|im_start|>user...<|im_start|>seco\n{respuesta}<|im_end|>`.
- `scripts/gen_orpo_v8.py:25` (`RX_SERVIL`): patrón de complacencia y servilismo (usado en los pares ORPO v8).
  - Está versionado en git (commit `08f0ebd`) porque el reward depende de él.
  - `criterios.py` lo importa duro: si falta, el import falla con error, no degrada en silencio (sin él la separación medida cae de 63.64% a 8.36%).
- `scripts/gen_toolcalling_dataset.py:142, 145, 149` (`RX_TUTEO_INEQUIVOCO`, `RX_TUTEO_HOMOGRAFO`, `RX_TUTEO_GUARDA`): detección de tuteo con normalización NFC y guarda de homógrafos (`_check_tuteo()`).
- `scripts/analyze_eval29_coherence.py:28` (`check_degenerate(text) -> tuple[bool, int]`): detección de bucle y repeticiones de n-gramas (heurístico de la celda 9 del kernel eval29).
- `scripts/validate_orpo_pairs.py:29` (`check(row, lineno)`): validador estructural de pares con voseo y delimitadores ChatML.

---

## 2. Tabla de criterios y pesos

| Criterio | Tipo | Penalización / Efecto | Fuente (`archivo:línea`) |
|---|---|---|---|
| `purga:charla_de_asistente` | Dura | `score = 0.0` | `pipelines/kateto_purge.py:298` |
| `purga:placeholder_degenerado` | Dura | `score = 0.0` | `pipelines/kateto_purge.py:298` |
| `purga:contrato_en_dialogo` | Dura | `score = 0.0` | `pipelines/kateto_purge.py:298` |
| `purga:andamiaje_en_dialogo` | Dura | `score = 0.0` | `pipelines/kateto_purge.py:298` |
| `servil` | Dura | `score = 0.0` | `scripts/gen_orpo_v8.py:25` |
| `tuteo` | Dura | `score = 0.0` | `scripts/gen_toolcalling_dataset.py:142` |
| `degenerado` | Dura | `score = 0.0` | `scripts/analyze_eval29_coherence.py:28` |
| `centinela` | Dura | `score = 0.0` | `scripts/rl/criterios.py:126` (`<\|wait\|>`, `<\|no_response\|>`, vacío) |
| `purga:exclamacion_prosodia` | Blanda | `-0.2` | `pipelines/kateto_purge.py:298` |
| `purga:simil_repetido` | Blanda | `-0.2` | `pipelines/kateto_purge.py:298` |
| `largo_fuera_de_banda` | Blanda | `-0.2` | `scripts/rl/criterios.py:147` (`< 40` o `> 900` caracteres) |

- **Regla dura:** Si cualquier regla dura dispara (o cualquier regla no prevista agregada dinámicamente a `kateto_purge.REGLAS`), `score = 0.0`.
- **Reglas blandas:** Cada regla blanda resta `0.2` desde `1.0`, con piso en `0.0`.
- **Limpia:** Sin reglas disparadas -> `score = 1.0`.

---

## 3. Rúbrica de persona (4 criterios, 1..5)

Evaluada mediante `score_persona_judge(respuesta, contexto, *, judge=None) -> dict | None`. `judge` es un callable inyectable. Si `judge is None` o falla, devuelve `None` de forma segura sin propagar errores ni bloquear.

1. `registro_seco`:
   - **1:** Empalagoso, servicial de chatbot genérico, edulcorado o excesivamente formal ("¡Hola! Con mucho gusto...").
   - **5:** Rioplatense natural, seco, directo, sin cortesías vacías ni relleno.
2. `no_servil`:
   - **1:** Sumiso, disculpándose innecesariamente, halagando al usuario o dándole la razón a ciegas ("como asistente", "tenés toda la razón").
   - **5:** Firme, sobrio, no adula ni pide disculpas indebidas.
3. `adversarial_honesta`:
   - **1:** Acepta premisas falsas, valida datos erróneos o se deja manipular para complacer al usuario.
   - **5:** Defiende la verdad factual con firmeza, marca errores sin agresividad pero sin concesiones falsas.
4. `coherencia_con_turno_previo`:
   - **1:** Se va de tema, contradice el diálogo anterior o alucina incoherencias.
   - **5:** Estrictamente relevante y perfectamente hilado con lo que el usuario acaba de decir.

### Combinación Total
$$\text{score\_total} = \begin{cases} 0.7 \times \text{score\_det} + 0.3 \times \frac{\text{judge\_medio} - 1.0}{4.0} & \text{si hay juez válido} \\ \text{score\_det} & \text{sin juez} \end{cases}$$

---

## 4. Guarda anti-reward-hack

Implementada en `scripts/rl/anti_hack.py::evaluar_progreso(baseline, candidato, n_seeds)`:

- **Multi-semilla obligatoria:** `n_seeds < 3` -> `"no_concluyente"`.
- **Progreso en reward:** `candidato.reward_medio <= baseline.reward_medio` -> `"rechaza"`.
- **No regresión en coherencia:** `candidato.coherentes_pct < baseline.coherentes_pct` -> `"rechaza"`.
- **Control de verborragia (longitud):** `candidato.longitud_media_chars > 1.2 * baseline.longitud_media_chars` -> `"rechaza"`.
- **Seguridad en datos:** Claves faltantes producen `"rechaza"` con `dato_faltante:<clave>`, nunca `KeyError`.

---

## 5. Números del Replay

Comando exacto de medición:
```bash
venv/bin/python scripts/rl/validar_reward.py --json
```

### Resultados sobre `kateto_orpo_v8.jsonl` (550 pares)

| Categoría | Pares | Separación estricta (`chosen > rej`) | Tasa de separación | Falsos positivos (`chosen < 1.0`) | Rejected detectados |
|---|---|---|---|---|---|
| `anti_servilismo` | 200 | 200 | **100.0%** | 0 | 200 |
| `reparacion` | 150 | 150 | **100.0%** | 0 | 150 |
| `debate` | 200 | 0 | **0.0%** | 1 | 0 |
| **TOTAL** | **550** | **350** | **63.64%** | **1** (0.18%) | **350** |

### Resultados sobre `gen29_rows.jsonl` (eval29)

| Conjunto | `degenerate: True` (score medio) | `degenerate: False` (score medio) | Filas deg | Filas non-deg |
|---|---|---|---|---|
| `eval_prompts_jsonl` | **0.0000** | **0.9500** | 20 | 40 |
| `prompts_v1_generation` | **0.0000** | **0.8333** | 24 | 36 |
| `dummy` | **0.0000** | - | 1 | 0 |
| **TOTAL** | **0.0000** | **0.8947** | **45** | **76** |

Separación perfecta: el 100% de las respuestas degeneradas obtienen score `0.0`.

---

## 6. Qué falta

1. **Juez de persona conectado:** En la categoría `debate` (200 pares, 36.4% del dataset), el rechazado afirma un dato falso con buena prosodia y sin servilismo léxico. Por diseño, las regex deterministas no detectan falsedades semánticas. La separación en esta categoría requiere conectar `score_persona_judge` con un endpoint de LLM juez (criterio `adversarial_honesta`).
2. **Entrenamiento de RL:** Este brief implementa el reward, el replay de validación, la exportación de pares y las guardas anti-hack. La corrida de entrenamiento con RL/ORPO/DPO se ejecuta en la card siguiente.
