# Brief Eval-C — Métricas de generación (§2.2–§2.5, §2.8)

> **Repo:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Especificación completa:** `docs/eval-behavior-brief.md` §2.2 a §2.5 y §2.8
> **Salida:** `scripts/eval_generation.py` + `out/eval/run1/gen_<ckpt>.json`

## REGLA 0 — PROHIBIDO EXPLORAR

**No lances subagentes. No hagas fase de reconocimiento. No escribas un plan en prosa.**
Todos los archivos que necesitás están listados acá con ruta exacta y ya fueron verificados.
**Escribí el archivo y corré el smoke test.** Una corrida anterior se quemó entera explorando y
entregó cero archivos.

## Insumos YA EXISTENTES y VERIFICADOS (no los recrees, no los edites)

- `config/behavior_patterns.yaml` — patrones regex con campo `why`. 10/18 validados como vivos.
- `out/eval/prompts_v1.json` — set congelado. Usá la clave `generation` (80 items: 60 de CBB con
  `acto_de_habla` + 20 curados por registro). Cada item tiene `id` y `prompt`.
- `scripts/eval_behavior.py` — **reusá, no copies**: de acá importás
  `load_student`, `free_gpu`, `ensure_eval`, `percentile`, y las constantes `PROJECT`, `PROBE_SEED`.
- `rwkv_pipeline/opd_rollout.py` — `rollout_from_prompt(model, tok, prompt_text, max_len=..., temperature=..., top_p=...)`
  devuelve un objeto con `.response_ids`, `.finished`. `_decode_str(tok, ids)` decodifica.
  **Ojo**: `rollout_from_prompt` NO formatea el prompt — le pasás el texto ya formateado.

## Qué construir

`scripts/eval_generation.py` con CLI:

```
--ckpt <path>            (obligatorio)
--chat-format {world,chatml}
--prompts out/eval/prompts_v1.json
--patterns config/behavior_patterns.yaml
--out out/eval/run1/gen_<nombre>.json
--max-len 64 --temperature 0.80 --top-p 0.70 --seed 1337
--n-items N              (default: todos)
```

### Formateo del prompt (crítico, ya decidido — no lo rediseñes)

- `world`:  `f"User: {prompt}\n\nAssistant:"`
- `chatml`: `f"<|im_user|>{prompt.strip()}\n<|im_start|>seco\n"`

Justificación (no la re-litigues): cada checkpoint se evalúa con el formato en que fue entrenado.

### Métricas a computar sobre las respuestas generadas

1. **§2.2 Marcadores de vicio de asistente** — para cada patrón del YAML con `category: assistant_vice`
   (aperturas enlatadas, preguntas reflexivas, cierre de manual, `not_x_but_y`, hedging):
   `rate` = fracción de respuestas que matchean, más `n_matches` crudo.
   **Un rate alto es malo.** Reportá el `why` de cada patrón en la salida.
2. **§2.3 Marcadores de persona** — categoría `persona` (voseo, lunfardo, postura):
   `rate` por patrón + `density` = matches cada 100 tokens.
3. **§2.4 Distribución de longitud** — sobre `n_tokens` de respuesta:
   `median`, `p10`, `p90`, **`iqr`**, `mean`, `stdev`, y `cv = stdev/mean`.
   **Un IQR colapsado es un fallo**, aunque la mediana esté bien.
4. **§2.5 Disciplina de tool-call** — usá los items de `out/eval/prompts_v1.json` con
   `expects_toolcall: false` (registro debate/charla): reportá `toolcall_rate`, que **debe ser 0**.
   Contá cualquier substring `<tool_call`, `<|tool_call|>`, o `"name":` con `{`.
5. **§2.8 Degeneración** — `degenerate_rate`: fracción con un substring de 8+ chars repetido ≥3 veces.
   Reportá también `finished_rate` (¿terminó sola?) y `filler_rate` (¿terminó en el max_len?).

### Obligatorio en el JSON de salida

- `arch`, `chat_format`, `ckpt`, `seed`, `temperature`, `top_p`, `max_len`, `n_items`.
- `n_scored` **crudo** por métrica (no sólo la tasa).
- **`raw_responses`**: para CADA item, `{id, prompt, n_tokens, finished, text}`. Sin esto no se puede
  auditar a mano por qué se movió una métrica. Es obligatorio, no opcional.
- Un campo `notes` con cualquier cosa que no hayas podido medir.

## Verificación obligatoria antes de dar por terminado

1. **Smoke test**: corré con `--n-items 4` y confirmá que el JSON existe, tiene `raw_responses` con 4
   entradas y texto no vacío. Pegá el comando y el resultado en la respuesta final.
2. **`--n-items 4` con `--seed` fijo, dos veces → mismo `n_tokens` por item.** Si no, hay no-determinismo
   en el sampling y hay que declararlo en `notes`.
3. **Reportá `n_scored` crudo** junto a cada tasa.
4. Si una métrica da **0 en todos los items**, NO la borres: reportala con `n_scored` y decí en `notes`
   si creés que es un patrón muerto o un resultado real.

## Restricciones de entorno (ya sufridas, no las descubras de nuevo)

- Exportá antes de importar torch: `HSA_OVERRIDE_GFX_VERSION=10.3.0`,
  `PYTORCH_ALLOC_CONF=expandable_segments:True`, `TORCH_COMPILE_DISABLE=1`.
- **El background de Hermes está capado a 4 GiB** → cargar modelos >1B muere con exit 137 sin traceback.
  Corré con `systemd-run --user --scope -p MemoryMax=infinity --unit=<nombre> bash -c '...'`.
- Los checkpoints son chicos (0.4B ~900 MB, 1.5B ~3.05 GB). La VRAM es 3.98 GB.
- `WKV="triton"` (ya resuelto en `opd_rollout.py`; si ves JIT de CUDA, algo lo pisó).

## NO hagas

- No toques `config/behavior_patterns.yaml` ni `out/eval/prompts_v1.json`: están congelados.
- No implementes §2.1 (CBB) — ya está hecho en `scripts/eval_behavior.py`.
- No implementes §2.6 (separación de voces) ni §2.7 (anti-sycophancy) — son la Parte D.
- No reentrenes nada. No hagas OPD.
