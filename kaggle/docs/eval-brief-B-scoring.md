# Brief Eval-B — Scoring de CBB (§2.1) + Gates

> **Repo:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Especificación completa:** `docs/eval-behavior-brief.md` §2.1 y §3
> **Insumo ya listo:** `out/eval/prompts_v1.json` (Parte A, verificada)
> **Este brief es la PARTE B de tres.** Generación + reporte son la Parte C — **no las hagas**.

---

## 0. Reglas de ejecución

1. **PROHIBIDO explorar.** No subagentes, no "fase de reconocimiento". Rutas exactas abajo. **Escribí.**
2. **PROHIBIDO planificar en prosa.** No escribas un plan largo.
3. Alcance: **UN script** y su salida JSON.
4. **PROHIBIDO asumir** los resultados de §3.1 y §3.2 — **se miden** y se reportan con el número.

---

## 1. Entregables

| # | Archivo |
|---|---|
| 1 | `scripts/eval_behavior.py` — **solo la parte de scoring** (dejá placeholders `NotImplementedError` para las métricas de generación de las §2.2–§2.8; las hace la Parte C) |
| 2 | `out/eval/run1/cbb.json` — accuracy + márgenes de §2.1, baseline vs candidato |
| 3 | `out/eval/run1/probes.md` — resultados de §3.1 (lo_r) y §3.2 (formato) |

---

## 2. Qué computar (§2.1 — la métrica estrella)

Para cada par de `data/hf/che-boludo-benchmark/datasets/dataset_cbb_dpo.json` (**808**):

```
score(seq) = media de log P(token_i | prompt + seq[:i])   sobre los tokens de `seq`
accuracy   = fraccion de pares donde score(chosen) > score(rejected)
```

**Reusar** `rwkv_pipeline/opd_rollout.py`:
- `build_student(ckpt, device="cuda", lo_r=0)` → pero ver §3.1 **antes** de fijar `lo_r`.
- `rescore_response(student, tokenizer, prompt_ids, response_ids, ctx_len=512)` → devuelve las filas de
  logits alineadas a la respuesta. De ahí sale `log P`.
- **Regla del kernel, no negociable**: todo forward pasa por `_pad_len` (bucket 64) y se lee la
  **posición real**. Sin eso el kernel triton devuelve NaN en la cola cuando `T % 8 != 0`
  (`out/opd/tlen_probe.md`, medido).

Reportar en `cbb.json`:
- `accuracy` global, para **baseline** y **candidato**.
- `n_pairs`, y `n_scored` (si alguno se descarta, **decir por qué** y cuántos).
- **El margen**, no sólo el signo: `media`, `mediana`, `p10`, `p90` de `score(chosen) - score(rejected)`.
  Un accuracy de 0.5 con margen 0.001 es azar; con margen grande es preferencia real.
- **Estratificado** por `pragmatica.acto_de_habla` y por `metadata.generacion` (las etiquetas están en
  `out/eval/prompts_v1.json` → `cbb_pairs.strata`, indexadas por `id`). Un estrato con `n < 20` va
  marcado **`underpowered`**.
- `seed` y los parámetros de sampling usados.

**Checkpoints** (arquitecturas medidas — misma arch, no mezclar):
- baseline: `models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth` (24L/1024)
- candidato: `out/rwkv_kateto_base/rwkv-1.pth` (24L/1024)

---

## 3. Los dos experimentos que hay que MEDIR (no asumir)

### 3.1 ¿`lo_r>0` perturba el modelo? ¿`model.train()` rompe el determinismo?

`opd_rollout.build_student` tiene **dos** cosas sospechosas para una eval:

```python
def build_student(..., lo_r: int = 16, ...):   # default 16 -> envuelve en LoRA
    model.to(...)
    model.train()                              # <- deja DROPOUT ACTIVO
```

**Experimento (una corrida, 8 pares, semilla fija):**

| variante | `lo_r` | modo | score esperado |
|---|---|---|---|
| A | **0** | `.eval()` | referencia |
| B | **0** | `.train()` | ¿difiere? (dropout) |
| C | **16** | `.eval()` | ¿difiere? (LoRA a init) |

Reportar si `score(A) == score(B)` y si `score(A) == score(C)`, **con los números** (máx abs diff).

Interpretación esperada **a confirmar**: en PEFT estándar `lora_B` se inicializa en ceros, así que C
debería coincidir con A. **Si no coincide, es un hallazgo y hay que usar `lo_r=0` obligatoriamente.**
Y `model.train()` activa dropout (`lora_dropout=0.01`), así que B probablemente difiera — **si difiere,
el script tiene que forzar `.eval()`** o toda la eval deja de ser determinista.

**No escribas la conclusión antes de correr el experimento.**

### 3.2 ¿Qué formato habla cada checkpoint?

Hay una contradicción real en el repo que hay que resolver con una medición, no con un argumento:

- `data/rwkv_kateto_base_train.jsonl` está en **ChatML** (`<|im_user|>` / `<|im_start|>{voice}`).
- Pero el modelo **base** con ChatML produce un **bucle degenerado** (verificado en
  `out/opd/phase01_localgen.md` §5.1); el base habla **RWKV World** (`User: {q}\n\nAssistant:`).

**Sonda de auto-detección**: para cada checkpoint, scorear la **misma** respuesta canónica bajo los dos
formatos de prompt y **quedarse con el de mayor logprob medio**. Es determinista y barato.
Reportar para cada checkpoint: `logprob_world`, `logprob_chatml`, formato elegido, y el margen.

**Si el margen es chico (< 0.05 nat/token), decirlo** — significa que la sonda no discrimina y hay que
elegir el formato por otra vía (documentando cuál).

Usá el formato detectado en §2.1 para cada checkpoint. **Reportá cuál usaste para cada uno.**

---

## 4. Gates obligatorios

- **Gate 3 (el más importante) — auto-consistencia.** Correr §2.1 con **el mismo checkpoint en los dos
  lados** (`--baseline X --candidate X`). El accuracy **tiene que dar ~0.5**. Si da 0.9, hay un bug en
  el scoring y **todo lo demás es papel mojado**. Reportar el número.
  *Ojo: los dos lados tienen que usar el MISMO formato detectado, si no la comparación se contamina.*
- **Gate 1 — determinismo.** Dos corridas idénticas del scoring dan `cbb.json` idéntico (salvo
  timestamps). Reportar el diff.

---

## 5. Entorno

- **`systemd-run --user --scope -p MemoryMax=infinity`** para todo lo que cargue un modelo. El
  background de Hermes está capado a **4 GiB** por cgroup → **exit 137**, que no es un bug del código.
- **`WKV=triton`** (el import del modelo lo setea solo; no lo pises). `cuda` cuelga en un lock stale y
  no compila en gfx1034; `fla` no está instalado.
- 0.4B en bf16 entra en VRAM (pico medido 1.01 GB). Un solo proceso usando la GPU a la vez.
- **No tocar `venv-unsloth-qwen`, `base/`, ni hacer push.**
- **Presupuesto**: medí el wall por par con 8 pares **antes** de correr los 808. 808 pares × 2 forwards
  de scoring es barato comparado con generar, pero dimensioná con el número, no con la intuición.

---

## 6. Autochequeo antes de cerrar

- ¿`cbb.json` tiene `accuracy` **con `n` crudo**, o un número pelado?
- ¿Corriste **Gate 3** y está el número?
- ¿Los tres resultados de §3.1 están **medidos** (con valores), o afirmados?
- ¿La sonda de §3.2 reporta los dos logprobs y el margen?
- ¿`mtime` de `cbb.json` posterior al de `scripts/eval_behavior.py`?

Si alguno falla: **no reportes aterrizaje.** Reportá el bloqueo con el número que lo demuestra.
