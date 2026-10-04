# Brief Fase 3 — Diagnosticar el `coverage` bajo y cerrar Gates B3/C/D

> **Repo:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Antecede:** `docs/opd-brief-phase2-metrics.md`
> **Estado:** la Fase 2 **NO aterrizó**. Leer §1 antes de tocar nada.

---

## 1. Veredicto de la Fase 2 (verificado por el orquestador, con evidencia)

| Gate | Estado | Evidencia |
|---|---|---|
| B3 (loop ≥30 pasos) | ❌ **NO** | `out/opd/run0/metrics.jsonl` tiene **1 línea**, no 30 |
| C (decisión por `coverage`) | ⚠️ sin dato válido | el único número válido es `0.0256` (2.6%) con loss **NaN** |
| D (baseline `lr=0`) | ❌ **NO** | no existe `metrics_baseline.jsonl` ni `run0-baseline/` |
| Reporte | ❌ **NO** | `out/opd/REPORT.md` no existe |

### 1.1 El "landed" del runner fue un **falso positivo** (importante)

Timestamps:

```
out/opd/run0/metrics.jsonl    04:16:00
out/opd/run0_train.log        04:16:00
rwkv_pipeline/opd_train.py    04:17:40   <- el fix es POSTERIOR
```

`metrics.jsonl` fue escrito **1 minuto 40 segundos ANTES** del fix que el propio harness aplicó después.
Su contenido es la salida del **bug que se corrigió**:

```json
{"step":1,"loss_rkl":126.68357,"coverage":19.6097,...}
```

Y el comentario del código (`opd_train.py`, ~línea 93) lo documenta textualmente:

> *"we must NOT replace -inf logits with 0 before exp() — exp(0)==1.0 would inject a fake unit
> probability at every teacher-only entry, inflating coverage > 1 and the RKL (measured:
> coverage=19.6, loss=126)."*

Además `coverage` se computa como `(ps * teacher_known * known_s).sum()` sobre una softmax: **vive en
`[0,1]`**. Un valor de `19.6` es **físicamente imposible** → confirma que el archivo es pre-fix.

**Acción obligatoria:** mover `out/opd/run0/` a `out/opd/run0-stale-prefix/` con un `README.md` que
explique por qué, y **no escribir nunca más en `out/opd/run0/`**. Correr todo en directorios **nuevos**.

### 1.2 El único dato válido que hay

`out/opd/_gateB3_train.log` (03:56, corrida de 5 pasos):

```
[opd] step 1/5 lr=1e-05 {"loss_rkl": NaN, "coverage": 0.0256, "teacher_entropy": 1.3136,
                         "student_entropy": 0.9119, "response_tokens": 24, "think": 0.0312}
```

Escala correcta (`0.0256` = 2.6% ✓, una fracción) pero **loss NaN**. Y **`coverage = 2.6%` está muy por
debajo del umbral de Gate C (`< 0.2` → parar y cambiar de backend)**. Ese es el resultado real de la
Fase 2: **el soporte común entre el vocab de RWKV y el del teacher es casi vacío.**

---

## 2. EL TRABAJO

### Fase 3.1 — Separar las dos causas del `coverage` bajo (barato, decidible)

`coverage` bajo puede venir de **dos** causas distintas y hay que distinguirlas **antes** de cambiar de
backend:

- **(a) Truncamiento del top-k del teacher.** El backend `router` solo devuelve `top_logprobs=K`
  (default 20). Si K es chico, la mayoría de los tokens del student no aparecen en la distribución del
  teacher **por truncamiento**, no por desalineación.
- **(b) Desalineación real de tokenizadores.** El vocab del student (65536, RWKV) y el del teacher
  (BPE) no coinciden a nivel de byte-string.

**Experimento (1 paso, mismo prompt fijo):** barrer `--teacher-topk` en `{20, 50, 200, 1000}` y reportar
`coverage` para cada valor.

- Si `coverage` **sube fuerte** con K → era truncamiento. Subir el default y seguir con `router`.
- Si `coverage` **casi no se mueve** → es desalineación de tokenizadores. Documentarlo y pasar al
  backend `hf` (§2.2), donde el teacher da logits completos y la probabilidad de un string arbitrario
  se puede calcular sin depender de que esté en su top-k.

Entregable: `out/opd/coverage_vs_topk.md` con la tabla medida.

**Ojo con el costo**: cada request del router devuelve `top_logprobs` más grande; con K=1000 puede
pesar y tardar. Medir el wall por request también.

### 2.2 Backend `hf` (si el barrido dice desalineación)

- El teacher `hf` necesita un modelo **en formato HF (no GGUF)**. Verificar **antes** cuál carga con el
  `transformers` instalado (`venv-unsloth-qwen` tiene **5.5.0**). Ojo: `MiniCPM5-1B` pide
  `transformers>=5.6`.
- **Regla dura: NO tocar `venv-unsloth-qwen`.** Si hace falta otra versión de transformers, crear un
  venv aparte (`venv-opd-teacher`) y documentarlo. No upgradear el venv de entrenamiento.
- Si ningún teacher HF es viable sin romper el venv, **reportarlo como bloqueo** y proponer alternativas
  con evidencia. No forzar.

### 2.3 Arreglar el loss **NaN**

`loss_rkl = NaN` en la corrida post-fix. Antes de cualquier run larga, un test de 1 paso que **asierte
finitud**:

```python
assert torch.isfinite(loss), f"loss no finito: {loss}"
```

Causas probables a descartar (no asumir): `-inf` residual en `t_raw`/`s_logits` que llega a
`log_softmax`; `EPS` insuficiente; `teacher_logp` vacío en toda la posición; `union` vacía.
**Reportar la causa encontrada con el valor que la delata**, no un "arreglado".

### 2.4 Recién ahora: la corrida completa (Gate B3 + C + D)

Con el coverage explicado y el loss finito:

- **Presupuesto de tiempo, calcularlo ANTES.** Medido: **~133 s por paso** (24 tokens, 8 prompts,
  backend router, padding bucket 64). Entonces `pasos × 133 s` tiene que entrar **holgado** en el
  `--timeout` del runner. La Fase 2 pidió 30 pasos (≈ 66 min) con un timeout de 45 min: **se cortó por
  eso.** O bajar a ~12 pasos, o subir el timeout, o reducir el trabajo por paso. Justificar la elección
  en el reporte.
- **Directorios nuevos**: `out/opd/run1/` (entrenado) y `out/opd/run1-baseline/` (lr=0).
- **Mismo set de prompts, mismo orden, misma semilla** en las dos. Guardar los prompts usados.
- **Gate D es obligatorio**: sin la curva `lr=0`, "el KL bajó" no significa nada.

---

## 3. Entregables

| # | Artefacto | Criterio |
|---|---|---|
| 1 | `out/opd/coverage_vs_topk.md` | tabla K → coverage, medida |
| 2 | `out/opd/run1/metrics.jsonl` | ≥12 líneas con `loss_rkl` **finito** y `coverage` en `[0,1]` |
| 3 | `out/opd/run1-baseline/metrics.jsonl` | misma cantidad de pasos, `lr=0` |
| 4 | `out/opd/REPORT.md` | las dos curvas, coverage, causa del NaN, veredicto de Gate C, **"Qué NO funcionó"** |

**Autochequeo antes de dar por cerrado** (esto es lo que falló la vez pasada):
- ¿`metrics.jsonl` tiene la cantidad de líneas que pide el brief, o quedó una sola?
- ¿`coverage` es una **fracción** (0–1)? Si da >1, el código es pre-fix → no vale.
- ¿El `mtime` del artefacto es **posterior** al `mtime` del código que lo produce?
- ¿Existe `REPORT.md`? ¿Existen **los dos** `metrics.jsonl`?

Si alguno falla: **no reportar aterrizaje.** Reportar el bloqueo con el número que lo demuestra.

---

## 4. Restricciones duras

1. **Todo con `systemd-run --user --scope -p MemoryMax=infinity`.** El background de Hermes está capado
   a **4 GiB** (`ProcessRegistry._worker_memory_max_bytes()`, techo duro) → **exit 137 = cgroup, no bug**.
   En la Fase 2 `run0_train.log` **no** tiene la línea del unit → esa corrida violó la regla.
2. **`WKV=triton`.** `cuda` cuelga en un lock stale y después no compila; `fla` no está instalado.
3. **Student en `cuda` (3.15/3.98 GB VRAM), teacher en CPU.** El router ya corre en CPU por default.
4. **No tocar `venv-unsloth-qwen`, `base/`, ni hacer push.**
5. **No inventar métricas.** Lo no medido se reporta "no medido" + motivo.
6. **Separar hecho de hipótesis.** El NaN y el coverage bajo son **hechos**; sus causas son
   **hipótesis** hasta que el experimento las confirme.
7. **No reusar `out/opd/run0/`** (contiene salida pre-fix).
