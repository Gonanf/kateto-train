# Brief Fase 2 — Gates B3/C/D: el loop de OPD con métricas

> **Repo:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Antecede:** `docs/opd-brief-phase1-unblock.md` (Gates A/B1/B2 ya resueltos — ver §1)
> **Ejecutor:** harness externo · **Entregable clave:** `out/opd/run0/metrics.jsonl` con curva KL + baseline

---

## 1. Ya resuelto — NO re-hacer

| Gate | Estado | Evidencia |
|---|---|---|
| **A** (teacher) | ✅ VERDE | `teacher_self_coverage: 0.9946`, 0.55s, backend router `MiniCPM5-1B` |
| **B1** (longitud T) | ✅ HECHO | `out/opd/tlen_probe.md` + `_tlen_probe_run.log` (verbatim, 56 líneas) |
| **B2** (rollout) | ✅ VERDE | 8 prompts, `mean_response_len=24.0`, `mean_tokens_per_s=2.1`, `think_fraction_mean=0.0312`, `finished_fraction=0.0`, exit 0 |

**Hallazgo de B1 que hay que respetar (medido, no teorizado):** el chunk del kernel es **K=8**, y `y = th.empty_like(v)` con `T//K` truncado deja las posiciones de cola **sin escribir** cuando `T % 8 != 0`. `_next_logits` tomaba `logits[0,-1]` = justo una posición de cola → **NaN** → `softmax` NaN → `multinomial` cuelga. Ese era el hang. El fix (`_pad_len` a múltiplo de 64 + leer la posición real) ya está aplicado y B2 lo confirma.

**Costo del fix, ya visible:** el padding a bucket 64 bajó el throughput a **2.1 tok/s** (contra 4.4–4.6 sin padear) porque acorta las secuencias al bucket. Es el precio documentado; no es un bug.

---

## 2. EL TRABAJO: correr el loop y medir

### Gate B3 — el loop corre y produce métricas

```
out/opd/run0/metrics.jsonl   con: step, loss_rkl, coverage, teacher_entropy,
                                   student_entropy, response_tokens, think_frac, wall_s
```

Pasos: **≥30**, sobre un set de prompts **fijo y reproducible** (mismo orden, misma semilla).
Backend de teacher: `router` (Gate A lo validó). `--teacher-model MiniCPM5-1B`.
`--steps 30 --max-len 24 --n-prompts 8`.

**Gate C — decisión sobre `coverage`.** Mirar los primeros 5 pasos:
- `>= 0.5` → seguir
- `0.2 – 0.5` → seguir, documentar el techo
- `< 0.2` → **parar**, cambiar de backend de teacher (probar `hf`), y reportar. **No maquillar el número.**

`coverage` es la fracción de masa del teacher representable en el soporte del student, alineada por byte-strings (recordá: el student tiene vocab 65536 propio de RWKV, el teacher uno BPE distinto; el solapamiento puede ser bajo y **eso es un resultado, no un fracaso**).

### Gate D — baseline obligatorio `lr=0`

**Correr exactamente el mismo loop con `--lr 0`** y guardar la curva aparte (`metrics_baseline.jsonl`).
Sin esa curva, "el KL bajó" no significa nada: el KL puede bajar sola porque el student se mueve hacia el teacher por el propio optimizador o porque las respuestas cambian de longitud.

Reportar las **dos curvas superpuestas** o en una tabla paralela.

---

## 3. Observaciones de B2 que van a afectar la Fase 2 (mirarlas, no ignorarlas)

1. **`finished_fraction = 0.0`**: las 8 respuestas pegaron `max_len=24`, ninguna terminó sola. Es esperable — el base G1j en formato World no emite el token de fin en este setup — pero significa que **toda respuesta está truncada a 24 tokens**. Si el KL se mide sobre texto truncado, hay que decirlo en el reporte. Considerar subir `--max-len` a 48 como corrida secundaria y reportar si cambia la conclusión.
2. **`think_fraction_mean = 0.0312`** (3.1%). El canal `<think>` del 1.5B se dispara con input en español (documentado en `out/opd/phase01_localgen.md` §5.1). Está bajo, pero **hay que reportar el número y probar `--block-think`** para ver el efecto en el KL. No elegir un default en silencio.
3. **El prompt set importa.** `data/kateto_qa.jsonl` tiene 500 líneas con `question`/`answer`. Usar un subconjunto **fijo** (primeras N, o semilla fija) y **reportar cuales**. Si el set cambia entre la corrida entrenada y el baseline, la comparación no vale.

---

## 4. Entregables

| # | Artefacto | Verificación esperada |
|---|---|---|
| 1 | `out/opd/run0/metrics.jsonl` | ≥30 líneas, cada una con `loss_rkl` y `coverage` no nulos |
| 2 | `out/opd/run0/metrics_baseline.jsonl` | misma cantidad de pasos, `lr=0` |
| 3 | `out/opd/REPORT.md` | las dos curvas, coverage, entropías, tok/s, device de student y teacher, y una sección **"Qué NO funcionó"** |

El reporte tiene que responder, con números: **¿bajó el KL entrenado respecto del baseline `lr=0`, o no?** Si no bajó, eso es el resultado y se reporta así.

---

## 5. Restricciones duras (recordatorio — ya se rompieron una vez)

1. **TODO corre con `systemd-run --user --scope -p MemoryMax=infinity`.** El background de Hermes está capado a **4 GiB** por cgroup (`ProcessRegistry._worker_memory_max_bytes()`, techo duro, `TERMINAL_LOCAL_MEMORY_MAX_MB` sólo lo aprieta). Sin esto: **exit 137 = SIGKILL de cgroup, no bug del código**.
2. **`WKV` = `triton`.** `cuda` cuelga en un lock stale y después no compila (nvcc flags a hipcc, sin gfx1034, `amd_hip_bf16.h` roto). `fla` no está instalado.
3. **El student vive en `cuda` (3.15/3.98 GB VRAM); el teacher en CPU.** El router ya corre en CPU por default — no cambiar presets para forzar GPU.
4. **No tocar `venv-unsloth-qwen`, `base/`, ni hacer push.**
5. **No inventar métricas.** Lo no medido se reporta como "no medido" + motivo.
6. **Separar hecho de hipótesis** en cada afirmación del reporte.
7. **Ningún KL se reporta sin el baseline `lr=0`** (Gate D).

---

## 6. Plantilla de corrida

```bash
systemd-run --user --scope -p MemoryMax=infinity --unit=opd-$(date +%s) bash -c '
  cd /run/media/chaos/terciario/proyectos/kateto-train
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TORCH_COMPILE_DISABLE=1
  # entrenado
  venv-unsloth-qwen/bin/python -u rwkv_pipeline/opd_train.py \
    --backend router --teacher-model MiniCPM5-1B \
    --steps 30 --max-len 24 --n-prompts 8 --lr 1e-5 \
    --run-dir out/opd/run0
  # baseline (mismo set, mismos prompts, lr=0)
  venv-unsloth-qwen/bin/python -u rwkv_pipeline/opd_train.py \
    --backend router --teacher-model MiniCPM5-1B \
    --steps 30 --max-len 24 --n-prompts 8 --lr 0 \
    --run-dir out/opd/run0-baseline
'
```
