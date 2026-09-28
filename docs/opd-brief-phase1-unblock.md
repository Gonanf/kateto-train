# Brief de continuación — OPD Fase 0.2–0.5 (desbloqueo del rollout)

> **Repo:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Creado:** 2026-09-12 · **Ejecutor:** harness externo
> **Antecede:** `docs/opd-brief-phase0.md` (leer §1 y §2 primero — hechos verificados del entorno)
> **Estado:** el primer harness escribió los 3 módulos pero **nunca los corrió**. Los probes que dejó (`out/opd/_*.log`) están **vacíos**. Todo lo de abajo es de ejecución real posterior.

---

## 1. Lo que YA está arreglado (no re-tocar, no re-descubrir)

El orquestador corrió los gates y encontró una cadena de 4 bloqueos reales. **Estos 4 ya están parchados en el árbol; no los reviertas ni los "redescubras".** Leé los diffs antes de tocar esos archivos.

| # | Archivo | Bug | Fix aplicado |
|---|---|---|---|
| 1 | `teacher_client.py` | `RouterTEACHERBackend._chat` mandaba `top_logprobs` **sin `logprobs: true`** → llama.cpp responde **HTTP 400**. | Se agregó `"logprobs": True` al payload. |
| 2 | `teacher_client.py` | `decode_token` hacía `token.encode("utf-8")` sobre el campo **string**. llama.cpp devuelve tokens **parciales de UTF-8**, así que eso corrompe bytes. Todo el alineamiento del reverse KL es por byte-strings (brief §2) → la métrica `coverage` habría sido basura. | Nuevo helper `_token_str()` que empaqueta el campo **`bytes`** (autoritativo) como latin-1; `decode_token` hace `.encode("latin-1")`. Round-trip verificado byte-exacto (33/33 entradas, 0 mismatches). |
| 3 | `opd_rollout.py` línea ~38 | `os.environ.setdefault("WKV", "cuda")` — **este era el bloqueo principal.** | Ahora `"triton"`. Ver §2. |
| 4 | `opd_rollout.py` `_get_rwkv7()` | No seteaba el contrato de entorno que `train.py` sí setea antes de importar el modelo. | `setdefault` de `WKV`/`RWKV_MY_TESTING`/`RWKV_TRAIN_TYPE`/`FUSED_KERNEL`/`RWKV_FLOAT_MODE`/`RWKV_JIT_ON`. |

También se limpiaron **locks stale** de `~/.cache/torch_extensions/py311_cpu/*/lock` (del 2026-09-03, 0 bytes, en directorios de build vacíos).

### 1.1 Gate A — VERDE ✅

```
$ python rwkv_pipeline/teacher_client.py --backend router --teacher-model MiniCPM5-1B --top-k 10
GATE A RESULT: {"backend":"router","model":"MiniCPM5-1B","top_k":10,
                "wall_s":0.552,"teacher_self_coverage":0.9946}
```

### 1.2 Gate B — ROJO ❌ (es lo único que falta)

`opd_rollout.py` llega hasta el rollout y **se cuelga en el primer forward**.

```
[03:25:33] 3. _infer_arch  -> {'n_embd': 2048, 'n_layer': 24, 'head_size': 64, 'vocab': 65536}
[03:25:33] 4. _get_rwkv7() -> x070 Wind Triton Kernel Mode; clase OK
[03:25:57] 5. build_student -> OK en 23.8s
             trainable params: 14,155,776 || all params: 1,541,824,512 || trainable%: 0.9181
             info={'device':'cuda','dtype':'bf16','missing':0,'unexpected':0}
[03:25:57] 6. n_trainable_tensores=288, primer param device=cuda:0 dtype=bfloat16
[03:25:57] 7. rollout de 4 tokens -> COLGADO
Timeout (0:02:00)!
  File "rwkv_pipeline/opd_rollout.py", line 235 in _sample_next
  File "rwkv_pipeline/opd_rollout.py", line 257 in rollout_from_prompt
```

El proceso queda a **0.0% CPU y 0% GPU** con el modelo cargado: no computa, espera. `wchan=hrtimer_nanosleep` en el thread principal.

---

## 2. Por qué `WKV="cuda"` era el bloqueo principal (contexto que hace falta entender)

`RWKV-PEFT/rwkvt/operator/rwkvop.py` decide **en tiempo de import**:

```
line  28:  if os.environ["WKV"] == 'fla':        # -> from rwkvfla.ops.rwkv7 import chunk_rwkv7
line  90:  elif os.environ["WKV"] == 'triton':   # -> kernel Triton (el que anda en este box)
line 296:  else:                                 # -> torch.utils.cpp_extension.load(...) JIT NATIVO
```

Con `WKV="cuda"` se toma el `else` y ocurren **dos desastres en secuencia**:

1. **Primero cuelga.** `load()` en `rwkvop.py:380` entra a `torch.utils.cpp_extension._jit_compile` → `torch/utils/file_baton.py:50` `wait()` y espera **para siempre** un lock de build que quedó stale. Eso explica exactamente por qué el `_opcompile.log` del harness anterior cortaba en `importing rwkvt.rwkv7.model ...` y por qué el rollout se colgaba a 0% de CPU. **El harness anterior pisó este hang y siguió sin diagnosticarlo.**

2. **Después falla la compilación** (verificado, con los locks limpios, `rwkvop.py:380`):

```
clang++: error: unknown argument: '-res-usage'          <- flags de nvcc pasados a hipcc
clang++: error: unknown argument: '--use_fast_math'
clang++: error: unknown argument: '-Xptxas'
clang++: error: unknown argument: '--extra-device-vectorization'
  + --offload-arch=gfx900,gfx906,...,gfx1201  <- NO incluye gfx1034 (esta GPU)
  + /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h:350  static_assert(sizeof(__hip_bfloat16[2]) == sizeof(__bf16_2))  [4 == 2]
RuntimeError: Error building extension 'rwkv7_clampw'
```

**El path nativo `cuda` es inusable en este box. No volver a intentarlo.** Las tres opciones reales:

| `WKV` | Estado |
|---|---|
| `fla` | ❌ `ModuleNotFoundError: No module named 'rwkvfla'` (verificado; no está en `venv-unsloth-qwen`) |
| **`triton`** | ✅ **la que funciona.** Es lo que usa `train_lora_base.sh` (`--op triton`); el patch de LDS `K=8`/`num_stages=1` para gfx1034 vive ahí |
| `cuda` | ❌ hang + build imposible |

---

## 3. La restricción que hay que respetar en TODA corrida: cgroup de 4 GiB

**Los procesos de background de Hermes corren en un cgroup capado a 4 GiB.** Verificado:

```
Memory cgroup out of memory: Killed process (python)
  total-vm:19.6 GB, anon-rss:4.2 GB
  oom_memcg=.../hermes-worker-proc_7c47d96f3066.scope
```

El límite lo pone `ProcessRegistry` (`_worker_memory_max_bytes()` en `tools/process_registry.py`) y **es un techo duro**: `TERMINAL_LOCAL_MEMORY_MAX_MB` solo puede *apretarlo*, nunca aflojarlo (`min(memory.max del cgroup padre, mitad de la RAM física)`).

Cargar el checkpoint 1.5B (3.05 GB en CPU + copia) **revienta ese límite**. Todas las corridas del student tienen que ir fuera del cgroup:

```bash
systemd-run --user --scope -p MemoryMax=infinity --unit=opd-$(date +%s) bash -c '<comando>'
```

Verificado que sobrevive 7 GB de asignación. **Sin esto, la corrida muere con exit 137 (SIGKILL), no con un error de Python** — si ves 137, no es un bug del código.

---

## 4. EL TRABAJO: desbloquear el rollout

### 4.1 Hipótesis principal (fundamentada, no confirmada)

`rollout_from_prompt` llama `_next_logits(student, token_ids, ctx_len)` **una vez por token generado**, y `_next_logits` hace `forward_normal` sobre **la secuencia completa** (prompt + generado). Dos problemas:

1. **Es O(T²)** y re-procesa el prefijo entero en cada paso. En un modelo **recurrente** esto es, además de lento, conceptualmente incorrecto: el estado recurrente es lo que evita reprocesar.
2. **La longitud T de cada forward es arbitraria** (15, 16, 17, 18, …) porque crece de a un token. El kernel Triton x070 es *chunked* con `CHUNK_LEN = 16` y `assert T % CHUNK_LEN == 0` aparece en los paths CUDA de `rwkvop.py`. `train.py` **siempre** entrena con `T = ctx_len = 512` (múltiplo de 16). Es muy plausible que un T no múltiplo de 16 cuelgue o corrompa el kernel chunked.

**El síntoma calza**: el hang aparece en el **primer punto de sincronización** (`F.softmax` en `_sample_next`), no en el forward — porque torch encola async y el deadlock del kernel se manifiesta en el primer sync.

### 4.2 Qué hay que hacer

1. **Confirmar o refutar la hipótesis ANTES de reescribir.** Experimento mínimo y barato:
   - `forward_normal` con `T=512` (múltiplo de 16) → ¿funciona?
   - `forward_normal` con `T=16, 32, 48` → ¿funcionan?
   - `forward_normal` con `T=15, 17, 23` → ¿cuelgan?
   - Reportar la tabla. **Si T=15 cuelga y T=16 no, la hipótesis queda confirmada con evidencia**, no por argumento.
2. **Reescribir el rollout con forward de 1 token y estado recurrente que persiste** (que es el motivo por el que RWKV existe y lo que hace `infer_kateto.RWKV_RNN`, pero **con gradiente**):
   - Un forward del prompt (T = len(prompt), paddeado a múltiplo de `CHUNK_LEN` si hace falta).
   - Después, **un forward por token** con el estado recurrentre arrastrado.
   - `rescore_response` ya hace un forward grad-enabled sobre prompt+respuesta: **también tiene longitud arbitraria** → mismo problema. Padear a múltiplo de 16 y **enmascarar la loss en el padding**.
3. Si padding a múltiplo de 16 no alcanza porque el modelo exige `T <= ctx_len` y/o el buffer es fijo: documentarlo y proponer la alternativa (fijar T=ctx_len con máscara).

### 4.3 Gates que hay que pasar (en este orden, con output real)

- **Gate B1:** el experimento de §4.2.1, con tabla de `T` → funciona/cuelga. **Gate duro**: sin esta tabla no se toca el rollout.
- **Gate B2:** `opd_rollout.py --ckpt models/rwkv7-1.5b/rwkv7-g1j-1.5b-20260831-ctx16384.pth --device cuda --max-len 24 --n-prompts 8` completa y reporta longitud media de respuesta, tok/s y **fracción de tokens de `<think>`**.
- **Gate B3:** `opd_train.py --selftest` (si existe) o el loop de OPD con `--steps 30` produce `out/opd/run0/metrics.jsonl` con `loss_rkl`, `coverage`, `response_tokens`.
- **Gate C (decisión):** mirar `coverage` de los primeros 5 pasos.
  - `>= 0.5` → seguir.
  - `0.2–0.5` → seguir, documentar el techo.
  - `< 0.2` → **parar**, cambiar de backend de teacher y reportar. No maquillar.
- **Gate D:** **baseline obligatorio con `lr=0`** sobre el mismo set. Sin esa curva, "el KL bajó" no significa nada.

---

## 5. Datos duros del entorno (medidos hoy, usarlos, no re-medirlos)

| Cosa | Valor |
|---|---|
| Student 1.5B | `n_layer=24 n_embd=2048 n_head=32 head_size=64 vocab=65536`, 798 tensores |
| Student 1.5B en cuda | 4.4–4.6 tok/s, **VRAM pico 3.15 GB / 3.98 GB** |
| 0.4B en cuda (referencia) | 12–13 tok/s, VRAM pico 1.01 GB |
| LoRA del student | r=16, alpha=32 → **14.155.776 params entrenables (0.918%)**, 288 tensores |
| Descarga del 1.5B | 3.055.444.605 bytes, MATCH con HF. Usar `aria2c -x8`, **`curl -C -` muere con exit 56** |
| Teacher (router) | `http://127.0.0.1:11434/v1`, **CPU por default** (los presets no tienen `n-gpu-layers`; VRAM usada 16 MB) — es lo que queremos |
| Teacher: `MiniCPM5-1B` | carga y devuelve logprobs; `teacher_self_coverage: 0.9946` |
| Formato del student base | **RWKV World** `User: {q}\n\nAssistant:` — el ChatML de Kateto produce bucle degenerado en el base (verificado) |
| `<think>` del 1.5B | se **dispara con input en español** y deriva en meta-análisis del prompt. Hay que controlarlo y **medir su fracción** |
| RAM del sistema | 31 GB totales, ~20 GB disponibles |
| Venv | `venv-unsloth-qwen/bin/python` (torch 2.9.1+rocm6.3, transformers 5.5.0, peft 0.20.0, triton 3.5.1) |

---

## 6. Restricciones duras

1. **No revertir los 4 fixes de §1.** Están verificados con output real.
2. **No intentar `WKV=cuda`.** Ver §2.
3. **No correr el student en background de Hermes** sin `systemd-run --user --scope`. Ver §3. Exit 137 = cgroup, no bug.
4. **No tocar `venv-unsloth-qwen`.** Si un teacher necesita otra versión de transformers, venv aparte.
5. **No tocar `base/`, no push, no deploy.** Todo el output a `out/opd/`.
6. **No inventar métricas.** Lo que no se pueda medir se reporta como "no medido" con el motivo.
7. **Separar hecho de hipótesis en cada reporte.** La §4.1 es una **hipótesis con evidencia circunstancial**, no un hecho: el Gate B1 existe para confirmarla o refutarla. No la escribas como confirmada sin la tabla.
8. **Ningún KL se reporta sin el baseline `lr=0`.**

---

## 7. Entregables

| # | Artefacto | Ubicación |
|---|---|---|
| 1 | Tabla `T` → funciona/cuelga del kernel (Gate B1) | `out/opd/tlen_probe.md` |
| 2 | Rollout corregido con estado recurrente persistente | `rwkv_pipeline/opd_rollout.py` |
| 3 | Gate B2/B3 con output verbatim | `out/opd/phase02_rollout.md` |
| 4 | Curvas KL (baseline `lr=0` + entrenado) + coverage + entropías | `out/opd/run0/metrics.jsonl` |
| 5 | Reporte honesto con sección **"Qué NO funcionó"** | `out/opd/REPORT.md` |

---

## 8. Cómo correr (plantilla — el cgroup es obligatorio)

```bash
systemd-run --user --scope -p MemoryMax=infinity --unit=opd-$(date +%s) bash -c '
  cd /run/media/chaos/terciario/proyectos/kateto-train
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export TORCH_COMPILE_DISABLE=1
  venv-unsloth-qwen/bin/python -u rwkv_pipeline/opd_rollout.py \
    --ckpt models/rwkv7-1.5b/rwkv7-g1j-1.5b-20260831-ctx16384.pth \
    --device cuda --max-len 24 --n-prompts 8
'
```
