# Wave 4 — Kaggle dry-run (todo 19, blocks todo 20)

Date: 2026-09-06. Local feasibility probes + quota check. No Kaggle credentials used; nothing uploaded; no notebook created (todo 20 owns it).

## 1. Local GPU probe (`nvidia-smi`)

```
$ nvidia-smi
/usr/bin/bash: línea 1: nvidia-smi: orden no encontrada   (exit 127)
```

Expected: this is an AMD ROCm box (RX 6500 XT / gfx1034, 4 GB). `nvidia-smi`
is absent by construction. `rocm-smi` confirms the local GPU is alive
(47 °C, VRAM 33 %) — local box can only do 0.4B smoke / CPU probes, never
2.9B training. All CUDA/NCCL validation is therefore deferred to the Kaggle
cell itself (todo 20 cell 1 runs `nvidia-smi` on 2×T4).

## 2. NCCL expectation (2×T4, no NVLink)

Kaggle 2×T4 has **no NVLink** — GPUs talk over PCIe. Todo 20 must set in the
setup cell, before any distributed init:

```
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
```

Strategy is DDP (one full-model replica per GPU, gradient allreduce over
PCIe), not tensor parallelism — RWKV-7 does not shard trivially. No local
NCCL probe is possible (no NVIDIA hardware, no `nccl-tests` installed);
bandwidth validation happens live in the notebook via a first-epoch `it/s`
reading, not here.

## 3. VRAM math: 2.9B PiSSA-r32 SFT, micro-batch 4, accum 8, ctx 1024 → T4-16GB

Reference points (alljeb readable in-repo, no SPEC pseudocode used as proof):

- Trainable params (measured, `out/pissa_dryrun.log`): **47.2M**
  (PiSSA-r32 fallback ≡ LoRA-r32 by construction; 4 targets
  `receptance,key,value,output`).
- SPEC §6 prose: 2.3B full weights ~4.6 GB + activations ~3 GB ≈ 7.6 GB/GPU;
  SPEC §6 table: SFT 2.3B ≈ 10 GB/GPU, ORPO ≈ 12 GB/GPU, states ≈ 6 GB/GPU.
- Local VRAM peaks: **none recorded** — learnings.md holds no training VRAM
  peak (all local training probes were CPU or dry-run; the 0.4B ORPO spike ran
  float32 on CPU, `out/orpo_spike.log`). The one local figure is inference:
  ~2.19 GB / 4 GB on RX 6500 XT (SPEC §1). Stated honestly as absent, not zero.

Extrapolation 2.3B → 2.9B (param ratio 2.9/2.3 ≈ 1.26×), bf16, DDP replica
per GPU (grad-accum adds steps, not memory):

| component (per GPU) | 2.3B (SPEC) | 2.9B (scaled) |
|---|---|---|
| frozen weights bf16 | ~4.6 GB | **~5.8 GB** (2.9B×2 B) |
| trainable params + grads (47.2M, bf16) | ~0.1 GB | **~0.19 GB** |
| Adam states fp32 on trainable only | ~0.3 GB | **~0.38 GB** (47.2M×8 B) |
| activations (micro-bsz 4, ctx 1024) | ~3 GB | **~3–3.8 GB** |
| **SFT total / GPU** | **~10 GB** | **≈ 9.4–10.2 GB → fits 16 GB** |

- ORPO cell (double forward, chosen+rejected): +~3 GB activations →
  ≈ 12–13 GB/GPU. Fits, but tight; Kaggle deferral on OOM per plan.
  **Moot for now**: ORPO spike was NO-GO (`docs/orpo-spike.md`, 0/8) →
  the ORPO cell in todo 20 is **conditional** (SFT-only unless re-spike GO).
- State tuning: ≈ 6 GB/GPU, comfortable.
- No packing: kernel gate FAIL (todo 14) → `data_type jsonl`, no-pack;
  ctx 1024 in todo 20 is plain per-example length, not packed batches.

## 4. Quota acknowledgment

Kaggle free tier: **~30 h/week T4** (~40 h with verified account). Budget vs
SPEC §6 estimates:

| task | estimate | running total |
|---|---|---|
| SFT Capa 1 (PiSSA r=32, 3 ep) | 4–6 h | 4–6 h |
| ORPO (2 ep, conditional — currently NO-GO) | 2–3 h | 6–9 h |
| State tuning (5 voices) | ~2 h | **8–11 h** |

8–11 h < 30 h with margin for one full retry (≈ 19–22 h worst case).
Quota does **not** block. 7B is excluded from this budget regardless (stretch).

## 5. Model-source resolution (REQUIRED for todo 20)

Re-probed 2026-09-06 via `HfApi().model_info` (metadata only, no download):

| repo | result |
|---|---|
| `aabbdev/RWKV7-2.3B-20260805` (SPEC default) | **404 again** (RepositoryNotFound, confirmed twice) |
| `RWKV/rwkv-7-world-2.3b` (SPEC alt) | **404** |
| `RWKV/rwkv-7-world-7b` (SPEC 7B) | **404** |
| **`BlinkDL/rwkv7-g1`** | **OK** (sha `53b2ec91…`, 14 files) — hosts **`rwkv7-g1j-2.9b-20260831-ctx16384.pth`**, the exact pinned filename (local sha `966f3420…`, 5 896 273 469 B ≈ 5.5 GiB) |

**Decision for todo 20 — option (a), verified reachable HF id:**

```python
hf_hub_download(repo_id="BlinkDL/rwkv7-g1",
                filename="rwkv7-g1j-2.9b-20260831-ctx16384.pth")
```

This is exactly what the existing stub (`rwkv_pipeline/kaggle_rwkv7_g1j.py:29-42`)
already downloads — the stub's source assumption is validated. No upload plan
needed (option (b) rejected: 5.5 GiB upload to a Kaggle dataset is unnecessary
when the byte-identical filename is publicly reachable). Todo 20 targets 2.9B
as the default, not 2.3B — the SPEC 2.3B id is dead and has no live replacement.

## 6. 7B stretch: explicitly unvalidated

DeepSpeed ZeRO-3 / FSDP for 7.2B (`rwkv7-g1j-7.2b-20260831-ctx16384.pth`,
also present in `BlinkDL/rwkv7-g1`) has **zero local validation** — no import
probe, no sharding smoke, no VRAM measurement. 7B stays stretch regardless:
fenced behind its own VRAM math + a 2.9B-pass precondition in todo 20.

## Verdict

**GO (conditional): SFT + states on 2.9B via `BlinkDL/rwkv7-g1`; ORPO cell
conditional on re-spike GO (SFT-only otherwise); 7B fenced as stretch;
no-pack (`data_type jsonl`) per kernel-gate FAIL.**
Reason: model source resolved to a verified reachable HF id, SFT VRAM
≈ 10 GB < 16 GB per T4, and the 8–11 h budget fits the ~30 h/week quota —
with the three conditions above carried as hard constraints into todo 20.
