# Todo 14 — Kernel-decision gate — VERDICT: FAIL (default FORBID packing stands)

Date: 2026-09-06 · Scope: DOCS-ONLY, zero kernel change. Binding decision for all local training (todos 18+).

## 1. Verdict

**FAIL — packing FORBIDDEN.** No `reset_flag`/`reset_mask` threading proceeds.
All training runs no-pack (`data_type jsonl`). `rwkv_pipeline/sequence_packer.py`
is code-complete but QUARANTINED (FORBIDDEN notice at lines 8–12, confirmed
present). This verdict MUST NOT be flipped to PASS inside this task under any
circumstance; unblock conditions in §5 require a fresh proof run first.

## 2. Evidence (from `docs/verify-wave0-kernel.md`, todo 2)

- **State-update site:** `RWKV-PEFT/rwkvt/operator/rwkvop.py:158`, inside
  `fw_attn_triton` (lines 117–159, K=8 tile):
  `state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)`
  (re-verified read-only on 2026-09-06 — line byte-identical).
- **Zero reset path:** `grep -rn "reset_mask|reset_flag|rwkv7_attn_one_kv_head" RWKV-PEFT/rwkvt`
  → zero hits (todo 2 §2). SPEC `rwkv7_attn_one_kv_head` (SPEC §2.1:100-110) is
  phantom; SPEC path `rwkv_pipeline/rwkvop.py` absent — real kernel is
  `RWKV-PEFT/rwkvt/operator/rwkvop.py`. Any future edit is greenfield threading
  through `fw_attn_triton` + `TritonRWKV7.forward/backward` + both `RUN_*`
  wrappers + all three `att.py` call sites (178/223/317).
- **Import baseline missing:** mandated env-prefixed form
  `cd RWKV-PEFT && WKV=fla RWKV_MY_TESTING=x070 RWKV_TRAIN_TYPE=none
  ../venv-unsloth-qwen/bin/python -c "import rwkvt.operator.rwkvop; print('rwkvop-ok')"`
  → exit 1, `ModuleNotFoundError: No module named 'rwkvfla'`
  (raised at `rwkvop.py:30`, `from rwkvfla.ops.rwkv7 import chunk_rwkv7`;
  log `out/verify_kernel.log`). LDS math passes with headroom (~35 KB steady-state
  < 64 KB gfx1034; mask adds ≈32 B) — the gate fails purely on the missing
  import baseline, not on LDS.

## 3. Enforcement (code + process, no kernel edit)

- **No-pack fallback:** all training uses `data_type jsonl` (unpacked). No packed
  batch may reach a DataLoader or train script while this gate is FAIL.
- **Packer quarantined:** `SequencePacker` lives only in
  `rwkv_pipeline/sequence_packer.py`; its docstring carries the FORBIDDEN notice,
  and any future packed-batch training MUST re-run the todo-2 proof first.
  Future `PackedBatchesForbidden` enforcement (data-loader assert `gate == PASS`
  else raise + fall back to `data_type jsonl`) is owned by the PASS-track work,
  not by this FAIL decision.
- **Zero kernel change in this task:** no edit to `rwkvop.py`, packer, train
  scripts, or any product file. Package installs out of scope.

## 4. Consumer proof (2026-09-06, this task)

Mandated command (scoped with `--exclude-dir` for `.git/base/venvs/.cache/__pycache__`
after the unscoped form timed out on ~45k files; exclusions match the task's
`.omo/`+venvs+self-file+docs rule):

```
grep -rn "SequencePacker\|sequence_packer" --include="*.py" --include="*.sh" \
  --exclude-dir=.git --exclude-dir=.omo --exclude-dir=base \
  --exclude-dir=.venv --exclude-dir=venv-unsloth-qwen --exclude-dir=venv \
  --exclude-dir=.cache --exclude-dir=__pycache__ --exclude-dir=node_modules .
```

Result: **zero consumer hits.** Only matches are inside `sequence_packer.py`
itself (line 18 `class SequencePacker:`, line 38 skip-warning string, line 59
self-check `packer = SequencePacker(max_len=16)`) — the file the task excludes.
No DataLoader, train script, shell script, or any other module imports or
references the packer. Companion sweep over `docs/` + `.omo/` for the same
symbols → zero hits (exit 1). No BLOCKING consumer found; nothing to escalate.

## 5. Unblock conditions (ALL required before any PASS-track work)

1. `rwkv-fla` installed in the training venv
   (declared in `RWKV-PEFT/pyproject.toml:16` + `requirements.txt:6`).
2. The exact env-prefixed import above prints `rwkvop-ok` with exit 0
   (bare `import` form proves nothing — unguarded `os.environ["WKV"]` reads
   raise `KeyError: WKV`).
3. A gfx1034 (or T4) **device** compile of the edited `fw_attn_triton` passes.
   CPU-only success must never flip this verdict without the device caveat stated.
4. Fresh re-run of the todo-2 proof against the edited kernel, evidence logged
   under `out/`, before any packed batch reaches training.

## 6. Polarity pin (for the future PASS track — NOT applied now)

When unblock conditions are met, thread `reset_flag` with polarity PINNED:
`1.0` on every non-boundary token, `0.0` exactly on the boundary token —
`state = w * (state * reset_flag) + k·vᵀ`. Any inversion silently leaks
cross-document state; the PASS smoke (`it/s` 1.1→2.0+ → `out/pack_bench.log`)
must assert this polarity. Recorded here so the future edit has no ambiguity.

## 7. Downstream pointers

- Todos 18/20 MUST cite this doc for no-pack operation (`data_type jsonl`).
- Todo 7's `SequencePacker` stays quarantined until this gate flips via §5.
- Rollback of any future kernel edit: file revert + re-run todo-2 smoke.

Skipped: kernel edit, package installs, training runs (all out of scope for a
FAIL gate). Add mask threading only when §5 is green.
