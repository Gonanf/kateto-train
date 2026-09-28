# Wave 0 — Todo 2: kernel reset mechanism + compile safety — VERDICT: FAIL

Date: 2026-09-06 · Scope: READ-ONLY probe, no product file modified.

## 1. Recurrent state-update site (Triton branch, `WKV=triton`)

File: `RWKV-PEFT/rwkvt/operator/rwkvop.py`, branch at lines 90–295, K=8 tile.

- Kernel: `fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr)` — lines 117–159.
- Exact recurrent update — line 158:
  `state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)`
- Tile driver: `TritonRWKV7.forward` (lines 250–261), `K = 8` (line 252), grid `[(H,B)]` (line 258).
- Python entry points (Triton branch): `RUN_CUDA_RWKV7g(r,w,k,v,a,b, HEAD_SIZE=64, dot_prec='fp32')` lines 282–288; `RUN_RWKV7_STATE(...)` lines 289–295.
- Call sites in `RWKV-PEFT/rwkvt/rwkv7/att.py`: line 178 (`RUN_CUDA_RWKV7g`), line 223 (`RUN_RWKV7_STATE`), line 317 (`RUN_RWKV7_STATE`).
- SPEC symbol `rwkv7_attn_one_kv_head` (SPEC §2.1:100-110) **does not exist** in this repo — confirmed phantom. SPEC paths `rwkv_pipeline/rwkvop.py` also absent; the real kernel lives in `RWKV-PEFT/rwkvt/operator/rwkvop.py`.
- fla branch (`WKV=fla`, lines 28–58): no explicit state-update line in repo code — all three defs delegate to `chunk_rwkv7(...)` from the external `rwkvfla` package.

## 2. `reset_mask` / `reset_flag` — prior finding CONFIRMED

`grep -rn "reset_mask|reset_flag|rwkv7_attn_one_kv_head" RWKV-PEFT/rwkvt` → **zero hits**. No reset path exists anywhere in the operator or `att.py`. The todo-14 edit would be greenfield threading through `fw_attn_triton` + `TritonRWKV7.forward/backward` + both `RUN_*` wrappers + all three `att.py` call sites.

## 3. LDS budget math (gfx1034, 64 KB)

Per program `(bi,hi)`, HEAD_SIZE C=64, tile dT=K=8, fp32: state tile C×C = 16 KB; six input tiles (w,q,k,v,a,b) dT×C = 2 KB each = 12 KB; `u`/`ab_u` dT×C = 2 KB each; `ab/ak/qk/qb` dT×dT ≈ 0.25 KB each. Steady-state ≈ 35 KB < 64 KB. A per-token mask adds one dT-vector (≈32 B fp32, or one extra dT×C scaling multiply if broadcast) — **negligible, fits with wide headroom**. LDS alone does NOT forbid the mask. (Caveat: static tile accounting only; real occupancy/`num_stages=1` spilling on gfx1034 needs a device compile to confirm — which is exactly what the proof below attempted.)

## 4. Compile proof (mandated env-prefixed form ONLY)

Command: `cd RWKV-PEFT && WKV=fla RWKV_MY_TESTING=x070 RWKV_TRAIN_TYPE=none ../venv-unsloth-qwen/bin/python -c "import rwkvt.operator.rwkvop; print('rwkvop-ok')"`
(Bare `import` form deliberately NOT used — `os.environ["WKV"]` reads at lines 28/90 are unguarded and raise `KeyError: WKV`.)

Result: **exit 1**, stdout/stderr captured in `out/verify_kernel.log`:
`ModuleNotFoundError: No module named 'rwkvfla'` (raised at `rwkvop.py:30`, `from rwkvfla.ops.rwkv7 import chunk_rwkv7`).
Environment notes: `venv-unsloth-qwen` has torch 2.9.1+rocm6.3, triton 3.5.1, `torch.cuda.is_available()==True`; but `rwkv-fla` (declared in `RWKV-PEFT/pyproject.toml:16` + `requirements.txt:6`) was never installed there, and package installs are out of scope for this probe. So even the *unmodified* kernel does not import under the mandated form on this box — no baseline exists against which to prove a mask edit safe.

## 5. Verdict: FAIL (default FORBID packing stands)

FAIL — not on LDS math (which passes with headroom), but on the compile-proof gate: the mandated import exits 1 with `ModuleNotFoundError: No module named 'rwkvfla'`, so **no mask threading may proceed**. Todo 14 is BLOCKED: zero kernel change, keep no-pack (`data_type jsonl`) fallback, `PackedBatchesForbidden` enforcement still required. Revisit only after `rwkv-fla` is installed and the same command prints `rwkvop-ok` with exit 0, plus a gfx1034 (or T4) device compile of the edited `fw_attn_triton`. CPU-only success must never flip this verdict without the device caveat stated.

Skipped: full training run, package installs, kernel edit itself (todo 14). Add mask threading only when the import baseline is green + device compile passes.
