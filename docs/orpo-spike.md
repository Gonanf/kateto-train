# ORPO forward-pass spike (todo 16) — NO-GO

Command: `venv/bin/python rwkv_pipeline/orpo_trainer.py --smoke --pairs 8`
(8 pairs from `data/orpo/anti_sycophancy_rioplatense.jsonl`)
Model: `models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth`, CPU, float32,
24 layers / 1024 embd. Elapsed 252.7s. Full log: `out/orpo_spike.log`.

## Results

- Finite loss on all 8 pairs: YES (L_total 3.00–4.21).
- chosen logP > rejected logP: **0/8** — base 0.4B systematically prefers the
  short servile rejected strings over the longer firm chosen ones
  (e.g. pair 0: −2.97 vs −2.62). Expected pre-training; the margin is what
  ORPO training would have to create, it is not present at init.
- State-reset proof: YES — pair B fresh1=−3.106320, on dirty state
  (pair A run through without reset) −1.880014 (|Δ|=1.23), after reset
  fresh2=−3.106320, |fresh1−fresh2|=0.00e+00 ≤ 1e−6. Reset is both
  necessary (dirty contaminates) and sufficient (bitwise order-independence).

## Verdict

**NO-GO — todo 17 STAYS BLOCKED, SFT-only path continues.**
Mechanics proven (two forwards + reset work on RWKV-7 RNN state), but the
gate requires chosen>rejected and the untrained base gives 0/8. Re-spike
after SFT warmup or on the retrained base; do not commit a trainer against
this signal. Exit code 5 (wins gate) by design — never faked.

## Re-spike 2026-09-07 (on SFT-warmed base) — NO-GO, todo 17 STAYS BLOCKED

Commands (weights via `--model`, no default touched; venv is
`venv-unsloth-qwen/bin/python`, CPU float32, 24L/1024, same 8 pairs):

- baseline: `.../orpo_trainer.py --smoke --pairs 8 --model out/rwkv_kateto_base/rwkv-1.pth` → 0/8, exit 5 (replicates original NO-GO)
- retrained: `.../orpo_trainer.py --smoke --pairs 8 --model out/rwkv_kateto_retrain/rwkv-1.pth` → **1/8**, exit 5

Retrained per-pair margins (chosen − rejected logP): p0 −0.1656, **p1 +0.1561**,
p2 −0.6951, p3 −0.5784, p4 −0.2030, p5 −0.3532, p6 −1.0374, p7 −0.8844.
Finite loss 8/8 both runs (retrained L_total 3.96–4.82, higher than
baseline's 3.02–4.09 — the SFT mix contained no ORPO-style pairs, so NLL on
these pairs rose even as one margin flipped). Reset proof bitwise-exact both
runs (|fresh1−fresh2|=0.00e+00; dirty gaps 1.44 / 1.21).

Gate was ≥5/8 for GO: retrained 1/8 < 5/8 → **NO-GO, BLOCKED-continues**.
SFT warmup moved exactly one margin (pair 1); the base still systematically
prefers the short servile rejected strings. Mark todo 17 `- [~]` terminal;
do not build the trainer. Full runs appended in `out/orpo_spike.log`.
