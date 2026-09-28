#!/usr/bin/env python3
"""ORPO forward-pass spike (todo 16, research-only).

Smoke path ONLY: loads RWKV-7, runs chosen/rejected forward passes with
recurrent state RESET between pairs, prints per-pair logP/odds/L_OR/L_total
plus a mandatory state-reset proof, and writes out/orpo_spike.log.

No training loop, no optimizer, no checkpointing here (todo 17 owns those).
Any non-smoke invocation exits nonzero with a pointer to this docstring.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Pre-import env so torch picks up the same allocator defaults as infer.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "RWKV-PEFT"))
sys.path.insert(0, str(PROJECT / "rwkv_pipeline"))
from infer_kateto import RWKV_RNN, RWKV_TOKENIZER  # reuse tokenizer + RNN forward

LAMBDA_OR = 0.1
RESET_TOL = 1e-6


def main():
    ap = argparse.ArgumentParser(description="ORPO forward-pass spike (smoke only)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--pairs-file", type=str,
                    default=str(PROJECT / "data/orpo/anti_sycophancy_rioplatense.jsonl"))
    ap.add_argument("--log", type=str, default=str(PROJECT / "out/orpo_spike.log"))
    args = ap.parse_args()

    if not args.smoke:
        print("ORPO trainer (full loop) not implemented — todo 17 owns it. "
              "This file is smoke-only: rerun with --smoke.", file=sys.stderr)
        return 2

    model_path = args.model or str(PROJECT / "models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth")
    vocab_path = PROJECT / "RWKV-PEFT/rwkv_vocab_v20230424.txt"
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.pairs_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[:args.pairs]
    assert len(rows) == args.pairs, f"wanted {args.pairs} pairs, file has {len(rows)}"

    t0 = time.time()
    tok = RWKV_TOKENIZER(str(vocab_path))
    ckpt = torch.load(model_path, map_location="cpu")
    n_embd = ckpt["emb.weight"].shape[1]
    n_layer = sum(1 for k in ckpt if k.startswith("blocks.") and k.endswith(".att.r_k"))
    head_size = ckpt["blocks.0.att.r_k"].shape[1]
    del ckpt
    device = "cpu"  # ponytail: CPU-only spike; GPU path is todo 17's problem
    dtype = torch.float32
    model = RWKV_RNN(model_path, n_layer, n_embd, head_size, dtype=dtype, device=device)
    model.eval()

    def fresh_state():
        H, N = n_embd // head_size, head_size
        st = [None] * (n_layer * 3)
        for i in range(n_layer):
            st[i * 3 + 0] = torch.zeros(n_embd, dtype=dtype)
            st[i * 3 + 1] = torch.zeros((H, N, N), dtype=torch.float)
            st[i * 3 + 2] = torch.zeros(n_embd, dtype=dtype)
        return st

    def mean_logp_response(prompt, response):
        """Fresh-state forward; mean logP over response tokens only."""
        p_ids = tok.encode(prompt)
        r_ids = tok.encode(response)
        full = p_ids + r_ids
        st = fresh_state()
        out, st = model.forward(full[0], st)
        lps, n = 0.0, 0
        for i in range(1, len(full)):
            lp = float(F.log_softmax(out.float(), dim=-1)[full[i]].item())
            if i >= len(p_ids):  # response positions only
                lps += lp
                n += 1
            out, st = model.forward(full[i], st)
        return lps / max(n, 1), st  # return terminal state for dirty-state proof

    def odds_of(mean_lp):
        p = min(math.exp(mean_lp), 1.0 - 1e-9)
        return p / (1.0 - p)

    def lor_from_logps(lp_c, lp_r):
        d = math.log(odds_of(lp_c)) - math.log(odds_of(lp_r))
        return math.log1p(math.exp(-d))  # -log(sigmoid(d)), stable

    lines = []
    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    log(f"# ORPO forward-pass spike — model={model_path} device=cpu "
        f"dtype=float32 layers={n_layer} embd={n_embd} pairs={len(rows)}")
    per_pair, wins = [], 0
    for i, r in enumerate(rows):
        lp_c, _ = mean_logp_response(r["prompt"], r["chosen"])
        lp_r, _ = mean_logp_response(r["prompt"], r["rejected"])
        lor = lor_from_logps(lp_c, lp_r)
        nll = -lp_c
        total = nll + LAMBDA_OR * lor
        ok_finite = all(math.isfinite(v) for v in (lp_c, lp_r, lor, total))
        win = lp_c > lp_r
        wins += bool(win)
        per_pair.append(total)
        log(f"pair {i}: logP_c={lp_c:.4f} logP_r={lp_r:.4f} "
            f"odds_c={odds_of(lp_c):.3e} odds_r={odds_of(lp_r):.3e} "
            f"L_OR={lor:.4f} L_SFT={nll:.4f} L_total={total:.4f} "
            f"finite={ok_finite} chosen>rejected={win}")
    log(f"loss curve: {' '.join(f'{v:.4f}' for v in per_pair)}")
    log(f"chosen>rejected: {wins}/{len(rows)}")

    # --- MANDATORY state-reset proof (crux for recurrent ORPO) ---
    b = rows[1 % len(rows)]
    lp_b_fresh1, _ = mean_logp_response(b["prompt"], b["chosen"])
    # Contaminate: run pair 0 fully, keep dirty state, eval B on it.
    a = rows[0]
    _, dirty = mean_logp_response(a["prompt"], a["chosen"])
    p_ids = tok.encode(b["prompt"])
    r_ids = tok.encode(b["chosen"])
    full = p_ids + r_ids
    out, dirty = model.forward(full[0], dirty)
    lps = 0.0
    n = 0
    for i in range(1, len(full)):
        lp = float(F.log_softmax(out.float(), dim=-1)[full[i]].item())
        if i >= len(p_ids):
            lps += lp
            n += 1
        out, dirty = model.forward(full[i], dirty)
    lp_b_dirty = lps / max(n, 1)
    lp_b_fresh2, _ = mean_logp_response(b["prompt"], b["chosen"])
    gap_fresh = abs(lp_b_fresh1 - lp_b_fresh2)
    gap_dirty = abs(lp_b_fresh1 - lp_b_dirty)
    reset_ok = gap_fresh <= RESET_TOL and math.isfinite(lp_b_dirty)
    log(f"reset proof: fresh1={lp_b_fresh1:.6f} dirty={lp_b_dirty:.6f} "
        f"fresh2={lp_b_fresh2:.6f}")
    log(f"reset proof: |fresh1-fresh2|={gap_fresh:.2e} (tol {RESET_TOL:.0e}), "
        f"|fresh-dirty|={gap_dirty:.2e} -> order-independent={reset_ok}")

    all_finite = all(math.isfinite(v) for v in per_pair)
    go = all_finite and reset_ok and wins == len(rows)
    verdict = "GO" if go else "NO-GO"
    log(f"verdict: {verdict} — finite={all_finite} reset-proof={reset_ok} "
        f"wins={wins}/{len(rows)} elapsed={time.time()-t0:.1f}s "
        f"(todo 17 {'UNBLOCKED' if go else 'STAYS BLOCKED'})")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Honest exit code: 0 only on full pass; NEVER fake it.
    if not all_finite:
        return 3
    if not reset_ok:
        return 4
    if wins != len(rows):
        return 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
