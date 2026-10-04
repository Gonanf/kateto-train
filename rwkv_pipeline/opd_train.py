#!/usr/bin/env python3
"""OPD loop: byte-string mapping + reverse KL + optimizer — Phase 0.4/0.5.

Implements the MiniCPM5 recipe (top-k student logits + teacher distribution,
reverse KL over the union support) with the Kateto twist: student (RWKV) and
teacher (BPE) tokenizers differ, so the support is aligned on **byte-strings**
(§2): two symbols are "the same" if decode() gives the same bytes. Mass that
cannot be represented is discarded & renormalized; the discarded fraction is
reported as `coverage` (a deliverable, not a detail).

    L = Σ_t  Σ_{v∈U_t} P_s(v) · [ log P_s(v) − log P_t(v) ]

with P_s a softmax over U_t (with grad) and P_t detached.

The pure alignment/RKL math lives in the standalone functions below so the
mechanism can be verified on synthetic logits (--selftest) independently of any
checkpoint/VRAM/toolchain constraints.

Rules honoured: student on cuda + teacher on CPU (8); 1.5B weights don't fit a
full fine-tune so the trainable params are LoRA adapters (matching the repo);
baseline lr=0 is mandatory (5); no metric is invented (3).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))           # for rwkv_pipeline / RWKV-PEFT

from rwkv_pipeline import opd_rollout as OR  # noqa: E402
from rwkv_pipeline.teacher_client import TeacherClient  # noqa: E402

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Byte-string alignment + reverse KL (pure math, testable standalone)
# --------------------------------------------------------------------------- #
def student_topk_indexes(student_logits_row: torch.Tensor, top_k: int,
                         tokenizer) -> Dict[bytes, int]:
    """Map the student's top-K indices for one position to byte-strings.

    Returns {bytes: student_index_into_row} keeping the highest-logit index
    per byte-string (row keeps autograd).
    """
    v, i = torch.topk(student_logits_row, min(top_k, student_logits_row.shape[0]))
    out: Dict[bytes, int] = {}
    for idx in i.tolist():                       # idx in descending logit order
        b = tokenizer.decodeBytes([idx])
        if b not in out:                          # first = best
            out[b] = idx
    return out


def reverse_kl_step(student_row: torch.Tensor, student_idx: Dict[bytes, int],
                    teacher_logp: Dict[bytes, float], top_k: int,
                    tokenizer) -> Tuple[torch.Tensor, Dict[str, float]]:
    """One position of reverse KL over the union support U_t (autograd-safe).

    student_row:      (vocab,) student logits WITH grad.
    student_idx:      {bytes: index_into_student_row} (its top-K support).
    teacher_logp:     {bytes: logprob} teacher distribution over ITS support.
    Returns (loss, metrics) — loss carries grad through student_row.
    """
    union = list(dict.fromkeys(list(student_idx.keys()) + list(teacher_logp.keys())))
    if not union:
        return torch.tensor(0.0, device=student_row.device), {
            "loss_rkl": 0.0, "coverage": 1.0, "student_entropy": 0.0,
            "teacher_entropy": 0.0, "n_union": 0}

    # P_s over U_t with grad (softmax restricted to the student's known bytes).
    s_logits = torch.full((len(union),), float("-inf"), device=student_row.device)
    for k, b in enumerate(union):
        if b in student_idx:
            s_logits[k] = student_row[student_idx[b]]
    s_logp = F.log_softmax(s_logits, dim=-1)
    # P_s mass: 0 exactly where the student cannot represent the byte (logit
    # == -inf). NOTE: we must NOT replace -inf logits with 0 before exp() —
    # exp(0)==1.0 would inject a fake unit probability at every teacher-only
    # entry, inflating coverage > 1 and the RKL (measured: coverage=19.6,
    # loss=126). Instead we keep the raw softmax and mask the PRODUCTS to 0 at
    # student-unknown entries, so each 0 * (-inf) term contributes exactly 0.
    ps = s_logp.exp()                       # sum(ps) over union == 1.0 (in [0,1])
    known_s = torch.isfinite(s_logp).to(ps.dtype)

    # P_t over U_t (detached; eps floor where the teacher cannot represent).
    dev = student_row.device
    t_raw = torch.tensor([teacher_logp.get(b, float("-inf")) for b in union],
                         device=dev)
    t_logp = torch.log(F.softmax(t_raw, dim=-1) + EPS).detach()

    teacher_known = torch.tensor(
        [b in teacher_logp for b in union], dtype=torch.float32, device=dev)

    # Reverse KL over the union: terms at student-unknown entries are masked.
    term = ps * (s_logp - t_logp)
    term = torch.where(known_s > 0, term, torch.zeros_like(term))
    loss = term.sum().clamp(min=0.0)

    coverage = float((ps * teacher_known * known_s).sum().item() or 0.0)

    se_term = -ps * s_logp
    se_term = torch.where(known_s > 0, se_term, torch.zeros_like(se_term))
    student_entropy = float(se_term.sum().item())

    tprobs = F.softmax(t_raw, dim=-1)
    tlog = F.log_softmax(t_raw, dim=-1)
    t_known = torch.isfinite(tlog).to(tlog.dtype)
    te_term = -tprobs * tlog
    te_term = torch.where(t_known > 0, te_term, torch.zeros_like(te_term))
    teacher_entropy = float(te_term.sum().item())
    metrics = {"loss_rkl": float(loss.item()), "coverage": coverage,
               "student_entropy": student_entropy, "teacher_entropy": teacher_entropy,
               "n_union": len(union)}
    return loss, metrics


def _teacher_dist_hf(client, prompt_text: str, response_text: str,
                     teacher_topk: int, T_resp: int) -> List[Dict[bytes, float]]:
    """Mode A: one forward over prompt+response; per-position teacher logp."""
    tlogits = client.logits_at(prompt_text, response_text)   # (T_resp, vocab)
    lp = torch.log_softmax(tlogits.float(), dim=-1)
    out: List[Dict[bytes, float]] = []
    for j in range(T_resp):
        colp = lp[j]
        _, idx = torch.topk(colp, min(teacher_topk, colp.shape[0]))
        m: Dict[bytes, float] = {}
        for t in idx.tolist():
            b = client.decode_teacher_token(t)
            if b not in m:
                m[b] = float(colp[t].item())    # first (best) logp wins
        out.append(m)
    return out


def _teacher_dist_router(client, prefixes: List[str],
                         teacher_topk: int) -> List[Dict[bytes, float]]:
    """Mode B: one HTTP request per position; top-K next-token logp."""
    out: List[Dict[bytes, float]] = []
    for pf in prefixes:
        top = client.top_logprobs_at(pf)          # [(token_str, logprob)]
        m: Dict[bytes, float] = {}
        for tok, lpx in top:
            b = client.decode_teacher_token(tok)
            if b not in m:
                m[b] = lpx
        out.append(m)
    return out


def step_losses(student, tokenizer, teacher_client, prompt_text: str,
                max_len: int, top_k: int, teacher_topk: int,
                ctx_len: int, block_think: bool, backend: str,
                temperature: float, top_p: float):
    """One prompt: rollout -> rescore -> per-position reverse KL -> total loss."""
    roll = OR.rollout_from_prompt(student, tokenizer, prompt_text,
                                  max_len=max_len, temperature=temperature,
                                  top_p=top_p, block_think=block_think,
                                  ctx_len=ctx_len)
    if not roll.response_ids:
        return {"loss_rkl": 0.0, "coverage": 0.0, "student_entropy": 0.0,
                "teacher_entropy": 0.0, "response_tokens": 0, "think": 0.0,
                "tok_per_s": 0.0}, torch.tensor(0.0)
    rows, _ = OR.rescore_response(student, tokenizer, roll.prompt_ids,
                                  roll.response_ids, ctx_len=ctx_len)
    resp_text = OR._decode_str(tokenizer, roll.response_ids)
    if backend == "hf":
        tdist = _teacher_dist_hf(teacher_client, prompt_text, resp_text,
                                 teacher_topk, len(roll.response_ids))
    else:
        tdist = _teacher_dist_router(teacher_client, roll.prefixes, teacher_topk)

    total = torch.tensor(0.0, device=rows.device)
    acc = {"loss_rkl": 0.0, "coverage": 0.0, "student_entropy": 0.0,
           "teacher_entropy": 0.0}
    for j in range(rows.shape[0]):
        if not tdist[j]:
            # Teacher support empty at this position (router returned no
            # top_logprobs): t_raw would be all -inf -> softmax(t_raw) = NaN ->
            # t_logp NaN -> loss NaN. HYPOTHESIS for the Phase-2 NaN, to be
            # confirmed by out/opd/_topk_sweep.py; skip either way because the
            # position carries no teacher signal and its coverage contribution
            # is 0 by construction.
            acc["n_skipped_no_teacher"] = acc.get("n_skipped_no_teacher", 0) + 1
            continue
        student_idx = student_topk_indexes(rows[j], top_k, tokenizer)
        loss_j, met = reverse_kl_step(rows[j], student_idx, tdist[j], top_k,
                                      tokenizer)
        total = total + loss_j
        for k in acc:
            acc[k] += met[k] / rows.shape[0]
    acc["response_tokens"] = rows.shape[0]
    acc["think"] = roll.think_fraction
    acc["tok_per_s"] = roll.tok_per_s
    return acc, total


def wire_rosa(student, enable: bool) -> int:
    """Engancha ROSA al head del student. Devuelve los params anadidos.

    Con `enable=False` (default del flag) no importa `rosa_module` ni instancia
    nada: es un no-op, 0 params. Con `enable=True` el import es lazy y la rama
    queda como `model.head`, asi que sus params entran solas en
    `OR._trainable_params` (todo `requires_grad` del arbol crudo).
    """
    if not enable:
        return 0
    from rwkv_pipeline.rosa_module import attach_rosa
    head = attach_rosa(getattr(student, "_raw", student))
    return sum(p.numel() for p in head.rosa.parameters())


def run_opd(ckpt_path: str, run_dir: str, prompts: List[str],
            backend: str = "router", teacher_model: Optional[str] = None,
            base_url: str = "http://127.0.0.1:11434/v1",
            device: str = "cuda", steps: int = 30, max_len: int = 24,
            top_k: int = 20, teacher_topk: int = 20, lr: float = 1e-5,
            lo_r: int = 16, ctx_len: int = 512, block_think: bool = False,
            ckpt_every: int = 10, temperature: float = 0.8, top_p: float = 0.8,
            rosa: bool = False):
    """OPD loop -> metrics.jsonl + checkpoints (deliverable 5)."""
    import random
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    student, tokenizer, info = OR.build_student(ckpt_path, device=device, lo_r=lo_r)
    n_rosa = wire_rosa(student, rosa)
    if rosa:
        print(f"[opd] ROSA encendida en el head: +{n_rosa:,} params trainables "
              f"({info['arch']['vocab']} vocab, n_embd={info['arch']['n_embd']})")
    teacher = TeacherClient(backend=backend, model=teacher_model,
                            base_url=base_url, top_k=teacher_topk)
    trainable = list(OR._trainable_params(student))
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    metrics_path = run_dir / "metrics.jsonl"

    random.seed(1337)
    pool = list(prompts)
    for step in range(1, steps + 1):
        t0 = time.time()
        random.shuffle(pool)
        batch = pool[: min(8, len(pool))]
        opt.zero_grad(set_to_none=True)
        agg = {"loss_rkl": 0.0, "coverage": 0.0, "student_entropy": 0.0,
               "teacher_entropy": 0.0, "think": 0.0, "tok_per_s": 0.0}
        total = torch.tensor(0.0, device=device)
        n = 0
        for p in batch:
            try:
                acc, t = step_losses(student, tokenizer, teacher, p,
                                     max_len, top_k, teacher_topk, ctx_len,
                                     block_think, backend, temperature, top_p)
            except Exception as e:   # no metric invented: drop & report
                print(f"[opd] prompt failed ({type(e).__name__}: {e}); skipping")
                continue
            if acc["response_tokens"] == 0:
                continue
            total = total + t
            for k in ("loss_rkl", "coverage", "student_entropy",
                      "teacher_entropy", "think", "tok_per_s"):
                agg[k] += acc[k]
            n += 1
        if n == 0:
            raise RuntimeError("all prompts failed; aborting run")
        if lr > 0:
            assert torch.isfinite(total).item(), (
                f"loss no finito en step {step}: {float(total)} — aborting "
                "(no metric invented, rule 3)")
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        wall = round(time.time() - t0, 2)
        rec = {"step": step, "loss_rkl": round(agg["loss_rkl"] / n, 5),
               "coverage": round(agg["coverage"] / n, 4),
               "teacher_entropy": round(agg["teacher_entropy"] / n, 4),
               "student_entropy": round(agg["student_entropy"] / n, 4),
               "response_tokens": acc["response_tokens"],
               "think_frac": round(agg["think"] / n, 4),
               "tok_per_s": round(agg["tok_per_s"] / n, 2),
               "wall_s": wall}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[opd] step {step}/{steps} lr={lr} " + json.dumps(rec))
        if ckpt_every and (step % ckpt_every == 0 or step == steps):
            raw = getattr(student, "_raw", student)
            sd = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()
                  if "lora" in k}
            torch.save({"step": step, "state_dict": sd}, run_dir / f"ckpt_{step}.pth")
    return student, metrics_path


def selftest() -> None:
    """Mechanism proof of byte-string mapping + reverse KL math (pure)."""
    random = __import__("random")
    random.seed(0)
    torch.manual_seed(0)
    vocab = 64
    row = torch.randn(vocab, requires_grad=True)          # grad-bearing row
    _, topi = torch.topk(row, 10)
    student_idx: Dict[bytes, int] = {}
    for t in topi.tolist():
        b = f"s{t}".encode()
        if b not in student_idx:
            student_idx[b] = t
    teacher_logp = {k: -0.3 * (vi + 1) for vi, k in
                    enumerate(list(student_idx.keys())[:6])}  # ~0.6 coverage
    from rwkv_pipeline import infer_kateto
    assert (PROJECT / "RWKV-PEFT/rwkv_vocab_v20230424.txt").exists()
    tok = infer_kateto.RWKV_TOKENIZER(
        str(PROJECT / "RWKV-PEFT/rwkv_vocab_v20230424.txt"))
    loss, m = reverse_kl_step(row, student_idx, teacher_logp, 10, tok)
    loss.backward()
    assert row.grad is not None and float(row.grad.abs().sum()) > 0
    assert 0.0 <= m["coverage"] <= 1.0
    print("SELFTEST reverse-KL OK")
    print("  coverage=%.3f loss=%.3f student_entropy=%.3f teacher_entropy=%.3f "
          "n_union=%d grad_flow=%.3f"
          % (m["coverage"], m["loss_rkl"], m["student_entropy"],
             m["teacher_entropy"], m["n_union"], float(row.grad.abs().sum())))


def main() -> None:
    ap = argparse.ArgumentParser(description="OPD prototype loop (Phase 0.4)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--run-dir", default=str(PROJECT / "out/opd/run0"))
    ap.add_argument("--backend", choices=["hf", "router"], default="router")
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--teacher-topk", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lo-r", type=int, default=16)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--qa", default=str(PROJECT / "data/kateto_qa.jsonl"))
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--block-think", action="store_true")
    ap.add_argument("--rosa", action="store_true",
                    help="engancha ROSA al head del student (apagado por default)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.ckpt:
        raise SystemExit("--ckpt required unless --selftest")

    prompts = []
    with open(args.qa, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line).get("question", "")
            if q and not q.startswith("[DRY"):
                prompts.append(f"User: {q}\n\nAssistant:")
    prompts = prompts[: args.n_prompts]
    if args.lr <= 0:
        print("Running BASELINE lr=0 (no weight update) — rule 5")
    run_opd(args.ckpt, args.run_dir, prompts, backend=args.backend,
            teacher_model=args.teacher_model, base_url=args.base_url,
            device=args.device, steps=args.steps, max_len=args.max_len,
            top_k=args.top_k, teacher_topk=args.teacher_topk, lr=args.lr,
            lo_r=args.lo_r, ctx_len=args.ctx_len, block_think=args.block_think,
            rosa=args.rosa)


if __name__ == "__main__":
    main()