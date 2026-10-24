#!/usr/bin/env python3
"""Banco de medición comparativo de pérdidas ORPO: SFT baseline, RWKV-PEFT y RWKV-LM-RLHF.

Uso:
  scripts/orpo_variantes_bench.py \
    --data data/kateto_orpo_v8.jsonl --vocab RWKV-PEFT/rwkv_vocab_v20230424.txt \
    --arms sft,ours,theirs --seeds 0,1,2 \
    --steps 900 --max-len 192 \
    --lr 3e-4 --lambda 0.1 --alpha 0.01 \
    --log-every 20 --out /tmp/orpo-variantes.json
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

# Limitar hilos para CPU amigable con otros procesos en la máquina
import torch

torch.set_num_threads(2)

import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rwkv_pipeline.infer_kateto import RWKV_TOKENIZER
from rwkv_pipeline.mini_rwkv import MiniRWKV


def resolve_path(p: str) -> str:
    path = Path(p)
    if path.exists():
        return str(path)
    for base in [REPO_ROOT, Path("/run/media/chaos/terciario/proyectos/kateto-train")]:
        cand = base / p
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"No se encuentra el archivo requerido: {p}")


def _a_log_odds(logp: torch.Tensor) -> torch.Tensor:
    """log(p/(1-p)) a partir de log p, fórmula canónica de RWKV-PEFT."""
    logp_fp32 = logp.float()
    logp_fp32 = torch.clamp(logp_fp32, min=-20.0, max=-1e-4)
    return logp_fp32 - torch.log1p(-torch.exp(logp_fp32))


def forward_response(model: MiniRWKV, prompt_ids: list[int], resp_ids: list[int]) -> torch.Tensor:
    """Forward eficiente que calcula la cabeza de vocabulario únicamente sobre la respuesta."""
    full_ids = torch.tensor([prompt_ids + resp_ids], dtype=torch.long)
    m = len(prompt_ids)
    k = len(resp_ids)
    x = model.emb(full_ids)
    for blk in model.blocks:
        x = blk(x)
    h = model.ln_out(x)
    # Token resp[0] es predicho por la posición m - 1; resp[k-1] por m + k - 2
    h_resp = h[0, (m - 1) : (m + k - 1), :]
    logits_resp = model.head(h_resp)
    return logits_resp


def run_arm(arm: str, seed: int, pairs: list[dict], args) -> dict:
    torch.manual_seed(seed)
    model = MiniRWKV(vocab=65536, n_embd=96, n_layer=2, ctx_len=512)
    model.train()

    emb_bytes = model.emb.weight.detach().cpu().numpy().tobytes()
    emb_sha = hashlib.sha256(emb_bytes).hexdigest()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps_records = []
    ema_ce_history = []
    grad_pref_norms = []
    current_ema_ce = None
    current_pref_acc_ema = None

    n_pairs = len(pairs)
    print(f"[{arm.upper()} seed={seed}] Iniciando corrida de {args.steps} pasos...")

    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        pair = pairs[(step - 1) % n_pairs]
        pw = pair["prompt_w"]
        pl = pair["prompt_l"]
        cw = pair["chosen"]
        cr = pair["rejected"]

        yw = torch.tensor(cw, dtype=torch.long)
        yl = torch.tensor(cr, dtype=torch.long)

        # Forward elegido (común a todos los brazos para CE_w)
        logits_w = forward_response(model, pw, cw)
        ce_w = F.cross_entropy(logits_w, yw, reduction="mean")
        ce_val = float(ce_w.item())

        if arm == "sft":
            loss = ce_w
            pref_loss_val = 0.0
            total_val = ce_val
            grad_pref_norm = 0.0

            # Medición de s_w y s_l para reporte comparativo (sin gradiente)
            with torch.no_grad():
                logits_l = forward_response(model, pl, cr)
                ce_l = F.cross_entropy(logits_l, yl, reduction="mean")
                s_w = _a_log_odds(-ce_w).clamp(-20.0, 20.0)
                s_l = _a_log_odds(-ce_l).clamp(-20.0, 20.0)
                s_w_val = float(s_w.item())
                s_l_val = float(s_l.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif arm == "ours":
            # RWKV-PEFT: log-odds de la log-probabilidad media
            mean_logp_w = -ce_w
            s_w = _a_log_odds(mean_logp_w)
            s_w = torch.nan_to_num(s_w, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

            logits_l = forward_response(model, pl, cr)
            mean_logp_l = -F.cross_entropy(logits_l, yl, reduction="mean")
            s_l = _a_log_odds(mean_logp_l)
            s_l = torch.nan_to_num(s_l, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

            s_w_val = float(s_w.item())
            s_l_val = float(s_l.item())

            lam = float(args.orpo_lambda)
            c_l = (lam * torch.sigmoid(s_l.detach() - s_w.detach())).clamp(0.0, lam)
            pref_loss = c_l * (s_l - s_w)
            loss = ce_w + pref_loss

            pref_loss_val = float(pref_loss.item())
            total_val = float(loss.item())

            # Gradiente exclusivo del término de preferencia
            params = [p for p in model.parameters() if p.requires_grad]
            pref_grads = torch.autograd.grad(pref_loss, params, retain_graph=True, allow_unused=True)
            sq_sum = sum(g.detach().pow(2).sum().item() for g in pref_grads if g is not None)
            grad_pref_norm = math.sqrt(sq_sum)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif arm == "theirs":
            # RWKV-LM-RLHF (mode 0): log-odds sobre la suma de log-probabilities
            tok_logp_w = F.log_softmax(logits_w.float(), dim=-1).gather(1, yw.unsqueeze(1)).squeeze(1)
            sp_pos = tok_logp_w.sum()

            logits_l = forward_response(model, pl, cr)
            tok_logp_l = F.log_softmax(logits_l.float(), dim=-1).gather(1, yl.unsqueeze(1)).squeeze(1)
            sp_neg = tok_logp_l.sum()

            sp_pos_c = sp_pos.clamp(max=-1e-7)
            sp_neg_c = sp_neg.clamp(max=-1e-7)

            s_w = sp_pos - torch.log1p(-torch.exp(sp_pos_c))
            s_l = sp_neg - torch.log1p(-torch.exp(sp_neg_c))
            ratio = s_w - s_l

            alpha = float(args.orpo_alpha)
            pref_loss = -alpha * F.logsigmoid(ratio)
            loss = ce_w * (1.0 - alpha) + pref_loss

            s_w_val = float(s_w.item())
            s_l_val = float(s_l.item())
            pref_loss_val = float(pref_loss.item())
            total_val = float(loss.item())

            params = [p for p in model.parameters() if p.requires_grad]
            pref_grads = torch.autograd.grad(pref_loss, params, retain_graph=True, allow_unused=True)
            sq_sum = sum(g.detach().pow(2).sum().item() for g in pref_grads if g is not None)
            grad_pref_norm = math.sqrt(sq_sum)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            raise ValueError(f"Brazo desconocido: {arm}")

        sec = time.perf_counter() - t0
        pref_ok = bool(s_w_val > s_l_val)

        # Actualizar EMA de CE y de pref_acc
        if current_ema_ce is None:
            current_ema_ce = ce_val
            current_pref_acc_ema = 1.0 if pref_ok else 0.0
        else:
            current_ema_ce = 0.95 * current_ema_ce + 0.05 * ce_val
            current_pref_acc_ema = 0.95 * current_pref_acc_ema + 0.05 * (1.0 if pref_ok else 0.0)

        ema_ce_history.append(current_ema_ce)
        grad_pref_norms.append(grad_pref_norm)

        record = {
            "step": step,
            "ce": ce_val,
            "pref_loss": pref_loss_val,
            "total": total_val,
            "s_w": s_w_val,
            "s_l": s_l_val,
            "pref_ok": pref_ok,
            "sec": sec,
        }
        steps_records.append(record)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                f"[{arm:<6} s={seed} {step:4d}/{args.steps}] "
                f"CE: {ce_val:6.4f} | pref_loss: {pref_loss_val:8.4f} | total: {total_val:6.4f} | "
                f"gap: {(s_w_val - s_l_val):7.3f} | ok: {str(pref_ok):<5} | "
                f"grad_pref: {grad_pref_norm:7.4f} | {sec:5.3f} s/paso"
            )

    # Checkpoints para ce_at, ema_ce y total_at
    ce_steps = [1, 40, 50, 100, 300, 900]
    ce_at = {str(s): steps_records[s - 1]["ce"] for s in ce_steps if s <= args.steps}

    ema_steps = [40, 100, 300, 900]
    ema_ce = {str(s): ema_ce_history[s - 1] for s in ema_steps if s <= args.steps}
    total_at = {str(s): steps_records[s - 1]["total"] for s in ema_steps if s <= args.steps}

    n_tot = len(steps_records)
    k100 = min(100, n_tot)
    pref_acc_first100 = sum(1.0 for r in steps_records[:k100] if r["pref_ok"]) / k100
    pref_acc_last100 = sum(1.0 for r in steps_records[-k100:] if r["pref_ok"]) / k100

    pref_gap_median = float(statistics.median(r["s_w"] - r["s_l"] for r in steps_records))
    grad_pref_norm_median = float(statistics.median(grad_pref_norms))
    sec_per_step_median = float(statistics.median(r["sec"] for r in steps_records))
    sec_total = sum(r["sec"] for r in steps_records)

    run_dict = {
        "seed": seed,
        "arm": arm,
        "steps": steps_records,
        "ce_at": ce_at,
        "ema_ce": ema_ce,
        "total_at": total_at,
        "pref_acc_ema": float(current_pref_acc_ema),
        "pref_acc_first100": float(pref_acc_first100),
        "pref_acc_last100": float(pref_acc_last100),
        "pref_gap_median": pref_gap_median,
        "grad_pref_norm_median": grad_pref_norm_median,
        "sec_per_step_median": sec_per_step_median,
        "sec_total": sec_total,
    }
    return run_dict, emb_sha


def main():
    parser = argparse.ArgumentParser(description="Banco de medición comparativo de pérdidas ORPO")
    parser.add_argument("--data", type=str, default="data/kateto_orpo_v8.jsonl")
    parser.add_argument("--vocab", type=str, default="RWKV-PEFT/rwkv_vocab_v20230424.txt")
    parser.add_argument("--arms", type=str, default="sft,ours,theirs")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--max-len", dest="max_len", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda", dest="orpo_lambda", type=float, default=0.1)
    parser.add_argument("--alpha", dest="orpo_alpha", type=float, default=0.01)
    parser.add_argument("--log-every", dest="log_every", type=int, default=20)
    parser.add_argument("--out", type=str, default="/tmp/orpo-variantes.json")
    args = parser.parse_args()

    data_file = resolve_path(args.data)
    vocab_file = resolve_path(args.vocab)
    tok = RWKV_TOKENIZER(vocab_file)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Cargar y pre-tokenizar dataset
    raw_pairs = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_pairs.append(json.loads(line))

    pairs_total = len(raw_pairs)
    pairs_truncated = 0
    tokenized_pairs = []

    for item in raw_pairs:
        p_ids = tok.encode(item["prompt"])
        w_ids = tok.encode(item["chosen"])
        l_ids = tok.encode(item["rejected"])

        trunc_w = False
        trunc_l = False

        if len(p_ids) + len(w_ids) > args.max_len:
            p_keep_w = max(1, args.max_len - len(w_ids))
            pw = p_ids[-p_keep_w:]
            trunc_w = True
        else:
            pw = p_ids

        if len(p_ids) + len(l_ids) > args.max_len:
            p_keep_l = max(1, args.max_len - len(l_ids))
            pl = p_ids[-p_keep_l:]
            trunc_l = True
        else:
            pl = p_ids

        if trunc_w or trunc_l:
            pairs_truncated += 1

        tokenized_pairs.append({
            "prompt_w": pw,
            "prompt_l": pl,
            "chosen": w_ids,
            "rejected": l_ids,
        })

    print(
        f"Configuración del banco: arms={arms}, seeds={seeds}, steps={args.steps}, "
        f"max_len={args.max_len}, pairs={pairs_total} (truncados={pairs_truncated})"
    )

    init_sha256 = {}
    runs = []

    for seed in seeds:
        init_sha256[str(seed)] = {}
        for arm in arms:
            run_dict, emb_sha = run_arm(arm, seed, tokenized_pairs, args)
            init_sha256[str(seed)][arm] = emb_sha
            runs.append(run_dict)

    # Armar summary por brazo
    summary = {}
    for arm in arms:
        arm_runs = [r for r in runs if r["arm"] == arm]
        ce_finals = [r["ce_at"][str(max(int(k) for k in r["ce_at"]))] for r in arm_runs]
        summary[arm] = {
            "ce_final_median": float(statistics.median(ce_finals)),
            "pref_gap_median": float(statistics.median(r["pref_gap_median"] for r in arm_runs)),
            "grad_pref_norm_median": float(statistics.median(r["grad_pref_norm_median"] for r in arm_runs)),
            "pref_acc_ema_mean": float(statistics.mean(r["pref_acc_ema"] for r in arm_runs)),
            "sec_per_step_median": float(statistics.median(r["sec_per_step_median"] for r in arm_runs)),
            "sec_total": sum(r["sec_total"] for r in arm_runs),
        }

    output_data = {
        "meta": {
            "steps": args.steps,
            "seeds": seeds,
            "arms": arms,
            "max_len": args.max_len,
            "lr": args.lr,
            "lambda": args.orpo_lambda,
            "alpha": args.orpo_alpha,
            "pairs_total": pairs_total,
            "pairs_truncated": pairs_truncated,
            "init_sha256": init_sha256,
        },
        "runs": runs,
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResultados guardados exitosamente en {out_path}")
    print("\n" + "=" * 70)
    print("RESUMEN DE RITMO Y CONVERGENCIA:")
    print("=" * 70)
    print(f"{'Brazo':<8} | {'CE Final':<10} | {'Pref Gap':<10} | {'Grad Pref':<10} | {'Ritmo (s/paso)':<15}")
    print("-" * 70)
    for arm, s in summary.items():
        print(
            f"{arm:<8} | {s['ce_final_median']:<10.4f} | {s['pref_gap_median']:<10.3f} | "
            f"{s['grad_pref_norm_median']:<10.4f} | {s['sec_per_step_median']:<15.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
