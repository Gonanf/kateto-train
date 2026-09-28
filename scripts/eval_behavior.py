#!/usr/bin/env python3
"""Eval de comportamiento — PARTE B: scoring CBB (§2.1) + probes (§3.1, §3.2) + Gates 1/3.

Las métricas de generación (§2.2–§2.8) son PARTE C: placeholders NotImplementedError.

Uso:
  systemd-run --user --scope -p MemoryMax=infinity \
    venv-unsloth-qwen/bin/python scripts/eval_behavior.py \
    --baseline models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
    --candidate out/rwkv_kateto_base/rwkv-1.pth \
    --stages probes,cbb,gate3 --out-dir out/eval/run1 [--n-pairs N]

Todo scoring es determinista (sin sampling). Cada forward pasa por _pad_len
(bucket 64) vía rescore_response; se lee la posición real (rescore_response
devuelve filas alineadas a la respuesta, no la cola).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "rwkv_pipeline"))

from chat_template import render_chat

CBB_DPO = PROJECT / "data/hf/che-boludo-benchmark/datasets/dataset_cbb_dpo.json"
CBB_MAIN = PROJECT / "data/hf/che-boludo-benchmark/datasets/dataset_cbb.json"
PROMPTS_V1 = PROJECT / "out/eval/prompts_v1.json"

# Respuesta canónica para la sonda de formato (§3.2): neutra, en la voz del set.
CANONICAL_RESPONSE = "Es un genio, me río mucho con las anécdotas"
PROBE_N_PAIRS = 8
PROBE_SEED = 1337


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fmt_prompt(prompt_text: str, response_text: str, chat_format: str) -> str:
    """Formato de prompt por checkpoint (§1.3). chat_format in {world, chatml}."""
    if chat_format == "world":
        return f"User: {prompt_text}\n\nAssistant: {response_text}"
    if chat_format == "chatml":
        return render_chat([], prompt_text, "seco") + response_text
    raise ValueError(chat_format)


def encode_prompt(tok, prompt_text: str, chat_format: str):
    """prompt_ids = tokens del prompt SIN la respuesta (para rescore_response)."""
    if chat_format == "world":
        ptxt = f"User: {prompt_text}\n\nAssistant:"
    else:
        ptxt = render_chat([], prompt_text, "seco")
    return tok.encode(ptxt)


def token_logprob_mean(model, tok, prompt_ids, response_text: str, ctx_len: int = 512):
    """score = media de log P(token_i | prompt + seq[:i]) sobre tokens de la respuesta.

    Devuelve (score, n_tokens) o (None, motivo) si se descarta.
    """
    import torch
    from rwkv_pipeline.opd_rollout import rescore_response
    response_ids = tok.encode(response_text)
    if len(response_ids) == 0:
        return None, "empty_response"
    if len(prompt_ids) + len(response_ids) > ctx_len:
        # rescore_response trunca por la izquierda pero su slice de filas usa
        # len(prompt_ids)-1 como offset: con truncación ese offset queda mal.
        return None, f"too_long({len(prompt_ids)}+{len(response_ids)})"
    rows, _ = rescore_response(model, tok, prompt_ids, response_ids, ctx_len=ctx_len)
    if bool(torch.isnan(rows).any()):
        return None, "nan_logits"
    # detach: rescore_response corre bajo enable_grad() (lo necesita el path de
    # entrenamiento), asi que `rows` trae grafo. Convertir a float() sin detach
    # tira UserWarning y retiene el grafo al pedo en 808x2 forwards.
    with torch.no_grad():
        logp = torch.log_softmax(rows.detach().float(), dim=-1)
        tgt = torch.tensor(response_ids, device=logp.device)
        tok_logp = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
        score = float(tok_logp.mean().item())
    return score, len(response_ids)


def percentile(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_cbb(n_pairs, seed):
    """[(pair_id, prompt, chosen, rejected)]; ids cruzados con dataset_cbb.json
    por (prompt, chosen, rejected) para los strata de prompts_v1.json."""
    import random
    dpo = json.loads(CBB_DPO.read_text(encoding="utf-8"))
    main = json.loads(CBB_MAIN.read_text(encoding="utf-8"))
    by_content = {}
    for row in main:
        by_content[(row["prompt"], row["chosen"], row["rejected"])] = row["id"]
    strata = json.loads(PROMPTS_V1.read_text(encoding="utf-8"))["cbb_pairs"]["strata"]
    items, n_missing = [], 0
    for row in dpo:
        key = (row["prompt"], row["chosen"], row["rejected"])
        pid = by_content.get(key)
        if pid is None:
            n_missing += 1
            pid = f"UNMATCHED_{len(items):06d}"
        items.append((pid, row["prompt"], row["chosen"], row["rejected"]))
    random.Random(seed).shuffle(items)
    if n_pairs is not None:
        items = items[:n_pairs]
    return items, strata, {"n_pairs_in_file": len(dpo), "n_unmatched_id": n_missing}


def ensure_eval(model):
    """Anti-dropout: build_student deja .train(); el scoring exige .eval()."""
    getattr(model, "_raw", model).eval()
    return model


def load_student(ckpt: str, lo_r: int = 0, device: str = "cuda"):
    from rwkv_pipeline.opd_rollout import build_student
    model, tok, info = build_student(ckpt, device=device, lo_r=lo_r)
    ensure_eval(model)
    return model, tok, info


def free_gpu(model):
    import gc
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# §3.1 probe: lo_r / train-mode
# --------------------------------------------------------------------------- #

def probe_lo_r(ckpt: str, items, device="cuda") -> dict:
    import torch
    torch.manual_seed(PROBE_SEED)
    subset = items[:PROBE_N_PAIRS]
    out = {}
    for tag, lo_r, force_train in (("A_eval", 0, False), ("B_train", 0, True),
                                   ("C_lora16_eval", 16, False)):
        model, tok, _ = load_student(ckpt, lo_r=lo_r, device=device)
        if force_train:  # B: build_student ya deja .train() — dropout activo
            getattr(model, "_raw", model).train()
        scores = []
        for pid, prompt, chosen, rejected in subset:
            s, _ = token_logprob_mean(model, tok, encode_prompt(tok, prompt, "world"), chosen)
            if s is not None:
                scores.append(s)
        out[tag] = {"n": len(scores), "mean": round(statistics.fmean(scores), 6),
                    "per_pair": [round(x, 6) for x in scores]}
        free_gpu(model)
    res = {
        "ckpt": ckpt, "seed": PROBE_SEED, "n_pairs": PROBE_N_PAIRS,
        "variants": out,
        "A_vs_B_max_abs_diff": max(abs(a - b) for a, b in
                                   zip(out["A_eval"]["per_pair"], out["B_train"]["per_pair"])),
        "A_vs_C_max_abs_diff": max(abs(a - b) for a, b in
                                   zip(out["A_eval"]["per_pair"], out["C_lora16_eval"]["per_pair"])),
    }
    res["A_equals_B"] = res["A_vs_B_max_abs_diff"] == 0.0
    res["A_equals_C"] = res["A_vs_C_max_abs_diff"] == 0.0
    res["verdict"] = {
        "train_mode_breaks_determinism": not res["A_equals_B"],
        "lora16_perturbs_at_init": not res["A_equals_C"],
        "scoring_mode": ".eval() forzado tras build_student (obligatorio si B difiere)",
        "lo_r_for_scoring": 0 if res["A_equals_C"] else 0,
        "lo_r_note": "lora16 coincide con A en init (lora_B en ceros)" if res["A_equals_C"]
        else "HALLAZGO: lora16 NO coincide con A -> lo_r=0 obligatorio",
    }
    return res


# --------------------------------------------------------------------------- #
# §3.2 probe: formato world vs chatml
# --------------------------------------------------------------------------- #

def probe_format(ckpt: str, items, device="cuda") -> dict:
    model, tok, _ = load_student(ckpt, device=device)
    res = {"ckpt": ckpt, "canonical_response": CANONICAL_RESPONSE, "n_prompts": PROBE_N_PAIRS}
    acc = {"world": [], "chatml": []}
    for pid, prompt, _, _ in items[:PROBE_N_PAIRS]:
        for cf in ("world", "chatml"):
            s, _ = token_logprob_mean(model, tok, encode_prompt(tok, prompt, cf),
                                      CANONICAL_RESPONSE)
            if s is not None:
                acc[cf].append(s)
    lw = statistics.fmean(acc["world"]) if acc["world"] else float("nan")
    lc = statistics.fmean(acc["chatml"]) if acc["chatml"] else float("nan")
    res["logprob_world"] = round(lw, 6)
    res["logprob_chatml"] = round(lc, 6)
    res["margin"] = round(abs(lw - lc), 6)
    res["chosen_format"] = "world" if lw >= lc else "chatml"
    res["probe_discriminates"] = res["margin"] >= 0.05
    res["note"] = ("margen >= 0.05 nat/token: sonda discriminante" if res["probe_discriminates"]
                   else "margen < 0.05 nat/token: sonda NO discrimina; formato elegido por "
                        "mayor logprob; si se disputa, elegir por otra vía documentada")
    free_gpu(model)
    return res


# --------------------------------------------------------------------------- #
# §3.2bis probe de formato POR GENERACION (reemplaza al de logprob)
# --------------------------------------------------------------------------- #

def _degenerate(text: str) -> dict:
    """Loop detection sobre el texto generado: substrings de 8+ chars repetidos."""
    t = text.strip()
    if len(t) < 8:
        return {"degenerate": False, "reason": "too_short"}
    reps = 0
    worst = ""
    for n in (8, 12, 16, 24):
        for i in range(0, max(0, len(t) - 2 * n)):
            sub = t[i:i + n]
            c = t.count(sub)
            if c > reps:
                reps, worst = c, sub
    return {"degenerate": reps >= 3, "max_repeat": reps,
            "worst_sub": worst[:40] if worst else None}


def probe_format_generation(ckpt: str, items, device="cuda",
                            max_len: int = 24, n_prompts: int = 8) -> dict:
    """Sonda de formato POR GENERACION — reemplaza a probe_format (logprob).

    Por que el reemplazo (medido): probe_format compara el logprob de la MISMA
    continuacion bajo dos prompts de distinta estructura y largo, asi que mide
    perplejidad del prompt, no competencia de formato. Evidencia de que esta
    confundida: los DOS checkpoints eligieron `chatml` con margenes casi
    identicos (0.692 base / 0.683 Kateto), incluido el base que DEGENERA con
    chatml al generar (verificado en out/opd/phase01_localgen.md §5.1).

    Pregunta correcta: bajo que formato el modelo PRODUCE texto coherente?
    Metrica: repeticion de substrings + si termina en marcador de fin.
    """
    import torch
    from rwkv_pipeline.opd_rollout import rollout_from_prompt, _decode_str
    model, tok, _ = load_student(ckpt, device=device)
    res = {"ckpt": ckpt, "max_len": max_len, "n_prompts": n_prompts,
           "temperature": 0.80, "top_p": 0.70, "seed": PROBE_SEED, "by_format": {}}
    for cf in ("world", "chatml"):
        torch.manual_seed(PROBE_SEED)
        rows = []
        for pid, prompt, _, _ in items[:n_prompts]:
            ptext = (f"User: {prompt}\n\nAssistant:" if cf == "world"
                     else render_chat([], prompt, "seco"))
            try:
                r = rollout_from_prompt(model, tok, ptext, max_len=max_len,
                                        temperature=0.80, top_p=0.70)
                text = _decode_str(tok, list(r.response_ids))
            except Exception as e:  # un formato puede reventar; se reporta, no se oculta
                rows.append({"id": pid, "error": f"{type(e).__name__}: {e}"})
                continue
            dg = _degenerate(text)
            rows.append({"id": pid, "n_tokens": len(r.response_ids),
                         "finished": bool(r.finished), "text": text[:160],
                         **dg})
        ok = [r for r in rows if "error" not in r]
        n = max(1, len(ok))
        res["by_format"][cf] = {
            "n": len(ok), "n_errors": len(rows) - len(ok),
            "mean_tokens": round(statistics.fmean([r["n_tokens"] for r in ok]), 2) if ok else None,
            "finished_rate": round(sum(1 for r in ok if r["finished"]) / n, 3) if ok else None,
            "degenerate_rate": round(sum(1 for r in ok if r["degenerate"]) / n, 3) if ok else None,
            "distinct_texts": len({r["text"] for r in ok}),
            "samples": [{"id": r["id"], "degenerate": r["degenerate"],
                         "text": r["text"]} for r in ok[:3]],
        }
    free_gpu(model)
    w, c = res["by_format"]["world"], res["by_format"]["chatml"]
    dw = w["degenerate_rate"] if w["degenerate_rate"] is not None else 1.0
    dc = c["degenerate_rate"] if c["degenerate_rate"] is not None else 1.0
    if abs(dw - dc) >= 0.25:
        res["chosen_format"] = "world" if dw < dc else "chatml"
        res["discriminates"] = True
        res["note"] = (f"discrimina por degeneracion: world={dw} chatml={dc} "
                       f"-> nativo = {res['chosen_format']}")
    else:
        res["chosen_format"] = None
        res["discriminates"] = False
        res["note"] = (f"NO discrimina: degeneracion world={dw} vs chatml={dc} "
                       "(diferencia < 0.25). Sin formato nativo claro por esta via; "
                       "elegir explicitamente y documentar el criterio.")
    return res


# --------------------------------------------------------------------------- #
# §2.1 scoring CBB
# --------------------------------------------------------------------------- #

def _summarize_discards(discarded):
    reasons = {}
    for d in discarded:
        for k in ("chosen", "rejected"):
            v = d.get(k)
            if v:
                reasons[v] = reasons.get(v, 0) + 1
    return reasons


def score_side(model, tok, items, chat_format: str, label: str):
    """-> (per_pair pid->(score_chosen, score_rejected), discarded, wall_s)"""
    per_pair, discarded = {}, []
    t0 = time.time()
    for i, (pid, prompt, chosen, rejected) in enumerate(items):
        pids = encode_prompt(tok, prompt, chat_format)
        sc, why_c = token_logprob_mean(model, tok, pids, chosen)
        sr, why_r = token_logprob_mean(model, tok, pids, rejected)
        if sc is None or sr is None:
            discarded.append({"id": pid, "chosen": why_c if sc is None else None,
                              "rejected": why_r if sr is None else None})
            continue
        per_pair[pid] = (sc, sr)
        if (i + 1) % 50 == 0:
            wall = time.time() - t0
            print(f"[{label}] {i+1}/{len(items)} scored, wall={wall:.1f}s "
                  f"({wall/(i+1):.3f}s/par)", flush=True)
    return per_pair, discarded, time.time() - t0


def aggregate(margins, per_pair, strata):
    m = list(margins)
    res = {
        "n_scored": len(per_pair),
        "accuracy": round(sum(1 for x in m if x > 0) / len(m), 6) if m else None,
        "n_correct": sum(1 for x in m if x > 0),
        "n_ties": sum(1 for x in m if x == 0),
        "margin": {
            "mean": round(statistics.fmean(m), 6),
            "median": round(statistics.median(m), 6),
            "p10": round(percentile(m, 10), 6),
            "p90": round(percentile(m, 90), 6),
        },
    }
    by_act, by_gen = {}, {}
    for pid, (sc, sr) in per_pair.items():
        st = strata.get(pid)
        if st is None:
            continue
        by_act.setdefault(st["acto"], []).append(sc - sr)
        by_gen.setdefault(st["generacion"], []).append(sc - sr)

    def strat(groups):
        out = {}
        for k, v in sorted(groups.items()):
            out[k] = {"n": len(v),
                      "accuracy": round(sum(1 for x in v if x > 0) / len(v), 4),
                      "margin_mean": round(statistics.fmean(v), 6),
                      **({"underpowered": True} if len(v) < 20 else {})}
        return out

    res["strata_acto_de_habla"] = strat(by_act)
    res["strata_generacion"] = strat(by_gen)
    return res


def run_side(ckpt: str, items, strata, device="cuda", label="cbb"):
    chat_format = FORMATS[ckpt]
    model, tok, info = load_student(ckpt, device=device)
    per_pair, discarded, wall = score_side(model, tok, items, chat_format, label)
    margins = [sc - sr for sc, sr in per_pair.values()]
    res = {"ckpt": ckpt, "chat_format": chat_format, "arch": info["arch"],
           "wall_s": round(wall, 1), **aggregate(margins, per_pair, strata),
           "n_discarded": len(discarded),
           "discard_reasons": _summarize_discards(discarded)}
    free_gpu(model)
    return res


def run_cbb(baseline: str, candidate: str, items, strata, device="cuda") -> dict:
    result = {"created": now_iso(), "seed": PROBE_SEED,
              "sampling": {"scoring": "determinista (media de log P por token, sin sampling)",
                           "generation_defaults_recorded": "temp 0.80 / top_p 0.70 (referencia PARTE C)"}}
    result["baseline"] = run_side(baseline, items, strata, device, "baseline")
    result["candidate"] = run_side(candidate, items, strata, device, "candidate")
    result["delta_accuracy"] = round(
        result["candidate"]["accuracy"] - result["baseline"]["accuracy"], 6)
    return result


def run_gate3(ckpt: str, items, strata, device="cuda") -> dict:
    """Gate 3 — auto-consistencia: mismo checkpoint en AMBOS lados.

    Lo que se testea es DETERMINISMO, no ausencia de preferencia: con el mismo
    checkpoint, los dos pases de scoring tienen que dar scores IDENTICOS
    (max_abs_diff == 0). NOTA: la expectativa original del brief -- "accuracy ~0.5"
    -- era incorrecta; el accuracy mide la preferencia REAL del modelo entre
    chosen y rejected, y un modelo base perfectamente puede preferir la respuesta
    estilo-assistente. Se reporta como HALLAZGO, no como gate.
    """
    chat_format = FORMATS[ckpt]
    model, tok, info = load_student(ckpt, device=device)
    p1, d1, w1 = score_side(model, tok, items, chat_format, "gate3_a")
    p2, d2, w2 = score_side(model, tok, items, chat_format, "gate3_b")
    free_gpu(model)
    common = sorted(set(p1).intersection(p2))
    diffs = []
    for pid in common:
        diffs.append(abs(p1[pid][0] - p2[pid][0]))
        diffs.append(abs(p1[pid][1] - p2[pid][1]))
    max_abs = max(diffs) if diffs else None
    m = [sc - sr for sc, sr in p1.values()]
    acc = (sum(1 for x in m if x > 0) / len(m)) if m else None
    ok = (max_abs == 0.0)
    return {
        "ckpt": ckpt, "chat_format": chat_format, "created": now_iso(),
        "arch": info["arch"],
        "n_pairs_scored": len(common),
        "rescore_max_abs_diff": max_abs,
        "deterministic": ok,
        "pass": ok,
        "accuracy_same_ckpt": round(acc, 6) if acc is not None else None,
        "accuracy_note": ("HALLAZGO, no gate: preferencia real del modelo entre "
                          "chosen y rejected. 0.5 = sin preferencia; != 0.5 es informativo."),
        "note": ("determinismo OK: dos pases con el mismo checkpoint dan scores identicos"
                 if ok else
                 f"FALLO: dos pases con el mismo checkpoint difieren (max_abs={max_abs}) "
                 "=> el scoring no es determinista, revisar antes de confiar en cualquier metrica"),
    }


# El formato por checkpoint se fija con la sonda §3.2 antes del scoring.
FORMATS: dict = {}


# --------------------------------------------------------------------------- #
# PARTE C — placeholders (§2.2–§2.8)
# --------------------------------------------------------------------------- #

def generation_metrics(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("§2.2–§2.8: métricas de generación — PARTE C")


# === SECTION_MAIN ===

def _md_probes(probes: dict) -> str:
    L = ["# Probes — Parte B (§3.1 lo_r / §3.2 formato)", "",
         f"Generado: {now_iso()}", ""]
    for ckpt, pf in probes.get("format", {}).items():
        L += [f"## §3.2 Formato — `{Path(ckpt).name}`", "",
              f"- respuesta canonica: `{pf['canonical_response']}`",
              f"- logprob_world  : **{pf['logprob_world']}**",
              f"- logprob_chatml : **{pf['logprob_chatml']}**",
              f"- margen         : {pf['margin']}",
              f"- elegido        : **{pf['chosen_format']}**",
              f"- sonda discrimina (>=0.05): **{pf['probe_discriminates']}**",
              f"- {pf['note']}", ""]
    for ckpt, pl in probes.get("lo_r", {}).items():
        v = pl["variants"]
        L += [f"## §3.1 lo_r / train-mode — `{Path(ckpt).name}`", "",
              f"seed={pl['seed']} n_pairs={pl['n_pairs']}", "",
              "| variante | n | mean logP |", "|---|---|---|"]
        for tag, d in v.items():
            L.append(f"| {tag} | {d['n']} | {d['mean']} |")
        L += ["",
              f"- `A_vs_B_max_abs_diff` (dropout en .train()): **{pl['A_vs_B_max_abs_diff']}**"
              f" -> `A_equals_B = {pl['A_equals_B']}`",
              f"- `A_vs_C_max_abs_diff` (LoRA r=16 a init): **{pl['A_vs_C_max_abs_diff']}**"
              f" -> `A_equals_C = {pl['A_equals_C']}`",
              "",
              f"- **train_mode_breaks_determinism**: {pl['verdict']['train_mode_breaks_determinism']}",
              f"- **lora16_perturbs_at_init**: {pl['verdict']['lora16_perturbs_at_init']}",
              f"- lo_r usado para scoring: **{pl['verdict']['lo_r_for_scoring']}**",
              f"- {pl['verdict']['lo_r_note']}", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Eval de comportamiento — Parte B (scoring CBB + gates)")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--stages", default="probes,cbb,gate3",
                    help="subconjunto separado por comas de: probes,cbb,gate3")
    ap.add_argument("--out-dir", default=str(PROJECT / "out/eval/run1"))
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="limite de pares CBB (None = los 808). Usar chico para dimensionar.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--baseline-format", default="world",
                    help="formato del baseline cuando se saltea la etapa probes")
    ap.add_argument("--candidate-format", default="chatml",
                    help="formato del candidato cuando se saltea la etapa probes")
    ap.add_argument("--probe-ckpt", default=None,
                    help="checkpoint para los probes (default: --candidate)")
    args = ap.parse_args()

    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items, strata, meta = load_cbb(args.n_pairs, PROBE_SEED)
    probe_ckpt = args.probe_ckpt or args.candidate
    print(f"[main] pares={len(items)} {meta} | stages={sorted(stages)}", flush=True)

    # --- La sonda de formato §3.2 tiene que correr ANTES del scoring: run_side
    #     lee FORMATS[ckpt] y sin esto da KeyError.
    probes: dict = {"format": {}, "lo_r": {}, "formatgen": {}, "meta": meta}
    probes_path = out_dir / "probes.md"
    if "probes" in stages:
        for ckpt in dict.fromkeys([args.baseline, args.candidate]):
            print(f"[probes] formato de {ckpt}", flush=True)
            pf = probe_format(ckpt, items, device=args.device)
            probes["format"][ckpt] = pf
            FORMATS[ckpt] = pf["chosen_format"]
            print(f"  -> {pf['chosen_format']} (world={pf['logprob_world']}, "
                  f"chatml={pf['logprob_chatml']}, margen={pf['margin']})", flush=True)
        print(f"[probes] lo_r/train-mode en {probe_ckpt}", flush=True)
        probes["lo_r"][probe_ckpt] = probe_lo_r(probe_ckpt, items, device=args.device)
        # §3.2bis: la sonda de logprob esta confundida (ver docstring). Si la de
        # generacion discrimina, ES SU resultado el que fija FORMATS.
        if "formatgen" in stages:
            for ckpt in dict.fromkeys([args.baseline, args.candidate]):
                print(f"[probes] formato POR GENERACION de {ckpt}", flush=True)
                pg = probe_format_generation(ckpt, items, device=args.device)
                probes["formatgen"][ckpt] = pg
                print(f"  -> discrimina={pg['discriminates']} "
                      f"chosen={pg['chosen_format']} | {pg['note']}", flush=True)
                if pg["discriminates"]:
                    FORMATS[ckpt] = pg["chosen_format"]
                    print(f"  -> FORMATS[{Path(ckpt).name}] := {pg['chosen_format']} "
                          f"(por generacion, no por logprob)", flush=True)
        (out_dir / "probes.json").write_text(
            json.dumps(probes, ensure_ascii=False, indent=1), encoding="utf-8")
        probes_path.write_text(_md_probes(probes), encoding="utf-8")
        print(f"[probes] escrito {probes_path}", flush=True)
    else:
        # Sin etapa probes: formato EXPLICITO por checkpoint. Necesario porque el
        # criterio decidido es heterogeneo: base crudo = world, Kateto = chatml
        # (evaluar cada uno con el formato en que fue entrenado). Forzar el mismo
        # para los dos haria incomparable el score.
        FORMATS[args.baseline] = args.baseline_format
        FORMATS[args.candidate] = args.candidate_format
        print(f"[info] FORMATS: baseline={args.baseline_format} "
              f"candidate={args.candidate_format}", flush=True)

    rc = 0
    if "cbb" in stages:
        print("[cbb] scoring baseline vs candidato", flush=True)
        res = run_cbb(args.baseline, args.candidate, items, strata, device=args.device)
        res["n_pairs_requested"] = len(items)
        res["cbb_source"] = str(CBB_DPO.relative_to(PROJECT))
        (out_dir / "cbb.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        b, c = res["baseline"], res["candidate"]
        print(f"[cbb] baseline  acc={b['accuracy']} n={b['n_scored']} fmt={b['chat_format']}",
              flush=True)
        print(f"[cbb] candidato acc={c['accuracy']} n={c['n_scored']} fmt={c['chat_format']}",
              flush=True)
        print(f"[cbb] delta={res['delta_accuracy']}", flush=True)

    if "gate3" in stages:
        print("[gate3] determinismo (mismo ckpt, dos pases)", flush=True)
        g3 = run_gate3(args.baseline, items, strata, device=args.device)
        (out_dir / "gate3.json").write_text(
            json.dumps(g3, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[gate3] pass={g3['pass']} max_abs_diff={g3['rescore_max_abs_diff']} "
              f"accuracy(mismo ckpt)={g3['accuracy_same_ckpt']}", flush=True)
        if not g3["pass"]:
            rc = 1

    return rc


if __name__ == "__main__":
    sys.exit(main())


