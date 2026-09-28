#!/usr/bin/env python3
"""§2.1bis — CBB con margen CORREGIDO por tipicalidad (PMI).

Problema medido en §2.1: accuracy ~0.19 en AMBOS modelos y delta ~0.004.
Sintoma: los actos de habla mas idiomaticos (insulto_afiliativo 0.13,
reproche 0.08) tienen el accuracy mas bajo -> cuanto menos formuláica la
respuesta humana, peor puntua. Eso es tipicalidad, no preferencia.

Definiciones:
  L(r | p)  = media de log P(token_i | p + r[:i])           (lo que mide §2.1)
  L(r | n)  = idem con n = andamiaje de formato SIN pregunta (neutro)
  margin_raw = L(chosen|p) - L(rejected|p)                  <- contaminado
  margin_pmi = [L(c|p) - L(c|n)] - [L(r|p) - L(r|n)]        <- cancelado

margin_pmi pregunta: "cuanto favorece LA PREGUNTA a cada respuesta?", que es
la pregunta que queriamos hacer. Lo tipico se cancela porque aparece igual
en L(·|p) y L(·|n).
"""
import argparse, json, os, statistics as st, sys
from pathlib import Path

ROOT = Path(os.environ.get("KATETO_HOME", "/run/media/chaos/terciario/proyectos/kateto-train"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.eval_behavior import (  # noqa: E402
    CBB_DPO, FORMATS, load_cbb, load_student, free_gpu,
    encode_prompt, token_logprob_mean, percentile, ensure_eval)

import torch  # noqa: E402


def score_pair(model, tok, prompt_text, chosen, rejected, chat_format, ctx_len=512):
    """Devuelve (L(c|p), L(r|p), L(c|n), L(r|n)) o None si se descarta."""
    p_ids = encode_prompt(tok, prompt_text, chat_format)
    n_ids = encode_prompt(tok, "", chat_format)          # neutro: sin pregunta
    out = []
    for ids in (p_ids, n_ids):
        for resp in (chosen, rejected):
            s, meta = token_logprob_mean(model, tok, ids, resp, ctx_len=ctx_len)
            out.append(s)
    if any(x is None for x in out):
        return None
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chat-format", default="world")
    ap.add_argument("--n-pairs", type=int, default=808)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    items, strata, meta = load_cbb(a.n_pairs, 1337)
    name = Path(a.ckpt).name
    print(f"[pmi] ckpt={name} fmt={a.chat_format} pares={len(items)}", flush=True)

    model, tok, info = load_student(a.ckpt, device=a.device)
    ensure_eval(model)
    rows, disc = [], {}
    for i, (pid, prompt, chosen, rejected) in enumerate(items):
        r = score_pair(model, tok, prompt, chosen, rejected, a.chat_format)
        if r is None:
            disc["discarded"] = disc.get("discarded", 0) + 1
            continue
        lcp, lrp, lcn, lrn = r
        rows.append({"id": pid, "L_c_p": lcp, "L_r_p": lrp, "L_c_n": lcn,
                     "L_r_n": lrn, "margin_raw": lcp - lrp,
                     "margin_pmi": (lcp - lcn) - (lrp - lrn)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)
    free_gpu(model)

    def acc(key):
        m = [r[key] for r in rows]
        return (sum(1 for x in m if x > 0) / len(m), m)

    acc_raw, mraw = acc("margin_raw")
    acc_pmi, mpmi = acc("margin_pmi")
    res = {
        "ckpt": a.ckpt, "ckpt_name": name, "chat_format": a.chat_format,
        "arch": info["arch"], "n_scored": len(rows), "discards": disc, "meta": meta,
        "accuracy_raw": round(acc_raw, 6),
        "accuracy_pmi": round(acc_pmi, 6),
        "margin_raw": {"mean": round(st.fmean(mraw), 6),
                       "median": round(st.median(mraw), 6)},
        "margin_pmi": {"mean": round(st.fmean(mpmi), 6),
                       "median": round(st.median(mpmi), 6),
                       "p10": round(percentile(mpmi, 10), 6),
                       "p90": round(percentile(mpmi, 90), 6)},
        "mean_logp": {
            "chosen_prompt": round(st.fmean([r["L_c_p"] for r in rows]), 6),
            "rejected_prompt": round(st.fmean([r["L_r_p"] for r in rows]), 6),
            "chosen_neutral": round(st.fmean([r["L_c_n"] for r in rows]), 6),
            "rejected_neutral": round(st.fmean([r["L_r_n"] for r in rows]), 6),
        },
    }
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
