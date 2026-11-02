#!/usr/bin/env python3
"""EVAL-1 — gemelo HTTP de `scripts/eval_cbb_pmi.py`.

Misma metrica (margen PMI del CBB, held-out), mismo esquema de salida, mismo set
y mismo render, pero puntuando un modelo servido por `llama-server` en vez de un
checkpoint RWKV con torch.

  L(r|p)      = media de log P(token_i | head + r[:i])   (sonda /completion n_predict=0)
  margin_raw  = L(c|p) - L(r|p)
  margin_pmi  = [L(c|p) - L(c|n)] - [L(r|p) - L(r|n)]
  accuracy    = fraccion de margenes > 0

Lo congelado NO se reimplementa: el set y el render salen de `scripts.eval_behavior`
(`load_cbb(n, 1337)` y `fmt_prompt(..., "chatml")` = render_chat([], texto, "seco") + respuesta),
asi que los bytes del prompt son identicos al camino RWKV y el set es comparable
entre motores (de ahi el `--manifest`, que solo depende del set).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.eval_behavior import fmt_prompt, load_cbb, percentile  # noqa: E402

# Congelado por consigna. No parametrizar: queremos los mismos bytes que el RWKV.
CHAT_FORMAT = "chatml"
VOICE = "seco"
SEED = 1337

SCHEMA = ("ckpt ckpt_name chat_format arch n_scored discards meta accuracy_raw "
          "accuracy_pmi margin_raw margin_pmi mean_logp pairs latencia").split()
SCHEMA_ARCH = ("motor device base_url model_id voice parallel").split()
SCHEMA_LAT = ("ttft_frio_ms ttft_caliente_p50_ms n_frio n_caliente ttfab_ms "
              "ttfab_motivo").split()


class RuntError(RuntimeError):
    """Fallo de red/HTTP. Descarta el par, no aborta la corrida."""


# --------------------------------------------------------------------------- #
# cliente HTTP (urllib + stdlib)
# --------------------------------------------------------------------------- #
class Client:
    def __init__(self, base_url: str, timeout: float):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._first_request = True

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _cache(self, override: bool | None = None) -> bool:
        """`cache_prompt=False` SOLO en el primer request de toda la corrida.

        `override` fija el modo a proposito (self-check: probe y generacion tienen
        que compararse en el MISMO modo — cold y warm dan logprobs que difieren
        en ~1e-2, no en ruido de punto flotante).
        """
        first = self._first_request
        self._first_request = False
        return first if override is None else override

    def model_id(self) -> str:
        return self._get("/v1/models")["data"][0]["id"]

    def tokenize(self, text: str):
        return self._post("/tokenize", {"content": text})["tokens"]

    def detokenize(self, ids):
        return self._post("/detokenize", {"tokens": list(ids)})["content"]

    def probe(self, prefix_txt: str, n_probs: int, cache: bool | None = None):
        """Sonda: distribucion del token SIGUIENTE a prefix_txt."""
        return self._post("/completion", {
            "prompt": prefix_txt, "n_predict": 0, "temperature": 0, "seed": 1,
            "n_probs": n_probs, "cache_prompt": self._cache(cache)})

    def greedy1(self, prompt_txt: str, n_probs: int | None = None,
                cache: bool | None = None):
        """Una generacion greedy de 1 token. Devuelve (completion_probabilities, ms).

        `n_probs` solo se manda cuando se necesita leer (id, logprob) del token
        generado: sin `n_probs` el server no incluye `completion_probabilities`.
        No altera el sampling (temperature 0, seed fijo).
        """
        payload = {"prompt": prompt_txt, "n_predict": 1, "temperature": 0, "seed": 1,
                   "cache_prompt": self._cache(cache)}
        if n_probs is not None:
            payload["n_probs"] = n_probs
        t0 = time.perf_counter()
        out = self._post("/completion", payload)
        return out, (time.perf_counter() - t0) * 1000.0


def call(fn, *a, **kw):
    """Envuelve un request: red/HTTP -> RuntError (el par se descarta)."""
    try:
        return fn(*a, **kw)
    except RuntError:
        raise
    except Exception as e:  # URLError, HTTPError, JSONDecodeError, ...
        raise RuntError(f"{type(e).__name__}: {e}") from e


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def margin(lcp, lrp, lcn, lrn):
    """La formula de eval_cbb_pmi.py:33-44/69-71, sin tocar."""
    return (lcp - lrp), ((lcp - lcn) - (lrp - lrn))


def score_seq(cli, prompt_txt: str, resp: str, n_probs: int, ctx_len: int):
    """Media de log P(token_i | prefijo) sobre los tokens de la respuesta.

    Devuelve (score, n_tokens) o (None, motivo).
    """
    head = fmt_prompt(prompt_txt, "", CHAT_FORMAT)
    full = fmt_prompt(prompt_txt, resp, CHAT_FORMAT)
    ids_head = call(cli.tokenize, head)
    ids_full = call(cli.tokenize, full)
    if ids_full[:len(ids_head)] != ids_head:
        return None, "bpe_boundary"
    resp_ids = ids_full[len(ids_head):]
    if not resp_ids:
        return None, "empty_response"
    if len(ids_full) > ctx_len:
        return None, "too_long"

    logps = []
    for k, tid in enumerate(resp_ids):
        prefix_ids = ids_full[:len(ids_head) + k]
        prefix_txt = call(cli.detokenize, prefix_ids)
        if call(cli.tokenize, prefix_txt) != prefix_ids:
            return None, "bpe_prefix"
        top = call(cli.probe, prefix_txt, n_probs)["completion_probabilities"][0]["top_logprobs"]
        hit = next((t for t in top if t["id"] == tid), None)
        if hit is None:
            # No inventes un logprob: el token salio del top-K -> secuencia muerta.
            return None, "token_off_topk"
        logps.append(hit["logprob"])
    return st.fmean(logps), len(resp_ids)


def score_pair(cli, prompt, chosen, rejected, n_probs, ctx_len):
    """(lcp, lrp, lcn, lrn) o (None, motivo). Las 4 secuencias o nada."""
    out = []
    for ptxt in (prompt, ""):
        for resp in (chosen, rejected):
            s, why = score_seq(cli, ptxt, resp, n_probs, ctx_len)
            if s is None:
                return None, why
            out.append(s)
    return tuple(out), None


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
P0 = "User: hola\n\nAssistant:"
SELFCHECK_NPROBS = 20000  # top-K del mini-run de (e); el default de --n-probs es 1000
FIXTURE = [  # (L_c_p, L_r_p, L_c_n, L_r_n, margin_raw, margin_pmi) — a mano
    (-1.0, -2.0, -0.5, -1.5, 1.0, 0.0),   # tipicalidad identica: PMI la cancela
    (-1.0, -2.0, -2.0, -0.5, 1.0, 2.5),   # la pregunta favorece a la rejected
]


def self_check(args) -> int:
    def fail(step, msg):
        print(f"SELF-CHECK FAIL [{step}]: {msg}", flush=True)
        return 1

    try:
        cli = Client(args.base_url, args.req_timeout)

        # (a) /v1/models responde y trae el model id
        mid = cli.model_id()
        if not mid:
            return fail("a", "/v1/models sin data[0].id")
        print(f"  (a) /v1/models id={mid}", flush=True)

        # (b) round-trip detokenize(tokenize(S)) == S, con marcador chatml
        s = "User: hola\n\nAssistant: <|im_user|> che, todo bien?"
        rt = cli.detokenize(cli.tokenize(s))
        if rt != s:
            return fail("b", f"round-trip != S\n  in ={s!r}\n  out={rt!r}")
        print("  (b) round-trip tokenize/detokenize OK", flush=True)

        # (c) sonda == generacion greedy en P0. Ambos en cache=False: cold y warm
        # difieren ~1e-2 (medido), comparar cruzados daria falso negativo.
        gen, _ = cli.greedy1(P0, n_probs=1, cache=False)
        gid, glp = gen["completion_probabilities"][0]["id"], \
            gen["completion_probabilities"][0]["logprob"]
        probe = cli.probe(P0, args.n_probs, cache=False)["completion_probabilities"][0]
        hit = next((t for t in probe["top_logprobs"] if t["id"] == gid), None)
        if hit is None:
            return fail("c", f"token greedy {gid} ausente del top-{args.n_probs} de la sonda")
        if abs(hit["logprob"] - glp) >= 1e-4:
            return fail("c", f"|lp_sonda-lp_gen|={abs(hit['logprob'] - glp):.2e} >= 1e-4 "
                             f"(sonda={hit['logprob']} gen={glp})")
        print(f"  (c) sonda==generacion (id={gid}, lp={glp:.6f})", flush=True)

        # (d) formula del margen contra el fixture
        for lcp, lrp, lcn, lrn, e_raw, e_pmi in FIXTURE:
            raw, pmi = margin(lcp, lrp, lcn, lrn)
            if raw != e_raw or pmi != e_pmi:
                return fail("d", f"margin({lcp},{lrp},{lcn},{lrn}) = ({raw},{pmi}) != ({e_raw},{e_pmi})")
        print(f"  (d) margen OK sobre {len(FIXTURE)} fixtures a mano", flush=True)

        # (e) corrida real --n-pairs 2 escribiendo --out
        # Top-K amplio: a K=1000 los 2 pares caen en token_off_topk (vocab 262k,
        # distribucion peaked) y (e) no ejercitaria el scoring. El default
        # --n-probs sigue en 1000 para la corrida real.
        saved_k = args.n_probs
        args.n_probs = max(saved_k, SELFCHECK_NPROBS)
        args.n_pairs = 2
        try:
            rc = run(args)
        finally:
            args.n_probs = saved_k
        if rc != 0:
            return fail("e", "la corrida minima devolvio rc != 0")
        res = json.loads(Path(args.out).read_text(encoding="utf-8"))
        missing = [k for k in SCHEMA if k not in res]
        if missing:
            return fail("e", f"claves faltantes: {missing}")
        for path, keys in (("arch", SCHEMA_ARCH), ("latencia", SCHEMA_LAT)):
            miss = [k for k in keys if k not in res[path]]
            if miss:
                return fail("e", f"{path} sin claves: {miss}")
        if not 0 < res["n_scored"] <= 2:
            return fail("e", f"n_scored={res['n_scored']}, se esperaba 0 < n <= 2")
        a = res["accuracy_pmi"]
        if a is None or not 0.0 <= a <= 1.0:
            return fail("e", f"accuracy_pmi={a} fuera de [0,1]")
        print(f"  (e) corrida n=2 OK (n_scored={res['n_scored']}, acc_pmi={a})", flush=True)

    except Exception as e:
        print(f"SELF-CHECK FAIL [{type(e).__name__}]: {e}", flush=True)
        return 1

    print("SELF-CHECK OK", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# corrida
# --------------------------------------------------------------------------- #
def run(args) -> int:
    try:
        cli = Client(args.base_url, args.req_timeout)
        mid = cli.model_id()
    except Exception as e:
        print(f"[http] no se pudo leer /v1/models: {type(e).__name__}: {e}", flush=True)
        return 1

    items, strata, meta = load_cbb(args.n_pairs, SEED)
    n_req = len(items)
    print(f"[pmi-http] label={args.label} model={mid} pares={n_req} "
          f"n_probs={args.n_probs} ctx={args.ctx_len} fmt={CHAT_FORMAT}", flush=True)

    rows, disc, lat_rows = [], {}, []
    for i, (pid, prompt, chosen, rejected) in enumerate(items):
        # Latencia por fila: UNA generacion greedy de 1 token sobre el head de la
        # condicion-pregunta. Reloj del cliente.
        turno = "frio" if not lat_rows else "caliente"
        try:
            _, ttft_ms = cli.greedy1(fmt_prompt(prompt, "", CHAT_FORMAT))
        except Exception as e:
            disc[f"latency_net_error:{type(e).__name__}"] = \
                disc.get(f"latency_net_error:{type(e).__name__}", 0) + 1
            print(f"  [{pid}] descartado: latencia net_error {e}", flush=True)
            continue

        try:
            r, why = score_pair(cli, prompt, chosen, rejected, args.n_probs, args.ctx_len)
        except RuntError as e:
            disc["net_error"] = disc.get("net_error", 0) + 1
            print(f"  [{pid}] descartado: net_error {e}", flush=True)
            continue
        if r is None:
            disc[why] = disc.get(why, 0) + 1
            print(f"  [{pid}] descartado: {why}", flush=True)
            continue

        lcp, lrp, lcn, lrn = r
        raw, pmi = margin(lcp, lrp, lcn, lrn)
        rows.append({"id": pid, "L_c_p": lcp, "L_r_p": lrp, "L_c_n": lcn, "L_r_n": lrn,
                     "margin_raw": raw, "margin_pmi": pmi})
        lat_rows.append({"id": pid, "prompt_sha256": sha256_text(prompt),
                         "turno": turno, "ttft_ms": round(ttft_ms, 3)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_req} scored={len(rows)}", flush=True)

    if not rows:
        print("[pmi-http] ningun par scored; no se puede promediar", flush=True)
        return 1

    mraw = [r["margin_raw"] for r in rows]
    mpmi = [r["margin_pmi"] for r in rows]
    acc_raw = sum(1 for x in mraw if x > 0) / len(mraw)
    acc_pmi = sum(1 for x in mpmi if x > 0) / len(mpmi)

    cal = [x["ttft_ms"] for x in lat_rows if x["turno"] == "caliente"]
    frio = [x for x in lat_rows if x["turno"] == "frio"]
    res = {
        "ckpt": args.label, "ckpt_name": args.label, "chat_format": CHAT_FORMAT,
        "arch": {"motor": "llama.cpp", "device": args.device, "base_url": args.base_url,
                 "model_id": mid, "voice": VOICE, "parallel": 1},
        "n_scored": len(rows), "discards": disc,
        "meta": {**meta, "n_pairs_requested": n_req, "ctx_len": args.ctx_len,
                 "n_probs": args.n_probs, "prompt_sha256_set": set_sha(items)},
        "accuracy_raw": round(acc_raw, 6),
        "accuracy_pmi": round(acc_pmi, 6),
        "margin_raw": {"mean": round(st.fmean(mraw), 6), "median": round(st.median(mraw), 6)},
        "margin_pmi": {"mean": round(st.fmean(mpmi), 6), "median": round(st.median(mpmi), 6),
                       "p10": round(percentile(mpmi, 10), 6),
                       "p90": round(percentile(mpmi, 90), 6)},
        "mean_logp": {
            "chosen_prompt": round(st.fmean([r["L_c_p"] for r in rows]), 6),
            "rejected_prompt": round(st.fmean([r["L_r_p"] for r in rows]), 6),
            "chosen_neutral": round(st.fmean([r["L_c_n"] for r in rows]), 6),
            "rejected_neutral": round(st.fmean([r["L_r_n"] for r in rows]), 6),
        },
        "pairs": lat_rows,
        # TTFAB no es medible aca: no hay device ni audio en esta corrida.
        "latencia": {"ttft_frio_ms": frio[0]["ttft_ms"] if frio else None,
                     "ttft_caliente_p50_ms": round(percentile(cal, 50), 3) if cal else None,
                     "n_frio": len(frio), "n_caliente": len(cal), "ttfab_ms": None,
                     "ttfab_motivo": "sin device ni audio: corrida texto-solo"},
    }
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    if args.manifest:
        # El set completo, no solo lo scored: el manifest identifica el set para
        # comparar motores, y un motor distinto puede discardear otros pares.
        mf = {"voice": VOICE, "chat_format": CHAT_FORMAT, "n_pairs": n_req,
              "prompt_sha256_set": res["meta"]["prompt_sha256_set"],
              "pairs": [{"id": pid, "prompt_sha256": sha256_text(p)}
                        for pid, p, _, _ in items]}
        Path(args.manifest).write_text(json.dumps(mf, ensure_ascii=False, indent=1),
                                       encoding="utf-8")

    print(json.dumps({k: res[k] for k in ("n_scored", "discards", "accuracy_raw",
                                          "accuracy_pmi", "margin_pmi", "latencia")},
                     ensure_ascii=False, indent=1), flush=True)
    return 0


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def set_sha(items) -> str:
    return sha256_text("\n".join(p for _, p, _, _ in items))


def main() -> int:
    ap = argparse.ArgumentParser(description="CBB margen PMI sobre llama-server (HTTP)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8092")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-pairs", type=int, default=30)
    ap.add_argument("--n-probs", type=int, default=1000)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--req-timeout", type=float, default=120)
    ap.add_argument("--device", default="Vulkan1 (Intel Iris Xe DG1)",
                    help="solo informativo, va al JSON de arch")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.manifest:
        Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    return self_check(a) if a.self_check else run(a)


if __name__ == "__main__":
    sys.exit(main())