#!/usr/bin/env python3
"""Eval de comportamiento — PARTE C: métricas de generación (§2.2–§2.5, §2.8).

Genera respuestas con sampling y computa métricas sobre ellas.
Reutiliza load_student, free_gpu, ensure_eval, percentile, PROJECT, PROBE_SEED
de eval_behavior.py.

Uso (local):
  systemd-run --user --scope -p MemoryMax=infinity \
    venv-unsloth-qwen/bin/python scripts/eval_generation.py \
    --ckpt out/rwkv_kateto_base/rwkv-1.pth \
    --chat-format chatml \
    --n-items 4 \
    --out out/eval/run1/gen_test.json

Uso (http — llama.cpp router):
  systemd-run --user --scope -p MemoryMax=infinity \
    python scripts/eval_generation.py \
    --backend http --model RWKV7-Kateto-Retrain \
    --chat-format chatml --n-items 4 \
    --out out/eval/run1/gen_http_kateto.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ROCm/torch pre-import env (must precede torch import via opd_rollout).
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "rwkv_pipeline"))

from chat_template import render_chat
from scripts.eval_behavior import load_student, free_gpu, ensure_eval, percentile, PROBE_SEED


# --------------------------------------------------------------------------- #
# Prompt formatting (§1.3 — already decided, do not re-litigate)
# --------------------------------------------------------------------------- #

def format_prompt(prompt_text: str, chat_format: str) -> str:
    if chat_format == "world":
        return f"User: {prompt_text}\n\nAssistant:"
    if chat_format == "chatml":
        return render_chat([], prompt_text, "seco")
    raise ValueError(f"unknown chat_format: {chat_format}")


# --------------------------------------------------------------------------- #
# Truncado por marcador de STOP (§2.4bis)
# --------------------------------------------------------------------------- #

STOP_MARKERS = {
    "world": ["\n\nUser:", "\nUser:", "\n\nAssistant:", "<|endoftext|>"],
    "chatml": ["<|im_end|>", "<|im_start|>", "\n<|im_user|>", "<|endoftext|>"],
}

# Marcadores que se le pasan al `stop` del SERVIDOR (llama.cpp).
#
# Por que NO son los mismos que STOP_MARKERS: pasar `<|im_start|>` como stop del
# servidor DESTRUYE LA EVIDENCIA. Medido 2026-09-12: cuando el modelo cae al eco
# del template, su primer token es el propio `<|im_start|>`; llama.cpp corta ahi y
# devuelve `content=""` con `tokens_predicted=7`, asi que el eco llega ya borrado.
# Resultado: 27/80 respuestas parecian "vacias" (silencio) cuando en realidad eran
# `>seco` en loop, y el detector de degeneracion — que exige len(texto)>=8 — no
# las veia. El truncado en Python (`truncate_at_stop`) ya hace ese trabajo, asi que
# el stop del servidor es redundante y encima borra el diagnostico.
#
# Solo se corta por el fin de turno REAL: eso evita que el modelo siga generando
# turnos falsos infinitos sin ocultar el eco.
SERVER_STOP_MARKERS = {
    "world": ["\n\nUser:", "\nUser:", "\n\nAssistant:", "<|endoftext|>"],
    "chatml": ["<|im_end|>", "<|endoftext|>"],
}


def truncate_at_stop(text: str, chat_format: str):
    """Corta la respuesta en el primer marcador de STOP. Devuelve (cortado, marcador).

    Por que existe: en el smoke test TODAS las respuestas pegaron en max_len
    (filler=1.0, finished=0.0), asi que §2.4 medía el techo y no la respuesta
    (median=max_len, iqr=0.0, cv=0.0) y §2.2/§2.3 leian texto de relleno.
    Truncar en el marcador hace que longitud y rates se midan sobre la respuesta
    REAL. El texto crudo completo se guarda igual en `text_raw`, y la fraccion
    que alcanzo un marcador se reporta como `stop_marker_rate` (mide terminacion).
    """
    best_i, best_m = len(text), None
    for m in STOP_MARKERS.get(chat_format, []):
        i = text.find(m)
        if i != -1 and i < best_i:
            best_i, best_m = i, m
    return text[:best_i], best_m


# --------------------------------------------------------------------------- #
# Pattern loading from YAML
# --------------------------------------------------------------------------- #

# Keys that map to §2.2 (assistant vice)
ASSISTANT_VICE_KEYS = {"slop_opening", "reflexive_question", "not_x_but_y",
                       "hedge_close", "list_format",
                       # think_leak: razonamiento filtrado a la respuesta visible,
                       # ademas en INGLES. Medido en el base 0.4B: emite bloques
                       # <think> narrando al usuario ("the user shares a story").
                       # Se agrego DESPUES del primer YAML y por eso estaba siendo
                       # ignorado en silencio: esta lista es la que decide que se mide.
                       "think_leak"}
# Keys that map to §2.3 (persona)
PERSONA_KEYS = {"voseo", "opinion"}


def load_patterns(yaml_path: str) -> dict:
    """Load YAML, return {category_key: compiled_patterns}.

    Each pattern dict: {compiled_re, original_str, why}
    list_format is special: it has bullet/numbered/threshold.
    """
    try:
        import yaml
    except ImportError:
        # Fallback: minimal YAML parser for this specific file structure
        return _load_patterns_fallback(yaml_path)

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    out = {}
    for key, val in raw.items():
        if key in ASSISTANT_VICE_KEYS:
            if key == "list_format":
                # Special structure: {bullet, numbered, threshold}
                out[key] = {"type": "list_format",
                            "bullet": re.compile(val["bullet"]),
                            "numbered": re.compile(val["numbered"]),
                            "threshold": val["threshold"],
                            "why": "formato lista (más del 40% de líneas)"}
            else:
                entries = []
                for item in val:
                    entries.append({"compiled": re.compile(item["pattern"]),
                                    "original": item["pattern"],
                                    "why": item.get("why", "")})
                out[key] = entries
        elif key in PERSONA_KEYS:
            entries = []
            for item in val:
                entries.append({"compiled": re.compile(item["pattern"]),
                                "original": item["pattern"],
                                "why": item.get("why", "")})
            out[key] = entries
        elif key == "lunfardo":
            # List of literal words/regexes
            entries = []
            for item in val:
                entries.append({"compiled": re.compile(item, re.IGNORECASE),
                                "original": item})
            out[key] = entries
    return out


def _load_patterns_fallback(yaml_path: str) -> dict:
    """Minimal fallback if PyYAML not installed."""
    text = Path(yaml_path).read_text(encoding="utf-8")
    # This is fragile — prefer pip install pyyaml
    raise ImportError(
        f"PyYAML not installed. Install with: pip install pyyaml\n"
        f"Attempted to load {yaml_path}"
    )


# --------------------------------------------------------------------------- #
# §2.8 — Degeneracy detection
# --------------------------------------------------------------------------- #

def _check_degenerate(text: str) -> bool:
    """True if any substring of 8+ chars is repeated ≥3 times."""
    t = text.strip()
    if len(t) < 8:
        return False
    for n in range(8, min(len(t) // 3 + 1, 65)):
        for i in range(len(t) - 2 * n + 1):
            sub = t[i:i + n]
            if t.count(sub) >= 3:
                return True
    return False


# Fragmentos que el modelo emite cuando cae al eco del template en vez de responder.
# `<|im_start|>` no es un token especial en este vocab (son 7 tokens: [61,125,1949,
# 96,35691,125,63]), asi que el eco arranca en `<|im_start` y sigue con `>seco`.
# Medido 2026-09-12: es el modo de fallo dominante con prompts cortos.
# `[...]` es un marcador del DATASET (aparece en el train) que el modelo regurgita.
_TEMPLATE_FRAGMENTS = ("<|im_start|>", "<|im_end|>", "<|im_user|>", ">seco",
                       "<|endoftext|>", "[...]")
# El marcador arrastra el nombre de la voz pegado (`<|im_start|>seco`). Si solo se
# saca el marcador, queda `seco` suelto y el texto NO queda vacio -> se clasificaba
# como respuesta valida. Medido: '<|im_start|>seco\n<|im_start|>seco\n' -> None.
_TEMPLATE_RX = re.compile(r"<\|im_(?:start|end|user)\|>\s*\w*|<\|endoftext\|>|\[\.\.\.\]|>seco")


def classify_nonanswer(text: str, prompt: str) -> str | None:
    """Clasifica una respuesta que NO es una respuesta. None si es utilizable.

    - "template_echo": lo unico que hay es el marcador del template o su cola
      (`>seco`), o sea el modelo reconstruye la estructura en vez de contestar.
    - "prompt_echo": repite las palabras del usuario (>70%).

    Por que existe: sin esto, un eco de template llega sin texto (si el server lo
    corto por `stop`) o entra en `degenerate_rate` de casualidad, y se reporta como
    "vacia" — que se lee como "el modelo se callo" cuando en realidad ECO el
    template. Son fallos distintos y se arreglan distinto.

    OJO: se exige que NO QUEDE NADA despues de sacar los fragmentos de template.
    Un umbral de largo (`< 8 chars`) marcaba como eco respuestas cortas pero
    VALIDAS (`"2"`, `"Si"`, `"No"`), que es peor que no medir: reporta como fallo
    del modelo una respuesta correcta. Medido 2026-09-12 con `"2 + 2?" -> "2"`.
    """
    t = (text or "").strip()
    rest = _TEMPLATE_RX.sub("", t)
    for frag in _TEMPLATE_FRAGMENTS:
        rest = rest.replace(frag, "")
    rest = "".join(ch for ch in rest if not ch.isspace())
    if not rest:
        return "template_echo"
    pw = set((prompt or "").lower().split())
    tw = t.lower().split()
    if tw and len(tw) > 3 and sum(1 for w in tw if w in pw) / len(tw) > 0.70:
        return "prompt_echo"
    return None


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def compute_metrics(responses: list, patterns: dict, tc_responses: list) -> dict:
    """Compute §2.2–§2.5, §2.8 from raw responses.

    responses: list of {id, prompt, n_tokens, finished, text} — gen items
    tc_responses: list of {id, prompt, n_tokens, finished, text} — debate items for §2.5
    patterns: from load_patterns()
    """
    n = len(responses)
    if n == 0:
        return {"error": "no responses"}

    # Usar los tokens TRUNCADOS (post stop-marker) para las densidades: si no, el
    # relleno hasta max_len diluye la densidad de lunfardo.
    total_tokens = sum(r.get("n_tokens_truncated", r["n_tokens"]) for r in responses)

    # --- §2.2 Assistant vice ---
    vice_rates = {}
    for key in ASSISTANT_VICE_KEYS:
        if key not in patterns:
            continue
        pat_def = patterns[key]
        if isinstance(pat_def, dict) and pat_def.get("type") == "list_format":
            # List format: proportion of lines matching bullet/numbered
            n_match = 0
            for r in responses:
                lines = r["text"].strip().split("\n")
                if not lines:
                    continue
                match_count = sum(
                    1 for ln in lines
                    if pat_def["bullet"].search(ln) or pat_def["numbered"].search(ln)
                )
                if match_count / max(len(lines), 1) >= pat_def["threshold"]:
                    n_match += 1
            rate = n_match / n
            vice_rates[key] = {"rate": round(rate, 4), "n_match": n_match,
                               "n_total": n, "why": pat_def["why"]}
        else:
            # List of regex patterns
            for pi, pat in enumerate(pat_def):
                n_match = sum(1 for r in responses if pat["compiled"].search(r["text"]))
                rate = n_match / n
                tag = f"{key}_{pi}" if len(pat_def) > 1 else key
                vice_rates[tag] = {"rate": round(rate, 4), "n_match": n_match,
                                   "n_total": n, "why": pat.get("why", ""),
                                   "pattern": pat["original"]}

    # --- §2.3 Persona ---
    persona_rates = {}
    for key in PERSONA_KEYS:
        if key not in patterns:
            continue
        for pi, pat in enumerate(patterns[key]):
            n_match = sum(1 for r in responses if pat["compiled"].search(r["text"]))
            rate = n_match / n
            total_matches = sum(len(pat["compiled"].findall(r["text"]))
                                for r in responses)
            density = (total_matches / total_tokens * 100) if total_tokens > 0 else 0.0
            tag = f"{key}_{pi}" if len(patterns[key]) > 1 else key
            persona_rates[tag] = {"rate": round(rate, 4), "n_match": n_match,
                                  "n_total": n, "density": round(density, 4),
                                  "why": pat.get("why", ""),
                                  "pattern": pat["original"]}

    # Lunfardo density
    if "lunfardo" in patterns:
        total_lunf = 0
        per_item_lunf = []
        for r in responses:
            item_lunf = sum(len(p["compiled"].findall(r["text"]))
                           for p in patterns["lunfardo"])
            total_lunf += item_lunf
            per_item_lunf.append(item_lunf)
        lunf_density = (total_lunf / total_tokens * 100) if total_tokens > 0 else 0.0
        persona_rates["lunfardo_density"] = {
            "density_per_100tok": round(lunf_density, 4),
            "total_matches": total_lunf,
            "n_total": n,
        }

    # --- §2.4 Length distribution ---
    # Longitud de la RESPUESTA (truncada en el stop marker), no del relleno hasta
    # max_len. Con relleno la mediana es max_len por construccion e IQR=0.
    lens = [r.get("n_tokens_truncated", r["n_tokens"]) for r in responses]
    len_median = statistics.median(lens)
    len_p10 = percentile(lens, 10)
    len_p90 = percentile(lens, 90)
    len_iqr = (percentile(lens, 75) or 0) - (percentile(lens, 25) or 0)
    len_mean = statistics.fmean(lens)
    len_stdev = statistics.stdev(lens) if len(lens) > 1 else 0.0
    len_cv = (len_stdev / len_mean) if len_mean > 0 else 0.0

    length_dist = {
        "n_scored": n,
        "median": round(len_median, 2),
        "p10": round(len_p10, 2),
        "p90": round(len_p90, 2),
        "iqr": round(len_iqr, 2),
        "mean": round(len_mean, 2),
        "stdev": round(len_stdev, 2),
        "cv": round(len_cv, 4),
    }

    # --- §2.5 Tool-call discipline ---
    tc_matches = 0
    for r in tc_responses:
        txt = r["text"]
        if ("<tool_call" in txt or "<|tool_call|>" in txt
                or ('"name"' in txt and "{" in txt)):
            tc_matches += 1
    tc_rate = (tc_matches / len(tc_responses)) if tc_responses else 0.0

    toolcall = {
        "toolcall_rate": round(tc_rate, 4),
        "n_match": tc_matches,
        "n_scored": len(tc_responses),
        "n_total_debate": len(tc_responses),
    }

    # --- §2.8 Degeneracy ---
    n_degen = sum(1 for r in responses if _check_degenerate(r["text"]))
    n_finished = sum(1 for r in responses if r["finished"])
    # filler = finished=False (hit max_len without eos)
    n_filler = n - n_finished

    n_stop = sum(1 for r in responses if r.get("stop_marker"))

    # --- §2.8bis: POR QUE no es una respuesta ---
    kinds = [classify_nonanswer(r.get("text", ""), r.get("prompt", "")) for r in responses]
    n_template_echo = sum(1 for k in kinds if k == "template_echo")
    n_prompt_echo = sum(1 for k in kinds if k == "prompt_echo")
    n_usable = sum(1 for k in kinds if k is None) if n else 0
    n_nonanswer = n - n_usable

    degen = {
        "degenerate_rate": round(n_degen / n, 4),
        "n_degenerate": n_degen,
        "n_usable": n_usable,
        "usable_rate": round(n_usable / n, 4) if n else 0.0,
        "nonanswer_rate": round(n_nonanswer / n, 4) if n else 0.0,
        "template_echo_rate": round(n_template_echo / n, 4),
        "n_template_echo": n_template_echo,
        "prompt_echo_rate": round(n_prompt_echo / n, 4),
        "n_prompt_echo": n_prompt_echo,
        "finished_rate": round(n_finished / n, 4),
        "n_finished": n_finished,
        "filler_rate": round(n_filler / n, 4),
        "n_filler": n_filler,
        "stop_marker_rate": round(n_stop / n, 4),
        "n_stop_marker": n_stop,
        "stop_markers_seen": {m: sum(1 for r in responses if r.get("stop_marker") == m)
                              for m in {r.get("stop_marker") for r in responses} if m},
        "n_total": n,
    }

    return {
        "assistant_vice": {"n_scored": n, **vice_rates},
        "persona": {"n_scored": n, **persona_rates},
        "length_distribution": length_dist,
        "toolcall_discipline": toolcall,
        "degeneracy": degen,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Eval generación — §2.2–§2.5, §2.8")
    ap.add_argument("--backend", default="local", choices=["local", "http"],
                    help="local = .pth directo, http = llama.cpp router")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434",
                    help="llama.cpp router base URL (http backend only)")
    ap.add_argument("--model", default=None,
                    help="model preset name for http backend (e.g. RWKV-2.9B)")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint path (local backend only)")
    # default "chatml": el formato propio de Kateto (<|im_user|>...<|im_seco|>).
    # "world" (User:/Assistant:) es el formato OFICIAL de RWKV-G1, que el modelo
    # nunca vio en entrenamiento: con ese prompt narra las reglas en vez de hablar
    # (medido 2026-10-10) -> el falso "no habla". No cambiar el default a world.
    ap.add_argument("--chat-format", default="chatml", choices=["world", "chatml"],
                    help="formato del prompt (§1.3)")
    ap.add_argument("--prompts", default=str(PROJECT / "out/eval/prompts_v1.json"))
    ap.add_argument("--patterns", default=str(PROJECT / "config/behavior_patterns.yaml"))
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.80)
    ap.add_argument("--top-p", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--repeat-penalty", type=float, default=1.30,
                    help="OBLIGATORIO para este modelo: sin el se traba repitiendo el "
                         "marcador del template y las respuestas se reportan como vacias "
                         "(usable 0.57 -> 0.93 medido). 0 lo desactiva.")
    ap.add_argument("--n-items", type=int, default=None,
                    help="limit de prompts (None = todos)")
    args = ap.parse_args()

    if args.backend == "local" and not args.ckpt:
        ap.error("--ckpt is required for local backend")
    if args.backend == "http" and not args.model:
        ap.error("--model is required for http backend")

    ckpt_path = args.ckpt
    out_path = Path(args.out) if args.out else (
        PROJECT / f"out/eval/run1/gen_{args.model or Path(ckpt_path).stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load prompts
    prompts_data = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    gen_items = prompts_data["generation"]
    debate_tc_items = prompts_data.get("debate_for_toolcall", [])
    if args.n_items is not None:
        gen_items = gen_items[:args.n_items]

    # Load patterns
    patterns = load_patterns(args.patterns)

    # Generate responses
    use_http = args.backend == "http"
    info = {}
    server_stop: list[str] = []

    if use_http:
        from scripts.eval_http_backend import generate_http, wait_for_model
        print(f"[gen] http backend: waiting for model {args.model}...", flush=True)
        if not wait_for_model(args.base_url, args.model):
            print(f"[gen] WARNING: model {args.model} did not load in time", flush=True)
        stop_markers = STOP_MARKERS.get(args.chat_format, [])          # truncado en Python
        server_stop = SERVER_STOP_MARKERS.get(args.chat_format, [])    # stop del servidor
        print(f"[gen] server stop={server_stop} (el truncado completo lo hace Python)", flush=True)
    else:
        import torch
        torch.manual_seed(args.seed)
        print(f"[gen] loading {ckpt_path}", flush=True)
        model, tok, info = load_student(ckpt_path, device="cuda")
        ensure_eval(model)
        from rwkv_pipeline.opd_rollout import rollout_from_prompt, _decode_str

    responses = []
    t0 = time.time()
    for i, item in enumerate(gen_items):
        ptext = format_prompt(item["prompt"], args.chat_format)
        try:
            if use_http:
                r = generate_http(
                    args.base_url, args.model, ptext,
                    max_len=args.max_len, temperature=args.temperature,
                    top_p=args.top_p, seed=args.seed, stop=server_stop,
                    repeat_penalty=(args.repeat_penalty or None),
                )
                text = r["text"]
                text_cut, stop_m = truncate_at_stop(text, args.chat_format)
                responses.append({
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "n_tokens": r["n_tokens"],
                    "n_chars_truncated": len(text_cut),
                    "finished": r["stopped"],
                    "stop_marker": stop_m,
                    # Evidencia del stop del SERVIDOR: sin esto no se puede distinguir
                    # "el modelo se callo" de "el server corto por palabra prohibida".
                    "server_stop_type": r.get("stop_type"),
                    "server_stopping_word": r.get("stopping_word"),
                    "text": text_cut,
                    "text_raw": text,
                })
            else:
                r = rollout_from_prompt(
                    model, tok, ptext,
                    max_len=args.max_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                text = _decode_str(tok, list(r.response_ids))
                text_cut, stop_m = truncate_at_stop(text, args.chat_format)
                responses.append({
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "n_tokens": len(r.response_ids),
                    "n_tokens_truncated": len(tok.encode(text_cut)),
                    "finished": bool(r.finished),
                    "stop_marker": stop_m,
                    "text": text_cut,
                    "text_raw": text,
                })
        except Exception as e:
            responses.append({
                "id": item["id"],
                "prompt": item["prompt"],
                "n_tokens": 0,
                "finished": False,
                "text": "",
                "error": f"{type(e).__name__}: {e}",
            })
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[gen] {i+1}/{len(gen_items)} done, {elapsed:.1f}s", flush=True)

    wall = time.time() - t0
    print(f"[gen] {len(gen_items)} responses in {wall:.1f}s", flush=True)

    # Generate §2.5 debate responses
    tc_items_active = [it for it in debate_tc_items
                       if not it.get("expects_toolcall", True)]
    tc_responses = []
    if tc_items_active:
        print(f"[gen] generating {len(tc_items_active)} debate responses for §2.5",
              flush=True)
        for i, item in enumerate(tc_items_active):
            ptext = format_prompt(item["prompt"], args.chat_format)
            try:
                if use_http:
                    r = generate_http(
                        args.base_url, args.model, ptext,
                        max_len=args.max_len, temperature=args.temperature,
                        top_p=args.top_p, seed=args.seed, stop=stop_markers,
                    )
                    tc_responses.append({
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "n_tokens": r["n_tokens"],
                        "finished": r["stopped"],
                        "text": r["text"],
                    })
                else:
                    r = rollout_from_prompt(
                        model, tok, ptext,
                        max_len=args.max_len,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    text = _decode_str(tok, list(r.response_ids))
                    tc_responses.append({
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "n_tokens": len(r.response_ids),
                        "finished": bool(r.finished),
                        "text": text,
                    })
            except Exception as e:
                tc_responses.append({
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "n_tokens": 0,
                    "finished": False,
                    "text": "",
                    "error": f"{type(e).__name__}: {e}",
                })

    if not use_http:
        free_gpu(model)

    # Compute metrics
    metrics = compute_metrics(responses, patterns, tc_responses)

    # Assemble output
    result = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arch": info.get("arch", {}) if not use_http else {},
        "backend": "http" if use_http else "local",
        "model": args.model if use_http else ckpt_path,
        "chat_format": args.chat_format,
        "repeat_penalty": args.repeat_penalty,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_len": args.max_len,
        "n_items": len(gen_items),
        "wall_s": round(wall, 1),
        **metrics,
        "raw_responses": responses,
        "raw_responses_debate": tc_responses,
        "notes": [],
    }

    # Check determinism note
    n_ok = sum(1 for r in responses if not r.get("error"))
    if n_ok < len(responses):
        result["notes"].append(
            f"{len(responses) - n_ok} items errored during generation")

    if metrics.get("toolcall_discipline", {}).get("n_scored", 0) == 0:
        result["notes"].append(
            "§2.5: no items with expects_toolcall=false found in prompts file")

    if use_http:
        result["notes"].append(
            "§2.1 (CBB/PMI) not runnable via HTTP backend: llama.cpp router "
            "cannot score arbitrary continuations (echo:true + max_tokens:0 + "
            "n_probs ignored). Only generation metrics §2.2–§2.5, §2.8 measured.")
        result["notes"].append(
            "n_tokens_truncated replaced by n_chars_truncated (char count) "
            "when backend=http: no tokenizer available server-side.")

    # Write output
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"[gen] written {out_path}", flush=True)

    # Summary
    v = metrics.get("assistant_vice", {})
    p = metrics.get("persona", {})
    ld = metrics.get("length_distribution", {})
    tc = metrics.get("toolcall_discipline", {})
    dg = metrics.get("degeneracy", {})
    print(f"\n=== Summary ===")
    print(f"Responses: {len(responses)} ({n_ok} ok)")
    print(f"§2.2 assistant_vice: ", end="")
    for k, v2 in v.items():
        if isinstance(v2, dict) and "rate" in v2:
            print(f"{k}={v2['rate']:.2f} ", end="")
    print()
    print(f"§2.3 persona: ", end="")
    for k, v2 in p.items():
        if isinstance(v2, dict) and "rate" in v2:
            print(f"{k}={v2['rate']:.2f} ", end="")
    print()
    print(f"§2.4 length: median={ld.get('median')} iqr={ld.get('iqr')} "
          f"cv={ld.get('cv')}")
    print(f"§2.5 toolcall_rate={tc.get('toolcall_rate')} "
          f"(n_scored={tc.get('n_scored')})")
    print(f"§2.8 degen={dg.get('degenerate_rate')} "
          f"finished={dg.get('finished_rate')} "
          f"filler={dg.get('filler_rate')}")
    _n_tot = (dg.get('n_usable', 0) + dg.get('n_template_echo', 0) + dg.get('n_prompt_echo', 0))
    print(f"§2.8bis UTILIZABLES={dg.get('n_usable')}/{_n_tot} ({dg.get('usable_rate')}) | "
          f"eco_template={dg.get('n_template_echo')} ({dg.get('template_echo_rate')}) | "
          f"eco_prompt={dg.get('n_prompt_echo')} ({dg.get('prompt_echo_rate')})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
