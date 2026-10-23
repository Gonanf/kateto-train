#!/usr/bin/env python3
"""Evaluador de coherencia larga sobre documentos de 1M tokens (Kateto RWKV29).

Mide la coherencia por par (turno frío vs turno caliente) utilizando el endpoint
/completion de llama.cpp con reuso de prefijo vía cache_prompt=true.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).parent))
from analyze_eval29_coherence import build_pairs, check_degenerate, summarize

PROBES = [
    "Che, ¿de qué se habló hasta acá? Contame de qué viene la charla.",
    "Bueno, y el invitado, ¿qué contó?",
    "¿Qué se dijo del doctor, te acordás?",
    "Retomá el tema que veníamos charlando, ¿por dónde íbamos?",
    "¿Hay algo que ya hayamos hablado y no quieras repetir?",
    "Cerrá con un resumen corto de todo lo anterior.",
]

TURNOS = ("frio", "caliente")


def file_sha256(path: Path) -> str:
    """Calcula SHA-256 en streaming (bloques de 1 MiB)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def post_json(url: str, payload: dict) -> dict:
    """Envía un POST JSON al backend llama.cpp y retorna la respuesta deserializada."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        sys.stderr.write(f"Error communicating with llama.cpp at {url}: {e}\n")
        sys.exit(1)


def count_tokens(base_url: str, text: str) -> int:
    """Tokeniza el texto vía POST /tokenize y retorna la cantidad de tokens."""
    if not text:
        return 0
    url = f"{base_url.rstrip('/')}/tokenize"
    res = post_json(url, {"content": text})
    return len(res.get("tokens", []))


def truncate_doc(base_url: str, text: str, max_tokens: int) -> tuple[str, int]:
    """Trunca el texto usando búsqueda binaria sobre caracteres hasta que quepa en max_tokens."""
    total_tokens = count_tokens(base_url, text)
    if total_tokens <= max_tokens:
        return text, total_tokens

    low = 0
    high = len(text)
    best_idx = 0
    best_tokens = 0
    while low <= high:
        mid = (low + high) // 2
        toks = count_tokens(base_url, text[:mid])
        if toks <= max_tokens:
            best_idx = mid
            best_tokens = toks
            low = mid + 1
        else:
            high = mid - 1
    return text[:best_idx], best_tokens


def format_probe_chatml(probe: str) -> str:
    """Formatea el probe en ChatML (voz seco)."""
    return f"<|im_start|>user\n{probe}<|im_end|>\n<|im_start|>assistant\n"


def enrich_rows_for_coherence(rows: list[dict]) -> None:
    """Agrega campos recalculados necesarios para build_pairs."""
    for r in rows:
        deg, max_rep = check_degenerate(r.get("text") or "")
        r["_deg_recalc"] = deg
        r["_max_repeat_recalc"] = max_rep
        r["_chars"] = len((r.get("text") or "").strip())


def run_self_check() -> None:
    """Ejecuta self-check sin red usando filas sintéticas."""
    def mk_dummy_row(id_: str, turno: str, prompt: str, text: str, deg: bool, rep: int, seed: int = 1337) -> dict:
        return {
            "set": "doc00", "id": id_, "turno": turno, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "text": text, "degenerate": deg, "max_repeat": rep, "n_tokens": 10,
            "ttfab_s": "0.100", "wall_s": "0.200", "model": "kateto-rwkv29-Q4_K_M.gguf",
            "motor": "llama.cpp", "device": "127.0.0.1:8199", "chat_format": "chatml",
            "doc_id": "doc00", "doc_tokens": 100, "seed": seed,
        }

    rows = [
        # Par coherente: ambos turnos presentes y ninguno degenerado
        mk_dummy_row("p01-s1337", "frio", PROBES[0], "hola que tal", False, 0),
        mk_dummy_row("p01-s1337", "caliente", PROBES[0], "hola que tal mundo", False, 0),
        # Par con turno degenerado (frio degenerado)
        mk_dummy_row("p02-s1337", "frio", PROBES[1], "bueno" * 20, True, 5),
        mk_dummy_row("p02-s1337", "caliente", PROBES[1], "otra cosa distinta", False, 0),
        # Par sin turno caliente (solo frio)
        mk_dummy_row("p03-s1337", "frio", PROBES[2], "solo frio presente", False, 0),
        # Dummy
        mk_dummy_row("dummy_00", "frio_proceso", "warmup", "warmup test", False, 0),
    ]
    enrich_rows_for_coherence(rows)
    pairs = build_pairs(rows)
    assert len(pairs) == 3, f"Esperado 3 pares, obtenido {len(pairs)}"
    summ = summarize(pairs)["total"]
    assert summ["pares"] == 3
    assert summ["pares_con_los_dos_turnos"] == 2
    assert summ["pares_con_ge1_turno_degenerado"] == 1
    assert summ["pares_coherentes_en_ambos"] == 1
    assert summ["turnos_frio_degenerados"] == 1
    assert summ["turnos_caliente_degenerados"] == 0

    metric = round(summ["pares_coherentes_en_ambos"] / summ["pares"], 4)
    assert metric == round(1 / 3, 4)

    frio_chars = [r["_chars"] for r in rows if r.get("turno") == "frio"]
    cal_chars = [r["_chars"] for r in rows if r.get("turno") == "caliente"]
    mean_frio = statistics.mean(frio_chars)
    mean_cal = statistics.mean(cal_chars)
    ratio = round(mean_cal / mean_frio, 4)
    assert ratio < 0.8
    assert check_degenerate("hola que tal")[0] is False
    assert check_degenerate("bueno" * 20)[0] is True
    print("self-check OK")
    sys.exit(0)


def evaluate_turn(
    base_url: str,
    prompt: str,
    probe_text: str,
    seed: int,
    n_predict: int,
    turno: str,
    doc_id: str,
    doc_tokens: int,
    probe_idx: int,
    model_label: str,
    device_str: str,
) -> dict:
    url = f"{base_url.rstrip('/')}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.8,
        "top_p": 0.7,
        "repeat_penalty": 1.3,
        "seed": seed,
        "cache_prompt": True,
        "stop": ["<|im_end|>"],
    }
    t0 = time.time()
    res = post_json(url, payload)
    wall_s = time.time() - t0

    content = res.get("content") or ""
    text = content.strip()
    deg, max_rep = check_degenerate(text)
    timings = res.get("timings") or {}
    prompt_ms = timings.get("prompt_ms") or 0.0
    ttfab_s = f"{prompt_ms / 1000.0:.4f}"
    tokens_predicted = res.get("tokens_predicted") or timings.get("predicted_n") or 0

    row_id = f"p{probe_idx + 1:02d}-s{seed}"
    prompt_sha = hashlib.sha256(probe_text.encode("utf-8")).hexdigest()

    return {
        "set": doc_id,
        "id": row_id,
        "turno": turno,
        "prompt": probe_text,
        "prompt_sha256": prompt_sha,
        "text": text,
        "degenerate": deg,
        "max_repeat": max_rep,
        "n_tokens": int(tokens_predicted),
        "ttfab_s": ttfab_s,
        "wall_s": f"{wall_s:.4f}",
        "model": model_label,
        "motor": "llama.cpp",
        "device": device_str,
        "chat_format": "chatml",
        "doc_id": doc_id,
        "doc_tokens": doc_tokens,
        "seed": seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluador de coherencia larga")
    ap.add_argument("--docs", help="Ruta al archivo JSONL con documentos")
    ap.add_argument("--out-dir", help="Directorio de salida")
    ap.add_argument("--max-tokens", type=int, default=1000000, help="Tokens máximos por doc")
    ap.add_argument("--seeds", default="1337,7,2024", help="Semillas separadas por coma")
    ap.add_argument("--max-probes", type=int, default=6, help="Máximo de probes a usar")
    ap.add_argument("--max-docs", type=int, default=None, help="Máximo de documentos a evaluar")
    ap.add_argument("--base-url", default="http://127.0.0.1:8199", help="URL base del backend llama.cpp")
    ap.add_argument("--n-predict", type=int, default=64, help="Tokens a predecir por turno")
    ap.add_argument("--model-label", default="kateto-rwkv29-Q4_K_M.gguf", help="Etiqueta del modelo")
    ap.add_argument("--model-path", default=None, help="Ruta al archivo del modelo")
    ap.add_argument("--rows", default=None, help="Ruta de salida para rows.jsonl")
    ap.add_argument("--report", default=None, help="Ruta de salida para eureka-largo.json")
    ap.add_argument("--self-check", action="store_true", help="Ejecutar self-check local sin red")
    args = ap.parse_args()

    if args.self_check:
        run_self_check()
        return 0

    if not args.docs:
        ap.error("falta --docs (o usar --self-check)")
    if not args.out_dir:
        ap.error("falta --out-dir (o usar --self-check)")

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    out_dir = Path(args.out_dir)
    rows_path = Path(args.rows) if args.rows else (out_dir / "rows.jsonl")
    report_path = Path(args.report) if args.report else (out_dir / "eureka-largo.json")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    probes = PROBES[: args.max_probes] if args.max_probes else PROBES

    docs_file = Path(args.docs)
    raw_lines = [l for l in docs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    docs_raw = [json.loads(l) for l in raw_lines]

    # Para el smoke de prueba rápida, si no se especifica --max-docs y se limitaron los probes,
    # se evalúa 1 doc.
    if args.max_docs is not None:
        docs_raw = docs_raw[: args.max_docs]
    elif args.max_probes is not None and args.max_probes < len(PROBES):
        docs_raw = docs_raw[:1]

    device_str = (
        args.base_url.replace("http://", "").replace("https://", "").rstrip("/")
    )

    all_rows: list[dict] = []
    docs_summary: list[dict] = []
    docs_processed: list[dict] = []

    print(f"Iniciando evaluación: {len(docs_raw)} doc(s), {len(probes)} probe(s), {len(seeds)} seed(s)")

    for doc_idx, doc_item in enumerate(docs_raw):
        doc_id = f"doc{doc_idx:02d}"
        doc_text, doc_tokens = truncate_doc(args.base_url, doc_item.get("text", ""), args.max_tokens)
        print(f"[{doc_id}] Chars: {len(doc_text)} | Tokens: {doc_tokens}")

        doc_rows: list[dict] = []

        # 1. Turnos caliente: el doc va adelante, crudo, reutilizando el prefijo con cache_prompt=true
        for seed in seeds:
            for p_idx, probe in enumerate(probes):
                probe_chatml = format_probe_chatml(probe)
                prompt = doc_text + probe_chatml
                r = evaluate_turn(
                    base_url=args.base_url,
                    prompt=prompt,
                    probe_text=probe,
                    seed=seed,
                    n_predict=args.n_predict,
                    turno="caliente",
                    doc_id=doc_id,
                    doc_tokens=doc_tokens,
                    probe_idx=p_idx,
                    model_label=args.model_label,
                    device_str=device_str,
                )
                doc_rows.append(r)
                print(f"  [caliente] {r['id']} | wall: {r['wall_s']}s | ttfab: {r['ttfab_s']}s | deg: {r['degenerate']}")

        # 2. Turnos frío: prompt_del_probe solo (sin doc, estado limpio)
        for seed in seeds:
            for p_idx, probe in enumerate(probes):
                probe_chatml = format_probe_chatml(probe)
                prompt = probe_chatml
                r = evaluate_turn(
                    base_url=args.base_url,
                    prompt=prompt,
                    probe_text=probe,
                    seed=seed,
                    n_predict=args.n_predict,
                    turno="frio",
                    doc_id=doc_id,
                    doc_tokens=doc_tokens,
                    probe_idx=p_idx,
                    model_label=args.model_label,
                    device_str=device_str,
                )
                doc_rows.append(r)
                print(f"  [frio]     {r['id']} | wall: {r['wall_s']}s | ttfab: {r['ttfab_s']}s | deg: {r['degenerate']}")

        all_rows.extend(doc_rows)
        docs_processed.append({
            "doc_id": doc_id,
            "chars": len(doc_text),
            "doc_tokens": doc_tokens,
            "doc_rows": doc_rows,
        })

    # Ordenar filas agrupadas por par (frio antes de caliente) para rows.jsonl
    order_turno = {"frio": 0, "caliente": 1}
    all_rows.sort(key=lambda r: (r["set"], r["id"], order_turno.get(r["turno"], 2)))

    with rows_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Filas escritas -> {rows_path}")

    # Enriquecer para build_pairs
    enrich_rows_for_coherence(all_rows)
    all_pairs = build_pairs(all_rows)
    summ = summarize(all_pairs)
    tot = summ["total"]

    # Desglose por documento
    for dp in docs_processed:
        doc_id = dp["doc_id"]
        doc_pairs = [p for p in all_pairs if p["set"] == doc_id]
        doc_rows = dp["doc_rows"]
        doc_frio_chars = [r["_chars"] for r in doc_rows if r.get("turno") == "frio"]
        doc_cal_chars = [r["_chars"] for r in doc_rows if r.get("turno") == "caliente"]
        doc_mean_frio = statistics.mean(doc_frio_chars) if doc_frio_chars else 0.0
        doc_mean_cal = statistics.mean(doc_cal_chars) if doc_cal_chars else 0.0

        per_seed = {}
        for seed in seeds:
            s_pairs = [p for p in doc_pairs if p["id"].endswith(f"-s{seed}")]
            s_cnt = len(s_pairs)
            s_coh = sum(1 for p in s_pairs if p["coherent_both"])
            s_ratio = round(s_coh / s_cnt, 4) if s_cnt else 0.0
            per_seed[str(seed)] = {
                "pairs": s_cnt,
                "coherent_both": s_coh,
                "ratio": s_ratio,
            }

        doc_pairs_cnt = len(doc_pairs)
        doc_coh = sum(1 for p in doc_pairs if p["coherent_both"])
        doc_ratio = round(doc_coh / doc_pairs_cnt, 4) if doc_pairs_cnt else 0.0

        docs_summary.append({
            "doc_id": doc_id,
            "path": str(Path(args.docs).resolve()),
            "chars": dp["chars"],
            "doc_tokens": dp["doc_tokens"],
            "pairs": doc_pairs_cnt,
            "coherent_both": doc_coh,
            "ratio": doc_ratio,
            "mean_chars_frio": round(doc_mean_frio, 4),
            "mean_chars_caliente": round(doc_mean_cal, 4),
            "per_seed": per_seed,
        })

    # Anti-hack
    frio_chars_all = [r["_chars"] for r in all_rows if r.get("turno") == "frio"]
    cal_chars_all = [r["_chars"] for r in all_rows if r.get("turno") == "caliente"]
    mean_chars_frio = statistics.mean(frio_chars_all) if frio_chars_all else 0.0
    mean_chars_cal = statistics.mean(cal_chars_all) if cal_chars_all else 0.0
    ah_ratio = (mean_chars_cal / mean_chars_frio) if mean_chars_frio > 0 else 0.0
    anti_hack = {
        "mean_chars_frio": round(mean_chars_frio, 4),
        "mean_chars_caliente": round(mean_chars_cal, 4),
        "ratio": round(ah_ratio, 4),
        "threshold": 0.8,
        "pass": ah_ratio >= 0.8,
    }

    metric = round(tot["pares_coherentes_en_ambos"] / tot["pares"], 4) if tot["pares"] else 0.0
    metric_detail = {
        "pares": tot["pares"],
        "pares_coherentes_en_ambos": tot["pares_coherentes_en_ambos"],
        "turnos_frio_degenerados": tot["turnos_frio_degenerados"],
        "turnos_caliente_degenerados": tot["turnos_caliente_degenerados"],
    }

    model_sha = file_sha256(Path(args.model_path)) if args.model_path else None
    model_path_str = str(Path(args.model_path).resolve()) if args.model_path else None
    model_entry = {
        "label": args.model_label,
        "path": model_path_str,
        "sha256": model_sha,
    }

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    report_data = {
        "benchmark": "coherencia-larga",
        "direction": "max",
        "metric_name": "pares_coherentes_en_ambos_ratio",
        "metric": metric,
        "metric_detail": metric_detail,
        "anti_hack": anti_hack,
        "seeds": seeds,
        "docs": docs_summary,
        "model": model_entry,
        "started_at": started_at,
        "finished_at": finished_at,
        "argv": sys.argv,
        "rows_jsonl": str(rows_path.resolve()),
    }

    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Reporte escrito -> {report_path}")
    print(f"Métrica: {metric} ({metric_detail['pares_coherentes_en_ambos']}/{metric_detail['pares']} pares coherentes)")
    print(f"Anti-hack pass: {anti_hack['pass']} (ratio {anti_hack['ratio']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
