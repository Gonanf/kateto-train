#!/usr/bin/env python3
"""Medidor de niveles de cuantización (f16, Q8_0, Q4_K_M).

Subcomandos: prompts, run, report.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(REPO_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "scripts"))

from analyze_eval29_coherence import check_degenerate
from eval_coherence_branches import TPL_VOICE
from rwkv_pipeline.chat_template import render_chat

DEFAULT_SOURCE = Path("/home/chaos/kaggle-out-eval29-v5/out-v5/out-eval29/gen29_rows.jsonl")
DEFAULT_OUT_DIR = REPO_DIR / "out" / "quant_niveles"


def cmd_prompts(args: argparse.Namespace) -> None:
    source_path = Path(args.source)
    out_dir = Path(args.out_dir)

    if not source_path.exists():
        sys.stderr.write(f"Error: fuente no encontrada: {source_path}\n")
        sys.exit(2)

    with open(source_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    eval_rows = [r for r in rows if r.get("set") == "eval_prompts_jsonl"]

    unique_prompts: list[tuple[str, str]] = []
    seen_shas: set[str] = set()

    for r in eval_rows:
        prompt_str = r.get("prompt", "")
        sha = r.get("prompt_sha256") or hashlib.sha256(prompt_str.encode("utf-8")).hexdigest()
        if sha not in seen_shas:
            seen_shas.add(sha)
            unique_prompts.append((prompt_str, sha))

    if len(unique_prompts) != 30:
        sys.stderr.write(
            f"Error: se esperaban exactamente 30 prompts únicos, se encontraron {len(unique_prompts)}\n"
        )
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    prompts30_path = out_dir / "prompts30.jsonl"

    with open(prompts30_path, "w", encoding="utf-8") as f:
        for idx, (p_str, p_sha) in enumerate(unique_prompts):
            rendered = render_chat([], p_str, TPL_VOICE)
            rendered_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            line_obj = {
                "idx": idx,
                "prompt": p_str,
                "prompt_sha256": p_sha,
                "voice": TPL_VOICE,
                "rendered": rendered,
                "rendered_sha256": rendered_sha,
            }
            f.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

    print("prompts30.jsonl: 30 lineas, 30 sha256 unicos")
    sys.exit(0)


def build_absolute_cmd() -> str:
    tokens = [sys.executable, str(Path(sys.argv[0]).resolve())]
    for arg in sys.argv[1:]:
        p = Path(arg)
        if (
            not arg.startswith("-")
            and not arg.startswith("http://")
            and not arg.startswith("https://")
            and (p.exists() or "/" in arg or "\\" in arg)
        ):
            tokens.append(str(p.resolve()))
        else:
            tokens.append(arg)
    return " ".join(tokens)


def make_http_request(
    endpoint_url: str, payload_dict: dict, timeout: float
) -> tuple[bool, int, str, int, float, dict | None]:
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    req = urllib.request.Request(
        endpoint_url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            wall_s = time.perf_counter() - t0
            status = resp.status
            if status != 200:
                return (
                    False,
                    status,
                    "",
                    0,
                    wall_s,
                    {"http_status": status, "message": f"HTTP status {status}"},
                )
            body_str = resp.read().decode("utf-8")
            data = json.loads(body_str)
            content = data.get("content", "")
            tokens_pred = data.get("tokens_predicted", 0)
            return True, status, content, tokens_pred, wall_s, None
    except urllib.error.HTTPError as e:
        wall_s = time.perf_counter() - t0
        return (
            False,
            e.code,
            "",
            0,
            wall_s,
            {"http_status": e.code, "message": str(e), "type": "HTTPError"},
        )
    except Exception as e:
        wall_s = time.perf_counter() - t0
        return False, 0, "", 0, wall_s, {"message": str(e), "type": type(e).__name__}


def cmd_run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    prompts_path = out_dir / "prompts30.jsonl"
    if not prompts_path.exists():
        sys.stderr.write(f"Error: {prompts_path} no existe\n")
        sys.exit(2)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    if len(prompts) != 30:
        sys.stderr.write(
            f"Error: {prompts_path} tiene {len(prompts)} lineas, se esperaban 30\n"
        )
        sys.exit(2)

    gguf_path = Path(args.gguf).resolve()
    if not gguf_path.exists():
        sys.stderr.write(f"Error: archivo GGUF no existe: {gguf_path}\n")
        sys.exit(2)

    gguf_bytes = gguf_path.stat().st_size
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        sys.stderr.write("Error: lista de seeds vacia\n")
        sys.exit(2)

    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_dir / f"{args.level}.meta.json"
    raw_path = raw_dir / f"{args.level}.jsonl"

    meta = {
        "level": args.level,
        "gguf": str(gguf_path),
        "gguf_bytes": gguf_bytes,
        "url": args.url,
        "argv": sys.argv,
        "cmd": build_absolute_cmd(),
        "cwd": str(Path.cwd().resolve()),
        "seeds": seeds,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "finished_at": None,
        "n_ok": 0,
        "n_fail": 0,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    endpoint_url = args.url.rstrip("/") + "/completion"
    n_ok = 0
    n_fail = 0

    with open(raw_path, "w", encoding="utf-8") as raw_f:
        for seed in seeds:
            for p_row in prompts:
                payload = {
                    "prompt": p_row["rendered"],
                    "max_tokens": args.max_tokens,
                    "n_predict": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repeat_penalty": args.repeat_penalty,
                    "seed": seed,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                    "cache_prompt": False,
                }

                ok, status, text, pred_n, wall_s, err = make_http_request(
                    endpoint_url, payload, args.timeout
                )
                if not ok:
                    # 1 reintento del mismo request
                    time.sleep(1.0)
                    ok, status, text, pred_n, wall_s, err = make_http_request(
                        endpoint_url, payload, args.timeout
                    )

                chars = len(text.strip()) if ok else 0
                tok_s = (pred_n / wall_s) if (ok and wall_s > 0) else 0.0

                if ok:
                    n_ok += 1
                else:
                    n_fail += 1

                rec = {
                    "level": args.level,
                    "seed": seed,
                    "idx": p_row["idx"],
                    "prompt_sha256": p_row["prompt_sha256"],
                    "rendered_sha256": p_row["rendered_sha256"],
                    "text": text,
                    "chars": chars,
                    "ok": ok,
                    "http_status": status,
                    "predicted_n": pred_n,
                    "wall_s": round(wall_s, 4),
                    "tok_s": round(tok_s, 2),
                    "error": err if not ok else None,
                }
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                raw_f.flush()

                sys.stderr.write(
                    f"[{args.level} {seed}/{p_row['idx']}] chars={chars} tok/s={tok_s:.2f}\n"
                )
                sys.stderr.flush()

                if n_fail > 5:
                    sys.stderr.write(
                        f"Abortando: mas de 5 fallos acumulados (n_fail={n_fail})\n"
                    )
                    meta["finished_at"] = datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                    meta["n_ok"] = n_ok
                    meta["n_fail"] = n_fail
                    with open(meta_path, "w", encoding="utf-8") as mf:
                        json.dump(meta, mf, indent=2, ensure_ascii=False)
                    sys.exit(3)

    meta["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["n_ok"] = n_ok
    meta["n_fail"] = n_fail
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)

    sys.stderr.write(f"[{args.level}] Resumen final: n_ok={n_ok}, n_fail={n_fail}\n")
    if n_fail > 5:
        sys.exit(3)
    sys.exit(0)


def calc_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "deg_rate": 0.0,
            "empty_rate": 0.0,
            "mediana_chars": 0.0,
            "chars_lt_10_rate": 0.0,
            "mediana_tok_s": 0.0,
        }

    deg_count = sum(1 for r in rows if check_degenerate(r.get("text", ""))[0])
    empty_count = sum(1 for r in rows if r.get("text", "").strip() == "")
    chars_list = [r.get("chars", len(r.get("text", "").strip())) for r in rows]
    chars_lt_10_count = sum(1 for c in chars_list if c < 10)
    tok_s_list = [float(r.get("tok_s", 0.0)) for r in rows]

    return {
        "n": n,
        "deg_rate": deg_count / n,
        "empty_rate": empty_count / n,
        "mediana_chars": float(statistics.median(chars_list)),
        "chars_lt_10_rate": chars_lt_10_count / n,
        "mediana_tok_s": float(statistics.median(tok_s_list)),
    }


def cmd_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    levels = ["f16", "Q8_0", "Q4_K_M"]

    raw_data: dict[str, list[dict]] = {}
    for lvl in levels:
        raw_path = raw_dir / f"{lvl}.jsonl"
        if not raw_path.exists():
            sys.stderr.write(f"Error: falta archivo requerido {raw_path}\n")
            sys.exit(2)
        with open(raw_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        if len(lines) != 90:
            sys.stderr.write(
                f"Error: {raw_path} tiene {len(lines)} lineas (se requerian 90)\n"
            )
            sys.exit(2)
        raw_data[lvl] = lines

    shas_file = out_dir / "gguf_shas.json"
    if not shas_file.exists():
        sys.stderr.write(f"Error: falta archivo requerido {shas_file}\n")
        sys.exit(2)

    with open(shas_file, "r", encoding="utf-8") as f:
        gguf_shas = json.load(f)

    meta_cmds: dict[str, str] = {}
    for lvl in levels:
        meta_file = raw_dir / f"{lvl}.meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta_cmds[lvl] = json.load(f).get("cmd", "")
        else:
            meta_cmds[lvl] = "N/A"

    seed_metrics = []
    for lvl in levels:
        seeds_seen: list[int] = []
        for r in raw_data[lvl]:
            s = r["seed"]
            if s not in seeds_seen:
                seeds_seen.append(s)
        for s in seeds_seen:
            sub_rows = [r for r in raw_data[lvl] if r["seed"] == s]
            m = calc_metrics(sub_rows)
            seed_metrics.append((lvl, s, m))

    agg_metrics: dict[str, dict] = {}
    for lvl in levels:
        agg_metrics[lvl] = calc_metrics(raw_data[lvl])

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    host_str = os.uname().nodename

    md: list[str] = []
    md.append("# Reporte de medicion de niveles de cuantizacion\n")
    md.append(f"- **Fecha**: {now_str}")
    md.append(f"- **Host**: {host_str}\n")
    md.append("### Modelos GGUF evaluados\n")
    md.append("| Nivel | Ruta | SHA256 | Tamaño (bytes) |")
    md.append("|:---|:---|:---|---:|")
    for lvl in levels:
        info = gguf_shas.get(lvl, {})
        p = info.get("path", "N/A")
        sha = info.get("sha256", "N/A")
        b = info.get("bytes", "N/A")
        md.append(f"| {lvl} | {p} | {sha} | {b} |")
    md.append("")

    md.append("## Metricas por nivel × seed\n")
    md.append(
        "| Nivel | Seed | N | Degenerados (%) | Vacios (%) | Mediana chars | chars < 10 (%) | Mediana tok/s |"
    )
    md.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for lvl, s, m in seed_metrics:
        md.append(
            f"| {lvl} | {s} | {m['n']} | {m['deg_rate'] * 100:.1f}% | {m['empty_rate'] * 100:.1f}% | "
            f"{m['mediana_chars']:.1f} | {m['chars_lt_10_rate'] * 100:.1f}% | {m['mediana_tok_s']:.2f} |"
        )
    md.append("")

    md.append("## Metricas agregadas por nivel\n")
    md.append(
        "| Nivel | N | Degenerados (%) | Vacios (%) | Mediana chars | chars < 10 (%) | Mediana tok/s |"
    )
    md.append("|:---|---:|---:|---:|---:|---:|---:|")
    for lvl in levels:
        m = agg_metrics[lvl]
        md.append(
            f"| {lvl} | {m['n']} | {m['deg_rate'] * 100:.1f}% | {m['empty_rate'] * 100:.1f}% | "
            f"{m['mediana_chars']:.1f} | {m['chars_lt_10_rate'] * 100:.1f}% | {m['mediana_tok_s']:.2f} |"
        )
    md.append("")

    md.append("## Comandos exactos\n")
    for lvl in levels:
        md.append(f"- **{lvl}**: `{meta_cmds.get(lvl, '')}`")
    md.append("")

    md.append("## Veredicto\n")

    m_f16 = agg_metrics["f16"]
    m_q8 = agg_metrics["Q8_0"]
    m_q4 = agg_metrics["Q4_K_M"]

    def serves(m: dict) -> bool:
        return (
            m["deg_rate"] <= 0.10
            and m["empty_rate"] <= 0.10
            and m["mediana_chars"] >= 40.0
        )

    f16_serves = serves(m_f16)
    q8_serves = serves(m_q8)

    diff_deg = abs(m_q8["deg_rate"] - m_f16["deg_rate"])
    chars_ratio = (
        (m_q8["mediana_chars"] / m_f16["mediana_chars"])
        if m_f16["mediana_chars"] > 0
        else 0.0
    )
    noise_ok = (diff_deg <= 0.05) and (
        m_q8["mediana_chars"] >= 0.90 * m_f16["mediana_chars"]
    )

    if f16_serves and q8_serves and noise_ok:
        md.append(
            "Se elige **Q8_0**. f16 y Q8_0 cumplen los criterios de utilidad "
            "(deg_rate <= 0.10, empty_rate <= 0.10, mediana_chars >= 40). "
            f"|deg_rate(Q8_0) - deg_rate(f16)| = |{m_q8['deg_rate']:.4f} - {m_f16['deg_rate']:.4f}| = {diff_deg:.4f} <= 0.05; "
            f"mediana_chars(Q8_0) = {m_q8['mediana_chars']:.1f} >= 0.90 * mediana_chars(f16) "
            f"({0.90 * m_f16['mediana_chars']:.1f}, ratio = {chars_ratio:.4f}). "
            "Q8_0 quedo dentro del ruido de f16 con un tamaño de 3,26 GB vs 5,93 GB del f16."
        )
    elif f16_serves:
        reasons = []
        if not q8_serves:
            if m_q8["deg_rate"] > 0.10:
                reasons.append(f"deg_rate = {m_q8['deg_rate']:.4f} > 0.10")
            if m_q8["empty_rate"] > 0.10:
                reasons.append(f"empty_rate = {m_q8['empty_rate']:.4f} > 0.10")
            if m_q8["mediana_chars"] < 40.0:
                reasons.append(f"mediana_chars = {m_q8['mediana_chars']:.1f} < 40.0")
        else:
            if diff_deg > 0.05:
                reasons.append(f"|deg_rate(Q8_0) - deg_rate(f16)| = {diff_deg:.4f} > 0.05")
            if m_q8["mediana_chars"] < 0.90 * m_f16["mediana_chars"]:
                reasons.append(
                    f"mediana_chars(Q8_0) = {m_q8['mediana_chars']:.1f} < 0.90 * mediana_chars(f16) "
                    f"({0.90 * m_f16['mediana_chars']:.1f})"
                )
        md.append(f"Se elige **f16**. Q8_0 no cumplio: {', '.join(reasons)}.")
    elif q8_serves:
        md.append(
            "Se elige **Q8_0**. f16 no cumplio los criterios de utilidad "
            f"(deg_rate = {m_f16['deg_rate']:.4f}, empty_rate = {m_f16['empty_rate']:.4f}, "
            f"mediana_chars = {m_f16['mediana_chars']:.1f})."
        )
    else:
        md.append(
            "Ningun nivel cumple los criterios de utilidad (deg_rate <= 0.10, empty_rate <= 0.10, mediana_chars >= 40). "
            f"No se elige ninguno (f16: deg_rate={m_f16['deg_rate']:.4f}, empty_rate={m_f16['empty_rate']:.4f}, "
            f"mediana_chars={m_f16['mediana_chars']:.1f}; Q8_0: deg_rate={m_q8['deg_rate']:.4f}, "
            f"empty_rate={m_q8['empty_rate']:.4f}, mediana_chars={m_q8['mediana_chars']:.1f})."
        )

    md.append("")
    if m_q4["mediana_chars"] < 40.0 or m_q4["chars_lt_10_rate"] >= 0.50:
        q4_premise = "Reproduce la premisa de basura corta (mediana_chars < 40 o chars_lt_10_rate >= 0.50 tipo '<|im')."
    else:
        q4_premise = "No reproduce la premisa de basura corta."

    md.append(
        f"Q4_K_M: deg_rate = {m_q4['deg_rate']:.4f}, empty_rate = {m_q4['empty_rate']:.4f}, "
        f"mediana_chars = {m_q4['mediana_chars']:.1f}, chars_lt_10_rate = {m_q4['chars_lt_10_rate']:.4f} "
        f"({m_q4['chars_lt_10_rate'] * 100:.1f}%), mediana_tok_s = {m_q4['mediana_tok_s']:.2f}. {q4_premise}"
    )

    report_path = out_dir / "REPORTE.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medidor de niveles de cuantizacion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1) prompts
    p_prompts = subparsers.add_parser("prompts", help="Generar prompts30.jsonl")
    p_prompts.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Ruta al archivo JSONL de fuente de prompts",
    )
    p_prompts.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directorio base de salida",
    )

    # 2) run
    p_run = subparsers.add_parser("run", help="Ejecutar mediciones de un nivel contra llama-server")
    p_run.add_argument(
        "--level",
        choices=["f16", "Q8_0", "Q4_K_M"],
        required=True,
        help="Nivel de cuantizacion",
    )
    p_run.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="Ruta al archivo GGUF evaluado",
    )
    p_run.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL base de llama-server (ej. http://127.0.0.1:8080)",
    )
    p_run.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directorio base de salida",
    )
    p_run.add_argument(
        "--seeds",
        type=str,
        default="1337,1338,1339",
        help="Lista de seeds separadas por coma",
    )
    p_run.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens a generar (default: 128)",
    )
    p_run.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature (default: 0.8)",
    )
    p_run.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        help="Top-p (default: 0.7)",
    )
    p_run.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.3,
        help="Repeat penalty (default: 1.3)",
    )
    p_run.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout en segundos por request (default: 300)",
    )

    # 3) report
    p_report = subparsers.add_parser("report", help="Generar REPORTE.md con resultados")
    p_report.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directorio base de salida",
    )

    args = parser.parse_args()

    if args.command == "prompts":
        cmd_prompts(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
