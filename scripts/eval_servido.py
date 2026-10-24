#!/usr/bin/env python3
"""Replay de filas del kernel contra GGUF servido por llama-server."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rwkv_pipeline.chat_template import render_chat


def load_check_degenerate():
    coherence_path = REPO_ROOT / "scripts" / "analyze_eval29_coherence.py"
    spec = importlib.util.spec_from_file_location("analyze_eval29_coherence", coherence_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {coherence_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_degenerate


check_degenerate = load_check_degenerate()


def map_stop_closure(stopping_word: str | None) -> tuple[str | None, bool]:
    """Mapeo de cierre estricto segun semantica del kernel."""
    if stopping_word == "<|im_end|>":
        return "id_end", True
    return None, False


def trim_stop_marker(text: str) -> str:
    """Recorta el marcador de cierre <|im_end|> si lo hubo, quedandose con lo anterior."""
    if "<|im_end|>" in text:
        return text.split("<|im_end|>")[0]
    return text


def compute_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def run_self_check() -> None:
    # 1. render_chat([], "hola", "seco")
    rendered = render_chat([], "hola", "seco")
    expected = "<|im_user|>hola<|im_end|>\n<|im_start|>seco\n"
    assert "<|im_user|>hola<|im_end|>" in rendered, f"Falta parte de usuario en: {rendered!r}"
    assert "<|im_start|>seco\n" in rendered, f"Falta parte de voz en: {rendered!r}"
    assert rendered == expected, f"Rendered {rendered!r} != expected {expected!r}"

    # 2. Closure mapping
    assert map_stop_closure("<|im_end|>") == ("id_end", True)
    assert map_stop_closure(None) == (None, False)
    assert map_stop_closure("stopped_limit") == (None, False)

    # 3. Marker trimming
    assert trim_stop_marker("hola<|im_end|>basura") == "hola"
    assert trim_stop_marker("hola") == "hola"

    # 4. check_degenerate
    assert check_degenerate("bueno" * 20)[0] is True
    assert check_degenerate("hola que tal")[0] is False

    # 5. sha256 prompt de juguete
    toy_prompt = "hola mundo de prueba"
    toy_sha = hashlib.sha256(toy_prompt.encode("utf-8")).hexdigest()
    assert toy_sha == hashlib.sha256(toy_prompt.encode()).hexdigest()

    print("SELF-CHECK OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay de filas del kernel contra GGUF servido por llama-server."
    )
    parser.add_argument("--src", type=str, default=None, help="Ruta al JSONL de origen del kernel")
    parser.add_argument("--src-manifest", type=str, default=None, help="Ruta al manifest.json del kernel")
    parser.add_argument("--gguf", type=str, default=None, help="Ruta al archivo .gguf servido")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8270", help="URL base del llama-server")
    parser.add_argument("--out-dir", type=str, default=None, help="Directorio de salida")
    parser.add_argument("--max-len", type=int, default=256, help="n_predict para /completion")
    parser.add_argument("--temperature", type=float, default=0.80, help="Temperatura")
    parser.add_argument("--top-p", type=float, default=0.70, help="Top-p")
    parser.add_argument("--repeat-penalty", type=float, default=1.3, help="Repeat penalty")
    parser.add_argument("--seed", type=int, default=1337, help="Seed para generación")
    parser.add_argument("--limit", type=int, default=None, help="Límite de filas a procesar")
    parser.add_argument("--resume", action="store_true", help="Saltear filas ya procesadas si ya existe")
    parser.add_argument("--self-check", action="store_true", help="Ejecutar self-check puro y salir")
    parser.add_argument("--timeout", type=int, default=900, help="Timeout HTTP en segundos")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        run_self_check()
        return 0

    if not args.src or not args.gguf or not args.out_dir:
        print("Error: --src, --gguf, y --out-dir son requeridos a menos que se use --self-check.", file=sys.stderr)
        return 1

    src_path = Path(args.src).expanduser().resolve()
    if not src_path.exists():
        print(f"Error: archivo origen no existe: {src_path}", file=sys.stderr)
        return 1

    gguf_path = Path(args.gguf).expanduser().resolve()
    if not gguf_path.exists():
        print(f"Error: archivo gguf no existe: {gguf_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows_path = out_dir / "gen29_rows.jsonl"
    out_manifest_path = out_dir / "manifest.json"

    with open(src_path, "r", encoding="utf-8") as f:
        src_rows = [json.loads(line) for line in f if line.strip()]

    if args.limit is not None:
        src_rows = src_rows[:args.limit]

    existing_rows = []
    keys_done = set()
    if args.resume and out_rows_path.exists():
        with open(out_rows_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing_rows.append(r)
                    keys_done.add((r["set"], r["id"], r["turno"]))
        all_rows = list(existing_rows)
        write_mode = "a"
    else:
        all_rows = []
        write_mode = "w"

    model_name = gguf_path.name
    errors_count = 0

    with open(out_rows_path, write_mode, encoding="utf-8") as out_f:
        for row in src_rows:
            row_key = (row["set"], row["id"], row["turno"])
            if row_key in keys_done:
                continue

            prompt_raw = row["prompt"]
            calc_sha = hashlib.sha256(prompt_raw.encode("utf-8")).hexdigest()
            if calc_sha != row.get("prompt_sha256"):
                raise ValueError(
                    f"Candado anti-hack: prompt_sha256 no coincide para row {row_key}: "
                    f"calculado {calc_sha} != {row.get('prompt_sha256')}"
                )

            prompt_render = render_chat([], prompt_raw, "seco")
            payload = {
                "prompt": prompt_render,
                "n_predict": args.max_len,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "repeat_penalty": args.repeat_penalty,
                "stream": False,
                "cache_prompt": False,
                "stop": ["<|im_end|>"],
            }

            t0 = time.time()
            error = None
            try:
                req = urllib.request.Request(
                    f"{args.url.rstrip('/')}/completion",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                wall_s = round(time.time() - t0, 3)

                timings = body.get("timings") or {}
                prompt_ms = timings.get("prompt_ms", 0.0)
                ttfab_s = round(prompt_ms / 1000.0, 4)
                n_tokens = timings.get("predicted_n")
                if n_tokens is None:
                    n_tokens = body.get("tokens_predicted", 0)

                stopping_word = body.get("stopping_word")
                stop_reason, finished = map_stop_closure(stopping_word)
                content = body.get("content", "")
                text = trim_stop_marker(content)
                degenerate, max_repeat = check_degenerate(text)
            except Exception as e:
                wall_s = round(time.time() - t0, 3)
                ttfab_s = 0.0
                n_tokens = 0
                finished = False
                stop_reason = None
                text = ""
                degenerate = False
                max_repeat = 0
                if isinstance(e, urllib.error.HTTPError):
                    err_body = e.read().decode("utf-8", errors="replace")
                    error = f"HTTP {e.code}: {err_body}"
                else:
                    error = str(e)
                errors_count += 1

            row_out = {
                "set": row["set"],
                "id": row["id"],
                "prompt": row["prompt"],
                "prompt_sha256": row["prompt_sha256"],
                "model": model_name,
                "motor": "llama-server /completion (GGUF)",
                "engine": "llama.cpp build 32 (f2cc728)",
                "device": "http:Vulkan1(DG1)",
                "turno": row["turno"],
                "chat_format": "chatml",
                "ttfab_s": ttfab_s,
                "wall_s": wall_s,
                "n_tokens": n_tokens,
                "finished": finished,
                "stop_reason": stop_reason,
                "degenerate": degenerate,
                "max_repeat": max_repeat,
                "text": text,
                "error": error,
            }

            out_f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            out_f.flush()
            all_rows.append(row_out)
            keys_done.add(row_key)

    if args.src_manifest:
        src_manifest_path = Path(args.src_manifest).expanduser().resolve()
    else:
        src_manifest_path = src_path.parent / "manifest.json"

    src_manifest = {}
    if src_manifest_path.exists():
        src_manifest = json.loads(src_manifest_path.read_text(encoding="utf-8"))

    frio_ttfab = [r["ttfab_s"] for r in all_rows if r.get("turno") == "frio" and r.get("ttfab_s") is not None]
    cal_ttfab = [r["ttfab_s"] for r in all_rows if r.get("turno") == "caliente" and r.get("ttfab_s") is not None]
    median_frio = round(float(statistics.median(frio_ttfab)), 4) if frio_ttfab else 0.0
    median_caliente = round(float(statistics.median(cal_ttfab)), 4) if cal_ttfab else 0.0

    manifest_data = {
        "env": {
            "python": sys.version,
            "hostname": socket.gethostname(),
            "url": args.url,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repeat_penalty": args.repeat_penalty,
                "max_len": args.max_len,
                "seed": args.seed,
            },
        },
        "base": src_manifest.get("base", {}),
        "adapter_sha256": src_manifest.get("adapter_sha256"),
        "merged": {
            "path": str(gguf_path),
            "bytes": gguf_path.stat().st_size,
            "sha256": compute_sha256(gguf_path),
            "recipe": "GGUF servido por llama-server; pesos = export de kateto-rwkv29 (ver provenance en el reporte)",
        },
        "arch": {
            "n_embd": 2560,
            "n_layer": 32,
            "head_size": 64,
            "vocab": 65536,
            "device": "vulkan(DG1)",
            "dtype": "Q4_K_M",
        },
        "gen_rows": {
            "n": len(all_rows),
            "median_ttfab_frio": median_frio,
            "median_ttfab_caliente": median_caliente,
        },
    }

    out_manifest_path.write_text(json.dumps(manifest_data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Completado: {len(all_rows)} filas escritas en {out_rows_path} (errores: {errors_count})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
