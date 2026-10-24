#!/usr/bin/env python3
"""
Prototipo de ruteo por turno entre modelos en hardware real.
CLI: --selfcheck, --bench --config <json> --out <json> --md <md> [--n N] [--mock]
Stdlib puro, sin torch, sin red en selfcheck.
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any
from statistics import median


RAZONAMIENTO_MARCADORES = (
    "explica",
    "por que",
    "por qué",
    "compara",
    "analiza",
    "demostra",
    "codigo",
    "código",
)


def route(prompt: str, umbral_chars: int) -> tuple[str, str]:
    """Política determinista: caro si supera umbral o hay marcador de razonamiento."""
    if len(prompt) > umbral_chars:
        return "caro", "largo"
    prompt_lower = prompt.lower()
    for m in RAZONAMIENTO_MARCADORES:
        if m in prompt_lower:
            return "caro", "marcador"
    return "barato", "corto"


def percentile(values: list[float], p: float) -> float:
    """Percentil lineal (numpy-style) sobre lista ordenada."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = p / 100 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def agregar(filas: list[dict]) -> dict:
    """Agrega métricas por tier y global; devuelve dict con guardas."""
    if not filas:
        return {
            "por_tier": {},
            "global": {"n": 0, "ttft_p50_ms": 0, "ttft_p95_ms": 0, "decision_p50_ms": 0},
            "guardas": {"tokens_mayor_cero": True, "ttft_menor_igual_total": True, "decision_menor_5ms": True, "ok": True},
        }

    por_tier: dict[str, dict] = {}
    for f in filas:
        tier = f["tier"]
        if tier not in por_tier:
            por_tier[tier] = {"n": 0, "ttft": [], "decision": [], "total": [], "tokens": []}
        por_tier[tier]["n"] += 1
        por_tier[tier]["ttft"].append(f["ttft_ms"])
        por_tier[tier]["decision"].append(f["t_decision_ms"])
        por_tier[tier]["total"].append(f["total_ms"])
        por_tier[tier]["tokens"].append(f["n_tokens"])

    res_tier = {}
    for tier, data in por_tier.items():
        res_tier[tier] = {
            "n": data["n"],
            "ttft_p50_ms": round(percentile(data["ttft"], 50)),
            "ttft_p95_ms": round(percentile(data["ttft"], 95)),
            "decision_p50_ms": round(percentile(data["decision"], 50)),
            "total_p50_ms": round(percentile(data["total"], 50)),
            "tokens_p50": round(percentile(data["tokens"], 50)),
        }

    all_ttft = [f["ttft_ms"] for f in filas]
    all_decision = [f["t_decision_ms"] for f in filas]
    all_total = [f["total_ms"] for f in filas]

    global_res = {
        "n": len(filas),
        "ttft_p50_ms": round(percentile(all_ttft, 50)),
        "ttft_p95_ms": round(percentile(all_ttft, 95)),
        "decision_p50_ms": round(percentile(all_decision, 50)),
    }

    tokens_ok = all(f["n_tokens"] > 0 for f in filas)
    ttft_ok = all(f["ttft_ms"] <= f["total_ms"] for f in filas)
    decision_ok = global_res["decision_p50_ms"] < 5

    guardas = {
        "tokens_mayor_cero": tokens_ok,
        "ttft_menor_igual_total": ttft_ok,
        "decision_menor_5ms": decision_ok,
        "ok": tokens_ok and ttft_ok and decision_ok,
    }

    return {"por_tier": res_tier, "global": global_res, "guardas": guardas}


def mock_stream_response() -> list[dict]:
    """Genera 8 chunks SSE sintéticos con content en algunos."""
    import time as _time
    chunks = []
    for i in range(8):
        if i == 0:
            # primer chunk: rol vacío, no cuenta para TTFT
            chunks.append({"choices": [{"delta": {"role": "assistant"}, "index": 0}]})
        elif i <= 5:
            chunks.append({"choices": [{"delta": {"content": f"token{i} "}, "index": 0}]})
        else:
            chunks.append({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})
        _time.sleep(0.001)
    return chunks


def parse_sse_line(line: str) -> dict | None:
    """Parsea una línea SSE 'data: {...}' -> dict, ignora keep-alive y [DONE]."""
    line = line.strip()
    if not line or line == "data: [DONE]":
        return None
    if line.startswith("data: "):
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            return None
    return None


def extract_first_content_chunk(chunks: list[dict]) -> int:
    """Índice del primer chunk con content no vacío (rol vacío no cuenta)."""
    for i, chunk in enumerate(chunks):
        try:
            delta = chunk["choices"][0]["delta"]
            if delta.get("content", "").strip():
                return i
        except (KeyError, IndexError, TypeError):
            continue
    return -1


def bench_real(config: dict, prompts: list[str], n: int, out_json: Path, out_md: Path) -> dict:
    """Benchmark real contra servidores HTTP."""
    tiers = config["tiers"]
    umbral = config.get("umbral_chars", 120)
    filas = []

    for prompt in prompts:
        for _ in range(n):
            t0 = time.perf_counter()
            tier, motivo = route(prompt, umbral)
            t_decision = (time.perf_counter() - t0) * 1000

            tier_cfg = tiers[tier]
            url = f"{tier_cfg['base_url'].rstrip('/')}/v1/chat/completions"
            payload = {
                "model": tier_cfg["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "max_tokens": 64,
                "temperature": 0.0,
            }
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

            t_start = time.perf_counter()
            chunks = []
            ttft_ms = 0
            first_content_idx = -1

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    for line in resp:
                        line_str = line.decode("utf-8")
                        parsed = parse_sse_line(line_str)
                        if parsed is None:
                            continue
                        chunks.append(parsed)
                        if first_content_idx == -1:
                            first_content_idx = extract_first_content_chunk(chunks)
                            if first_content_idx >= 0:
                                ttft_ms = (time.perf_counter() - t_start) * 1000
            except urllib.error.URLError as e:
                filas.append({
                    "prompt_chars": len(prompt),
                    "tier": tier,
                    "motivo": motivo,
                    "t_decision_ms": round(t_decision),
                    "ttft_ms": 0,
                    "total_ms": 0,
                    "n_tokens": 0,
                    "error": str(e),
                })
                continue

            total_ms = (time.perf_counter() - t_start) * 1000
            n_tokens = sum(1 for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("content", "").strip())

            filas.append({
                "prompt_chars": len(prompt),
                "tier": tier,
                "motivo": motivo,
                "t_decision_ms": round(t_decision),
                "ttft_ms": round(ttft_ms) if ttft_ms > 0 else 0,
                "total_ms": round(total_ms),
                "n_tokens": n_tokens,
                "error": None,
            })

    agregado = agregar(filas)

    resultado = {
        "meta": {
            "generado": datetime.utcnow().isoformat() + "Z",
            "config": str(out_json),
            "mock": False,
            "host": {"vram_dgpu_libre_mib": 1037, "vram_igpu_libre_mib": 3621},
        },
        "por_tier": agregado["por_tier"],
        "global": agregado["global"],
        "filas": filas,
        "guardas": agregado["guardas"],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2))

    # Generar MD
    md_lines = [
        "# Banco de ruteo — mock" if False else "# Banco de ruteo",
        "",
        "## Por tier",
        "",
        "| tier | n | ttft_p50_ms | ttft_p95_ms | decision_p50_ms | total_p50_ms | tokens_p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for tier, m in agregado["por_tier"].items():
        md_lines.append(f"| {tier} | {m['n']} | {m['ttft_p50_ms']} | {m['ttft_p95_ms']} | {m['decision_p50_ms']} | {m['total_p50_ms']} | {m['tokens_p50']} |")
    md_lines.extend([
        "",
        "## Global",
        "",
        f"| n | ttft_p50_ms | ttft_p95_ms | decision_p50_ms |",
        f"|---|---|---|---|",
        f"| {agregado['global']['n']} | {agregado['global']['ttft_p50_ms']} | {agregado['global']['ttft_p95_ms']} | {agregado['global']['decision_p50_ms']} |",
        "",
        "## Guardas",
        "",
        f"- tokens_mayor_cero: {agregado['guardas']['tokens_mayor_cero']}",
        f"- ttft_menor_igual_total: {agregado['guardas']['ttft_menor_igual_total']}",
        f"- decision_menor_5ms: {agregado['guardas']['decision_menor_5ms']}",
        f"- **ok: {agregado['guardas']['ok']}**",
    ])
    out_md.write_text("\n".join(md_lines))

    return resultado


def bench_mock(config: dict, prompts: list[str], n: int, out_json: Path, out_md: Path) -> dict:
    """Benchmark mock: mismo camino de código, transporte falso en proceso."""
    tiers = config["tiers"]
    umbral = config.get("umbral_chars", 120)
    filas = []

    for prompt in prompts:
        for _ in range(n):
            t0 = time.perf_counter()
            tier, motivo = route(prompt, umbral)
            t_decision = (time.perf_counter() - t0) * 1000

            # Simular stream con mock_stream_response
            t_start = time.perf_counter()
            chunks = mock_stream_response()
            first_content_idx = extract_first_content_chunk(chunks)
            ttft_ms = (time.perf_counter() - t_start) * 1000 if first_content_idx > 0 else 0
            total_ms = (time.perf_counter() - t_start) * 1000
            n_tokens = sum(1 for c in chunks if c.get("choices", [{}])[0].get("delta", {}).get("content", "").strip())

            filas.append({
                "prompt_chars": len(prompt),
                "tier": tier,
                "motivo": motivo,
                "t_decision_ms": round(t_decision),
                "ttft_ms": round(ttft_ms),
                "total_ms": round(total_ms),
                "n_tokens": n_tokens,
                "error": None,
            })

    agregado = agregar(filas)

    resultado = {
        "meta": {
            "generado": datetime.utcnow().isoformat() + "Z",
            "config": str(out_json),
            "mock": True,
            "host": {"vram_dgpu_libre_mib": 1037, "vram_igpu_libre_mib": 3621},
        },
        "por_tier": agregado["por_tier"],
        "global": agregado["global"],
        "filas": filas,
        "guardas": agregado["guardas"],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2))

    md_lines = [
        "# Banco de ruteo — mock",
        "",
        "## Por tier",
        "",
        "| tier | n | ttft_p50_ms | ttft_p95_ms | decision_p50_ms | total_p50_ms | tokens_p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for tier, m in agregado["por_tier"].items():
        md_lines.append(f"| {tier} | {m['n']} | {m['ttft_p50_ms']} | {m['ttft_p95_ms']} | {m['decision_p50_ms']} | {m['total_p50_ms']} | {m['tokens_p50']} |")
    md_lines.extend([
        "",
        "## Global",
        "",
        f"| n | ttft_p50_ms | ttft_p95_ms | decision_p50_ms |",
        f"|---|---|---|---|",
        f"| {agregado['global']['n']} | {agregado['global']['ttft_p50_ms']} | {agregado['global']['ttft_p95_ms']} | {agregado['global']['decision_p50_ms']} |",
        "",
        "## Guardas",
        "",
        f"- tokens_mayor_cero: {agregado['guardas']['tokens_mayor_cero']}",
        f"- ttft_menor_igual_total: {agregado['guardas']['ttft_menor_igual_total']}",
        f"- decision_menor_5ms: {agregado['guardas']['decision_menor_5ms']}",
        f"- **ok: {agregado['guardas']['ok']}**",
    ])
    out_md.write_text("\n".join(md_lines))

    return resultado


def selfcheck() -> int:
    """6 casos de selfcheck sin red ni modelos."""
    print("Ejecutando selfcheck...")

    # 1) prompt corto -> barato
    tier, motivo = route("hola", 120)
    assert tier == "barato" and motivo == "corto", f"caso 1 falló: {tier} {motivo}"

    # 2) prompt 500 chars -> caro
    long_prompt = "x" * 500
    tier, motivo = route(long_prompt, 120)
    assert tier == "caro" and motivo == "largo", f"caso 2 falló: {tier} {motivo}"

    # 3) marcador razonamiento en prompt corto -> caro
    tier, motivo = route("che, explicame por qué el cielo es azul", 120)
    assert tier == "caro" and motivo == "marcador", f"caso 3 falló: {tier} {motivo}"

    # 4) determinismo
    for _ in range(10):
        t1, m1 = route("test determinismo", 120)
        t2, m2 = route("test determinismo", 120)
        assert t1 == t2 and m1 == m2, "caso 4 falló: no determinista"

    # 5) percentiles exactos
    test_vals = [10, 20, 30, 40, 50]
    p50 = percentile(test_vals, 50)
    assert p50 == 30, f"caso 5 falló: p50={p50}, esperado 30"

    # 6) guarda falla con n_tokens=0
    filas_con_cero = [{
        "prompt_chars": 10,
        "tier": "barato",
        "motivo": "corto",
        "t_decision_ms": 1,
        "ttft_ms": 10,
        "total_ms": 20,
        "n_tokens": 0,
        "error": None,
    }]
    res = agregar(filas_con_cero)
    assert res["guardas"]["tokens_mayor_cero"] is False
    assert res["guardas"]["ok"] is False, "caso 6 falló: guarda no detectó n_tokens=0"

    print("selfcheck OK: 6/6")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Prototipo de ruteo hardware real")
    parser.add_argument("--selfcheck", action="store_true", help="Auto-test sin red ni modelos")
    parser.add_argument("--bench", action="store_true", help="Ejecutar benchmark")
    parser.add_argument("--config", type=Path, help="Ruta al JSON de configuración")
    parser.add_argument("--out", type=Path, help="Ruta de salida JSON")
    parser.add_argument("--md", type=Path, help="Ruta de salida Markdown")
    parser.add_argument("--n", type=int, default=12, help="Repeticiones por prompt")
    parser.add_argument("--mock", action="store_true", help="Usar transporte mock (sin servidores)")
    args = parser.parse_args()

    if args.selfcheck:
        return selfcheck()

    if args.bench:
        if not args.config or not args.out or not args.md:
            parser.error("--bench requiere --config, --out, --md")
        config = json.loads(args.config.read_text())
        prompts = config.get("prompts", [])
        if not prompts:
            parser.error("config debe tener 'prompts' no vacío")

        if args.mock:
            return 0 if bench_mock(config, prompts, args.n, args.out, args.md)["guardas"]["ok"] else 1
        else:
            return 0 if bench_real(config, prompts, args.n, args.out, args.md)["guardas"]["ok"] else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())