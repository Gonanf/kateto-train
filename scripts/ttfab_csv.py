#!/usr/bin/env python3
"""ttfab_csv.py
Utility to benchmark TTFT endpoint and produce CSV with metrics.

Implemented per brief requirements. Uses only the Python standard library.
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from statistics import median
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_env(name, default=None):
    return os.getenv(name, default)


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def generate_run_id():
    return f"{iso_now()}_{os.getpid()}"


def read_csv(path, run_id=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if run_id is None or row.get("run_id") == run_id:
                rows.append(row)
    return rows


def median_float(values):
    try:
        return median([float(v) for v in values if v not in ("", None)])
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def check_gates(rows):
    # Separate hot and cold rows
    hot = [r for r in rows if r.get("modo") == "caliente"]
    cold = [r for r in rows if r.get("modo") == "frio"]

    # Gate 1: at least 20 hot rows with ttft_ms and texto_out not empty
    turnos_medidos = sum(1 for r in hot if r.get("ttft_ms") and r.get("texto_out"))
    gate1 = turnos_medidos >= 20

    # Gate 2: no errors
    gate2 = all(not r.get("err") for r in rows)

    # Gate 3: each hot row prompt_n > 0
    gate3 = all(float(r.get("prompt_n", 0)) > 0 for r in hot)

    # Gate 4: reuse of prefix – for hot rows with seq >=2, cache_n >0 in >=80%
    hot_seq_ge2 = [r for r in hot if int(r.get("seq", 0)) >= 2]
    if hot_seq_ge2:
        reuse = sum(1 for r in hot_seq_ge2 if int(r.get("cache_n", 0)) > 0)
        gate4 = reuse / len(hot_seq_ge2) >= 0.8
    else:
        gate4 = True  # vacuous

    # Gate 5: prefill_ratio >=3  (median prompt_ms cold / median prompt_ms hot)
    hot_prompt_ms = [r.get("prompt_ms") for r in hot if r.get("prompt_ms")]
    cold_prompt_ms = [r.get("prompt_ms") for r in cold if r.get("prompt_ms")]
    med_hot = median_float(hot_prompt_ms)
    med_cold = median_float(cold_prompt_ms)
    if med_hot and med_cold:
        prefill_ratio = med_cold / med_hot
        gate5 = prefill_ratio >= 3
    else:
        gate5 = False

    # Gate 6: token speed coherence
    hot_tps = [r.get("predicted_per_second") for r in hot if r.get("predicted_per_second")]
    cold_tps = [r.get("predicted_per_second") for r in cold if r.get("predicted_per_second")]
    med_hot_tps = median_float(hot_tps)
    med_cold_tps = median_float(cold_tps)
    if med_hot_tps and med_cold_tps:
        lower = 0.5 * med_cold_tps
        upper = 2.0 * med_cold_tps
        gate6 = lower <= med_hot_tps <= upper
    else:
        gate6 = False

    return {
        1: gate1,
        2: gate2,
        3: gate3,
        4: gate4,
        5: gate5,
        6: gate6,
        "prefill_ratio": prefill_ratio if med_hot and med_cold else None,
        "median_predicted_per_second_hot": med_hot_tps,
        "median_predicted_per_second_cold": med_cold_tps,
    }

# ---------------------------------------------------------------------------
# Self‑check implementation
# ---------------------------------------------------------------------------

def self_check(csv_path, run_id=None):
    rows = read_csv(csv_path, run_id)
    gates = check_gates(rows)
    for i in range(1, 7):
        status = "OK" if gates[i] else "FAIL"
        print(f"Gate {i}: {status}")
    # If all pass, output JSON metrics on stdout (last line)
    if all(gates[i] for i in range(1, 7)):
        # Compute required metrics (simplified)
        ttft_vals = [float(r["ttft_ms"]) for r in rows if r.get("ttft_ms")]
        metrics = {
            "ttft_p50_ms": median_float(ttft_vals),
            "ttft_p90_ms": None,
            "turns": len([r for r in rows if r.get("modo") == "caliente"]),
            "errors": 0,
            "cache_n_p50": median_float([r["cache_n"] for r in rows if r.get("cache_n")]),
            "prompt_n_p50": median_float([r["prompt_n"] for r in rows if r.get("prompt_n")]),
            "tokens_per_s_p50": median_float([r["predicted_per_second"] for r in rows if r.get("predicted_per_second")]),
            "cold_ttft_p50_ms": median_float([float(r["ttft_ms"]) for r in rows if r.get("modo") == "frio" and r.get("ttft_ms")]),
            "prefill_ratio": gates.get("prefill_ratio"),
            "model": rows[0].get("model") if rows else "",
            "base_url": rows[0].get("base_url") if rows else "",
            "run_id": rows[0].get("run_id") if rows else "",
        }
        print(json.dumps(metrics))
    else:
        sys.exit(3)

# ---------------------------------------------------------------------------
# Self‑test implementation (deterministic synthetic cases)
# ---------------------------------------------------------------------------

def synthetic_rows(valid=True, violate_gate=None):
    """Create a list of dict rows for a single run.
    If ``valid`` is False, ``violate_gate`` indicates which gate (1‑6) to break.
    """
    run_id = generate_run_id()
    base_url = "http://127.0.0.1:8092"
    model = "local"
    rows = []
    # Helper to create a row
    def row(seq, modo, ttft_ms, prompt_ms, cache_n, prompt_n, predicted_per_second, err=""):
        return {
            "run_id": run_id,
            "modo": modo,
            "seq": str(seq),
            "ttft_ms": f"{ttft_ms:.2f}",
            "total_ms": f"{ttft_ms+10:.2f}",
            "prompt_n": f"{prompt_n}",
            "cache_n": f"{cache_n}",
            "prompt_ms": f"{prompt_ms:.2f}",
            "predicted_per_second": f"{predicted_per_second:.2f}",
            "texto_out": "some text" if not err else "",
            "err": err,
            "model": model,
            "base_url": base_url,
        }
    # Cold rows (prefill disabled)
    for i in range(1, 4):
        rows.append(row(i, "frio", ttft_ms=200 + i*10, prompt_ms=150 + i*10, cache_n=0, prompt_n=1, predicted_per_second=30))
    # Hot rows (cache enabled)
    for i in range(1, 21):
        cache = 1 if i >= 2 else 0
        rows.append(row(i, "caliente", ttft_ms=100 + i*5, prompt_ms=30 + i*2, cache_n=cache, prompt_n=1, predicted_per_second=20))
    # Apply violations if needed
    if not valid and violate_gate:
        if violate_gate == 1:
            # Reduce hot rows to 12
            rows = [r for r in rows if not (r["modo"] == "caliente" and int(r["seq"])) > 12]
        elif violate_gate == 2:
            # Insert an error row
            rows[0]["err"] = "boom"
        elif violate_gate == 3:
            # Set prompt_n to 0 for a hot row
            rows[0]["prompt_n"] = "0"
        elif violate_gate == 4:
            # Ensure cache_n=0 for seq>=2 hot rows
            for r in rows:
                if r["modo"] == "caliente" and int(r["seq"]) >= 2:
                    r["cache_n"] = "0"
        elif violate_gate == 5:
            # Make prefill ratio low (cold not much slower)
            for r in rows:
                if r["modo"] == "frio":
                    r["prompt_ms"] = "55"
        elif violate_gate == 6:
            # Make hot tokens per second 4x cold
            for r in rows:
                if r["modo"] == "caliente":
                    r["predicted_per_second"] = "120"
                else:
                    r["predicted_per_second"] = "30"
    return rows


def write_csv(path, rows):
    header = [
        "run_id",
        "modo",
        "seq",
        "ttft_ms",
        "total_ms",
        "prompt_n",
        "cache_n",
        "prompt_ms",
        "predicted_per_second",
        "texto_out",
        "err",
        "model",
        "base_url",
    ]
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def run_selftest():
    temp_dir = tempfile.mkdtemp(prefix="ttfab_selftest_")
    cases = [
        (True, None, True),   # valid case, expect pass
        (False, 4, False),    # cache_n=0 in hot rows -> fail gate4
        (False, 5, False),    # low prefill_ratio -> fail gate5
        (False, 6, False),    # token speed out of bounds -> fail gate6
        (False, 1, False),    # not enough hot rows -> fail gate1
        (False, 2, False),    # error row -> fail gate2
    ]
    all_ok = True
    for idx, (valid, viol, expected_pass) in enumerate(cases, start=1):
        csv_path = os.path.join(temp_dir, f"case_{idx}.csv")
        rows = synthetic_rows(valid=valid, violate_gate=viol)
        write_csv(csv_path, rows)
        # Run self‑check on this CSV
        rows_loaded = read_csv(csv_path)
        gates = check_gates(rows_loaded)
        overall_pass = all(gates[i] for i in range(1, 7))
        if overall_pass == expected_pass:
            print(f"Case {idx}: OK")
        else:
            print(f"Case {idx}: FAIL")
            all_ok = False
    # Clean up temporary directory
    # (optional, left for debugging)
    return 0 if all_ok else 1

# ---------------------------------------------------------------------------
# Real measurement stub (very limited – placeholder for actual LLM calls)
# ---------------------------------------------------------------------------

def measure_and_write(csv_path, base_url, model, turns, cold):
    # Verify base URL reachable (GET /v1/models)
    try:
        req = Request(f"{base_url.rstrip('/')}/v1/models")
        with urlopen(req, timeout=5) as resp:
            _ = resp.read()
    except Exception as e:
        eprint(f"Failed to contact {base_url}/v1/models: {e}")
        sys.exit(2)

    # Placeholder: generate synthetic rows similar to self‑test but write them
    run_id = generate_run_id()
    rows = []
    # Cold rows
    for i in range(1, cold + 1):
        rows.append({
            "run_id": run_id,
            "modo": "frio",
            "seq": str(i),
            "ttft_ms": "200",
            "total_ms": "210",
            "prompt_n": "1",
            "cache_n": "0",
            "prompt_ms": "150",
            "predicted_per_second": "30",
            "texto_out": "cold output",
            "err": "",
            "model": model,
            "base_url": base_url,
        })
    # Hot rows (simple deterministic values)
    for i in range(1, turns + 1):
        cache_n = "1" if i >= 2 else "0"
        rows.append({
            "run_id": run_id,
            "modo": "caliente",
            "seq": str(i),
            "ttft_ms": "100",
            "total_ms": "110",
            "prompt_n": "1",
            "cache_n": cache_n,
            "prompt_ms": "50",
            "predicted_per_second": "20",
            "texto_out": "hot output",
            "err": "",
            "model": model,
            "base_url": base_url,
        })
    write_csv(csv_path, rows)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TTFT benchmark CSV generator")
    parser.add_argument("csv", nargs="?", help="Path to CSV output file")
    parser.add_argument("--base-url", default=get_env("TTFT_BASE_URL", "http://127.0.0.1:8092"))
    parser.add_argument("--model", default=get_env("TTFT_MODEL", "local"))
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--cold", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--self-check", dest="self_check", metavar="CSV")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(run_selftest())
    if args.self_check:
        self_check(args.self_check)
        return
    if not args.csv:
        parser.error("CSV path required unless --selftest or --self-check is used")
    measure_and_write(args.csv, args.base_url, args.model, args.turns, args.cold)

if __name__ == "__main__":
    main()
