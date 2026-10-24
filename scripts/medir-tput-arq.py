#!/usr/bin/env python3
"""
scripts/medir-tput-arq.py — Driver de throughput para arquitecturas Mamba y RWKV
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

MODELOS = {
    "rwkv15": "/run/media/chaos/terciario/proyectos/kateto-train/out/kateto-15b-Q8_0.gguf",
    "rwkv29": "/run/media/chaos/terciario/proyectos/kateto-train/out/kateto-rwkv29-Q8_0.gguf",
    "mamba14": "/run/media/chaos/terciario/chaos-cache/mamba-bench/mamba-1.4b-hf.Q8_0.gguf",
    "mamba14inst": "/run/media/chaos/terciario/chaos-cache/mamba-bench/mamba-1.4b-instruct-hf.Q8_0.gguf",
    "mamba28": "/run/media/chaos/terciario/chaos-cache/mamba-bench/mamba-2.8b-hf-Q8_0.gguf",
}

BASE_ORDER = ["rwkv15", "mamba14", "mamba28", "rwkv29", "mamba14inst"]
BIN_PATH = "/home/chaos/.local/bin/llama-bench"


def get_binary_info():
    build_str = "b32-f2cc728"
    try:
        res = subprocess.run([BIN_PATH, "--version"], capture_output=True, text=True, timeout=10)
        out = (res.stdout + "\n" + res.stderr).strip()
        m = re.search(r"build (\d+),\s*commit ([0-9a-f]+)", out)
        if m:
            build_str = f"b{m.group(1)}-{m.group(2)}"
    except Exception:
        pass
    return BIN_PATH, build_str


def wait_for_file(path_str, timeout_sec=600):
    path = Path(path_str)
    start = time.time()
    while time.time() - start < timeout_sec:
        if path.is_file():
            # Check stability
            try:
                s1 = path.stat().st_size
                time.sleep(1)
                s2 = path.stat().st_size
                if s1 == s2 and s1 > 0:
                    return True
            except OSError:
                pass
        time.sleep(5)
    return path.is_file()


def main():
    parser = argparse.ArgumentParser(description="Medición de throughput Mamba vs RWKV")
    parser.add_argument("--out", default="./tput-arq.json", help="Ruta del JSON de salida")
    parser.add_argument("--reps", type=int, default=2, help="Repeticiones por test")
    parser.add_argument("--threads", type=int, default=4, help="Hilos de CPU")
    parser.add_argument("--pases", type=int, default=2, help="Cantidad de pases")
    parser.add_argument(
        "--modelos",
        default="rwkv15,mamba14,mamba28,rwkv29,mamba14inst",
        help="Modelos a medir separados por coma",
    )

    args = parser.parse_args()

    mod_keys = [m.strip() for m in args.modelos.split(",") if m.strip()]
    for m in mod_keys:
        if m not in MODELOS:
            sys.stderr.write(f"Error: modelo desconocido '{m}'. Válidos: {list(MODELOS.keys())}\n")
            sys.exit(2)

    bin_path, build_str = get_binary_info()
    nproc = os.cpu_count() or 4

    meta = {
        "fecha": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bin": bin_path,
        "build": build_str,
        "nproc": nproc,
        "reps": args.reps,
        "threads": args.threads,
        "pases": args.pases,
        "ngl": 0,
    }

    corridas = []

    for pase_idx in range(1, args.pases + 1):
        # Base order filtered by requested models
        order = [m for m in BASE_ORDER if m in mod_keys]
        if pase_idx % 2 == 0:
            order = list(reversed(order))

        for orden_idx, m_id in enumerate(order, start=1):
            gguf_path = MODELOS[m_id]
            file_ok = wait_for_file(gguf_path, timeout_sec=600)
            if not file_ok:
                corridas.append({
                    "modelo": m_id,
                    "pase": pase_idx,
                    "orden": orden_idx,
                    "gguf": gguf_path,
                    "ruta_existe": False,
                    "loadavg_antes": list(os.getloadavg()),
                    "loadavg_despues": list(os.getloadavg()),
                    "wall_s": 0.0,
                    "error": f"Archivo GGUF no existe tras esperar timeout: {gguf_path}",
                })
                continue

            cmd = [
                bin_path,
                "-m",
                gguf_path,
                "-p",
                "512",
                "-n",
                "128",
                "-r",
                str(args.reps),
                "-t",
                str(args.threads),
                "-ngl",
                "0",
                "-o",
                "json",
            ]

            load_antes = list(os.getloadavg())
            t0 = time.time()
            proc_err = None
            stdout_text = ""
            stderr_text = ""
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                stdout_text = p.stdout
                stderr_text = p.stderr
                if p.returncode != 0:
                    proc_err = f"llama-bench retornó código {p.returncode}"
            except subprocess.TimeoutExpired:
                proc_err = "llama-bench excedió timeout de 900s"
            except Exception as e:
                proc_err = str(e)

            t1 = time.time()
            load_desp = list(os.getloadavg())
            wall_s = round(t1 - t0, 3)

            if proc_err:
                stderr_lines = stderr_text.strip().splitlines()[-10:] if stderr_text else []
                err_detail = "\n".join(stderr_lines) if stderr_lines else proc_err
                corridas.append({
                    "modelo": m_id,
                    "pase": pase_idx,
                    "orden": orden_idx,
                    "gguf": gguf_path,
                    "ruta_existe": True,
                    "loadavg_antes": load_antes,
                    "loadavg_despues": load_desp,
                    "wall_s": wall_s,
                    "error": err_detail,
                })
                continue

            idx1 = stdout_text.find("[")
            idx2 = stdout_text.rfind("]")
            if idx1 == -1 or idx2 == -1 or idx2 <= idx1:
                stderr_lines = stderr_text.strip().splitlines()[-10:] if stderr_text else []
                err_detail = "\n".join(stderr_lines) or "No se encontró array JSON en la salida de llama-bench"
                corridas.append({
                    "modelo": m_id,
                    "pase": pase_idx,
                    "orden": orden_idx,
                    "gguf": gguf_path,
                    "ruta_existe": True,
                    "loadavg_antes": load_antes,
                    "loadavg_despues": load_desp,
                    "wall_s": wall_s,
                    "error": err_detail,
                })
                continue

            try:
                bench_entries = json.loads(stdout_text[idx1 : idx2 + 1])
            except Exception as e:
                corridas.append({
                    "modelo": m_id,
                    "pase": pase_idx,
                    "orden": orden_idx,
                    "gguf": gguf_path,
                    "ruta_existe": True,
                    "loadavg_antes": load_antes,
                    "loadavg_despues": load_desp,
                    "wall_s": wall_s,
                    "error": f"Error decodificando JSON: {e}",
                })
                continue

            parsed_tests = []
            for b in bench_entries:
                n_prompt = b.get("n_prompt", 0)
                n_gen = b.get("n_gen", 0)
                tipo = "pp" if n_prompt > 0 else "tg"
                n = n_prompt if n_prompt > 0 else n_gen
                parsed_tests.append({
                    "tipo": tipo,
                    "n": n,
                    "avg_ts": b.get("avg_ts"),
                    "stddev_ts": b.get("stddev_ts"),
                    "samples_ts": b.get("samples_ts", []),
                    "model_n_params": b.get("model_n_params"),
                    "model_size": b.get("model_size"),
                })

            corridas.append({
                "modelo": m_id,
                "pase": pase_idx,
                "orden": orden_idx,
                "gguf": gguf_path,
                "ruta_existe": True,
                "loadavg_antes": load_antes,
                "loadavg_despues": load_desp,
                "wall_s": wall_s,
                "tests": parsed_tests,
            })

    # Resumen
    resumen = []
    for m_id in [m for m in BASE_ORDER if m in mod_keys]:
        m_corridas = [c for c in corridas if c.get("modelo") == m_id]
        pp_vals = []
        tg_vals = []
        n_params = None
        error_msg = None

        for c in m_corridas:
            if "error" in c:
                error_msg = c["error"]
            for t in c.get("tests", []):
                if t.get("tipo") == "pp" and t.get("avg_ts") is not None:
                    pp_vals.append(t["avg_ts"])
                elif t.get("tipo") == "tg" and t.get("avg_ts") is not None:
                    tg_vals.append(t["avg_ts"])
                if t.get("model_n_params") is not None and n_params is None:
                    n_params = t["model_n_params"]

        if pp_vals and tg_vals:
            resumen.append({
                "modelo": m_id,
                "pp_avg_ts_min": min(pp_vals),
                "pp_avg_ts_max": max(pp_vals),
                "tg_avg_ts_min": min(tg_vals),
                "tg_avg_ts_max": max(tg_vals),
                "n_params": n_params,
                "nota": "",
            })
        else:
            err = error_msg or "sin corridas exitosas"
            resumen.append({
                "modelo": m_id,
                "error": err,
                "nota": f"no medido: {err}",
            })

    out_data = {
        "meta": meta,
        "corridas": corridas,
        "resumen": resumen,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
