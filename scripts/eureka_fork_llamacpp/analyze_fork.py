#!/usr/bin/env python3
"""
analyze_fork.py - Analizador de mediciones de llama-bench para fork de llama.cpp.
Solo stdlib de Python 3.
"""

import argparse
import json
import os
import sys

CONFIGS = [
    {
        "id": "c0",
        "file": "c0_cpu_rwkv29q4.json",
        "label": "A · RWKV 2.9B Q4_K_M · CPU (ngl 0, t8)",
        "device": "cpu",
        "config": "c0 (A)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "CPU (ngl 0, t8)",
    },
    {
        "id": "c1",
        "file": "c1_vk1_rwkv29q4.json",
        "label": "B · RWKV 2.9B Q4_K_M · DG1 (ngl 99)",
        "device": "Vulkan1",
        "config": "c1 (B)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "DG1 (ngl 99)",
    },
    {
        "id": "c2",
        "file": "c2_vk1_fa-on.json",
        "label": "B2 · B + -fa on",
        "device": "Vulkan1",
        "config": "c2 (B2)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "B + -fa on",
    },
    {
        "id": "c3",
        "file": "c3_vk1_ub1024.json",
        "label": "B3 · B con -ub 1024 -b 2048",
        "device": "Vulkan1",
        "config": "c3 (B3)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "B con -ub 1024 -b 2048",
    },
    {
        "id": "c4",
        "file": "c4_vk1_ub128.json",
        "label": "B4 · B con -ub 128 -b 512",
        "device": "Vulkan1",
        "config": "c4 (B4)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "B con -ub 128 -b 512",
    },
    {
        "id": "c5",
        "file": "c5_vk1_Q8_0.json",
        "label": "C · RWKV 2.9B Q8_0 · DG1",
        "device": "Vulkan1",
        "config": "c5 (C)",
        "model": "RWKV 2.9B Q8_0",
        "ngl_dev": "DG1",
    },
    {
        "id": "c6",
        "file": "c6_split.json",
        "label": "D · RWKV 2.9B Q4_K_M · device auto (AMD+DG1)",
        "device": "split",
        "config": "c6 (D)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "device auto (AMD+DG1)",
    },
    {
        "id": "c7",
        "file": "c7_vk1_t4.json",
        "label": "B5 · B con -t 4",
        "device": "Vulkan1",
        "config": "c7 (B5)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "B con -t 4",
    },
    {
        "id": "c8",
        "file": "c8_vk0_rwkv04b.json",
        "label": "E · RWKV7 0.4B Q4_K_M · RX 6500 XT",
        "device": "Vulkan0",
        "config": "c8 (E)",
        "model": "RWKV7 0.4B Q4_K_M",
        "ngl_dev": "RX 6500 XT",
    },
    {
        "id": "c9",
        "file": "c9_vk1_rwkv04b.json",
        "label": "E2 · RWKV7 0.4B Q4_K_M · DG1",
        "device": "Vulkan1",
        "config": "c9 (E2)",
        "model": "RWKV7 0.4B Q4_K_M",
        "ngl_dev": "DG1",
    },
    {
        "id": "c10",
        "file": "c10_vk1_ctkq8.json",
        "label": "B6 · B con -ctk q8_0 -ctv q8_0",
        "device": "Vulkan1",
        "config": "c10 (B6)",
        "model": "RWKV 2.9B Q4_K_M",
        "ngl_dev": "B con -ctk q8_0 -ctv q8_0",
    },
    {
        "id": "c11",
        "file": "c11_vk1_qwen06.json",
        "label": "F · Qwen3 0.6B Q4_K_M (transformer) · DG1",
        "device": "Vulkan1",
        "config": "c11 (F)",
        "model": "Qwen3 0.6B Q4_K_M (transformer)",
        "ngl_dev": "DG1",
    },
    {
        "id": "c12",
        "file": "c12_vk0_qwen06.json",
        "label": "F2 · Qwen3 0.6B Q4_K_M (transformer) · RX 6500 XT",
        "device": "Vulkan0",
        "config": "c12 (F2)",
        "model": "Qwen3 0.6B Q4_K_M (transformer)",
        "ngl_dev": "RX 6500 XT",
    },
    {
        "id": "c13",
        "file": "c13_cpu_vibethinker3b.json",
        "label": "G · VibeThinker 3B Q4_K_M (transformer) · CPU",
        "device": "cpu",
        "config": "c13 (G)",
        "model": "VibeThinker 3B Q4_K_M (transformer)",
        "ngl_dev": "CPU",
    },
]

TECHO = {
    "Vulkan0": 144.0,
    "Vulkan1": 68.26,
    "cpu": 41.6,
}


def read_sweep_commands():
    sweep_paths = [
        os.path.join(os.path.dirname(__file__), "run_sweep.sh"),
        "scripts/eureka_fork_llamacpp/run_sweep.sh",
    ]
    for sp in sweep_paths:
        if os.path.exists(sp):
            with open(sp, "r", encoding="utf-8") as f:
                lines = f.readlines()
            cmds = []
            for line in lines:
                s = line.strip()
                if s.startswith('"llama-bench') and s.endswith('",'):
                    cmds.append(s[1:-2])
                elif s.startswith('"llama-bench') and s.endswith('"'):
                    cmds.append(s[1:-1])
                elif s.startswith("llama-bench"):
                    cmds.append(s)
            if cmds:
                return cmds
    return []


def main():
    parser = argparse.ArgumentParser(description="Analizador de medicion fork llama.cpp")
    parser.add_argument("--dir", default="out/eureka-fork-llamacpp", help="Directorio con archivos JSON")
    parser.add_argument("--out", default="out/eureka-fork-llamacpp/MEDICION.md", help="Ruta archivo salida Markdown")
    args = parser.parse_args()

    input_dir = args.dir
    out_file = args.out

    pp_map = {}
    tg_map = {}
    gbs_tg_map = {}

    table_rows = []

    for cfg in CONFIGS:
        cid = cfg["id"]
        fname = cfg["file"]
        device = cfg["device"]
        fpath = os.path.join(input_dir, fname)

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Archivo requerido no encontrado: {fpath}")

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            n_prompt = entry.get("n_prompt", 0)
            avg_ts = entry.get("avg_ts", 0.0)
            stddev_ts = entry.get("stddev_ts", 0.0)
            model_size = entry.get("model_size", 0)

            kind = "pp" if n_prompt > 0 else "tg"
            gbs = model_size / 1e9 * avg_ts

            ref_device = "Vulkan1" if device == "split" else device
            ceiling = TECHO[ref_device]
            pct_techo = gbs / ceiling * 100.0

            if kind == "pp":
                pp_map[cid] = avg_ts
                gbs_str = "—"
                pct_str = "—"
            else:
                tg_map[cid] = avg_ts
                gbs_tg_map[cid] = gbs
                gbs_str = f"{gbs:.2f}"
                if device == "split":
                    pct_str = f"{pct_techo:.2f}%*"
                else:
                    pct_str = f"{pct_techo:.2f}%"

            table_rows.append({
                "config": cfg["config"],
                "model": cfg["model"],
                "ngl_dev": cfg["ngl_dev"],
                "test": kind,
                "ts": f"{avg_ts:.2f}",
                "sd": f"{stddev_ts:.2f}",
                "gbs": gbs_str,
                "pct_techo": pct_str,
            })

    # Calculo de derivados
    gain_pin_device = tg_map["c1"] / tg_map["c6"]
    gain_quant_q4_vs_q8 = tg_map["c1"] / tg_map["c5"]
    gain_fa = tg_map["c2"] / tg_map["c1"]
    gain_ub1024 = tg_map["c3"] / tg_map["c1"]
    gain_ub128 = tg_map["c4"] / tg_map["c1"]
    gain_threads_t4 = tg_map["c7"] / tg_map["c1"]
    gain_kvquant = tg_map["c10"] / tg_map["c1"]
    gain_pp_ub1024 = pp_map["c3"] / pp_map["c1"]
    gain_pp_ub128 = pp_map["c4"] / pp_map["c1"]
    rwkv_vs_transformer_amd = gbs_tg_map["c12"] / gbs_tg_map["c8"]
    rwkv_vs_transformer_intel = gbs_tg_map["c11"] / gbs_tg_map["c9"]
    rwkv_vs_transformer_cpu = gbs_tg_map["c0"] / gbs_tg_map["c13"]
    techo_amd_rwkv29_q4_ts = gbs_tg_map["c12"] / 1.874
    techo_amd_rwkv29_q8_ts = gbs_tg_map["c12"] / 3.258
    headroom_kernel_pct = (techo_amd_rwkv29_q4_ts / tg_map["c1"] - 1.0) * 100.0

    # Construir Markdown tabla mediciones
    md_table_lines = [
        "| config | modelo | ngl/dev | test | t/s | ±sd | GB/s efectivos | % techo |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in table_rows:
        md_table_lines.append(
            f"| {r['config']} | {r['model']} | {r['ngl_dev']} | {r['test']} | {r['ts']} | {r['sd']} | {r['gbs']} | {r['pct_techo']} |"
        )
    md_table = "\n".join(md_table_lines)

    # Derivados tabla
    derivados_items = [
        ("gain_pin_device", f"{gain_pin_device:.2f}"),
        ("gain_quant_q4_vs_q8", f"{gain_quant_q4_vs_q8:.2f}"),
        ("gain_fa", f"{gain_fa:.2f}"),
        ("gain_ub1024", f"{gain_ub1024:.2f}"),
        ("gain_ub128", f"{gain_ub128:.2f}"),
        ("gain_threads_t4", f"{gain_threads_t4:.2f}"),
        ("gain_kvquant", f"{gain_kvquant:.2f}"),
        ("gain_pp_ub1024", f"{gain_pp_ub1024:.2f}"),
        ("gain_pp_ub128", f"{gain_pp_ub128:.2f}"),
        ("rwkv_vs_transformer_amd", f"{rwkv_vs_transformer_amd:.2f}"),
        ("rwkv_vs_transformer_intel", f"{rwkv_vs_transformer_intel:.2f}"),
        ("rwkv_vs_transformer_cpu", f"{rwkv_vs_transformer_cpu:.2f}"),
        ("techo_amd_rwkv29_q4_ts", f"{techo_amd_rwkv29_q4_ts:.2f}"),
        ("techo_amd_rwkv29_q8_ts", f"{techo_amd_rwkv29_q8_ts:.2f}"),
        ("headroom_kernel_pct", f"{headroom_kernel_pct:.2f}%"),
    ]
    md_deriv_lines = [
        "| nombre | valor |",
        "|---|---|",
    ]
    for k, v in derivados_items:
        md_deriv_lines.append(f"| {k} | {v} |")
    md_deriv = "\n".join(md_deriv_lines)

    # Comandos sweep
    sweep_cmds = read_sweep_commands()
    sweep_block = "\n".join(sweep_cmds)

    # Contenido MEDICION.md
    medicion_content = f"""# Medición: Fork de llama.cpp

`pp` es compute-bound: los GB/s efectivos y el % del techo solo aplican a `tg` (decode lee todos los pesos una vez por token). Las filas `pp` muestran `—` en esas dos columnas.

{md_table}

## DERIVADOS

{md_deriv}

## Como se reprodujo

```bash
{sweep_block}
```

## Advertencias de medicion
- Corridas CPU: host compartido (loadavg 5-15). Los tg de CPU varian ±40% entre corridas; no sirven para comparar, solo para descartar la CPU como camino.
- Corridas GPU: ±0,1% entre corridas repetidas; son las que se usan para decidir.
"""

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(medicion_content)

    # JSON final de 23 claves
    final_json_data = {
        "gain_pin_device": round(float(gain_pin_device), 2),
        "gain_quant_q4_vs_q8": round(float(gain_quant_q4_vs_q8), 2),
        "gain_fa": round(float(gain_fa), 2),
        "gain_ub1024": round(float(gain_ub1024), 2),
        "gain_ub128": round(float(gain_ub128), 2),
        "gain_threads_t4": round(float(gain_threads_t4), 2),
        "gain_kvquant": round(float(gain_kvquant), 2),
        "gain_pp_ub1024": round(float(gain_pp_ub1024), 2),
        "gain_pp_ub128": round(float(gain_pp_ub128), 2),
        "rwkv_vs_transformer_amd": round(float(rwkv_vs_transformer_amd), 2),
        "rwkv_vs_transformer_intel": round(float(rwkv_vs_transformer_intel), 2),
        "rwkv_vs_transformer_cpu": round(float(rwkv_vs_transformer_cpu), 2),
        "techo_amd_rwkv29_q4_ts": round(float(techo_amd_rwkv29_q4_ts), 2),
        "techo_amd_rwkv29_q8_ts": round(float(techo_amd_rwkv29_q8_ts), 2),
        "headroom_kernel_pct": round(float(headroom_kernel_pct), 2),
        "tg_c1": round(float(tg_map["c1"]), 2),
        "tg_c5": round(float(tg_map["c5"]), 2),
        "tg_c6": round(float(tg_map["c6"]), 2),
        "gbs_c1": round(float(gbs_tg_map["c1"]), 2),
        "gbs_c5": round(float(gbs_tg_map["c5"]), 2),
        "gbs_c6": round(float(gbs_tg_map["c6"]), 2),
        "gbs_c12": round(float(gbs_tg_map["c12"]), 2),
        "gbs_c8": round(float(gbs_tg_map["c8"]), 2),
    }

    # Salida por stdout
    print(md_table)
    print("\n## DERIVADOS\n")
    print(md_deriv)
    print("")
    print(json.dumps(final_json_data, separators=(",", ":")))


if __name__ == "__main__":
    main()
