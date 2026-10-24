#!/usr/bin/env python3
"""Construye el notebook de evaluación A/B y metadatos para Kaggle.

Especificación: Brief t_7cbaf7c5 (SPEC 2).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def calcular_sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir kernel Kaggle A/B de cierre few-shot")
    parser.add_argument(
        "--v7",
        type=Path,
        default=Path("/tmp/kpull-v7/kateto-eval29-v7-2xt4.ipynb"),
        help="Ruta al notebook real v7 de Kaggle",
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=_REPO_ROOT / "kaggle/kernels/eval29_cierre_c1/prefix.json",
        help="Ruta a prefix.json generado",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_REPO_ROOT / "kaggle/kernels/eval29_cierre_c1",
        help="Directorio destino para kernel-metadata.json, eval29_cierre_c1.ipynb y reference",
    )
    args = parser.parse_args()

    assert args.v7.exists(), f"No existe el notebook v7: {args.v7}"
    assert args.prefix.exists(), f"No existe prefix.json: {args.prefix}"

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. Copia byte-exacta de v7 a reference-v7-as-run.ipynb
    ref_path = args.outdir / "reference-v7-as-run.ipynb"
    shutil.copy2(args.v7, ref_path)
    ref_sha256 = calcular_sha256_archivo(ref_path)

    # 2. Metadata para Kaggle
    meta = {
        "id": "gabrielsolotorevsky/kateto-eval29-cierre-c1-2xt4",
        "title": "kateto-eval29-cierre-c1-2xt4",
        "code_file": "eval29_cierre_c1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4x2",
        "dataset_sources": [
            "gabrielsolotorevsky/kateto-eval29",
            "gabrielsolotorevsky/kateto-eval29-v7-code",
        ],
    }
    meta_path = args.outdir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # 3. Leer prefix.json
    prefix_str = args.prefix.read_text(encoding="utf-8").strip()
    prefix_obj = json.loads(prefix_str)
    sha256_prefijo = prefix_obj["sha256_prefijo"]
    lineas_elegidas = prefix_obj["fuente"]["lineas"]

    # 4. Leer v7 notebook
    v7_nb = json.loads(args.v7.read_text(encoding="utf-8"))
    assert len(v7_nb["cells"]) == 8, f"v7 debe tener 8 celdas, tiene {len(v7_nb['cells'])}"

    # 5. Armar las 8 celdas
    # Celda 0: Markdown nuevo
    md_source = f"""# A/B cierre few-shot — 2 turnos cerrados en el prefijo (v7 + prefijo)

- **Checkpoint**: el mismo merged del v7 (mismo base HF `BlinkDL/rwkv7-g1::rwkv7-g1j-2.9b-20260831-ctx16384.pth`, misma revisión `cd67fb95fa9e2ce8757f8d21d713e74a9c788118`, mismo adapter `out/kateto-final/adapter_model.safetensors`).
- **Parámetros**: `max_len=256`, T 0.80, top_p 0.70, seed 1337, `WKV=cuda`, mismos 60 items, mismo motor (`rwkv-peft nativo (.pth) / rwkvt RWKV7, WKV=cuda`).
- **Brazo A**: `render_chat([], p, "seco")` (idéntico al v7).
- **Brazo B**: `render_chat(HIST, p, "seco")` con `HIST` = los 2 turnos del `prefix.json`.
- **SHA-256 del prefijo**: `{sha256_prefijo}`.
- **Nota**: El prefijo sale del set de ENTRENAMIENTO (`data/rwkv_kateto_seco.jsonl`, líneas {lineas_elegidas[0]} y {lineas_elegidas[1]}) y no de ningún set de evaluación.
"""
    cell_0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": md_source,
    }

    # Celdas 1..5 intactas del v7
    cells_1_to_5 = [copy.deepcopy(v7_nb["cells"][i]) for i in range(1, 6)]

    # Celda 6: Generación A/B
    c6_code = f"""import statistics
import os as _os
# Mismo motor que v5: rama CUDA de rwkvt/operator/rwkvop.py (no triton).
_os.environ["WKV"] = "cuda"  # antes de importar rwkv_pipeline (setdefault no pisa)
MOTOR = f"rwkv-peft nativo (.pth) / rwkvt RWKV7, WKV={{_os.environ.get('WKV', '?')}}"

pv1_path = PROJ / "out/eval/prompts_v1.json"
pv1_data = json.loads(pv1_path.read_text(encoding="utf-8"))
items = []
for it in pv1_data["generation"][:30]:
    items.append({{
        "set": "prompts_v1_generation",
        "id": it["id"],
        "prompt": it["prompt"],
    }})

ep_path = PROJ / "data/eval_prompts.jsonl"
ep_lines = [ln.strip() for ln in ep_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
for i, ln in enumerate(ep_lines[:30]):
    row_data = json.loads(ln)
    items.append({{
        "set": "eval_prompts_jsonl",
        "id": f"ep_{{i:02d}}",
        "prompt": row_data["prompt"],
    }})

if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
import rwkv_pipeline.opd_rollout as orl
from scripts.eval_behavior import load_student, ensure_eval
from chat_template import render_chat, render_historial

model, tok, info = load_student(str(MERGED), device="cuda")
ensure_eval(model)
STATE["arch"] = info

PREFIX = json.loads(r\"\"\"{prefix_str}\"\"\")
HIST = [tuple(t) for t in PREFIX["turnos"]]
PREFIJO_TXT = PREFIX["prefijo_texto"]
assert render_historial(HIST, "seco") == PREFIJO_TXT, "El historial no coincide con prefijo_texto"
assert PREFIX["sha256_prefijo"] == hashlib.sha256(PREFIJO_TXT.encode("utf-8")).hexdigest(), "SHA256 del prefijo no coincide"

n_prefijo_tokens = len(tok.encode(PREFIJO_TXT))
log(f"Prefijo validado: {{len(HIST)}} turnos, {{len(PREFIJO_TXT)}} chars, {{n_prefijo_tokens}} tokens")

def fmt(p, arm):
    if arm == "A":
        return render_chat([], p, "seco")
    elif arm == "B":
        return render_chat(HIST, p, "seco")
    raise ValueError(f"Brazo desconocido: {{arm}}")

_ORIG = orl._next_logits

def make_timer():
    box = {{"n": 0, "t0": None, "ttfab": None}}
    def wrapped(*args, **kwargs):
        if box["n"] == 0:
            box["t0"] = time.time()
        res = _ORIG(*args, **kwargs)
        if box["n"] == 0:
            box["ttfab"] = round(time.time() - box["t0"], 4)
        box["n"] += 1
        return res
    return box, wrapped

def check_degenerate(text):
    t = text.strip()
    if len(t) < 8:
        return False, 0
    reps = 0
    for n in (8, 12, 16, 24):
        for i in range(0, max(0, len(t) - 2 * n)):
            sub = t[i:i + n]
            c = t.count(sub)
            if c > reps:
                reps = c
    return (reps >= 3), reps

dev_str = "cuda0:" + torch.cuda.get_device_name(0)

def gen_row(item, turno, arm, rows_file):
    box, wrapped = make_timer()
    orl._next_logits = wrapped
    try:
        torch.manual_seed(1337)
        prompt_txt = item["prompt"]
        prompt_formatted = fmt(prompt_txt, arm)
        r = orl.rollout_from_prompt(
            model, tok, prompt_formatted,
            max_len=256, temperature=0.80, top_p=0.70
        )
        text = orl._decode_str(tok, list(r.response_ids))
        is_deg, max_rep = check_degenerate(text)
        row = {{
            "set": item["set"],
            "id": item["id"],
            "prompt": prompt_txt,
            "prompt_sha256": hashlib.sha256(prompt_txt.encode("utf-8")).hexdigest(),
            "model": "kateto-rwkv29-merged.pth",
            "motor": MOTOR,
            "engine": "rwkvt.rwkv7.model.RWKV7",
            "device": dev_str,
            "turno": turno,
            "chat_format": "chatml",
            "arm": arm,
            "prefix_sha256": PREFIX["sha256_prefijo"] if arm == "B" else None,
            "n_prefijo_tokens": n_prefijo_tokens if arm == "B" else 0,
            "ttfab_s": box["ttfab"],
            "wall_s": r.wall_s,
            "n_tokens": len(r.response_ids),
            "finished": bool(r.finished),
            "stop_reason": getattr(r, "stop_reason", None),
            "degenerate": is_deg,
            "max_repeat": max_rep,
            "text": text[:4000],
            "error": None,
        }}
    except Exception as e:
        prompt_txt = item["prompt"]
        row = {{
            "set": item["set"],
            "id": item["id"],
            "prompt": prompt_txt,
            "prompt_sha256": hashlib.sha256(prompt_txt.encode("utf-8")).hexdigest(),
            "model": "kateto-rwkv29-merged.pth",
            "motor": MOTOR,
            "engine": "rwkvt.rwkv7.model.RWKV7",
            "device": dev_str,
            "turno": turno,
            "chat_format": "chatml",
            "arm": arm,
            "prefix_sha256": PREFIX["sha256_prefijo"] if arm == "B" else None,
            "n_prefijo_tokens": n_prefijo_tokens if arm == "B" else 0,
            "ttfab_s": box["ttfab"],
            "wall_s": None,
            "n_tokens": 0,
            "finished": False,
            "stop_reason": None,
            "degenerate": False,
            "max_repeat": 0,
            "text": "",
            "error": str(e),
        }}
    finally:
        orl._next_logits = _ORIG

    with open(rows_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
        f.flush()
    return row

for arm in ("A", "B"):
    arm_dir = OUT / f"arm-{{arm}}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    rows_file = arm_dir / "gen29_rows.jsonl"
    if rows_file.exists():
        rows_file.unlink()

    log(f"=== INICIANDO BRAZO {{arm}} ===")
    arm_rows = []
    dummy_item = {{"set": "dummy", "id": "dummy_00", "prompt": "hola"}}
    r_dummy = gen_row(dummy_item, "frio_proceso", arm, rows_file)
    arm_rows.append(r_dummy)

    for idx, it in enumerate(items):
        rf = gen_row(it, "frio", arm, rows_file)
        rc = gen_row(it, "caliente", arm, rows_file)
        arm_rows.extend([rf, rc])
        if (idx + 1) % 10 == 0:
            log(f"[Brazo {{arm}}] Generados {{idx + 1}}/{{len(items)}} items...")

    assert len(arm_rows) == 121, f"Brazo {{arm}}: esperado 121 filas, obtenido {{len(arm_rows)}}"
    if arm == "B":
        tokens_set = {{r["n_prefijo_tokens"] for r in arm_rows}}
        assert len(tokens_set) == 1 and list(tokens_set)[0] > 0, f"Brazo B: tokens_set {{tokens_set}} invalido"
    else:
        assert all(r["n_prefijo_tokens"] == 0 for r in arm_rows), "Brazo A: tokens no son todos 0"

    val_frio = [r["ttfab_s"] for r in arm_rows if r["turno"] == "frio" and r["ttfab_s"] is not None and not r["error"]]
    val_caliente = [r["ttfab_s"] for r in arm_rows if r["turno"] == "caliente" and r["ttfab_s"] is not None and not r["error"]]
    med_frio = round(statistics.median(val_frio), 4) if val_frio else None
    med_caliente = round(statistics.median(val_caliente), 4) if val_caliente else None

    prefix_block = {{k: v for k, v in PREFIX.items() if k != "prefijo_texto"}} if arm == "B" else None
    arm_manifest = {{
        "arm": arm,
        "prefix": prefix_block,
        "merged": STATE.get("merged"),
        "base": STATE.get("base"),
        "adapter_sha256": STATE.get("adapter_sha256"),
        "arch": STATE.get("arch"),
        "sampling": {{
            "max_len": 256,
            "temperature": 0.80,
            "top_p": 0.70,
            "seed": 1337,
            "ngram_block": None,
        }},
        "gen_rows": {{
            "n": len(arm_rows),
            "median_ttfab_frio": med_frio,
            "median_ttfab_caliente": med_caliente,
        }},
        "n_prefijo_tokens": n_prefijo_tokens if arm == "B" else 0,
    }}
    (arm_dir / "manifest.json").write_text(json.dumps(arm_manifest, ensure_ascii=False, indent=1), encoding="utf-8")

STATE["arms"] = {{
    "arm-A": {{"path": str(OUT / "arm-A" / "gen29_rows.jsonl"), "n_filas": 121}},
    "arm-B": {{"path": str(OUT / "arm-B" / "gen29_rows.jsonl"), "n_filas": 121, "n_prefijo_tokens": n_prefijo_tokens, "sha256_prefijo": PREFIX["sha256_prefijo"]}},
}}

from scripts.eval_behavior import free_gpu
free_gpu(model)

flush_state()
"""
    cell_6 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": c6_code,
    }

    # Celda 7: Cierre / Manifest
    c7_code = f"""REFERENCE_V7_SHA256 = "{ref_sha256}"

sha_base = STATE.get("base", {{}}).get("sha256", "unknown")
sha_adapter = STATE.get("adapter_sha256", "unknown")
sha_merged = STATE.get("merged", {{}}).get("sha256", "unknown")

summary_ab = f\"\"\"# Resumen A/B Cierre Few-Shot

## Artefactos
- **Base SHA-256**: `{{sha_base}}`
- **Adapter SHA-256**: `{{sha_adapter}}`
- **Merged SHA-256**: `{{sha_merged}}`
- **Reference v7 SHA-256**: `{{REFERENCE_V7_SHA256}}`

## Prefijo Few-Shot
- **Prefijo SHA-256**: `{{PREFIX['sha256_prefijo']}}`
- **Prefijo tokens**: `{{n_prefijo_tokens}}`

## Filas Generadas
- **Brazo A**: `{{OUT / "arm-A" / "gen29_rows.jsonl"}}` (121 filas)
- **Brazo B**: `{{OUT / "arm-B" / "gen29_rows.jsonl"}}` (121 filas)
\"\"\"

(OUT / "SUMMARY-AB.md").write_text(summary_ab, encoding="utf-8")
log(f"SUMMARY-AB.md escrito en {{OUT / 'SUMMARY-AB.md'}}")
print("A/B DONE", flush=True)
flush_state()
"""
    cell_7 = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": c7_code,
    }

    new_cells = [cell_0] + cells_1_to_5 + [cell_6, cell_7]
    assert len(new_cells) == 8, f"Se esperaban 8 celdas, hay {len(new_cells)}"

    new_nb = {
        "cells": new_cells,
        "metadata": v7_nb.get("metadata", {}),
        "nbformat": v7_nb.get("nbformat", 4),
        "nbformat_minor": v7_nb.get("nbformat_minor", 5),
    }

    nb_out_path = args.outdir / "eval29_cierre_c1.ipynb"
    nb_out_path.write_text(json.dumps(new_nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Kernel generado exitosamente en {nb_out_path}")


if __name__ == "__main__":
    main()
