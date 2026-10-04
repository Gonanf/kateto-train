#!/usr/bin/env python3
"""todo18: JSON-valid rate on a FIXED tool-calling prompt set, baseline vs new weights.

Usage: venv-unsloth-qwen/bin/python scripts/eval_toolcall_json.py <baseline.pth> <candidate.pth> [--out out/retrain_eval.json]
Comparability: defaults temp 0.80 / top_p 0.70 (infer defaults since todo 12).
"""
import json
import re
import sys

if __name__ == "__main__" and len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <baseline.pth> <candidate.pth> [--out out/retrain_eval.json]", file=sys.stderr)
    sys.exit(2)

import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "RWKV-PEFT")
from rwkv_pipeline.infer_kateto import KatetoInferenceEngine  # noqa: E402

TOOL_PROMPTS = [
    "Leé el archivo prompt_builder.py, líneas 40 a 130.",
    "Mostrá el estado del repo con git status.",
    "Listá los archivos en el directorio data/.",
    "Escribí hola mundo en un archivo /tmp/saludo.txt.",
    "Buscá la función generate en infer_kateto.py.",
    "Contá cuántas líneas tiene el archivo README.md.",
    "Creá una carpeta /tmp/kateto_test.",
    "Leé las primeras 20 líneas de train.py.",
    "Verificá si existe el archivo config/state_tuning.yaml.",
    "Mostrá el uso de disco del directorio out/.",
    "Buscá todos los TODO en rwkv_pipeline/.",
    "Cloná el estado actual del repo en /tmp/repo_bak.",
]

DEBATE_PROMPTS = [
    "El mate amargo es mejor que el mate dulce.",
    "El fernet con coca es un invento sobrevalorado.",
    "El truco es el mejor juego de cartas argentino.",
    "La pizza argentina le gana a la italiana.",
    "El colectivo siempre llega tarde en hora pico.",
]

TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def tool_valid(text):
    m = TOOL_RE.search(text)
    if not m:
        return False
    try:
        obj = json.loads(m.group(1).strip())
    except Exception:
        return False
    return isinstance(obj, dict) and "name" in obj


def run(model_path):
    torch.manual_seed(7)
    eng = KatetoInferenceEngine(model_path, "RWKV-PEFT/rwkv_vocab_v20230424.txt", device="cuda")
    tools = [tool_valid(eng.generate(p, voice="seco")) for p in TOOL_PROMPTS]
    debates = [("<tool_call>" in eng.generate(p, voice="seco")) for p in DEBATE_PROMPTS]
    return {
        "model": model_path,
        "tool_valid": sum(tools),
        "tool_total": len(tools),
        "tool_rate": round(sum(tools) / len(tools), 4),
        "debate_toolcall_leaks": sum(debates),
        "debate_total": len(debates),
    }


if __name__ == "__main__":
    baseline, candidate = sys.argv[1], sys.argv[2]
    out = sys.argv[4] if len(sys.argv) > 4 and sys.argv[3] == "--out" else "out/retrain_eval.json"
    base = run(baseline)
    print("BASELINE", json.dumps(base), flush=True)
    cand = run(candidate)
    print("CANDIDATE", json.dumps(cand), flush=True)
    reg = float(cand["tool_rate"]) < float(base["tool_rate"])
    print(("REGRESSION" if reg else "NO-REGRESSION") + f" delta={float(cand['tool_rate']) - float(base['tool_rate']):+.4f}", flush=True)
    with open(out, "w") as f:
        json.dump({"baseline": base, "candidate": cand, "regression": reg}, f, indent=2)
    print(f"wrote {out}", flush=True)
