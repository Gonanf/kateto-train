#!/usr/bin/env python3
"""Exportador de pares ORPO a partir de candidatos usando reward de RL.

Filtra candidatos según margen mínimo entre respuestas y ausencia de reglas
duras en la respuesta elegida (chosen), validando cada par generado contra
scripts/validate_orpo_pairs.py::check.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for p in [_ROOT, _ROOT / "pipelines", _ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from .reward import REGLAS_DURAS, reward
except ImportError:
    from scripts.rl.reward import REGLAS_DURAS, reward

import validate_orpo_pairs as vop

MARGEN_MINIMO = 0.15
UMBRAL_REGLA_DURA = 0.0


def formatear_prompt(prompt: str) -> str:
    """Asegura que el prompt termine en \\n<|im_start|>seco\\n."""
    p = prompt.strip()
    if not p.endswith("<|im_start|>seco"):
        p = f"{p}\n<|im_start|>seco\n"
    else:
        p = f"{p}\n"
    return p


def formatear_respuesta(resp: str) -> str:
    """Asegura que la respuesta termine en <|im_end|>\\n."""
    r = resp.strip()
    if r.endswith("<|im_end|>"):
        r = r[: -len("<|im_end|>")].strip()
    return f"{r}<|im_end|>\n"


def procesar_candidatos(
    in_path: Path,
    out_path: Path,
    *,
    margen_minimo: float = MARGEN_MINIMO,
) -> dict[str, Any]:
    """Procesa candidatos, filtra por reward y exporta pares ORPO válidos."""
    lines = [l for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    candidatos = [json.loads(l) for l in lines]

    pares_validos: list[dict[str, str]] = []
    descartes: Counter[str] = Counter()

    for i, item in enumerate(candidatos, 1):
        prompt_raw = item.get("prompt", "")
        resps = item.get("respuestas", [])
        if len(resps) < 2:
            descartes["respuestas_insuficientes"] += 1
            continue

        scored = [(reward(r)["score"], reward(r), r) for r in resps]
        scored.sort(key=lambda x: x[0])

        sc_rej, meta_rej, text_rej = scored[0]
        sc_ch, meta_ch, text_ch = scored[-1]

        margen = sc_ch - sc_rej
        if margen < margen_minimo:
            descartes["margen_insuficiente"] += 1
            continue

        if any(err in REGLAS_DURAS for err in meta_ch.get("errores", [])):
            descartes["chosen_regla_dura"] += 1
            continue

        # Formato de salida requerido
        prompt_fmt = formatear_prompt(prompt_raw)
        chosen_fmt = formatear_respuesta(text_ch)
        rejected_fmt = formatear_respuesta(text_rej)

        row = {
            "prompt": prompt_fmt,
            "chosen": chosen_fmt,
            "rejected": rejected_fmt,
        }

        # Validar con scripts/validate_orpo_pairs.py::check adaptando delimiters para la voz
        check_row = {
            "prompt": row["prompt"],
            "chosen": (
                row["chosen"]
                if "<|im_start|>" in row["chosen"]
                else f"<|im_start|>seco\n{row['chosen']}"
            ),
            "rejected": (
                row["rejected"]
                if "<|im_start|>" in row["rejected"]
                else f"<|im_start|>seco\n{row['rejected']}"
            ),
        }
        err_msg = vop.check(check_row, i)
        if err_msg:
            motivo = err_msg.split(":", 1)[-1].strip()
            descartes[f"validador:{motivo}"] += 1
            continue

        pares_validos.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pares_validos:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return {
        "total_candidatos": len(candidatos),
        "pares_exportados": len(pares_validos),
        "descartes": dict(descartes),
        "out_path": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exportar pares ORPO filtrados por reward.")
    parser.add_argument("entrada", help="Ruta al archivo candidatos.jsonl")
    parser.add_argument("--out", required=True, help="Ruta de salida para pares jsonl")
    parser.add_argument("--margen", type=float, default=MARGEN_MINIMO, help="Margen mínimo entre chosen y rejected")
    args = parser.parse_args()

    in_path = Path(args.entrada)
    out_path = Path(args.out)
    res = procesar_candidatos(in_path, out_path, margen_minimo=args.margen)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
