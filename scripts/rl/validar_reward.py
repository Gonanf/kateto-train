#!/usr/bin/env python3
"""Replay del reward sobre datos reales (kateto_orpo_v8 y eval29).

Reporta tasas por categoría, detección de degeneradas y chequeos de reglas vacuas/asesinas.
Uso:
    venv/bin/python scripts/rl/validar_reward.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from .reward import reward
except ImportError:
    from scripts.rl.reward import reward

DEFAULT_PARES = "/run/media/chaos/terciario/proyectos/kateto-train/data/kateto_orpo_v8.jsonl"
DEFAULT_EVAL29 = "/home/chaos/kaggle-out-eval29-v5/out-v5/out-eval29/gen29_rows.jsonl"


def validar_orpo(pares_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Evalúa la separación score(chosen) > score(rejected) en los pares ORPO."""
    if not pares_path.exists():
        return {"error": f"Archivo no encontrado: {pares_path}"}, [f"Falta archivo {pares_path}"]

    lines = [l for l in pares_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]

    # Intentar cargar metadata complementaria si no viene en las filas
    meta_rows: list[dict[str, Any]] = []
    meta_candidates = [
        pares_path.parent / (pares_path.stem.replace("kateto_", "") + "_meta.jsonl"),
        pares_path.parent / "orpo_v8_meta.jsonl",
        pares_path.with_name(pares_path.stem + "_meta.jsonl"),
    ]
    for m_cand in meta_candidates:
        if m_cand.exists():
            try:
                meta_rows = [
                    json.loads(l)
                    for l in m_cand.read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
                break
            except Exception:
                pass

    by_cat: dict[str, dict[str, Any]] = {}
    total_pares = len(rows)
    total_separados = 0
    total_chosen_fp = 0
    total_rej_det = 0

    for i, r in enumerate(rows):
        cat = (
            r.get("cat")
            or r.get("categoria")
            or r.get("_cat")
            or (meta_rows[i].get("categoria") if i < len(meta_rows) else "general")
        )
        if cat not in by_cat:
            by_cat[cat] = {
                "pares": 0,
                "separacion_estricta": 0,
                "tasa_separacion": 0.0,
                "chosen_falsos_positivos": 0,
                "rejected_detectados": 0,
            }
        by_cat[cat]["pares"] += 1

        c_sc = reward(r.get("chosen", ""))["score"]
        rej_sc = reward(r.get("rejected", ""))["score"]

        if c_sc < 1.0:
            total_chosen_fp += 1
            by_cat[cat]["chosen_falsos_positivos"] += 1
        if rej_sc < 1.0:
            total_rej_det += 1
            by_cat[cat]["rejected_detectados"] += 1
        if c_sc > rej_sc:
            total_separados += 1
            by_cat[cat]["separacion_estricta"] += 1

    for cat, st in by_cat.items():
        st["tasa_separacion"] = round(100.0 * st["separacion_estricta"] / st["pares"], 2) if st["pares"] else 0.0

    tasa_total = round(100.0 * total_separados / total_pares, 2) if total_pares else 0.0
    orpo_out = {
        "total": {
            "pares": total_pares,
            "separacion_estricta": total_separados,
            "tasa_separacion": tasa_total,
            "chosen_falsos_positivos": total_chosen_fp,
            "rejected_detectados": total_rej_det,
        },
        "por_categoria": by_cat,
    }

    # Chequeo de reglas vacuas / asesinas
    alertas: list[str] = []
    for cat, st in by_cat.items():
        tasa = st["tasa_separacion"]
        if tasa == 100.0:
            alertas.append(f"Regla asesina/100% separación en categoría '{cat}' (separación determinista completa)")
        elif tasa == 0.0:
            if cat == "debate":
                alertas.append("Categoría 'debate': separación 0.0% (hueco conocido sin juez de persona)")
            else:
                alertas.append(f"Regla vacua/0% separación en categoría '{cat}'")

    return orpo_out, alertas


def validar_eval29(eval29_path: Path) -> dict[str, Any]:
    """Evalúa scores en eval29 contrastando filas con degenerate: True vs False."""
    if not eval29_path.exists():
        return {"error": f"Archivo no encontrado: {eval29_path}"}

    lines = [l for l in eval29_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in lines]

    by_set: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        s = r.get("set", "desconocido")
        is_deg = bool(r.get("degenerate"))
        sc = reward(r.get("text") or "")["score"]
        by_set.setdefault(s, {"deg": [], "non_deg": []})
        if is_deg:
            by_set[s]["deg"].append(sc)
        else:
            by_set[s]["non_deg"].append(sc)

    out_set: dict[str, Any] = {}
    all_deg: list[float] = []
    all_non_deg: list[float] = []

    for s, data in sorted(by_set.items()):
        all_deg.extend(data["deg"])
        all_non_deg.extend(data["non_deg"])
        mean_deg = round(sum(data["deg"]) / len(data["deg"]), 4) if data["deg"] else None
        mean_non = round(sum(data["non_deg"]) / len(data["non_deg"]), 4) if data["non_deg"] else None
        out_set[s] = {
            "score_medio_degeneradas": mean_deg,
            "score_medio_no_degeneradas": mean_non,
            "n_degeneradas": len(data["deg"]),
            "n_no_degeneradas": len(data["non_deg"]),
        }

    total_mean_deg = round(sum(all_deg) / len(all_deg), 4) if all_deg else None
    total_mean_non = round(sum(all_non_deg) / len(all_non_deg), 4) if all_non_deg else None

    return {
        "por_set": out_set,
        "total": {
            "score_medio_degeneradas": total_mean_deg,
            "score_medio_no_degeneradas": total_mean_non,
            "n_degeneradas": len(all_deg),
            "n_no_degeneradas": len(all_non_deg),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validación y replay de reward de RL.")
    parser.add_argument("--pares", default=DEFAULT_PARES, help="Ruta a kateto_orpo_v8.jsonl")
    parser.add_argument("--eval29", default=DEFAULT_EVAL29, help="Ruta a gen29_rows.jsonl")
    parser.add_argument("--json", action="store_true", help="Imprimir un JSON único")
    args = parser.parse_args()

    orpo_out, alertas = validar_orpo(Path(args.pares))
    eval29_out = validar_eval29(Path(args.eval29))

    payload = {
        "orpo_v8": orpo_out,
        "eval29": eval29_out,
        "alertas": alertas,
    }

    # Se imprime exactamente un JSON único
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
