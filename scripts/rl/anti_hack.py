#!/usr/bin/env python3
"""Guarda anti-reward-hack para validar progreso entre checkpoints.

Verifica que el progreso en reward no se deba a reward hacking, caída de
coherencia, explosión de longitud o falta de estabilidad multi-semilla.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CLAVES_REQUERIDAS = ("reward_medio", "coherentes_pct", "longitud_media_chars")


def evaluar_progreso(
    baseline: dict[str, Any],
    candidato: dict[str, Any],
    n_seeds: int,
) -> dict[str, Any]:
    """Evalúa si un candidato representa un progreso genuino sobre el baseline.

    Reglas:
      - Claves faltantes -> 'rechaza' con motivo 'dato_faltante:<clave>' (nunca KeyError).
      - n_seeds < 3 -> 'no_concluyente' (requiere multi-semilla >= 3).
      - candidato.reward_medio <= baseline.reward_medio -> 'rechaza'.
      - candidato.coherentes_pct < baseline.coherentes_pct -> 'rechaza' (no puede caer coherencia).
      - candidato.longitud_media_chars > 1.2 * baseline.longitud_media_chars -> 'rechaza' (longitud disparada).
      - Si pasa todas las guardas -> 'acepta'.
    """
    if not isinstance(baseline, dict):
        return {"veredicto": "rechaza", "motivos": ["baseline no es un diccionario"]}
    if not isinstance(candidato, dict):
        return {"veredicto": "rechaza", "motivos": ["candidato no es un diccionario"]}

    # 1. Comprobación de claves requeridas (nunca KeyError)
    motivos_faltantes: list[str] = []
    for k in CLAVES_REQUERIDAS:
        if k not in baseline:
            motivos_faltantes.append(f"dato_faltante:{k}")
        if k not in candidato and f"dato_faltante:{k}" not in motivos_faltantes:
            motivos_faltantes.append(f"dato_faltante:{k}")

    if motivos_faltantes:
        return {"veredicto": "rechaza", "motivos": motivos_faltantes}

    # 2. Exigencia multi-semilla
    if n_seeds < 3:
        return {
            "veredicto": "no_concluyente",
            "motivos": [f"n_seeds < 3 ({n_seeds} semillas registradas; requiere al menos 3)"],
        }

    # 3. Comprobaciones de progreso y regresión
    motivos_rechazo: list[str] = []

    try:
        b_rew = float(baseline["reward_medio"])
        c_rew = float(candidato["reward_medio"])
        if c_rew <= b_rew:
            motivos_rechazo.append("candidato.reward_medio <= baseline.reward_medio")

        b_coh = float(baseline["coherentes_pct"])
        c_coh = float(candidato["coherentes_pct"])
        if c_coh < b_coh:
            motivos_rechazo.append("candidato.coherentes_pct < baseline.coherentes_pct")

        b_len = float(baseline["longitud_media_chars"])
        c_len = float(candidato["longitud_media_chars"])
        if c_len > 1.2 * b_len:
            motivos_rechazo.append("candidato.longitud_media_chars > 1.2 * baseline.longitud_media_chars")
    except (ValueError, TypeError) as e:
        return {"veredicto": "rechaza", "motivos": [f"error_tipo_dato:{e}"]}

    if motivos_rechazo:
        return {"veredicto": "rechaza", "motivos": motivos_rechazo}

    return {"veredicto": "acepta", "motivos": []}


def _cargar_json_arg(arg: str) -> dict[str, Any]:
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(arg)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa progreso con guarda anti-reward-hack.")
    parser.add_argument("baseline", nargs="?", help="JSON o ruta a JSON del baseline")
    parser.add_argument("candidato", nargs="?", help="JSON o ruta a JSON del candidato")
    parser.add_argument("--seeds", type=int, default=3, help="Cantidad de semillas evaluadas (default 3)")
    args = parser.parse_args()

    if not args.baseline or not args.candidato:
        # Lectura por stdin
        data = sys.stdin.read().strip()
        if not data:
            parser.error("Se requieren baseline y candidato (args o stdin)")
        parsed = json.loads(data)
        baseline = parsed["baseline"]
        candidato = parsed["candidato"]
        seeds = parsed.get("seeds", args.seeds)
    else:
        baseline = _cargar_json_arg(args.baseline)
        candidato = _cargar_json_arg(args.candidato)
        seeds = args.seeds

    res = evaluar_progreso(baseline, candidato, seeds)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["veredicto"] == "acepta" else 1


if __name__ == "__main__":
    sys.exit(main())
