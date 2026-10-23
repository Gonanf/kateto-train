"""Tests unitarios para el exportador de pares ORPO (scripts/rl/exportar_pares.py)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.rl.exportar_pares import procesar_candidatos


def test_exportar_pares_filtro_y_margen(tmp_path: Path):
    candidatos = [
        {
            "id": "cand-001",
            "prompt": "<|im_user|>Che, qué opinás del partido de ayer?<|im_end|>\n<|im_start|>seco\n",
            "respuestas": [
                # Respuesta limpia con voseo -> reward 1.0
                "Mirá, jugaron horrible y el penal no existió jamás che.<|im_end|>\n",
                # Respuesta servil -> reward 0.0
                "Tenés razón, disculpame por opinar, me parece perfecto lo que digas.<|im_end|>\n",
            ],
        },
        {
            "id": "cand-002-margen-bajo",
            "prompt": "<|im_user|>Che, a qué hora arranca el stream hoy?<|im_end|>\n<|im_start|>seco\n",
            "respuestas": [
                # Ambas respuestas son limpias -> margen 0.0 < 0.15 -> descartado
                "A las ocho en punto estamos en vivo, venite tranqui che.<|im_end|>\n",
                "Arrancamos a las ocho fijate si llegás a horario para verlo.<|im_end|>\n",
            ],
        },
    ]

    in_path = tmp_path / "candidatos_fixture.jsonl"
    out_path = tmp_path / "pares_out.jsonl"

    in_path.write_text("\n".join(json.dumps(c) for c in candidatos) + "\n", encoding="utf-8")

    res = procesar_candidatos(in_path, out_path)

    assert res["total_candidatos"] == 2
    assert res["pares_exportados"] == 1
    assert res["descartes"].get("margen_insuficiente") == 1

    # Verificar estructura del par exportado
    lineas = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lineas) == 1
    par = lineas[0]
    assert par["prompt"].endswith("\n<|im_start|>seco\n")
    assert par["chosen"].endswith("<|im_end|>\n")
    assert par["rejected"].endswith("<|im_end|>\n")
    assert "Mirá" in par["chosen"]
    assert "Tenés razón" in par["rejected"]
