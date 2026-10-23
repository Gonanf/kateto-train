"""Tests unitarios para la guarda anti-reward-hack (scripts/rl/anti_hack.py)."""
from __future__ import annotations

import pytest

from scripts.rl.anti_hack import evaluar_progreso

BASELINE = {
    "reward_medio": 0.60,
    "coherentes_pct": 63.33,
    "longitud_media_chars": 100.0,
}


def test_camino_1_semillas_menor_a_3():
    candidato = {
        "reward_medio": 0.75,
        "coherentes_pct": 65.0,
        "longitud_media_chars": 105.0,
    }
    res = evaluar_progreso(BASELINE, candidato, n_seeds=2)
    assert res["veredicto"] == "no_concluyente"
    assert any("n_seeds < 3" in m for m in res["motivos"])


def test_camino_2_reward_baja_o_igual():
    candidato = {
        "reward_medio": 0.59,  # Menor que baseline 0.60
        "coherentes_pct": 65.0,
        "longitud_media_chars": 105.0,
    }
    res = evaluar_progreso(BASELINE, candidato, n_seeds=3)
    assert res["veredicto"] == "rechaza"
    assert "candidato.reward_medio <= baseline.reward_medio" in res["motivos"]


def test_camino_3_coherencia_cae():
    candidato = {
        "reward_medio": 0.80,
        "coherentes_pct": 60.0,  # Menor que baseline 63.33
        "longitud_media_chars": 105.0,
    }
    res = evaluar_progreso(BASELINE, candidato, n_seeds=3)
    assert res["veredicto"] == "rechaza"
    assert "candidato.coherentes_pct < baseline.coherentes_pct" in res["motivos"]


def test_camino_4_longitud_disparada():
    candidato = {
        "reward_medio": 0.85,
        "coherentes_pct": 65.0,
        "longitud_media_chars": 130.0,  # > 1.2 * 100.0 = 120.0 (+30%)
    }
    res = evaluar_progreso(BASELINE, candidato, n_seeds=3)
    assert res["veredicto"] == "rechaza"
    assert "candidato.longitud_media_chars > 1.2 * baseline.longitud_media_chars" in res["motivos"]


def test_camino_5_acepta_progreso():
    candidato = {
        "reward_medio": 0.75,
        "coherentes_pct": 64.0,
        "longitud_media_chars": 110.0,
    }
    res = evaluar_progreso(BASELINE, candidato, n_seeds=3)
    assert res["veredicto"] == "acepta"
    assert len(res["motivos"]) == 0


def test_claves_faltantes_rechaza_sin_keyerror():
    candidato_incompleto = {
        "reward_medio": 0.75,
        # Falta coherentes_pct y longitud_media_chars
    }
    res = evaluar_progreso(BASELINE, candidato_incompleto, n_seeds=3)
    assert res["veredicto"] == "rechaza"
    assert "dato_faltante:coherentes_pct" in res["motivos"]
    assert "dato_faltante:longitud_media_chars" in res["motivos"]
