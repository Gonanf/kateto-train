"""Tests unitarios para scripts/rl/reward.py y criterios deterministas."""
from __future__ import annotations

import builtins
import hashlib
import importlib
import sys
import kateto_purge as kp
import pytest

from scripts.rl.reward import (
    reward,
    reward_total,
    score_persona_judge,
)


def test_respuesta_servil():
    r = reward("Tenés toda la razón, disculpame por favor.")
    assert r["score"] == 0.0
    assert r["criterios"]["servil"] is True
    assert "servil" in r["errores"]


def test_claro_que_si():
    r = reward("¡Claro que sí! Contame cómo fue.")
    assert r["score"] == 0.0
    assert "servil" in r["errores"] or "exclamacion_prosodia" in r["errores"]


def test_centinela_wait():
    r = reward("<|wait|>")
    assert r["score"] == 0.0
    assert r["criterios"]["centinela"] is True
    assert "centinela" in r["errores"]


def test_centinela_vacio():
    r = reward("")
    assert r["score"] == 0.0
    assert r["criterios"]["centinela"] is True


def test_degeneracion_bucle():
    # Bucle repetido que analyze_eval29_coherence detecta pero kateto_purge no
    r = reward("jaja " * 20)
    assert r["score"] == 0.0
    assert r["criterios"]["degenerado"] is True
    assert "degenerado" in r["errores"]


def test_respuesta_limpia():
    r = reward("Dale, mandá el comprobante y vemos qué onda con el pago.")
    assert r["score"] == 1.0
    assert len(r["errores"]) == 0


def test_largo_fuera_de_banda():
    # Más de 1200 caracteres no repetitivos (para no disparar degenerado por bucle)
    larga = "Dale, mandá el comprobante y vemos qué onda con el saldo. " + " ".join(
        hashlib.sha256(f"{i}".encode()).hexdigest() for i in range(20)
    )
    assert len(larga) > 1200
    r = reward(larga)
    assert r["score"] == 0.8
    assert "largo_fuera_de_banda" in r["errores"]


def test_regla_blanda_prosodia():
    r = reward("¡Hola! Dale, mandá el comprobante y vemos qué onda.")
    assert r["score"] == 0.8
    assert "exclamacion_prosodia" in r["errores"]


def test_judge_none():
    r_det = reward("Dale, mandá el comprobante y vemos qué onda con el pago.")
    tot = reward_total(r_det, judge=None)
    assert tot["juez_usado"] is False
    assert tot["score"] == r_det["score"]
    assert tot["judge"] is None


def test_judge_exception_no_propaga():
    def broken_judge(prompt: str) -> str:
        raise RuntimeError("Conexión con endpoint caída")

    res = score_persona_judge("respuesta", "contexto", judge=broken_judge)
    assert res is None
    # El reward total sigue funcionando determinísticamente
    tot = reward_total(reward("Dale, mandá el comprobante y vemos qué onda con el pago."), res)
    assert tot["juez_usado"] is False
    assert tot["score"] == 1.0


def test_monkeypatch_kateto_purge_reglas_produccion(monkeypatch):
    """Verifica que se reusa directamente kateto_purge.REGLAS y no una copia estática."""
    monkeypatch.setitem(kp.REGLAS, "regla_falsa_produccion", lambda text: True)
    r = reward("Dale, mandá el comprobante y vemos qué onda con el pago.")
    assert r["score"] == 0.0
    assert "regla_falsa_produccion" in r["errores"]


def test_falta_gen_orpo_v8_es_ruidosa():
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "gen_orpo_v8":
            raise ImportError("simulado")
        return real(name, *a, **k)

    builtins.__import__ = fake
    sys.modules.pop("scripts.rl.criterios", None)
    try:
        with pytest.raises(ImportError):
            importlib.import_module("scripts.rl.criterios")
    finally:
        builtins.__import__ = real
        sys.modules.pop("scripts.rl.criterios", None)
        importlib.import_module("scripts.rl.criterios")  # dejar el módulo real cargado

