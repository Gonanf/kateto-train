#!/usr/bin/env python3
"""R7: el instrumento de estilo y el congelado de un solo tensor.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python -m pytest pipelines/tests -q

Los numeros del README salen de aca: si `fold`, el conteo o los trigramas se
rompen, los numeros pasan a ser mentira en silencio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parent.parent
if str(PIPELINES) not in sys.path:
    sys.path.insert(0, str(PIPELINES))
if str(PIPELINES / "estilo") not in sys.path:
    sys.path.insert(0, str(PIPELINES / "estilo"))

import metricas_estilo as me  # noqa: E402

try:
    import torch  # noqa: F401

    sin_torch = pytest.mark.skipif(False, reason="")
except ImportError:
    # El gate corre en un interprete sin torch: sin esto fallaria por interprete, no por codigo.
    sin_torch = pytest.mark.skipif(
        True, reason="el congelado del head necesita torch; este interprete no lo tiene"
    )


# --- normalizacion -----------------------------------------------------------


def test_fold_saca_tildes_y_mayusculas():
    assert me.fold("Tenés que ENTENDER") == "tenes que entender"
    assert me.fold("se le escapó   una sonrisa") == "se le escapo una sonrisa"


def test_lista_de_tells_cubre_la_especificacion():
    """10 marcadores de asistente + 13 genericos + 8 de comedia.md = 31."""
    assert len(me.todas_las_listas()) == 31
    for frase in ("en el fondo", "al fin y al cabo", "como ia", "espero que esto ayude"):
        assert me.fold(frase) in me.todas_las_listas()


# --- conteo -----------------------------------------------------------------


def test_conteo_de_cliches_ignora_tildes_y_mayusculas():
    texto = "En el fondo, TENES que entender. En el fondo."
    c = me.contar_cliches(me.fold(texto), me.todas_las_listas())
    assert c["en el fondo"] == 2
    assert c["tenes que entender"] == 1


def test_densidad_por_100_palabras():
    # "en el fondo" = 3 palabras, "sin embargo" = 2. Con 195 de relleno: 200 palabras.
    texto = " ".join(["palabra"] * 195 + ["en el fondo", "sin embargo"])
    p = me.perfil(texto, me.todas_las_listas())
    assert p["palabras"] == 200
    assert p["cliches_total"] == 2
    assert p["cliches_por_100"] == 1.0


# --- trigramas --------------------------------------------------------------


def test_trigramas_de_una_frase_corta():
    assert me.trigramas("a b c d") == {("a", "b", "c"), ("b", "c", "d")}


def test_jaccard_de_trigramas():
    a = me.perfil("a b c d", me.todas_las_listas())
    b = me.perfil("a b c d", me.todas_las_listas())
    r = me.comparar(a, b)
    assert r["jaccard_trigramas"] == 1.0
    assert r["compartidos_pct_de_b"] == 100.0


def test_sin_solapamiento():
    a = me.perfil("a b c d", me.todas_las_listas())
    b = me.perfil("w x y z", me.todas_las_listas())
    r = me.comparar(a, b)
    assert r["jaccard_trigramas"] == 0.0
    assert r["compartidos_pct_de_b"] == 0.0


# --- extraccion de la voz ---------------------------------------------------


def test_solo_voz_saca_el_turno_del_usuario():
    doc = "<|im_user|>hola usuario\nseco\nrespuesta de la voz"
    voz = me._solo_voz(doc)
    assert "respuesta de la voz" in voz
    assert "hola usuario" not in voz


def test_solo_voz_saca_el_nombre_de_la_voz():
    """El nombre ("seco") es metadata, no habla: no va al denominador."""
    doc = "<|im_user|>hola\nseco\nrespuesta"
    voz = me._solo_voz(doc)
    assert voz.strip() == "respuesta"
    assert "seco" not in voz


def test_solo_voz_con_varios_turnos():
    doc = "<|im_user|>u1\nseco\nr1\n<|im_user|>u2\nseco\nr2"
    voz = me._solo_voz(doc)
    assert "r1" in voz and "r2" in voz
    assert "u1" not in voz and "u2" not in voz


# --- congelado: el invariante central ---------------------------------------


def _cfg_minima() -> dict:
    return {
        "n_layer": 2, "n_embd": 32, "n_head": 4, "head_size": 8, "vocab": 256,
        "d_w": 16, "d_v": 8, "d_g": 32, "ffn": 128,
    }


@sin_torch
def test_congelar_deja_exactamente_un_tensor_entrenable():
    import style_tune_head as st

    cfg = _cfg_minima()
    modelo = st.SoloHead(st.pesos_mini(cfg), cfg)
    informe = st.congelar(modelo)

    con_grad = [n for n, p in modelo.named_parameters() if p.requires_grad]
    assert con_grad == ["head.weight"]
    assert informe["tensores_entrenables"] == 1
    assert informe["params_entrenables"] == cfg["vocab"] * cfg["n_embd"]
    assert not any(v.requires_grad for v in modelo.tronco.values())


@sin_torch
def test_el_trono_no_depende_del_head():
    """hidden_states no mira el head: es lo que hace seguro congelar el resto."""
    import style_tune_head as st

    cfg = _cfg_minima()
    pesos = st.pesos_mini(cfg)
    h1 = st.hidden_states(pesos, cfg, [1, 2, 3], st.estado_inicial(cfg))
    pesos["head.weight"] = pesos["head.weight"] + 999.0
    h2 = st.hidden_states(pesos, cfg, [1, 2, 3], st.estado_inicial(cfg))
    assert h1.shape == (3, cfg["n_embd"])
    assert (h1 == h2).all()