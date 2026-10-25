#!/usr/bin/env python
"""fix122: endeudamiento de la regla de anclaje.

Cuenta SOLO la prosa de la voz (no el material que llega por tool_result),
umbral por escenario (ancla_min) y diagnóstico con el máximo real (1, no 0).
Sin red, sin tocar data/."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "gen_tc", REPO / "scripts" / "gen_toolcalling_dataset.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

ANCLA = "bloque de juego de 20:00 a 22:00, despues corte para cenar"


def _sample(voz: str) -> str:
    return (f"<|im_user|>che contame que dijo el flaco ese del mate<|im_end|>\n"
            f"<|im_start|>seco\n{voz}<|im_end|>")


def _err(voz: str, **kw) -> str | None:
    errs = gen.validate(_sample(voz), ancla_payload=ANCLA, **kw)
    return next((e for e in errs if e.startswith("sin_anclaje_en_payload")), None)


# --- bug 1: el tool_result trae el ancla pero la prosa NO la toca -------------
# Hoy (regla vieja) la regla contaba el tool_result => no podia fallar. Con la
# regla que cuenta solo la prosa TIENE que dar sin_anclaje_en_payload.
def test_ancla_solo_en_tool_result_falla():
    voz = ("tool_call summarize: {\"video\": \"mate\"}\n"
           "tool_result summarize: bloque de juego de 20:00 a 22:00, despues corte para cenar\n"
           "Miralo, un puente de fideos mojados se banca menos que esta idea. Anotalo.")
    assert _err(voz) is not None


# --- 1 palabra del ancla en la prosa -----------------------------------------
def test_una_palabra_ancla_con_ancla_min_2_falla():
    voz = "dos horas de juego y se acabo. Anotalo."
    assert _err(voz, ancla_min=2) is not None


def test_una_palabra_ancla_con_ancla_min_1_pasa():
    voz = "dos horas de juego y se acabo. Anotalo."
    assert _err(voz, ancla_min=1) is None


# --- el mensaje reporta el número real (1, no 0) -------------------------------
def test_mensaje_reporta_contador_real():
    voz1 = "dos horas de juego y se acabo. Anotalo."
    assert _err(voz1, ancla_min=2) == "sin_anclaje_en_payload(1/2)"

    voz0 = "un puente de fideos mojados. Anotalo."
    assert _err(voz0, ancla_min=2) == "sin_anclaje_en_payload(0/2)"


# --- umbral por escenario ------------------------------------------------------
def _sc(id_: str) -> dict:
    for sc in gen.SCENARIOS:
        if sc["id"] == id_:
            return sc
    raise AssertionError(f"falta escenario {id_}")


def test_evento_ancla_min_1():
    assert _sc("bit_sobre_evento")["ancla_min"] == 1


def test_video_sin_clave_ancla_min():
    assert "ancla_min" not in _sc("bit_sobre_video")