#!/usr/bin/env python
"""fix121: escenarios de bit (video/evento), regla sin_anclaje_en_payload y
regla FORMA DEL BIT en los dos prompts. Sin red, sin tocar data/."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "gen_tc", REPO / "scripts" / "gen_toolcalling_dataset.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _sample(voz: str) -> str:
    return (f"<|im_user|>che contame que dijo el flaco ese del mate, quiero saber si el "
            f"problema es de la yerbera o del termo<|im_end|>\n"
            f"<|im_start|>seco\n{voz}<|im_end|>")


# --------------------------------------------------------------------------
# 1. los 2 escenarios existen, con ancla interpolada en el txt
# --------------------------------------------------------------------------
def _sc(id_: str) -> dict:
    for sc in gen.SCENARIOS:
        if sc["id"] == id_:
            return sc
    raise AssertionError(f"falta escenario {id_}")


def test_bit_sobre_video_existe():
    sc = _sc("bit_sobre_video")
    assert sc["cat"] == "A" and sc["tools"] == 2
    assert sc["ancla_payload"].strip()
    # el ancla aparece DENTRO del txt (interpolado), no como placeholder
    assert sc["ancla_payload"] in sc["txt"]
    assert "<ancla_payload>" not in sc["txt"]


def test_bit_sobre_evento_existe():
    sc = _sc("bit_sobre_evento")
    assert sc["cat"] == "C" and sc["contraparte"] == "evento" and sc["tools"] == 0
    assert sc["ancla_payload"] in sc["txt"]
    assert "<ancla_payload>" not in sc["txt"]


def test_otros_escenarios_sin_ancla():
    with_ancla = [sc["id"] for sc in gen.SCENARIOS if sc.get("ancla_payload")]
    assert sorted(with_ancla) == ["bit_sobre_evento", "bit_sobre_video"]


# --------------------------------------------------------------------------
# 2. regla sin_anclaje_en_payload
# --------------------------------------------------------------------------
ANCLA = "bloque de juego de 20:00 a 22:00, despues corte para cenar"


def test_anclaje_fallido():
    voz = ("Miralo, el mundo no para. Un puente de fideos mojados se banca menos que "
           "esta idea. Anotalo.")
    errs = gen.validate(_sample(voz), ancla_payload=ANCLA)
    assert any(e.startswith("sin_anclaje_en_payload") for e in errs), errs


def test_anclaje_exitoso():
    # comparte 2+ palabras de contenido del ancla (bloque, juego, cenar...)
    voz = ("Claro, dos horas de juego y despues corte para cenar, justo. Como una "
           "cañería que corta el agua justo cuando te metes en la ducha. Anotalo.")
    errs = gen.validate(_sample(voz), ancla_payload=ANCLA)
    assert not any(e.startswith("sin_anclaje_en_payload") for e in errs), errs


def test_anclaje_en_algun_turno_alcanza():
    voz1 = "Nada que ver con el bloque de juego, esto es otra cosa."
    voz2 = "Y aparte un puente de fideos mojados. Anotalo."
    sample = (f"<|im_user|>che que onda el bloque de juego de hoy, arranca a las 20:00 "
              f"o a las 22:00?<|im_end|>\n"
              f"<|im_start|>seco\n{voz1}<|im_end|>\n"
              f"<|im_user|>posta, decime de una vez<|im_end|>\n"
              f"<|im_start|>seco\n{voz2}<|im_end|>")
    errs = gen.validate(sample, ancla_payload=ANCLA)
    assert not any(e.startswith("sin_anclaje_en_payload") for e in errs), errs


def test_sin_ancla_la_regla_no_corre():
    voz = "Miralo, un puente de fideos mojados se banca menos que esta idea. Anotalo."
    sample = _sample(voz)
    for kwargs in ({}, {"ancla_payload": None}, {"ancla_payload": ""}):
        errs = gen.validate(sample, **kwargs)
        assert not any("ancla" in e for e in errs), (kwargs, errs)


def test_error_tiene_el_contador():
    voz = "Un puente de fideos mojados, anotalo."
    errs = gen.validate(_sample(voz), ancla_payload=ANCLA)
    e = next(x for x in errs if x.startswith("sin_anclaje_en_payload"))
    assert e in ("sin_anclaje_en_payload(0/2)", "sin_anclaje_en_payload(1/2)")


# --------------------------------------------------------------------------
# 3. FORMA DEL BIT en los dos prompts
# --------------------------------------------------------------------------
def test_forma_del_bit_en_los_dos_prompts():
    assert "FORMA DEL BIT" in gen.build_prompt(_sc("bit_sobre_video"), "", "seco")
    assert "FORMA DEL BIT" in gen.build_pair_prompt(_sc("bit_sobre_evento"), "seco", [], 1)
