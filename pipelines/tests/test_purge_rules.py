#!/usr/bin/env python3
"""R1: las reglas de purga descartan exactamente lo que deben.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest pipelines/tests -q

Mutacion: `PURGE_MUTAR=<regla>` invierte ese predicado al importar el modulo, y
la suite tiene que CAER. Si una suite mutada pasa, los tests son mentira.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parent.parent
if str(PIPELINES) not in sys.path:
    sys.path.insert(0, str(PIPELINES))

import kateto_purge as kp  # noqa: E402


# --- lineas reales de los shards que hoy se aceptan (no las toca ninguna regla) ---
# out/dataset/shards/parcial-w100.jsonl, voice=seco
REAL_1_TURNO = (
    '<|im_user|>Che, te puse un "dale" al stream y no pasó nada, boludo.<|im_end|>\n'
    "<|im_start|>seco\n<|no_response|><|im_end|>"
)
# mismo shard; 2 turnos de voz, un unico simil suelto -> se cuenta, no descarta
REAL_2_TURNOS = (
    "<|im_user|>Che, te puse el stream de la morocha que canta en el living, pero "
    'te dice que no hay audio. ¿Qué hago?<|im_end|>\n'
    "<|im_start|>seco\n<|wait|><|im_end|>\n"
    "<|im_user|>Che, ya lo probé tres veces y cada vez me tira \"Error 404: Audio no "
    "encontrado\", boludo. ¿Será que la morocha está muerta o solo es una pirata del "
    "streaming?<|im_end|>\n"
    "<|im_start|>seco\n(1) El stream está vivo, pero el audio lo mandó a la basura "
    "digital. (2) Es como cuando le pedís cerveza al boliche y te da agua con gas, te "
    "creés chino. (3) Dile a la morocha que se chequee los permisos del micrófono, "
    "genio.<|im_end|>"
)

LINEAS_REALES_ACEPTADAS = [REAL_1_TURNO, REAL_2_TURNOS]


# --- casos de descarte, uno por rama de cada regla ---
DESCARTE = {
    # rama 1: el placeholder literal (KATET + una o mas O)
    "placeholder_degenerado": (
        "<|im_user|>¿vos sos el robot?<|im_end|>\n"
        "<|im_start|>seco\nSoy Kateto, el seco.<|im_end|>"
    ),
    # rama 2: palabra con 3+ del mismo caracter al final. El marcador del
    # formato va pegado a proposito (`nooooo<|im_end|>`): sin el arreglo de R1b
    # ese marcador hacia que la palabra NUNCA cumpliera el patron. La
    # puntuacion suelta NO cuenta: ver test_ceiling_placeholder.
    "placeholder_degenerado_palabra": (
        "<|im_user|>¿que onda?\n"
        "seco\nte lo resumido noooooo<|im_end|>"
    ),
    "andamiaje_en_dialogo": (
        "<|im_user|>no me anda<|im_end|>\n"
        "<|im_start|>seco\nDiagnóstico: el audio no llega al canal.<|im_end|>"
    ),
    "contrato_en_dialogo": (
        "<|im_user|>no me anda<|im_end|>\n"
        "<|im_start|>seco\nVOS: responder seco y corto.<|im_end|>"
    ),
    "exclamacion_prosodia": (
        "<|im_user|>no me anda<|im_end|>\n"
        "<|im_start|>seco\n¡Vamos con todo, che!<|im_end|>"
    ),
    # 4 turnos de voz, simil en dos turnos consecutivos
    "simil_repetido_consecutivo": (
        "<|im_user|>?<|im_end|>\n"
        "<|im_start|>seco\nEs como cuando llueve.<|im_end|>\n"
        "<|im_user|>?<|im_end|>\n"
        "<|im_start|>seco\nEs como si nada.<|im_end|>\n"
        "<|im_start|>seco\nFin.<|im_end|>\n"
        "<|im_start|>seco\nListo.<|im_end|>"
    ),
    # 2 turnos de voz con 2 similes -> la rama de <= 3 turnos
    "simil_repetido_muestra_corta": (
        "<|im_user|>?<|im_end|>\n"
        "<|im_start|>seco\nEs como el pan.<|im_end|>\n"
        "<|im_user|>?<|im_end|>\n"
        "<|im_start|>seco\nEs como una piedra.<|im_end|>"
    ),
    # R8: la voz de asistente (aclaracion + autodeclaracion + rechazo). El patron
    # va sin acentos a proposito: el texto real trae `importante` con tilde y la
    # comparacion es sobre texto normalizado.
    "charla_de_asistente": (
        "<|im_user|>¿me arreglas la maquina?<|im_end|>\n"
        "<|im_start|>seco\nEs importante que lo tengas en cuenta: como asistente no "
        "puedo ayudarte con eso, lo siento, pero.<|im_end|>"
    ),
    # R8: rechazo en ingles. El apostrofo va tipografico a proposito ( asi esta en
    # el dato real): sin normalizar, `I’m sorry` no matchea `i'm sorry`.
    "charla_de_asistente_en_ingles": (
        "<|im_user|>¿me arreglas la maquina?<|im_end|>\n"
        "<|im_start|>seco\nAs an AI, I’m sorry, I cannot help with that.<|im_end|>"
    ),
}

# la clave es el nombre de la regla que tiene que disparar
CASOS = {
    "placeholder_degenerado": [
        DESCARTE["placeholder_degenerado"],
        DESCARTE["placeholder_degenerado_palabra"],
    ],
    "andamiaje_en_dialogo": [DESCARTE["andamiaje_en_dialogo"]],
    "contrato_en_dialogo": [DESCARTE["contrato_en_dialogo"]],
    "exclamacion_prosodia": [DESCARTE["exclamacion_prosodia"]],
    "simil_repetido": [
        DESCARTE["simil_repetido_consecutivo"],
        DESCARTE["simil_repetido_muestra_corta"],
    ],
    "charla_de_asistente": [
        DESCARTE["charla_de_asistente"],
        DESCARTE["charla_de_asistente_en_ingles"],
    ],
}


def comprobar(nombre: str, texto: str, debe_descartar: bool) -> None:
    """La asercion que usan todos los tests. La mutacion tiene que romperla."""
    disparos = kp.evaluar(texto)
    if debe_descartar:
        assert nombre in disparos, f"{nombre} no disparo: {disparos}"
    else:
        assert nombre not in disparos, f"{nombre} disparo sin motivo: {disparos}"


@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_descarta(nombre: str) -> None:
    for idx, texto in enumerate(CASOS[nombre]):
        assert nombre in kp.evaluar(texto), f"caso {idx} de {nombre} no disparo"


@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_no_descarta_linea_real(nombre: str) -> None:
    for idx, texto in enumerate(LINEAS_REALES_ACEPTADAS):
        assert nombre not in kp.evaluar(texto), f"linea real {idx} disparo {nombre}"


@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_lineas_reales_no_disparan_ninguna_regla(nombre: str) -> None:
    """Guardia: si las lineas de referencia estuvieran contaminadas, todos los
    'no' de arriba pasarian por el motivo equivocado."""
    for idx, texto in enumerate(LINEAS_REALES_ACEPTADAS):
        assert kp.evaluar(texto) == [], f"linea real {idx} dispara {kp.evaluar(texto)}"


@pytest.mark.parametrize("nombre", sorted(CASOS))
def test_mutacion_invierte_y_rompe_el_test(nombre: str, monkeypatch) -> None:
    """Invertir el predicado tiene que hacer caer el test de descarte."""
    original = kp.REGLAS[nombre]
    monkeypatch.setitem(kp.REGLAS, nombre, lambda t: not original(t))

    with pytest.raises(AssertionError):
        comprobar(nombre, CASOS[nombre][0], True)

    with pytest.raises(AssertionError):
        comprobar(nombre, LINEAS_REALES_ACEPTADAS[0], False)


@pytest.mark.parametrize(
    "cola,debe_descartar",
    [
        # `...`, `---` y `===` son puntuacion de la plantilla, no palabras
        # degeneradas: la rama NO las puede descartar.
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\nte lo resumido ... ", False),
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\n--- ", False),
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\n=== ", False),
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\nvalio 1000 ", False),
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\nche boludo ", False),
        # Estas SI: palabra con 3+ del mismo caracter al final, con el marcador
        # del formato pegado al token (que es justo lo que la anulaba).
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\nholaaaa<|im_end|>", True),
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\nte lo dije noooooo<|im_end|>", True),
        ("<|im_user|>/<|im_end|>\n<|im_start|>seco\nKATETOOOOO<|im_end|>", True),
    ],
)
def test_ceiling_placeholder(cola: str, debe_descartar: bool) -> None:
    """Semantica de la rama `token_repetido`, pineada a proposito.

    R1b: la rama es una PALABRA con 3+ del mismo caracter al final. Antes
    `(\\S)\\1{2,}$` con el marcador del formato pegado al token no podia verla
    nunca y solo pescaba puntuacion (`...`, `---`, `===`), llevandose miles de
    lineas limpias. Este test exige lo contrario: puntuacion no descarta, palabra
    degenerada si.
    """
    ramas = kp._clausula_placeholder(cola)
    if debe_descartar:
        assert "token_repetido" in ramas, f"no pesco la palabra degenerada: {ramas}"
    else:
        assert "token_repetido" not in ramas, f"pesco puntuacion: {ramas}"


def test_charla_solo_mira_la_voz() -> None:
    """El usuario puede decir "es importante" / "no puedo ayudarte" y no es voz de
    asistente: la regla se detiene en los turnos de la VOZ."""
    texto = (
        "<|im_user|>es importante que sepas que no puedo ayudarte con esto<|im_end|>\n"
        "<|im_start|>seco\n<|no_response|><|im_end|>"
    )
    assert kp.charla_de_asistente(texto) is False


def test_mutacion_por_variable_de_entorno() -> None:
    nombre = os.environ.get("PURGE_MUTAR")
    if nombre is None:
        pytest.skip("sin PURGE_MUTAR: la suite esta entera")
    assert kp.REGLAS[nombre](CASOS[nombre][0]) is False, (
        f"con PURGE_MUTAR={nombre} la regla deberia estar invertida, y el caso "
        "de descarte deberia haber dejado de descartar"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
