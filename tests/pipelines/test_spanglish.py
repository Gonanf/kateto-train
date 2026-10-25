#!/usr/bin/env python3
"""R16: el spanglish no va, excepto en los términos técnicos que se dicen en ingles.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest tests/pipelines -q

La whitelist vive en el modulo de la regla (`scripts/gen_toolcalling_dataset.py`)
y el gate de la purga entra por ahi: estos testsayan por `spanglish_en`, que es la
misma funcion que llama `validate`.
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parents[2] / "scripts" / "pipelines"
REPO = Path(__file__).resolve().parents[2]
for _p in (PIPELINES, PIPELINES.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402
import gen_toolcalling_dataset as ref  # noqa: E402

VOZ = "seco"
ESCENARIO = "solo_charla"


def errores_spanglish(turno_voz: str) -> list[str]:
    """Lo que la regla rechaza en un turno de voz."""
    return ref.spanglish_en(turno_voz)


def texto(turno_voz: str) -> str:
    """El dato armado, para pasar por el gate de verdad (`kateto_gen.validar`)."""
    return g.ensamblar([("user", "che seco, contame de la maquina que armoceayer"),
                        (VOZ, turno_voz),
                        ("user", "y de la GPU, eso si me interesa"),
                        (VOZ, "va, te lo resumo en dos lineas")], VOZ)


def spanglish_del_gate(turno_voz: str) -> list[str]:
    errs = g.validar(texto(turno_voz), ESCENARIO, VOZ)
    return [e for e in errs if e.startswith("spanglish(")]


# --------------------------------------------------------------------------- #
# los prestamos tecnicos pasan
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("termino", ["RAM", "GPU", "git", "github", "huggingface",
                                     "commit", "token", "prompt"])
def test_terminos_tecnicos_no_se_rechazan(termino: str) -> None:
    assert errores_spanglish(f"la {termino} del server, decime") == []


def test_whitelist_pasa_de_auno_con_ram() -> None:
    for t in ref.TERMINOS_TECNICOS:
        assert errores_spanglish(f"esto es un {t} de verdad") == [], t


def test_solo_una_fuente_de_verdad() -> None:
    """La lista del spanglish esta en UN solo modulo. Dos copias ya nos mordieron."""
    copias = []
    for py in sorted((PIPELINES.parent).rglob("*.py")):
        if re_compila(py.read_text(encoding="utf-8"), "RX_SPANGLISH"):
            copias.append(str(py.relative_to(REPO)))
    assert copias == ["scripts/gen_toolcalling_dataset.py"], copias


def re_compila(fuente: str, nombre: str) -> bool:
    return any(line.startswith(f"{nombre} = re.compile") for line in fuente.splitlines())


# --------------------------------------------------------------------------- #
# lo que el dueno NO autorizo como prestamo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("palabra", ["worries", "okay", "worry", "okey"])
def test_worries_y_okay_siguen_rechazados(palabra: str) -> None:
    """No son terminos tecnicos: tienen traduccion de uso ('preocupacion', 'dale')."""
    assert palabra in errores_spanglish(f"te lo digo {palabra}, sin mas")


@pytest.mark.parametrize("palabra", ["worries", "okay", "worry", "different",
                                     "differente"])
def test_palabras_sin_traduccion_no_estan_en_la_whitelist(palabra: str) -> None:
    assert palabra not in ref.TERMINOS_TECNICOS


def test_differente_no_se_rechaza_por_spanglish() -> None:
    """No es ingles: es una variante del castellano. Salió de la lista en R12."""
    assert errores_spanglish("un toque differente, como te lo dije") == []


# --------------------------------------------------------------------------- #
# el match es por palabra completa
# --------------------------------------------------------------------------- #
def test_programa_no_se_rechaza_por_ram() -> None:
    assert errores_spanglish("el programa ese no anda, fijate") == []


def test_chatear_no_se_rechaza_por_chat() -> None:
    """`chatear` es castellano, no spanglish."""
    assert errores_spanglish("estuve chatear con vos toda la tarde") == []


@pytest.mark.parametrize("variante", ["Gpu", "GITHUB", "ram", "RaM", "GitHub"])
def test_la_comparacion_es_case_insensitive(variante: str) -> None:
    assert errores_spanglish(f"miro el {variante} del servidor") == []


@pytest.mark.parametrize("termino", ref.TERMINOS_TECNICOS)
def test_ningun_termino_matchea_dentro_de_otra_palabra(termino: str) -> None:
    """`ram` no puede ser prefijo ni sufijo de otra palabra: `fullmatch`, no `search`
    (en "server-sidexyz" el `server` de la lista SI aparece, y esta bien)."""
    assert ref.RX_TECNICOS.fullmatch(termino + "xyz") is None, termino
    assert ref.RX_TECNICOS.fullmatch("z" + termino) is None, termino


@pytest.mark.parametrize("termino", ["scripts", "fine-tuning", "tokens", "commits",
                                     "chat de voz", "server-side", "fine tuning"])
def test_plural_y_guion(termino: str) -> None:
    assert errores_spanglish(f"con {termino} nous va") == []


# --------------------------------------------------------------------------- #
# la normalizacion: mismo criterio NFC que el validador de tuteo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("turno", ["la RAM de 32 GB", "te lo digo okay"])
def test_nfc_no_cambia_el_resultado(turno: str) -> None:
    assert errores_spanglish(unicodedata.normalize("NFD", turno)) == errores_spanglish(turno)


# --------------------------------------------------------------------------- #
# por el gate de verdad (purga + generador), no solo por la regla suelta
# --------------------------------------------------------------------------- #
def test_el_gate_deja_pasar_la_ram() -> None:
    assert spanglish_del_gate("la RAM anda al 90, el commit ya esta pusheado") == []


def test_el_gate_rechaza_el_okay() -> None:
    assert spanglish_del_gate("okay, despues vemos") == ["spanglish(okay)"]