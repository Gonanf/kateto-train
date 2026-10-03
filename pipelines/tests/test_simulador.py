#!/usr/bin/env python3
"""R9: el simulador de usuario que copia a la voz se rechaza y se reintenta.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest pipelines/tests -q

Teacher falso, sin red: devuelve como turno de usuario el turno de voz anterior.
Es el defecto MEDIDO en `smoke/teacher-pool` (R5): la conversacion no avanza, se
copia. El loop tiene que rechazarlo, reintentar, y la muestra no puede quedar con
el eco adentro.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent.parent
for _p in (PIPELINES, PIPELINES.parent / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402

VOZ = "seco"
RX_VOS = re.compile(r"^VOS: (.+)$", re.M)

# Los turnos REALES de `smoke/teacher-pool/muestras.jsonl` (R5). El detector se
# calibra contra esto, asi que el test falla si alguien cambia el umbral sin
# recalibrarlo con los numeros del bloque de `es_eco`.
ECO_CALCO = "Che, pero no te creas que andás tan solo con el cuento, que acá el stream está más quieto"
ECO_PARAFRASE = (
    "Che, arrancá contame qué te pasó que andás tan seco, que acá el stream está más quieto"
)
VOZ_DE_ECO = (
    "Che, pero no te creas que andás tan solo con el cuento, que acá el stream está más quieto"
)
CHATLA_REAL = (
    "Che, pero si el circo tiene platea llena, ¿no es por algo? Que la gente lo elige, boludo."
)

SC = {"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla de stream", "voseo": True}


def teacher_eco(*, sanar: bool) -> tuple:
    """Teacher que devuelve el turno de voz anterior como turno del USUARIO.

    El defecto de R5 es exactamente ese: el turno que pedimos nosotros sale
    calcado del turno de la voz. La voz, en cambio, contesta siempre con texto
    propio, asi que el eco no viene por ahi.

    El rol se lee del prompt ("le toca a la PERSONA" vs "le toca a la voz
    `seco`"): el teacher decide a quien esta respondiendo, no por numero de
    llamada.

    `sanar=True`: en el reintento devuelve algo que NO es eco, para probar el
    camino de "rechazo y sale bien". El reintento llega con el MISMO prompt (el
    transcript no cambio, el eco se rechazo antes de anexarlo), asi que el
    teacher se da cuenta por el prompt repetido.
    `sanar=False`: devuelve eco siempre, para probar el corte por reintentos
    agotados.
    """
    llamadas: list[str] = []
    ya_respondido: set[str] = set()
    voces_dadas = 0

    def teacher(prompt: str) -> str:
        nonlocal voces_dadas
        llamadas.append(prompt)
        if "le toca a la voz" in prompt:
            voces_dadas += 1
            return f"Voz {voces_dadas}: el stream arranco con medio dia de gente y ya se fue"
        voces = RX_VOS.findall(prompt)
        if not voces:
            return "che, arranquemos que el stream esta muerto"
        if sanar and prompt in ya_respondido:
            return "bueno, metele otra vez que no me entrainaste"
        ya_respondido.add(prompt)
        return voces[-1]

    return teacher, llamadas


def pares_usuario_voz(txt: str) -> list[tuple[str, str]]:
    """`[(turno_de_usuario, turno_de_voz_anterior)]` en orden de conversacion."""
    pares: list[tuple[str, str]] = []
    voz = ""
    for cab, cuerpo in g.desarmar(txt):
        if cab == g.IM_USER:
            pares.append((cuerpo, voz))
        else:
            voz = cuerpo
    return pares


def turnos_de_usuario(txt: str) -> list[str]:
    return [cuerpo for cab, cuerpo in g.desarmar(txt) if cab == g.IM_USER]


# --------------------------------------------------------------------------- #
# el detector, contra los turnos reales de R5
# --------------------------------------------------------------------------- #
def test_calco_literal_es_eco() -> None:
    assert g.es_eco(ECO_CALCO, VOZ_DE_ECO) is not None


def test_parafraseo_es_eco_aunque_no_comparta_el_arranque() -> None:
    """El eco de R5 que cambia el arranque y conserva la cola: lo caza el 3-grama."""
    voz = "Che, arrancá con el cuento que andás tan seco, que acá el stream está más quieto"
    motivo = g.es_eco(ECO_PARAFRASE, voz)
    assert motivo == "trigramas", f"el parafraseo tiene que entrar por trigramas, dio {motivo!r}"


def test_charla_real_no_es_eco() -> None:
    """La voz de R5 que SI debate. Si esto cae, el umbral esta apretado de mas."""
    voz = (
        "Che, River es un circo, pero el circo tiene platea llena y golpea fuerte, eso no se nega. "
        "El problema es que los hinchas ya no creen ni en los jugadores ni en los dirigentes."
    )
    assert g.es_eco(CHATLA_REAL, voz) is None


def test_sin_voz_previa_no_hay_eco() -> None:
    """Primer turno de la conversacion: no hay con que comparar."""
    assert g.es_eco(ECO_CALCO, "") is None


# --------------------------------------------------------------------------- #
# el prompt del simulador prohibe el eco
# --------------------------------------------------------------------------- #
def test_prompt_del_usuario_prohibe_copiar() -> None:
    prompt = g.build_user_prompt(SC, VOZ, [("user", "che"), (VOZ, "dale")], 3)
    assert g.REGLA_ANTIECO in prompt
    assert "PROHIBIDO reusar sus palabras" in prompt


def test_el_registro_no_va_al_prompt_del_usuario() -> None:
    """El registro es la ficha de Kateto; al simulador lo hace copy-pastear."""
    assert "Sos Kateto" not in g.build_user_prompt(SC, VOZ, [], 3)


# --------------------------------------------------------------------------- #
# el loop: rechazo, reintento, corte
# --------------------------------------------------------------------------- #
def test_el_loop_rechaza_el_eco_y_reintenta() -> None:
    teacher, _ = teacher_eco(sanar=True)
    stats: dict = {}
    txt, _ = g.generar_muestra(SC, VOZ, teacher, n_ex=2, stats=stats)

    assert stats["ecos"] > 0, "el eco tiene que quedar contado"
    assert stats["motivos_eco"], "y con motivo: trigramas o arranque"
    assert stats["corte_eco"] is False
    for u, v in pares_usuario_voz(txt):
        assert g.es_eco(u, v) is None, f"la muestra quedo con el eco adentro: {u!r} vs {v!r}"


def test_eco_siempre_corta_la_conversacion_y_no_guarda_el_eco() -> None:
    teacher, _ = teacher_eco(sanar=False)
    stats: dict = {}
    txt, _ = g.generar_muestra(SC, VOZ, teacher, n_ex=2, max_ecos=1, stats=stats)

    assert stats["corte_eco"] is True
    assert stats["ecos"] == 2, f"max_ecos=1 son 2 intentos: {[1]}"
    # se corta en el ultimo turno de voz, no con un eco al final
    assert txt.endswith(g.IM_END + "\n")
    usuarios = turnos_de_usuario(txt)
    assert len(usuarios) == 1, f"el corte tiene que dejar 1 turno de usuario: {usuarios}"


def test_max_ecos_cero_no_reintenta() -> None:
    teacher, _ = teacher_eco(sanar=False)
    stats: dict = {}
    _, _ = g.generar_muestra(SC, VOZ, teacher, n_ex=2, max_ecos=0, stats=stats)
    assert stats["ecos"] == 1, "sin reintentos, un solo intento por turno de usuario"


def test_sin_eco_el_contador_queda_en_cero() -> None:
    turnos = [
        "che, arranco el stream y no vino nadie",
        "che yo entre tarde, que se me perdio",
        "bueno, arranco yo entonces",
        "dale, contame",
    ]
    it = iter(turnos)
    stats: dict = {}
    txt, _ = g.generar_muestra(SC, VOZ, lambda p: next(it), n_ex=2, stats=stats)

    assert stats["ecos"] == 0
    assert stats["corte_eco"] is False
    assert txt.count(g.MARCADOR_VOZ.format(voz=VOZ)) == 2