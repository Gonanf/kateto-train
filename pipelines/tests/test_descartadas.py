#!/usr/bin/env python3
"""R19: una muestra descartada deja rastro. `intentos == a + r + d`.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest pipelines/tests -q

El bug que cierra esto: el bucle de reintentos tiraba la muestra y nadie anotaba
por que, asi que la corrida cerraba con `aceptadas: 0` y `rechazadas: 0` A LA
VEZ. Eso no distingue "el modelo no sirve" de "el pipeline se la comio", y el
conteo de la corrida no cierra.

Teacher stub, sin red: la suite no llama a freellmapi. Los tres defectos que
matan muestras — eco, turno vacio/corto, excepcion — se reproducen con stubs y
tienen que aparecer con nombre propio en `descartadas.jsonl`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parent.parent
for _p in (PIPELINES, PIPELINES.parent / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402

VOZ = "seco"
RX_VOS = re.compile(r"^VOS: (.+)$", re.M)
RX_TRASCRIPTO = re.compile(r"Lo que se dijo hasta ahora:\n(.*?)\n\n", re.S)

SC = {"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla de stream", "voseo": True}


# --------------------------------------------------------------------------- #
# stubs: los tres defectos, sin red
# --------------------------------------------------------------------------- #
def teacher_que_repite(prompt: str) -> str:
    """El stub del brief: SIEMPRE devuelve el turno anterior, del rol que sea.

    El rol se lee del prompt ("le toca a la voz" esta solo en el prompt de la
    voz) y lo anterior sale del transcript, que es lo unico que ve el teacher.
    En el primer intercambio no hay turno anterior, asi que arranca con algo
    propio y desde ahi el calco es automatico.
    """
    if "le toca a la voz" not in prompt:
        return "Che, arranco el stream y todavia no entro nadie"
    tras = RX_TRASCRIPTO.search(prompt)
    lineas = [l for l in (tras.group(1).splitlines() if tras else []) if not l.startswith("VOS: ")]
    return lineas[-1].split(": ", 1)[-1] if lineas else "la voz arranca sola"


def teacher_vacio(prompt: str) -> str:
    """No devuelve texto."""
    return ""


def teacher_corto(prompt: str) -> str:
    """Devuelve texto, pero mas corto que `MIN_LINEA_VOZ`."""
    return "ok"


def teacher_que_revienta(prompt: str) -> str:
    raise RuntimeError("el teacher cayo")


def teacher_bueno(prompts: list[str]) -> str:
    """Teacher sano: intercala usuario y voz, sin eco. Para el lado sano del
    invariante."""
    return f"turno {len(prompts) + 1} del teacher, con texto propio y holgado"


# --------------------------------------------------------------------------- #
# el archivo de descartadas: motivo, etapa y detalle, con nombre propio
# --------------------------------------------------------------------------- #
def fila(teacher, *, n_ex: int = 2, intento: int = 0, **kw) -> dict:
    """Una fila como la que arma `GenerarMuestras`: el loop mas el gate."""
    sc = dict(SC)
    st: dict = {}
    txt, _ = g.generar_muestra(sc, VOZ, teacher, n_ex, stats=st, intento=intento, **kw)
    return {
        "text": txt,
        "scenario": sc["id"],
        "cat": sc["cat"],
        "voice": VOZ,
        "fallo": "voz_eco" if st["descarte_voz_eco"] else ("" if txt else "no_se_pudo_armar"),
        "ecos": st["ecos"],
        "motivos_eco": st["motivos_eco"],
        "voz_eco": st["voz_eco"],
        "motivos_eco_voz": st["motivos_eco_voz"],
        "descartes": st["descartes"],
    }


def por_sink(filas: list[dict], tmp_path: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    """Corre las filas por el gate y el sink de verdad. Devuelve
    `(stats, dataset, rechazos, descartadas)`."""
    gate = g.AnotarValidacion(name="anotar_validacion")
    out = tmp_path / "ds.jsonl"
    rej = tmp_path / "ds_rejects.jsonl"
    desc = tmp_path / "ds_descartadas.jsonl"
    sink = g.EscribirDataset(name="escribir_dataset", out_path=str(out),
                             rejects_path=str(rej), descarta_path=str(desc))
    # `process` es un generador: sin consumirlo no escribe nada. Y el sink pide
    # un logger que distilabel le inyecta al correr dentro de una `Pipeline`.
    sink._logger = logging.getLogger("test_descartadas")
    for paso in gate.process(filas):
        for _ in sink.process(paso):
            pass
    sink.unload()
    leer = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]  # noqa: E731
    return json.loads(Path(str(out).replace(".jsonl", ".stats.json")).read_text()), \
        leer(out), leer(rej), leer(desc)


# --------------------------------------------------------------------------- #
# 1. el eco que descarta: la voz calcando al usuario
# --------------------------------------------------------------------------- #
def test_el_eco_que_descarta_queda_registrado(tmp_path: Path) -> None:
    """El stub del brief: el teacher repite el turno anterior.

    Del lado del usuario el eco CORTA la conversacion y la muestra se guarda
    (R9, `corte_eco`); del lado de la voz DESCARTA la muestra entera. La que
    tiene que aparecer en `descartadas.jsonl` es `eco_voz`.
    """
    stats, _, _, descartadas = por_sink([fila(teacher_que_repite)], tmp_path)

    assert len(descartadas) == 1, f"una muestra descartada deja una linea: {descartadas}"
    d = descartadas[0]
    assert d["motivo"] == "eco_voz", f"el motivo tiene nombre propio: {d}"
    assert d["etapa"]["rol"] == "voz"
    assert d["etapa"]["turno"] == 1, "murio en el primer intercambio de voz"
    assert d["etapa"]["intento"] == 0
    assert "calco" in d["detalle"], f"el detalle dice que paso: {d}"
    assert stats["descartadas"] == 1
    assert stats["motivos_descarte"] == {"eco_voz": 1}


def test_el_eco_del_usuario_no_es_un_descarte(tmp_path: Path) -> None:
    """El otro lado del `eco_*`: el usuario calcando a la voz CORTA, no descarta.

    Por eso `eco_usuario` no aparece nunca en `descartadas.jsonl`, y la
    invariante sigue cerrando: la muestra sale con texto y la cuenta el gate.
    """
    ya_respondido: set[str] = set()

    def solo_usuario(prompt: str) -> str:
        if "le toca a la voz" in prompt:
            n = len(RX_VOS.findall(prompt))
            return f"Voz {n}: el stream arranco con medio dia de gente y al final se fueron"
        voces = RX_VOS.findall(prompt)
        if prompt in ya_respondido:
            return "bueno, metele otra vez que no me entrainaste"
        ya_respondido.add(prompt)
        return voces[-1] if voces else "che arranquemos que el stream esta muerto"

    stats, _, _, descartadas = por_sink([fila(solo_usuario)], tmp_path)

    assert descartadas == [], "el corte del usuario guarda la muestra: no es un descarte"
    assert stats["descartadas"] == 0
    assert stats["intentos"] == stats["aceptadas"] + stats["rechazadas"] + stats["descartadas"]
    assert stats["motivos_descarte"] == {}


# --------------------------------------------------------------------------- #
# 2. turno vacio / corto
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "teacher,motivo",
    [(teacher_vacio, "turno_vacio"), (teacher_corto, "turnos_insuficientes")],
)
def test_turno_inutilizable_queda_registrado(teacher, motivo: str, tmp_path: Path) -> None:
    stats, _, _, descartadas = por_sink([fila(teacher)], tmp_path)

    assert len(descartadas) == 1, f"no se puede perder en silencio: {descartadas}"
    assert descartadas[0]["motivo"] == motivo
    assert descartadas[0]["etapa"]["rol"] == "usuario", "murio en el primer rol que se pide"
    assert stats["motivos_descarte"] == {motivo: 1}


def test_turno_vacio_detalle_no_inventa_nada(tmp_path: Path) -> None:
    _, _, _, descartadas = por_sink([fila(teacher_vacio)], tmp_path)
    assert descartadas[0]["detalle"] == "el teacher no devolvio texto"


# --------------------------------------------------------------------------- #
# 3. la excepcion del teacher, con su texto real
# --------------------------------------------------------------------------- #
def test_excepcion_del_teacher_queda_registrada(tmp_path: Path) -> None:
    stats, _, _, descartadas = por_sink([fila(teacher_que_revienta)], tmp_path)

    assert len(descartadas) == 1, f"una excepcion no puede comerse el lote: {descartadas}"
    d = descartadas[0]
    assert d["motivo"] == "excepcion_teacher"
    assert d["detalle"] == "RuntimeError: el teacher cayo", f"el texto real: {d}"
    assert stats["motivos_descarte"] == {"excepcion_teacher": 1}


def test_la_excepcion_no_se_reintenta(tmp_path: Path) -> None:
    """Una excepcion no es un turno invalido: se anota y se corta, igual que
    antes de R19 (donde la excepcion se escapaba del loop)."""
    llamadas: list[str] = []

    def cuenta(prompt: str) -> str:
        llamadas.append(prompt)
        raise RuntimeError("el teacher cayo")

    fila(cuenta)
    assert len(llamadas) == 1, f"sin reintentos: {len(llamadas)} llamadas al teacher"


def test_una_muestra_sin_texto_siempre_tiene_descarte(tmp_path: Path) -> None:
    """El canario: si alguna vez una muestra sin texto no tiene motivo, el bug
    esta en otro lado y el conteo cerraria sin querer."""
    stats, _, _, descartadas = por_sink([fila(teacher_vacio)], tmp_path)
    assert stats["descartadas"] == 1 and len(descartadas) == 1

    stats2, _, _, d2 = por_sink(
        [dict(fila(teacher_bueno), text="", descartes=[])], tmp_path)
    assert stats2["motivos_descarte"] == {g.MOTIVO_SIN_MOTIVO: 1}, "el canario suena"
    assert d2[0]["motivo"] == g.MOTIVO_SIN_MOTIVO


# --------------------------------------------------------------------------- #
# 4. la invariante del brief, con los tres casos
# --------------------------------------------------------------------------- #
def test_el_conteo_de_la_corrida_cierra(tmp_path: Path) -> None:
    """`intentos == aceptadas + rechazadas + descartadas`, con los tres casos
    de descarte y uno sano. Las cuatro filas pasan por el gate y el sink reales.
    """
    filas = [
        fila(teacher_que_repite, intento=0),   # eco_voz
        fila(teacher_vacio, intento=1),         # turno_vacio
        fila(teacher_corto, intento=2),         # turnos_insuficientes
        fila(teacher_que_revienta, intento=3),  # excepcion_teacher
    ]
    stats, dataset, rechazos, descartadas = por_sink(filas, tmp_path)

    assert stats["intentos"] == len(filas) == 4
    assert stats["aceptadas"] == len(dataset)
    assert stats["rechazadas"] == len(rechazos)
    assert stats["descartadas"] == len(descartadas)
    assert stats["intentos"] == stats["aceptadas"] + stats["rechazadas"] + stats["descartadas"]
    assert stats["motivos_descarte"] == {
        "eco_voz": 1, "turno_vacio": 1, "turnos_insuficientes": 1, "excepcion_teacher": 1,
    }
    # el `intento` de la corrida ordena el archivo
    assert [d["etapa"]["intento"] for d in descartadas] == [0, 1, 2, 3]
    assert all(d["scenario"] == SC["id"] for d in descartadas)


def test_el_lado_sano_cierra_tambien(tmp_path: Path) -> None:
    """Sin descartes, `intentos == aceptadas + rechazadas` como antes de R19."""
    turnos = [
        "Che, arranco el stream y no vino nadie",
        "Che, yo entre tarde, que se me perdio",
        "Bueno, arranco yo entonces",
        "Dale, contame",
    ]
    it = iter(turnos)
    stats, dataset, rechazos, descartadas = por_sink([fila(lambda p: next(it))], tmp_path)

    assert stats["intentos"] == 1
    assert stats["descartadas"] == 0 and descartadas == []
    assert stats["intentos"] == stats["aceptadas"] + stats["rechazadas"]
    assert len(dataset) + len(rechazos) == 1


def test_una_muestra_que_no_se_descarte_no_toca_descartadas(tmp_path: Path) -> None:
    """Una muestra que se arma y ademas el gate la rechaza va a `rechazos`, no a
    `descartadas`: los tres baldes son disjuntos."""
    turnos = iter(["Che, arranco el stream y no vino nadie",
                   "tienes",  # 6 chars: la voz no llega al minimo -> descarte, no rechazo
                   ])
    stats, dataset, rechazos, descartadas = por_sink([fila(lambda p: next(turnos))], tmp_path)

    assert stats["descartadas"] == 1
    assert stats["rechazadas"] == 0, "sin texto no hay nada que rechazar"
    assert rechazos == [] and dataset == [] and len(descartadas) == 1


def test_la_etapa_numera_las_muestras_de_la_corrida(monkeypatch) -> None:
    """El Step le pasa su indice de fila a `generar_muestra`: sin esto el archivo
    de descartadas tiene todos `intento: 0` y no se puede ordenar ni cruzar con
    la corrida."""
    monkeypatch.setattr(g, "TeacherHTTP", lambda *a, **k: teacher_que_revienta)
    step = g.GenerarMuestras(name="generar_turnwise", max_ecos=1)
    step._logger = logging.getLogger("test_descartadas")
    filas = [r for out in step.process([{"escenario": dict(SC)} for _ in range(3)]) for r in out]

    assert [r["descartes"][0]["etapa"]["intento"] for r in filas] == [0, 1, 2]
    assert all(r["descartes"][0]["motivo"] == "excepcion_teacher" for r in filas)
    assert all(r["descartes"][0]["detalle"] == "RuntimeError: el teacher cayo" for r in filas)