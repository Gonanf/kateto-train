#!/usr/bin/env python3
"""R2: el texto armado tiene el formato del dato, y el gate cae si se rompe.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest pipelines/tests -q

Teacher stub, sin red: la suite no llama a freellmapi.

Mutacion: `GEN_MUTAR=armado` saca el cierre del ultimo bloque al importar el
modulo, y la suite tiene que CAER. Un gate que no cae con el armado roto no
sirve.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parent.parent
for _p in (PIPELINES, PIPELINES.parent / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402
import gen_toolcalling_dataset as ref  # noqa: E402

VOZ = "seco"
# El marcador REAL de un turno de voz. Contar con esto, no con heuristicas.
MARCADOR = g.MARCADOR_VOZ.format(voz=VOZ)

HIST = [("user", "che seco, del video largo de ayer"), (VOZ, "dale, vi algo")]
TXT = g.ensamblar(HIST, VOZ)


def comprobar_formato(txt: str, voice: str = VOZ) -> None:
    """La asercion de formato. La mutacion tiene que romper ESTA."""
    assert txt.startswith(g.IM_USER), "el texto no arranca en el marcador de usuario"
    assert txt.endswith(g.IM_END), "el texto no cierra en el marcador de fin"
    assert not txt.endswith("\n"), "sobro un \\n: el dato no lo lleva"
    bloques = g.desarmar(txt)
    assert len(bloques) >= 2, f"no hay bloques que re-armar: {bloques}"
    for cab, cuerpo in bloques:
        assert cab in (g.IM_USER, g.MARCADOR_VOZ.format(voz=voice)), f"cabecera rara: {cab!r}"
        assert cuerpo.strip(), "bloque vacio"


# --------------------------------------------------------------------------- #
# stubs: teacher sin red
# --------------------------------------------------------------------------- #
def teacher_stub(respuestas: list[str]) -> tuple:
    """Devuelve `(teacher, llamadas)`. El teacher reparte la lista en orden."""
    it = iter(respuestas)
    llamadas: list[str] = []

    def teacher(prompt: str) -> str:
        llamadas.append(prompt)
        return next(it)

    return teacher, llamadas


ESCENARIO_TOOL = {
    "id": "video_rag_chain",
    "cat": "A",
    "tools": 2,
    "txt": "El usuario pregunta algo que esta en un video. seco usa list_videos y answer.",
    "video_id": "charla-rust.mp4",
    "video_fragmento": "el flaco del video dice que el mate lavado es culpa de la yerbera",
}

# 2 intercambios x (usuario + tool + comentario) = 6 llamadas.
# El 3er turno de la voz mete un `tool_result` INVENTADO: tiene que desaparecer.
TURNOS_TOOL = [
    "che seco, que dijo el del video largo",
    "Dejame ver que hay indexado.\ntool_call list_videos: {}",
    "Esta el de la charla. Lo busco.",
    "y que onda con el mate",
    "Ahora si.\ntool_call answer: {\"video\": \"charla-rust.mp4\", \"query\": \"mate\"}\n"
    "tool_result answer: el contenido que el modelo se invento",
    "Es la yerbera,Clearly. Todo el mundo lo sabe.",
]

TURNOS_SENTINELA = [
    "che, al final me voy a dormir",
    "Dale, descansa.",
    "mmm",
]


# --------------------------------------------------------------------------- #
# formato del dato
# --------------------------------------------------------------------------- #
def test_marcador_real_es_el_del_dato() -> None:
    assert g.IM_USER == f"{chr(60)}|im_user|{chr(62)}"
    assert g.IM_START == f"{chr(60)}|im_start|{chr(62)}"
    assert g.IM_END == f"{chr(60)}|im_end|{chr(62)}"
    assert MARCADOR == g.IM_START + f"{VOZ}\n"
    # el mismo marcador que usa el validador de la referencia
    assert g.MARCADOR_VOZ.format(voz=VOZ) in TXT


def test_texto_armado_cierra_el_formato() -> None:
    comprobar_formato(TXT)


def test_empieza_en_marcador_de_usuario() -> None:
    assert TXT.startswith(g.IM_USER)


def test_termina_en_marcador_de_fin_sin_sobras() -> None:
    assert TXT.endswith(g.IM_END)
    assert not TXT.endswith("\n")


def test_linea_jsonl_termina_en_nueva_linea(tmp_path: Path) -> None:
    """El `endswith("\\n")` del brief es el de la LINEA del jsonl, no el del texto.

    Medido sobre ~180 shards de produccion: ningun texto termina en `\\n`; lo que
    termina en `\\n` es la linea del jsonl. Si algum dia el `text` lleva `\\n`
    final, el formateador de ahi abajo se desalinea del shard.
    """
    out = tmp_path / "d.jsonl"
    out.write_text(json.dumps({"text": TXT}, ensure_ascii=False) + "\n", encoding="utf-8")
    linea = out.read_text(encoding="utf-8")
    assert linea.endswith("\n")
    assert json.loads(linea)["text"].endswith(g.IM_END)


def test_cuenta_bloques_con_el_marcador_real() -> None:
    largo = g.ensamblar(
        [("user", "a"), (VOZ, "b"), ("user", "c"), (VOZ, "d")], VOZ
    )
    assert largo.count(MARCADOR) == 2
    assert largo.count(g.IM_USER) == 2
    assert largo.count(g.IM_END) == 4
    # el marcador cuenta lo mismo que el regex del validador de la referencia
    assert largo.count(MARCADOR) == len(ref.RX_TURN.findall(largo))


def test_rearmar_devuelve_el_texto_exacto() -> None:
    assert g.rearmar(TXT) == TXT


def test_rearmar_no_deja_sobras() -> None:
    """Todo lo que `RX_BLOQUE` NO matchea tiene que ser solo el separador `\\n`."""
    largo = g.ensamblar(
        [("user", "a"), (VOZ, "b"), ("user", "c"), (VOZ, "d")], VOZ
    )
    sobras = g.RX_BLOQUE.sub("", largo)
    assert sobras == "\n" * sobras.count("\n"), f"sobras no separadores: {sobras!r}"
    assert len(g.desarmar(largo)) == largo.count(g.IM_END)


# --------------------------------------------------------------------------- #
# el loop de cuatro roles, con teacher stub
# --------------------------------------------------------------------------- #
def test_loop_arma_la_muestra_con_estado_real() -> None:
    teacher, _ = teacher_stub(TURNOS_TOOL)
    txt, estado = g.generar_muestra(ESCENARIO_TOOL, VOZ, teacher, n_ex=2)

    comprobar_formato(txt)
    assert txt.count(MARCADOR) == 2
    # el tool_result lo puso el pipeline, desde el estado; no el teacher
    assert "tool_result list_videos: [\"charla-rust.mp4\"]" in txt
    assert "el contenido que el modelo se invento" not in txt
    assert "yerbera" in txt
    assert estado.videos == {"charla-rust.mp4": ESCENARIO_TOOL["video_fragmento"]}


def test_el_teacher_nunca_escribe_el_tool_result() -> None:
    """Si el teacher inventa el resultado, se descarta: es una alucinacion con
    formato valido."""
    teacher, _ = teacher_stub(TURNOS_TOOL)
    txt, _ = g.generar_muestra(ESCENARIO_TOOL, VOZ, teacher, n_ex=2)
    resultados = [c for t, _, c in ref.extraer_bloques(txt) if t == "res"]
    assert len(resultados) == 2
    for cuerpo in resultados:
        assert "se invento" not in cuerpo


def test_cada_tool_call_aparece_una_sola_vez() -> None:
    """Bug del smoke: la linea cruda del teacher sobrevivia y la llamada
    quedaba duplicada (una del teacher, una del pipeline)."""
    teacher, _ = teacher_stub(TURNOS_TOOL)
    txt, _ = g.generar_muestra(ESCENARIO_TOOL, VOZ, teacher, n_ex=2)
    llamadas = [c for t, _, c in ref.extraer_bloques(txt) if t == "call"]
    resultados = [c for t, _, c in ref.extraer_bloques(txt) if t == "res"]
    assert len(llamadas) == 2, llamadas
    assert len(llamadas) == len(resultados), "cada llamada del pipeline tiene su resultado"
    for cuerpo in llamadas:
        assert ", " not in cuerpo or json.loads(cuerpo), f"args no canonicos: {cuerpo!r}"


def test_el_teacher_escribe_una_llamada_repetida() -> None:
    """Aunque el teacher repita la llamada, el turno lleva una sola."""
    turnos = list(TURNOS_TOOL)
    turnos[1] = "Dejame ver.\ntool_call list_videos: {}\ntool_call list_videos: {}"
    teacher, _ = teacher_stub(turnos)
    txt, _ = g.generar_muestra(ESCENARIO_TOOL, VOZ, teacher, n_ex=1)
    assert txt.count("tool_call list_videos:") == 2
    assert txt.count("tool_result list_videos:") == 2


def test_turno_solo_tool_call_no_comenta_dos_veces() -> None:
    teacher, llamadas = teacher_stub(TURNOS_TOOL)
    g.generar_muestra(ESCENARIO_TOOL, VOZ, teacher, n_ex=2)
    assert len(llamadas) == 6, f"esperaba 6 turnos, hubo {len(llamadas)}"
    assert any("PROTOCOLO DE TOOLS" in p for p in llamadas)
    assert any("que le toca a la voz" in p for p in llamadas)


def test_centinela_lo_pone_el_codigo() -> None:
    """`forzar_respuesta` por indice de intercambio: el teacher no lo pide."""
    sc = {
        "id": "no_llenar_silencio", "cat": "F", "tools": 0, "min_turns": 1,
        "voseo": False, "min_palabras_usuario": 1,
        "forzar_respuesta": [ref.SENTINELS[0]],
        "txt": "El usuario emite un ruido. seco no acusa recibo.",
    }
    teacher, llamadas = teacher_stub(TURNOS_SENTINELA[:2])
    txt, _ = g.generar_muestra(sc, VOZ, teacher, n_ex=1)

    assert f"{MARCADOR}{ref.SENTINELS[0]}{g.IM_END}" in txt
    assert len(llamadas) == 1, "el centinela no se le pide al teacher"


def test_juez_de_corte_reintenta_un_turno_vacio() -> None:
    respuestas = ["", "", "che seco una pregunta de verdad", "dale, mira"]
    teacher, llamadas = teacher_stub(respuestas)
    sc = {"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla", "voseo": True}
    txt, _ = g.generar_muestra(sc, VOZ, teacher, n_ex=1)
    assert txt.count(MARCADOR) == 1
    # 3 llamadas para el turno de usuario (2 vacias + 1 buena) y 1 para la voz
    assert len(llamadas) == 4, f"el juez tiene que reintentar el turno vacio: {llamadas}"


def test_sin_turno_armado_devuelve_vacio() -> None:
    teacher, _ = teacher_stub(["", "", "", "", "", ""])
    sc = {"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla", "voseo": True}
    txt, estado = g.generar_muestra(sc, VOZ, teacher, n_ex=1)
    assert txt == ""


# --------------------------------------------------------------------------- #
# estado autoritativo
# --------------------------------------------------------------------------- #
def test_tool_inexistente_da_unknown_tool() -> None:
    assert g.ToolState().call("websearch", {}) == "Error: unknown tool"


def test_tool_sin_requeridos_da_missing_required() -> None:
    assert g.ToolState().call("summarize", {}) == "Error: missing required argument"


def test_video_inexistente_da_file_not_found() -> None:
    est = g.ToolState()
    assert est.call("answer", {"video": "no.mp4", "query": "x"}) == "Error: file not found"


def test_gate_minecraft_doble() -> None:
    est = g.ToolState()
    assert est.call("voyager_mine_diamond", {}) == "Error: unknown tool"
    est.minecraft = True
    assert est.call("voyager_mine_diamond", {}) == "Error: unknown tool"
    est.skill_voyager = True
    assert "OK" in est.call("voyager_mine_diamond", {})


def test_request_generation_sin_voz_viva() -> None:
    est = g.ToolState(voces_vivas=("seco",))
    assert est.call("request_generation", {"target_voice": "doktor", "prompt": "x"}) == \
        "Error: no response from target voice"


def test_send_event_sin_receptores() -> None:
    est = g.ToolState()
    est.voces_vivas = set()
    assert est.call("send_event", {"event_name": "x", "data": {}}) == \
        "Error: event has no receivers"


def test_todo_error_de_estado_es_un_error_real_de_la_referencia() -> None:
    est = g.ToolState()
    for nombre, args in [
        ("websearch", {}),
        ("summarize", {}),
        ("answer", {"video": "no.mp4", "query": "x"}),
        ("voyager_mine_diamond", {}),
        ("request_generation", {"target_voice": "doktor", "prompt": "x"}),
        ("send_event", {"event_name": "x", "data": {}}),
        ("read_file", {"path": "/nope"}),
        ("run_command", {"command": "ls"}),
    ]:
        salida = est.call(nombre, args)
        assert salida in ref.REAL_ERRORS, f"{nombre}: {salida!r} no es un error real"


# --------------------------------------------------------------------------- #
# gate: validador de la referencia + reglas de R1
# --------------------------------------------------------------------------- #
def test_validador_engancha_las_dos_familias() -> None:
    # numero de turnos bajo + exclamacion (regla de R1)
    sucio = (
        f"{g.IM_USER}hola{g.IM_END}\n"
        f"{MARCADOR}Vamos con todo che!{g.IM_END}"
    )
    errs = g.validar(sucio, "video_rag_chain", VOZ)
    assert any(e.startswith("purga:") for e in errs), errs
    assert any(e.startswith("pocos_turnos") for e in errs), errs


def test_validador_rechaza_un_escenario_desconocido() -> None:
    assert g.validar(TXT, "no_existe", VOZ) == ["escenario_desconocido:no_existe"]


def test_validador_acepta_una_muestra_del_smoke() -> None:
    """Si el smoke real dejo muestras, tienen que pasar el gate."""
    smoke = PIPELINES / "smoke" / "tc_r2.jsonl"
    if not smoke.exists():
        pytest.skip("sin smoke: la suite esta entera")
    filas = [json.loads(l) for l in smoke.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert filas, "el smoke no dejo ninguna muestra"
    for fila in filas:
        comprobar_formato(fila["text"], fila["voice"])
        assert g.validar(fila["text"], fila["scenario"], fila["voice"]) == [], fila["scenario"]


# --------------------------------------------------------------------------- #
# mutacion: el armado roto tiene que hacer caer el gate de formato
# --------------------------------------------------------------------------- #
def test_mutacion_sacar_el_cierre_rompe_el_formato() -> None:
    roto = g.ensamblar(HIST, VOZ)[: -len(g.IM_END)]
    with pytest.raises(AssertionError):
        comprobar_formato(roto)


def test_mutacion_ensamblar_roto_hace_caer_la_suite(monkeypatch) -> None:
    original = g.ensamblar

    def roto(historial, voice):
        txt = original(historial, voice)
        return txt[: -len(g.IM_END)] if txt.endswith(g.IM_END) else txt

    monkeypatch.setattr(g, "ensamblar", roto)
    roto_txt = roto(HIST, VOZ)
    with pytest.raises(AssertionError):
        comprobar_formato(roto_txt)
    with pytest.raises(AssertionError):
        assert g.rearmar(roto_txt) == roto_txt


def test_mutacion_por_variable_de_entorno() -> None:
    if os.environ.get("GEN_MUTAR") != "armado":
        pytest.skip("sin GEN_MUTAR: la suite esta entera")
    with pytest.raises(AssertionError):
        comprobar_formato(TXT)
    assert g.rearmar(TXT) != TXT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))