"""Tests de fix120: tool NO siempre obligatoria (por escenario), escenarios de
silencio y reglas de contenido. Sin red."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_toolcalling_dataset as gen  # noqa: E402


SC_TOOLS = {"id": "video_rag_chain", "cat": "A", "tools": 2, "txt": "algo de un video"}


def _muestra_sin_tool() -> str:
    return ("<|im_user|>che seco, contame como viene el stream hoy<|im_end|>\n"
            "<|im_start|>seco\n"
            "Todo re bien pibe, posta que el laburo hoy fue un quilombo, che.<|im_end|>\n"
            "<|im_user|>dale, y que mas<|im_end|>\n"
            "<|im_start|>seco\n"
            "Posta, fijate que te deja mil cosas, che.<|im_end|>")


def _muestra_con_tool() -> str:
    return ("<|im_user|>che, que tenes de videos?<|im_end|>\n"
            "<|im_start|>seco\n"
            "Dale, dejame ver.\n"
            "tool_call list_videos: {}\n"
            "tool_result list_videos: [\"a.mp4\"]\n"
            "Tenemos ese, che.<|im_end|>")


def _muestra_3_turnos(tercero: str) -> str:
    return (f"<|im_user|>hola<|im_end|>\n"
            f"<|im_start|>seco\nChe, que onda?<|im_end|>\n"
            f"<|im_user|>bien<|im_end|>\n"
            f"<|im_start|>seco\nY vos?<|im_end|>\n"
            f"<|im_user|>todo bien<|im_end|>\n"
            f"<|im_start|>seco\n{tercero}<|im_end|>")


# ------------------------------------------------- tool_obligatoria por escenario

def test_tool_obligatoria_false_habilita_sin_tool():
    p = gen.build_pair_prompt(SC_TOOLS, "seco", [], 3, "  - list_videos()",
                              permite_tools=True, tool_obligatoria=False)
    assert "escribi solo tu mensaje" in p
    assert "EXIGE TOOL" not in p
    assert "CONTESTA DIRECTO" in p


def test_tool_obligatoria_true_exige_tool():
    p = gen.build_pair_prompt(SC_TOOLS, "seco", [], 3, "  - list_videos()",
                              permite_tools=True, tool_obligatoria=True)
    assert "EXIGE TOOL" in p
    assert "escribi solo tu mensaje" not in p


# ----------------------------------------------------- regla prohibir_tool

def test_prohibir_tool_con_llamada_da_error():
    errs = gen.validate(_muestra_con_tool(), voice="seco", prohibir_tool=True)
    assert any(e.startswith("tool_no_permitida(list_videos)")
               for e in errs)


def test_prohibir_tool_sin_llamada_no_da_error():
    errs = gen.validate(_muestra_sin_tool(), voice="seco", prohibir_tool=True)
    assert not any(e.startswith("tool_no_permitida") for e in errs)


# ---------------------------------------------- regla pregunta_en_todos_los_turnos

def test_pregunta_en_todas_las_voces_da_error():
    errs = gen.validate(_muestra_3_turnos("Seguro?"), voice="seco",
                        require_voseo=False)
    assert "pregunta_en_todos_los_turnos" in errs


def test_dos_de_tres_no_da_error():
    errs = gen.validate(_muestra_3_turnos("Bueno, dale de una."), voice="seco",
                        require_voseo=False)
    assert "pregunta_en_todos_los_turnos" not in errs


def test_dos_turnos_pregunta_no_da_error():
    s = _muestra_3_turnos("Seguro?")
    # recortar a 2 turnos de voz: dos preguntas seguidas es charla normal
    s = s.rsplit("<|im_user|>todo bien<|im_end|>", 1)[0]
    assert s.count("<|im_start|>seco") == 2
    errs = gen.validate(s, voice="seco", require_voseo=False)
    assert "pregunta_en_todos_los_turnos" not in errs


# ------------------------------------ escenarios nuevos con las claves exactas

def _por_id(esc_id: str) -> dict:
    for sc in gen.SCENARIOS:
        if sc["id"] == esc_id:
            return sc
    raise AssertionError(f"escenario {esc_id} no existe")


def test_escenario_pregunta_trivial_sin_tool():
    sc = _por_id("pregunta_trivial_sin_tool")
    assert sc == {
        "id": "pregunta_trivial_sin_tool", "cat": "B", "tools": 2,
        "tool_obligatoria": False, "prohibir_tool": True,
        "txt": "El usuario hace una pregunta corta y trivial que se contesta de una: una cuenta simple "
               "('cuanto es 2+2', 'que es el 20% de 300'), el significado de una palabra o un dato basico. "
               "seco responde DIRECTO, con su voz y su opinion, SIN llamar ninguna tool. Si llama una tool, "
               "la muestra se descarta.",
    }


def test_escenario_no_llenar_silencio():
    sc = _por_id("no_llenar_silencio")
    assert sc == {
        "id": "no_llenar_silencio", "cat": "F", "tools": 0,
        "min_turns": 1, "voseo": False, "n_ex": 1, "min_palabras_usuario": 1,
        "forzar_respuesta": ["<|no_response|>"],
        "txt": "El usuario emite un reconocimiento corto que NO pide nada ('aja', 'claro', 'mm', 'si', 'ok'): "
               "sigue mirando el video o esta pensando. seco NO acusa recibo, NO pregunta de nuevo y NO "
               "inventa tema: se queda callado. La respuesta es exactamente `<|no_response|>` y nada mas.",
    }


def test_escenario_silencio_charla_cerrada():
    sc = _por_id("silencio_charla_cerrada")
    assert sc == {
        "id": "silencio_charla_cerrada", "cat": "F", "tools": 0,
        "min_turns": 1, "voseo": True, "n_ex": 2,
        "forzar_respuesta": [None, "<|no_response|>"],
        "txt": "La conversacion YA se cerro: el usuario se despidio ('buenas noches', 'me voy a dormir', "
               "'despues te cuento') y seco le contesto la despedida. En el intercambio siguiente el usuario "
               "emite un ruido de fondo sin dirigirse a seco ('mmm', 'uy', un suspiro). seco NO vuelve a "
               "arrancar la charla: la respuesta es exactamente `<|no_response|>`.",
    }


# --------------------------------------- regla de silencio en los dos prompts

R_SILENCIO = "es exactamente `<|no_response|>`"


def test_prompt_pair_tiene_regla_de_silencio():
    p = gen.build_pair_prompt(SC_TOOLS, "seco", [], 3, "",
                              permite_tools=True, tool_obligatoria=True)
    assert R_SILENCIO in p
    assert "NO cierres todos los turnos con una pregunta" in p


def test_prompt_normal_tiene_regla_de_silencio():
    p = gen.build_prompt(SC_TOOLS, "", "seco", 3)
    assert R_SILENCIO in p
    assert "NO cierres todos los turnos con una pregunta" in p