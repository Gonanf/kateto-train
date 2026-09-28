"""Tests de parseo paired/turnwise y validacion de turno de usuario (sin red).

Caso raiz: el plugin de opencode escribe 'KATETOOOOO' por stdout ANTES del texto
del modelo; el parseo anclado a la primera linea convertia ese banner en el
turno de usuario del dataset (medido: 371/371 turnos 'KATETOOOOO').
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_toolcalling_dataset import (
    build_pair_prompt,
    clean_teacher_output,
    generate_paired,
    record_rejection,
    validate,
)

# Salida REAL medida del harness (primeras 3 lineas) + respuesta del modelo.
SALIDA_CON_BANNER = (
    "KATETOOOOO\n"
    "\n"
    "> Sisyphus - ultraworker · muse-spark-1.3-contributor-free\n"
    "\n"
    "PERSONA: che boludo, mira el quilombo que se armo\n"
    "###\n"
    "VOS: uh pibe, dejame ver que hay\n"
)

SALIDA_LIMPIA_TOOLS = (
    "PERSONA: che seco, acordate del video ese largo que vimos ayer\n"
    "###\n"
    "VOS: Dale, dejame ver que tengo indexado.\n"
    "tool_call list_videos: {}\n"
    'tool_result list_videos: ["charla-rust-2026.mp4"]\n'
    "Esta el de la charla de rust. Que querias saber?\n"
)


def test_banner_no_es_turno_de_usuario():
    out = clean_teacher_output(SALIDA_CON_BANNER)
    assert out.startswith("PERSONA:")
    assert "KATETOOOOO" not in out
    assert "Sisyphus" not in out


def test_paired_parsea_desde_el_contrato():
    sc = {"id": "x", "txt": "charla trivial en el stream.", "tools": 0}
    sample = generate_paired(sc, "seco", lambda p: SALIDA_CON_BANNER, n_ex=1)
    assert sample != ""
    assert "<|im_user|>che boludo, mira el quilombo que se armo<|im_end|>" in sample
    assert "KATETOOOOO" not in sample


def test_paired_limpio_sin_regresion():
    sc = {"id": "x", "txt": "video rag con tools.", "tools": 2}
    sample = generate_paired(sc, "seco", lambda p: SALIDA_LIMPIA_TOOLS, n_ex=1)
    esperado = (
        "<|im_user|>che seco, acordate del video ese largo que vimos ayer<|im_end|>\n"
        "<|im_start|>seco\n"
        "Dale, dejame ver que tengo indexado.\n"
        "tool_call list_videos: {}\n"
        'tool_result list_videos: ["charla-rust-2026.mp4"]\n'
        "Esta el de la charla de rust. Que querias saber?<|im_end|>"
    )
    assert sample == esperado
    assert validate(sample, require_tool=True, min_turns=1) == []


def test_voz_multilinea_con_tools_no_se_corta():
    sc = {"id": "x", "txt": "video rag con tools.", "tools": 2}
    sample = generate_paired(sc, "seco", lambda p: SALIDA_LIMPIA_TOOLS, n_ex=1)
    # el bloque de la voz conserva TODAS sus lineas (frase + tools + cierre)
    for frag in ("Dale, dejame ver que tengo indexado.",
                 "tool_call list_videos: {}",
                 'tool_result list_videos: ["charla-rust-2026.mp4"]',
                 "Esta el de la charla de rust. Que querias saber?"):
        assert frag in sample


def test_validador_usuario_degenerado():
    malo = ("<|im_user|>KATETOOOOO<|im_end|>\n"
            "<|im_start|>seco\n"
            "fijate el quilombo que hay en el medio, si me muevo ahora me matan<|im_end|>")
    errs = validate(malo)
    assert any(e.startswith("usuario_degenerado:KATETOOOOO") for e in errs), errs

    bueno = ("<|im_user|>che seco, acordate del video ese largo que vimos ayer<|im_end|>\n"
             "<|im_start|>seco\n"
             "Dale, dejame ver que tengo indexado.<|im_end|>")
    assert "usuario_degenerado" not in " ".join(validate(bueno))


# --------------------------------------------------------------------------- #
# REPORT6: `min_palabras_usuario` por escenario + usuario_repetido
# --------------------------------------------------------------------------- #

def _muestra_con_usuario(user: str, n: int = 1) -> str:
    resp = "fijate el quilombo que hay en el medio, si me muevo ahora me matan"
    partes = []
    for _ in range(n):
        partes.append(f"<|im_user|>{user}<|im_end|>")
        partes.append(f"<|im_start|>seco\n{resp}<|im_end|>")
    return "\n".join(partes)


def test_min_palabras_usuario_1_acepta_corto_completo():
    errs = validate(_muestra_con_usuario("dale"), min_turns=1,
                    min_palabras_usuario=1)
    assert "usuario_degenerado" not in " ".join(errs), errs


def test_min_palabras_usuario_default_3_rechaza_corto():
    errs = validate(_muestra_con_usuario("dale"), min_turns=1)
    assert any(e.startswith("usuario_degenerado:dale") for e in errs), errs


def test_patron_repetido_se_rechaza_con_min_1():
    errs = validate(_muestra_con_usuario("KATETOOOOO"), min_turns=1,
                    min_palabras_usuario=1)
    assert any(e.startswith("usuario_degenerado:KATETOOOOO") for e in errs), errs


def test_ok_se_acepta_con_min_1():
    errs = validate(_muestra_con_usuario("ok"), min_turns=1,
                    min_palabras_usuario=1)
    assert "usuario_degenerado" not in " ".join(errs), errs


def test_usuario_repetido_en_muestra_con_todos_iguales():
    errs = validate(_muestra_con_usuario("dale", n=3), min_turns=1,
                    min_palabras_usuario=1)
    assert any(e.startswith("usuario_repetido:dale") for e in errs), errs
    # y no es falso positivo si los turnos son distintos:
    distintos = ("<|im_user|>dale<|im_end|>\n<|im_start|>seco\n"
                 "fijate el quilombo que hay en el medio, si me muevo ahora me matan<|im_end|>\n"
                 "<|im_user|>no, quedate<|im_end|>\n<|im_start|>seco\n"
                 "fijate el quilombo que hay en el medio, si me muevo ahora me matan<|im_end|>")
    errs2 = validate(distintos, min_turns=1, min_palabras_usuario=1)
    assert "usuario_repetido" not in " ".join(errs2), errs2


def test_escenario_no_wait_declara_min_palabras_usuario_1():
    import gen_toolcalling_dataset as g
    sc = next(s for s in g.SCENARIOS if s["id"] == "no_wait_cuando_completo")
    assert sc.get("min_palabras_usuario") == 1, sc


# --------------------------------------------------------------------------- #
# Punto 5: Tests del contrato por escenario, chess y trazabilidad de rechazos
# --------------------------------------------------------------------------- #

def test_escenario_contraparte_agente_no_pide_persona_y_parsea():
    sc = {
        "id": "chess_uci",
        "cat": "E",
        "tools": 0,
        "min_turns": 2,
        "n_ex": 1,
        "contraparte": "agente",
        "txt": "Prompt de ajedrez con agente"
    }
    prompt = build_pair_prompt(sc, "seco", [], n_ex=1)
    assert "PERSONA:" not in prompt
    assert "AGENTE:" in prompt

    teacher_resp = (
        "AGENTE: [CHESS GameMode] fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
        "legal_moves=[e2e4, d2d4] request_id=1 e2e4\n"
        "###\n"
        "VOS: e7e5\n"
    )
    sample = generate_paired(sc, "seco", lambda p: teacher_resp, n_ex=1)
    assert sample != ""
    assert "<|im_user|>[CHESS GameMode]" in sample
    assert "<|im_start|>seco\ne7e5<|im_end|>" in sample


def test_escenario_contraparte_persona_sigue_pidiendo_persona():
    sc = {
        "id": "solo_charla",
        "cat": "B",
        "tools": 0,
        "contraparte": "persona",
        "txt": "Charla pura de stream"
    }
    prompt = build_pair_prompt(sc, "seco", [], n_ex=1)
    assert "PERSONA:" in prompt
    assert "AGENTE:" not in prompt

    teacher_resp = (
        "PERSONA: che seco cómo andas hoy?\n"
        "###\n"
        "VOS: todo piola che, acá stremeando\n"
    )
    sample = generate_paired(sc, "seco", lambda p: teacher_resp, n_ex=1)
    assert sample != ""
    assert "<|im_user|>che seco cómo andas hoy?<|im_end|>" in sample
    assert "<|im_start|>seco\ntodo piola che, acá stremeando<|im_end|>" in sample


def test_chess_uci_n_ex_2_y_validacion_uci_legal():
    import gen_toolcalling_dataset as g
    sc_chess = next(s for s in g.SCENARIOS if s["id"] == "chess_uci")
    assert sc_chess["n_ex"] == 2
    assert sc_chess.get("contraparte") == "agente"

    # Muestra legal (ambos turnos con UCI de legal_moves)
    sample_legal = (
        "<|im_user|>[CHESS GameMode] fen=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 "
        "legal_moves=[e2e4, d2d4, g1f3] request_id=1<|im_end|>\n"
        "<|im_start|>seco\ne2e4<|im_end|>\n"
        "<|im_user|>El UCI pedido es ilegal y seco corrige con uno legal. "
        "legal_moves=[c2c4, g1f3]<|im_end|>\n"
        "<|im_start|>seco\nc2c4<|im_end|>"
    )
    errs_ok = validate(sample_legal, voice="seco", min_turns=2, require_voseo=False, is_chess=True)
    assert not any(e.startswith("uci_ilegal") or e.startswith("sin_uci") for e in errs_ok), errs_ok

    # Muestra con UCI ilegal en turno 1
    sample_ilegal = (
        "<|im_user|>[CHESS GameMode] fen=... legal_moves=[e2e4, d2d4] request_id=1<|im_end|>\n"
        "<|im_start|>seco\nh7h5<|im_end|>\n"
        "<|im_user|>Error: requested move is not in legal_moves<|im_end|>\n"
        "<|im_start|>seco\nd2d4<|im_end|>"
    )
    errs_ilegal = validate(sample_ilegal, voice="seco", min_turns=2, require_voseo=False, is_chess=True)
    assert any(e == "uci_ilegal:h7h5" for e in errs_ilegal), errs_ilegal

    # Muestra sin UCI en seco
    sample_sin_uci = (
        "<|im_user|>[CHESS GameMode] fen=... legal_moves=[e2e4, d2d4] request_id=1<|im_end|>\n"
        "<|im_start|>seco\nque quilombo che no se que jugar<|im_end|>\n"
        "<|im_user|>Error: requested move is not in legal_moves<|im_end|>\n"
        "<|im_start|>seco\ne2e4<|im_end|>"
    )
    errs_sin_uci = validate(sample_sin_uci, voice="seco", min_turns=2, require_voseo=False, is_chess=True)
    assert any(e == "sin_uci:seco" for e in errs_sin_uci), errs_sin_uci


def test_muestra_no_armada_queda_en_rechazos_con_exc(tmp_path):
    import json

    rej_path = tmp_path / "rejects.jsonl"
    with rej_path.open("w", encoding="utf-8") as fr:
        record_rejection(
            fr,
            scenario_id="chess_uci",
            errors=["exc_RuntimeError"],
            raw="output crudo del teacher que no pudo armar",
            text=""
        )

    lines = rej_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    registro = json.loads(lines[0])
    assert registro["scenario"] == "chess_uci"
    assert any(e.startswith("exc_") for e in registro["errors"])
    assert registro["raw"] == "output crudo del teacher que no pudo armar"
    assert registro["text"] == ""

