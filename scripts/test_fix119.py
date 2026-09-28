"""Tests de fix119: prompt licencia tool + regla spanglish en la voz. Sin red."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_toolcalling_dataset as gen  # noqa: E402


def _muestra(sin_tool: bool) -> str:
    v1 = ("Todo re bien pibe, posta que el laburo del stream hoy fue un quilombo, che.<|im_end|>"
          if sin_tool else
          "Dale, dejame ver que tengo indexado.\n"
          "tool_call list_videos: {}\n"
          "tool_result list_videos: [\"charla-rust-2026.mp4\"]\n"
          "Esta el de la charla de rust, que querias saber?<|im_end|>")
    v2 = "Posta, fijate que la charla esa te deja mil cosas, che.<|im_end|>"
    return (f"<|im_user|>che seco, contame como viene el stream hoy<|im_end|>\n"
            f"<|im_start|>seco\n{v1}\n"
            f"<|im_user|>dale, y que mas queda del video ese<|im_end|>\n"
            f"<|im_start|>seco\n{v2}")


SC_TOOLS = {"id": "video_rag_chain", "cat": "A", "tools": 2, "txt": "algo de un video"}


# ---------------------------------------------------------------- spanglish

def test_spanglish_en_voz_da_error():
    s = _muestra(sin_tool=True).replace("Todo re bien", "No te worries, todo bien")
    errs = gen.validate(s, voice="seco")
    assert any(e.startswith("spanglish(") for e in errs)


def test_spanglish_entre_comillas_en_voz_no_da_error():
    s = (_muestra(sin_tool=True).replace(
        "posta que el laburo",
        "'no te worries dude', posta que el laburo"))
    assert gen.validate(s, voice="seco") == []


def test_spanglish_en_turno_de_usuario_no_da_error():
    s = _muestra(sin_tool=True).replace("contame como viene",
                                        "no te worries, contame como viene")
    assert gen.validate(s, voice="seco") == []


def test_turno_limpio_y_permitir_spanglish():
    assert gen.validate(_muestra(sin_tool=True), voice="seco") == []
    s = _muestra(sin_tool=True).replace("Todo re bien", "okay, todo bien")
    assert "spanglish(okay)" in gen.validate(s, voice="seco")
    assert gen.validate(s, voice="seco", permitir_spanglish=True) == []


# ---------------------------------------------------------------- prompt

def test_prompt_con_tools_exige_tool():
    p = gen.build_pair_prompt(SC_TOOLS, "seco", [], 3, "  - list_videos()",
                              permite_tools=True, tool_obligatoria=True)
    assert "ESTE ESCENARIO EXIGE TOOL" in p
    assert "sin tool_call se" in p
    assert "escribi solo tu mensaje" not in p


def test_prompt_con_hint():
    p = gen.build_pair_prompt(SC_TOOLS, "seco", [], 3, "", permite_tools=True, hint="X")
    assert "PISTA OBLIGATORIA:\nX" in p
    assert p.rstrip().endswith("X")


def test_prompt_sin_tools_no_exige():
    p = gen.build_pair_prompt({"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla"},
                              "seco", [], 3, "", permite_tools=False)
    assert "ESTE ESCENARIO EXIGE TOOL" not in p


# ---------------------------------------------------------------- validador

def test_validate_sample_ok_sigue_pasando():
    assert gen.validate(gen.SAMPLE_OK, voice="seco") == []


# ---------------------------------------------------------------- reintento

def test_main_reintenta_sin_tool_call(tmp_path, monkeypatch):
    llamadas: list[str] = []

    def fake_paired(sc, voice, teacher, n_ex, tools_txt, *, max_retries=3, hint="",
                    tool_obligatoria=True):
        llamadas.append(hint)
        return _muestra(sin_tool=len(llamadas) == 1)

    monkeypatch.setattr(gen, "generate_paired", fake_paired)
    monkeypatch.setattr(gen, "SCENARIOS", [SC_TOOLS])
    out = tmp_path / "out.jsonl"
    rej = tmp_path / "rej.jsonl"
    rc = gen.main(["--n", "1", "--teacher", "http", "--base-url", "http://127.0.0.1:1",
                   "--key", "x", "--out", str(out), "--rejects", str(rej)])
    assert rc == 0
    assert llamadas[0] == ""
    assert "OBLIGATORIO" in llamadas[1]
    assert len(llamadas) == 2
    lineas = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lineas) == 1
    assert "tool_call" in lineas[0]
    assert rej.read_text(encoding="utf-8").strip() == ""


def test_main_no_reintenta_otros_errores(tmp_path, monkeypatch):
    llamadas: list[str] = []

    def fake_paired(sc, voice, teacher, n_ex, tools_txt, *, max_retries=3, hint="",
                    tool_obligatoria=True):
        llamadas.append(hint)
        return "<|im_user|>che<|im_end|>"  # falla por varios motivos, no solo sin_tool_call

    monkeypatch.setattr(gen, "generate_paired", fake_paired)
    monkeypatch.setattr(gen, "SCENARIOS", [SC_TOOLS])
    out = tmp_path / "out.jsonl"
    rej = tmp_path / "rej.jsonl"
    gen.main(["--n", "1", "--teacher", "http", "--base-url", "http://127.0.0.1:1",
              "--key", "x", "--out", str(out), "--rejects", str(rej)])
    assert len(llamadas) == 1
    assert rej.read_text(encoding="utf-8").strip() != ""