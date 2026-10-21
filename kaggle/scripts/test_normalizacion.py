"""Tests de normalize_backticks en scripts/gen_toolcalling_dataset.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_toolcalling_dataset import SAMPLE_OK, normalize_backticks, validate  # noqa: E402

# caso inline del brief: tool_call y tool_result en medio de la prosa
SAMPLE_INLINE = (
    '<|im_user|>che seco, resumime el video del asado<|im_end|>\n'
    '<|im_start|>seco\n'
    'dale pibe, lo chusmeo un toque y te lo dejo clarito '
    'tool_call summarize: {"video": "asado.mp4"} '
    'tool_result summarize: {"summary": "asado largo"} posta que pinta quilombo'
    '<|im_end|>\n'
    '<|im_start|>seco\n'
    'Listo, ahi te lo deje resumido.\n<|im_end|>'
)

# regresion del fix anterior: backticks que envuelven tool_call/tool_result
SAMPLE_BT = (
    '<|im_user|>listame los videos<|im_end|>\n'
    '<|im_start|>seco\n'
    'Ahi voy.\n'
    '`tool_call list_videos: {}`\n'
    '`tool_result list_videos: ["a.mp4"]`\n'
    'Listo, hay dos.\n<|im_end|>\n'
    '<|im_start|>seco\n'
    'Te los dejo claritos.\n<|im_end|>'
)


def test_sample_ok_intacto():
    assert normalize_backticks(SAMPLE_OK) == SAMPLE_OK


def test_inline_pasa_validador():
    errs = validate(normalize_backticks(SAMPLE_INLINE), voice="seco")
    assert not [e for e in errs
                if e.startswith("sin_tool_call") or e.startswith("resultado_sin_llamada")], errs


def test_backticks_se_sacan():
    out = normalize_backticks(SAMPLE_BT)
    assert "`tool_call" not in out and "`tool_result" not in out
    assert "tool_call list_videos: {}" in out
    assert 'tool_result list_videos: ["a.mp4"]' in out


# --------------------------------------------------------------------------- #
# Punto 4: Normalizacion de saltos de linea dentro del JSON de tool_call
# --------------------------------------------------------------------------- #

def test_tool_call_salto_de_linea_adentro_de_string():
    sample = (
        '<|im_user|>che seco, que decia sobre los borrow checker?<|im_end|>\n'
        '<|im_start|>seco\n'
        'Dale, fijate lo que encontre.\n'
        'tool_call answer: {"video": "charla-rust-2026.mp4", "query": "borrow\nchecker", "k": 5}\n'
        'tool_result answer: postazo\n'
        'Ahi te lo dejo clarito boludo.\n<|im_end|>'
    )
    norm = normalize_backticks(sample)
    assert 'tool_call answer: {"video": "charla-rust-2026.mp4", "query": "borrow checker", "k": 5}' in norm
    errs = validate(norm, voice="seco", min_turns=1)
    assert not any(e.startswith("args_no_json") for e in errs), errs


def test_tool_call_json_valido_intacto():
    sample = (
        '<|im_user|>che seco, acordate del video ese largo<|im_end|>\n'
        '<|im_start|>seco\n'
        'Dale, ahi me fijo.\n'
        'tool_call list_videos: {}\n'
        'tool_result list_videos: ["charla-rust-2026.mp4"]\n'
        'tool_call answer: {"video": "charla-rust-2026.mp4", "query": "borrow checker", "k": 5}\n'
        'tool_result answer: postazo\n'
        'Ahi te lo dejo clarito boludo.\n<|im_end|>'
    )
    norm = normalize_backticks(sample)
    assert 'tool_call list_videos: {}' in norm
    assert 'tool_call answer: {"video": "charla-rust-2026.mp4", "query": "borrow checker", "k": 5}' in norm
    assert normalize_backticks(SAMPLE_OK) == SAMPLE_OK


def test_tool_call_json_roto_se_sigue_rechazando():
    sample = (
        '<|im_user|>che seco, tirame la data<|im_end|>\n'
        '<|im_start|>seco\n'
        'Dale boludo.\n'
        'tool_call answer: {"video": "charla.mp4", "query": }\n'
        'tool_result answer: error\n'
        'No salio che.\n<|im_end|>'
    )
    norm = normalize_backticks(sample)
    errs = validate(norm, voice="seco", min_turns=1)
    assert any(e.startswith("args_no_json:answer") for e in errs), errs

