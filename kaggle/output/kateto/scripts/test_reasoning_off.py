"""Test de fix-118: reasoning_effort en el cliente HTTP del teacher.

Sin red: monkeypatcheamos urllib.request.urlopen y capturamos el body que
arma generate(). No llama al gateway.
"""
import json
import sys
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_toolcalling_dataset as gen


class _FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": "<|im_user|> hola<|im_end|>"}}]}
        ).encode()


@pytest.fixture()
def captured_body(monkeypatch):
    bodies = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return bodies


def test_default_reasoning_none(captured_body):
    gen.generate("http://fake", "m", "prompt", None)
    assert captured_body[0]["reasoning_effort"] == "none"


def test_reasoning_high(captured_body):
    gen.generate("http://fake", "m", "prompt", None, reasoning="high")
    assert captured_body[0]["reasoning_effort"] == "high"


def test_parser_accepts_and_rejects_reasoning():
    ap = gen.build_parser()
    args = ap.parse_args(["--reasoning", "high"])
    assert args.reasoning == "high"
    args_def = ap.parse_args([])
    assert args_def.reasoning == "none"
    with pytest.raises(SystemExit):
        ap.parse_args(["--reasoning", "ultra"])


def test_main_passes_reasoning_to_generate(monkeypatch, captured_body):
    # el call site de main() (teacher http) tiene que propagar args.reasoning
    from unittest.mock import patch

    fake_key = "freellmapi-" + "a" * 48
    monkeypatch.setattr(gen, "get_freellm_key", lambda: fake_key)
    def fake_paired(sc, voice, maestro, n_ex, tools_txt, *, tool_obligatoria=True):
        maestro("p")  # fuerza la llamada real al teacher, como haria paired
        return "<|im_user|> x<|im_end|>"

    with patch.object(gen, "SCENARIOS", [{"id": "t", "cat": "x", "prompt": "p"}]), \
         patch.object(gen, "build_prompt", return_value="p"), \
         patch.object(gen, "generate_paired", side_effect=fake_paired), \
         patch.object(gen, "validate", return_value=[]):
        rc = gen.main(["--teacher", "http", "--reasoning", "medium",
                       "--out", "out/test_fix118.jsonl",
                       "--rejects", "out/test_fix118_rej.jsonl"])
    assert rc == 0
    assert captured_body[0]["reasoning_effort"] == "medium"
