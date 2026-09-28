"""Tests de teachers del generador de dataset (parseo y defaults, sin red)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_toolcalling_dataset import DEFAULT_MODEL, build_parser


def test_teachers_aceptados_y_default():
    p = build_parser()
    assert p.parse_args([]).teacher == "opencode"          # default
    assert p.parse_args(["--teacher", "opencode"]).teacher == "opencode"
    assert p.parse_args(["--teacher", "cline"]).teacher == "cline"


def test_teacher_cmd_rechazado(capsys):
    p = build_parser()
    with pytest.raises(SystemExit) as e:
        p.parse_args(["--teacher", "cmd"])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_modelos_default():
    assert DEFAULT_MODEL["opencode"] == "opencode/muse-spark-1.3-contributor-free"
    assert DEFAULT_MODEL["cline"] == "meta/muse-spark-1.3-contributor-free"