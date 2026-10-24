#!/usr/bin/env python3
"""Test deterministico del bloqueo de n-gramas (sin torch, sin GPU)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rwkv_pipeline._stop_helpers import repeticion_agotada  # noqa: E402


def test_repeticion_agotada_basic():
    """Test 1: repeticion_agotada([1, 2, 3] * 11, 3, 12) es False y repeticion_agotada([1, 2, 3] * 12, 3, 12) es True."""
    assert repeticion_agotada([1, 2, 3] * 11, 3, 12) is False, "11 repeticiones no deberian agotar con max_rep=12"
    assert repeticion_agotada([1, 2, 3] * 12, 3, 12) is True, "12 repeticiones deberian agotar con max_rep=12"


def test_repeticion_agotada_no_repeat():
    """Test 2: repeticion_agotada([1, 2, 3, 4, 5, 6], 3, 12) es False."""
    assert repeticion_agotada([1, 2, 3, 4, 5, 6], 3, 12) is False, "Secuencia sin repeticion no deberia agotar"


def test_repeticion_agotada_unigram():
    """Test 3: n=1: [7] * 11 -> False; [7] * 12 -> True."""
    assert repeticion_agotada([7] * 11, 1, 12) is False, "11 repeticiones unigram no deberian agotar"
    assert repeticion_agotada([7] * 12, 1, 12) is True, "12 repeticiones unigram deberian agotar"


def test_defaults_intact_ast():
    """Test 4: Verificar defaults por AST sin importar modulos (que arrastran torch)."""
    # infer_kateto.py -> generate(): ngram_n=3, ngram_repeat_max=12
    infer_path = ROOT / "rwkv_pipeline/infer_kateto.py"
    infer_src = infer_path.read_text(encoding="utf-8")
    infer_tree = ast.parse(infer_src)
    
    for node in ast.walk(infer_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate":
            defaults = node.args.defaults
            # ngram_n es el penultimo, ngram_repeat_max es el ultimo
            assert len(defaults) >= 2
            ngram_n_default = ast.literal_eval(defaults[-2])
            ngram_repeat_max_default = ast.literal_eval(defaults[-1])
            assert ngram_n_default == 3, f"infer_kateto.generate ngram_n default={ngram_n_default}, expected 3"
            assert ngram_repeat_max_default == 12, f"infer_kateto.generate ngram_repeat_max default={ngram_repeat_max_default}, expected 12"
            break
    else:
        raise AssertionError("No se encontro FunctionDef generate en infer_kateto.py")

    # opd_rollout.py -> rollout_from_prompt(): ngram_n=0, ngram_repeat_max=0
    opd_path = ROOT / "rwkv_pipeline/opd_rollout.py"
    opd_src = opd_path.read_text(encoding="utf-8")
    opd_tree = ast.parse(opd_src)
    
    for node in ast.walk(opd_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "rollout_from_prompt":
            defaults = node.args.defaults
            # ngram_n es el antepenultimo, ngram_repeat_max es el penultimo, ngram_max_intentos es el ultimo
            assert len(defaults) >= 3
            ngram_n_default = ast.literal_eval(defaults[-3])
            ngram_repeat_max_default = ast.literal_eval(defaults[-2])
            assert ngram_n_default == 0, f"opd_rollout.rollout_from_prompt ngram_n default={ngram_n_default}, expected 0"
            assert ngram_repeat_max_default == 0, f"opd_rollout.rollout_from_prompt ngram_repeat_max default={ngram_repeat_max_default}, expected 0"
            break
    else:
        raise AssertionError("No se encontro FunctionDef rollout_from_prompt en opd_rollout.py")


def test_ngram_block_v7_marker():
    """Test 5: NGRAM_BLOCK_V7 presente en ambos archivos."""
    opd_path = ROOT / "rwkv_pipeline/opd_rollout.py"
    stop_path = ROOT / "rwkv_pipeline/_stop_helpers.py"
    assert "NGRAM_BLOCK_V7" in opd_path.read_text(encoding="utf-8"), "Falta NGRAM_BLOCK_V7 en opd_rollout.py"
    assert "NGRAM_BLOCK_V7" in stop_path.read_text(encoding="utf-8"), "Falta NGRAM_BLOCK_V7 en _stop_helpers.py"


def test_rollout_dataclass_fields():
    """Test 6: Rollout declara ngram_bloqueos y ngram_deshacer por AST."""
    opd_path = ROOT / "rwkv_pipeline/opd_rollout.py"
    opd_src = opd_path.read_text(encoding="utf-8")
    opd_tree = ast.parse(opd_src)
    
    found_bloqueos = False
    found_deshacer = False
    
    for node in ast.walk(opd_tree):
        if isinstance(node, ast.ClassDef) and node.name == "Rollout":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id == "ngram_bloqueos":
                        found_bloqueos = True
                    elif item.target.id == "ngram_deshacer":
                        found_deshacer = True
            break
    
    assert found_bloqueos, "Rollout no declara ngram_bloqueos"
    assert found_deshacer, "Rollout no declara ngram_deshacer"


def main():
    test_repeticion_agotada_basic()
    test_repeticion_agotada_no_repeat()
    test_repeticion_agotada_unigram()
    test_defaults_intact_ast()
    test_ngram_block_v7_marker()
    test_rollout_dataclass_fields()
    print("OK ngram block")


if __name__ == "__main__":
    main()