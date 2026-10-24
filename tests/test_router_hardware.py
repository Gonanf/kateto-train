#!/usr/bin/env python3
"""
Tests para router_hardware.py
pytest, stdlib: route en 4 casos de política, determinismo, agregar() con percentiles exactos,
guarda que falla con n_tokens=0, y --mock end-to-end sobre 3 prompts con n=1.
Estilo del repo: docstring arriba, PROJECT al sys.path (mirá tests/test_state_bytes.py).
"""

import sys
import json
import tempfile
from pathlib import Path

# Agregar PROJECT al sys.path como en test_state_bytes.py
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.router_hardware import route, percentile, agregar, mock_stream_response, parse_sse_line, extract_first_content_chunk


def test_route_corto():
    """Prompt corto -> barato."""
    tier, motivo = route("hola", 120)
    assert tier == "barato"
    assert motivo == "corto"


def test_route_largo():
    """Prompt > umbral_chars -> caro."""
    tier, motivo = route("x" * 200, 120)
    assert tier == "caro"
    assert motivo == "largo"


def test_route_marcador_razonamiento():
    """Marcador de razonamiento en prompt corto -> caro."""
    tier, motivo = route("che, explicame por qué el cielo es azul", 120)
    assert tier == "caro"
    assert motivo == "marcador"


def test_route_determinismo():
    """route es determinista: misma entrada -> misma salida."""
    for _ in range(20):
        t1, m1 = route("test determinismo por qué", 120)
        t2, m2 = route("test determinismo por qué", 120)
        assert t1 == t2
        assert m1 == m2


def test_percentiles_exactos():
    """percentile con interpolación lineal (numpy-style)."""
    # p50 de [10,20,30,40,50] = 30 (índice 2)
    assert percentile([10, 20, 30, 40, 50], 50) == 30
    # p50 de [10,20,30,40] = 25 (interpolado entre 20 y 30)
    assert percentile([10, 20, 30, 40], 50) == 25
    # p95 de 5 elementos
    assert percentile([10, 20, 30, 40, 50], 95) == 48.0
    # un solo elemento
    assert percentile([42], 50) == 42
    assert percentile([42], 95) == 42
    # lista vacía
    assert percentile([], 50) == 0.0


def test_agregar_guardas_ok():
    """agregar() con n_tokens > 0 en todas las filas -> guardas.ok=True."""
    filas = [
        {"prompt_chars": 10, "tier": "barato", "motivo": "corto", "t_decision_ms": 1, "ttft_ms": 10, "total_ms": 20, "n_tokens": 5, "error": None},
        {"prompt_chars": 200, "tier": "caro", "motivo": "largo", "t_decision_ms": 2, "ttft_ms": 50, "total_ms": 100, "n_tokens": 8, "error": None},
    ]
    res = agregar(filas)
    assert res["guardas"]["tokens_mayor_cero"] is True
    assert res["guardas"]["ttft_menor_igual_total"] is True
    assert res["guardas"]["decision_menor_5ms"] is True  # p50 de [1,2] = 1.5 < 5
    assert res["guardas"]["ok"] is True


def test_agregar_guardas_falla_n_tokens_cero():
    """agregar() con una fila n_tokens=0 -> tokens_mayor_cero=False, ok=False."""
    filas = [
        {"prompt_chars": 10, "tier": "barato", "motivo": "corto", "t_decision_ms": 1, "ttft_ms": 10, "total_ms": 20, "n_tokens": 0, "error": None},
    ]
    res = agregar(filas)
    assert res["guardas"]["tokens_mayor_cero"] is False
    assert res["guardas"]["ok"] is False


def test_agregar_percentiles_por_tier():
    """agregar() calcula percentiles correctos por tier."""
    filas = [
        {"prompt_chars": 10, "tier": "barato", "motivo": "corto", "t_decision_ms": 1, "ttft_ms": 10, "total_ms": 20, "n_tokens": 5, "error": None},
        {"prompt_chars": 10, "tier": "barato", "motivo": "corto", "t_decision_ms": 2, "ttft_ms": 20, "total_ms": 30, "n_tokens": 6, "error": None},
        {"prompt_chars": 10, "tier": "barato", "motivo": "corto", "t_decision_ms": 3, "ttft_ms": 30, "total_ms": 40, "n_tokens": 7, "error": None},
        {"prompt_chars": 10, "tier": "barato", "motivo": "corto", "t_decision_ms": 4, "ttft_ms": 40, "total_ms": 50, "n_tokens": 8, "error": None},
        {"prompt_chars": 10, "tier": "barato", "motivo": "corto", "t_decision_ms": 5, "ttft_ms": 50, "total_ms": 60, "n_tokens": 9, "error": None},
    ]
    res = agregar(filas)
    barato = res["por_tier"]["barato"]
    assert barato["n"] == 5
    assert barato["ttft_p50_ms"] == 30  # mediana de [10,20,30,40,50]
    assert barato["ttft_p95_ms"] == 48  # p95
    assert barato["decision_p50_ms"] == 3  # mediana de [1,2,3,4,5]
    assert barato["total_p50_ms"] == 40
    assert barato["tokens_p50"] == 7


def test_mock_stream_response():
    """mock_stream_response genera 8 chunks con content en algunos."""
    chunks = mock_stream_response()
    assert len(chunks) == 8
    # primer chunk: rol, sin content
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assert not chunks[0]["choices"][0]["delta"].get("content")
    # chunks 1-5: content
    for i in range(1, 6):
        assert f"token{i} " in chunks[i]["choices"][0]["delta"]["content"]
    # últimos: finish_reason
    assert chunks[7]["choices"][0].get("finish_reason") == "stop"


def test_parse_sse_line():
    """parse_sse_line parsea data: {...} y ignora [DONE] y vacías."""
    assert parse_sse_line("data: {\"a\":1}") == {"a": 1}
    assert parse_sse_line("data: [DONE]") is None
    assert parse_sse_line("") is None
    assert parse_sse_line("  ") is None
    assert parse_sse_line("data: invalid json") is None


def test_extract_first_content_chunk():
    """extract_first_content_chunk ignora rol vacío y encuentra primer content."""
    chunks = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": ""}}]},
        {"choices": [{"delta": {"content": "hola"}}]},
        {"choices": [{"delta": {"content": " mundo"}}]},
    ]
    assert extract_first_content_chunk(chunks) == 2

    # sin content
    chunks_sin = [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {}}]},
    ]
    assert extract_first_content_chunk(chunks_sin) == -1


def test_mock_bench_end_to_end():
    """--mock end-to-end sobre 3 prompts con n=1 -> JSON tiene filas y guardas.ok=True."""
    import subprocess
    import tempfile

    config = {
        "tiers": {
            "barato": {"base_url": "http://127.0.0.1:1", "model": "test-barato", "ctx": 4096, "costo_rel": 1.0},
            "caro": {"base_url": "http://127.0.0.1:2", "model": "test-caro", "ctx": 4096, "costo_rel": 8.0},
        },
        "prompts": ["hola", "x" * 200, "che, explicame por qué"],
        "umbral_chars": 120,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        out_json = Path(tmpdir) / "banco.json"
        out_md = Path(tmpdir) / "banco.md"
        config_path.write_text(json.dumps(config))

        result = subprocess.run([
            sys.executable, str(PROJECT / "scripts/router_hardware.py"),
            "--bench", "--config", str(config_path),
            "--out", str(out_json), "--md", str(out_md),
            "--n", "1", "--mock"
        ], capture_output=True, text=True, cwd=PROJECT)

        assert result.returncode == 0, f"mock bench falló: {result.stderr}"

        data = json.loads(out_json.read_text())
        assert data["meta"]["mock"] is True
        assert data["global"]["n"] == 3
        assert len(data["filas"]) == 3
        assert data["guardas"]["ok"] is True

        # Verificar tier correcto por prompt
        tiers_esperados = ["barato", "caro", "caro"]
        for fila, esperado in zip(data["filas"], tiers_esperados):
            assert fila["tier"] == esperado

        # Verificar MD existe
        assert out_md.exists()
        md_content = out_md.read_text()
        assert "# Banco de ruteo — mock" in md_content
        assert "ok: True" in md_content


if __name__ == "__main__":
    # Permitir ejecución directa
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))