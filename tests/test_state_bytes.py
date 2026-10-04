#!/usr/bin/env python3
"""B2: el state buffer de llama.cpp se puede construir desde Python.

    venv/bin/python -m pytest tests/test_state_bytes.py -q

El round-trip contra el dump real (21627676 bytes) necesita que exista. El script
`script/probar-estado-python.sh` lo genera; si no esta, ese test se saltea y queda
el resto, que es sintetico y no depende de llama.cpp ni de la GPU.
"""
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rwkv_pipeline.state_bytes import IO_MAGIC, build_state_bytes, parse_state_bytes

# Hiperparametros reales de out/kateto-rwkv29-Q4_K_M.gguf, leidos de su metadata
# GGUF (general.architecture = rwkv7):
#   rwkv7.block_count            = 32    -> n_layer
#   rwkv7.embedding_length       = 2560  -> n_embd
#   rwkv7.wkv.head_size          = 64    -> wkv_head_size
# y token_shift_count = 2, el default de llama_hparams (llama-hparams.h:143).
N_LAYER = 32
N_EMBD = 2560
WKV_HEAD_SIZE = 64
TOKEN_SHIFT_COUNT = 2
N_EMBD_R = TOKEN_SHIFT_COUNT * N_EMBD        # 5120  (llama-hparams.cpp:208-211)
N_EMBD_S = N_EMBD * WKV_HEAD_SIZE             # 163840 (llama-hparams.cpp:236-239)
ROW_R = N_EMBD_R * 4                          # 20480 bytes, ggml_row_size(F32, n_embd_r)
ROW_S = N_EMBD_S * 4                          # 655360 bytes
# 8 (magic+seq) + 4 (cell_count) + 8 (celda) + 8 (s_trans+n_layer)
#   + 32*(4+8) + 32*ROW_R + 32*(4+8) + 32*ROW_S
SIZE_ESPERADO = 8 + 4 + 8 + 8 + 32 * 12 + 32 * ROW_R + 32 * 12 + 32 * ROW_S

DUMP = Path(os.environ.get("KATETO_STATE_DUMP", "/tmp/estado-orig.bin"))


def _dato_real() -> bytes:
    if not DUMP.exists():
        pytest.skip(
            f"no hay dump de estado en {DUMP}; generalo con "
            "`bash script/probar-estado-python.sh` o ponelo en $KATETO_STATE_DUMP"
        )
    return DUMP.read_bytes()


def _sintetico(capas: int = 3, celdas: int = 2, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "io_magic": IO_MAGIC,
        "seq_id": 3,
        "cell_count": celdas,
        "cells": [{"pos": i, "n_seq_id": 0} for i in range(celdas)],
        "capas": [
            {
                "r": rng.standard_normal((celdas, N_EMBD_R), dtype=np.float32),
                "p": None,
                "s": rng.standard_normal((celdas, N_EMBD_S), dtype=np.float32),
            }
            for _ in range(capas)
        ],
    }


# ---------------------------------------------------------------- round-trip


def test_roundtrip_sintetico_byte_a_byte():
    """parse -> build sobre un dict sintetico: los bytes vuelven identicos."""
    datos = _sintetico()
    crudo = build_state_bytes(datos, datos["seq_id"])
    vuelta = parse_state_bytes(crudo)
    assert build_state_bytes(vuelta, vuelta["seq_id"]) == crudo


def test_roundtrip_dump_real_byte_a_byte():
    """El entregable: el dump de llama.cpp -> parse -> build -> identico byte a byte."""
    buf = _dato_real()
    datos = parse_state_bytes(buf)
    rebuilt = build_state_bytes(datos, datos["seq_id"])

    assert len(rebuilt) == len(buf), f"tamanio: {len(rebuilt)} != {len(buf)}"
    diffs = [i for i, (a, b) in enumerate(zip(buf, rebuilt)) if a != b]
    assert diffs == [], f"{len(diffs)} bytes difieren, primero en {diffs[0]}"
    assert rebuilt == buf, "los buffers no son identicos"
    print(f"\nround-trip OK: {len(buf)} bytes, {len(diffs)} diferencias")


# ------------------------------------------------------------------- header


def test_header_io_magic():
    """io_magic == 0xaf143cd8 (llama-context.cpp:3093)."""
    buf = _dato_real()
    assert struct.unpack_from("<I", buf, 0)[0] == 0xAF143CD8 == IO_MAGIC
    assert parse_state_bytes(buf)["io_magic"] == 0xAF143CD8


def test_header_seq_id():
    """seq_id == 0: el dump lo produjo tools/gguf_state_roundtrip sobre la seq 0."""
    buf = _dato_real()
    assert struct.unpack_from("<i", buf, 4)[0] == 0
    assert parse_state_bytes(buf)["seq_id"] == 0


def test_cell_count_cuadra_con_las_dimensiones():
    """cell_count del header == la suma de las filas de cada tensor."""
    buf = _dato_real()
    datos = parse_state_bytes(buf)
    cell_count = datos["cell_count"]
    assert len(datos["cells"]) == cell_count
    assert cell_count == 1, "RWKV es recurrente: queda una sola celda, la ultima (pos 6)"

    off = 12 + 8 * cell_count + 8  # magic+seq, celdas, s_trans+n_layer
    for capa in datos["capas"]:
        _, row = struct.unpack_from("<iQ", buf, off)  # r_type, r_size_row
        off += 12 + cell_count * row
    for capa in datos["capas"]:
        _, row = struct.unpack_from("<iQ", buf, off)  # s_type, s_size_row
        off += 12 + cell_count * row
    assert off == len(buf), f"el recorrido deberia terminar en {len(buf)}, termino en {off}"


# ------------------------------------------------------------------- shapes


def test_shapes_contra_los_hiperparametros():
    """La cantidad de capas y el ancho de r y s salen de los hparams del modelo."""
    datos = parse_state_bytes(_dato_real())
    assert len(datos["capas"]) == N_LAYER, "rwkv7.block_count = 32"
    for il, capa in enumerate(datos["capas"]):
        assert capa["r"].shape == (datos["cell_count"], N_EMBD_R), f"capa {il} r"
        assert capa["s"].shape == (datos["cell_count"], N_EMBD_S), f"capa {il} s"
        assert capa["r"].dtype == np.float32 and capa["s"].dtype == np.float32
        assert capa["p"] is None, "este modelo no tiene p (sin PLE conv)"


def test_tamano_total_esperado():
    assert len(_dato_real()) == SIZE_ESPERADO == 21627676


# ------------------------------------------------------------- CLI y errores


def test_cli_rebuild(tmp_path):
    """El comando que usa el script: python -m rwkv_pipeline.state_bytes in out."""
    src, dst = tmp_path / "in.bin", tmp_path / "out.bin"
    src.write_bytes(_dato_real())
    r = subprocess.run(
        [sys.executable, "-m", "rwkv_pipeline.state_bytes", str(src), str(dst)],
        cwd=PROJECT, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert dst.read_bytes() == src.read_bytes()


def test_rechaza_lo_que_no_es_state_buffer():
    buf = _dato_real()
    with pytest.raises(ValueError, match="io_magic"):
        parse_state_bytes(b"\x00\x00\x00\x00" + buf[4:])
    with pytest.raises(ValueError):
        parse_state_bytes(buf[:-8])  # truncado
    with pytest.raises(ValueError, match="shape"):
        datos = parse_state_bytes(buf)
        datos["capas"][0]["r"] = np.zeros((9, 3), dtype=np.float32)
        build_state_bytes(datos, 0)
