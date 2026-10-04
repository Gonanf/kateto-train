#!/usr/bin/env python3
"""RWKV/llama.cpp: build and parse a recurrent state buffer, byte-exact.

El formato NO esta en ningun doc: se dedujo del codigo de llama.cpp
(arbol /run/media/chaos/terciario/proyectos/llama.cpp-master, commit ce8caa6).
Cada campo con la linea que lo prueba:

  offset  size        campo                     prueba
  0       4           uint32 io_magic           llama-context.cpp:3093 (constexpr 0xaf143cd8)
                                                  escrito en :3099 (get_size) y :3116 (get_data)
  4       4           int32  seq_id             llama-context.cpp:3100
  8       4           uint32 cell_count         llama-memory-recurrent.cpp:841
  12      8*cell_count  por celda:                llama-memory-recurrent.cpp:885-886
                    int32  pos                    (llama_pos)
                    uint32 n_seq_id = 0           :883 lo fuerza a 0 cuando seq_id != -1
  ...     4           uint32 s_trans = 0        llama-memory-recurrent.cpp:901 (const en :898)
  ...     4           uint32 n_layer             llama-memory-recurrent.cpp:902
  ...     32*12+...   por capa, SOLO los r:      llama-memory-recurrent.cpp:911-912 (r_type),
                    int32  r_type = 0 (F32)       :915-916 (r_size_row), :923 (payload)
                    uint64 r_size_row
                    cell_count*r_size_row bytes
                      y el p de la capa, si hay:  :928-929 (p_size_row), :933 (payload)
                        uint64 p_size_row
                        cell_count*p_size_row
  ...     32*12+...   por capa, los s AL FINAL:  :938 (bucle s separado), :944-945 (s_type),
                    int32  s_type = 0 (F32)       :948-949 (s_size_row), :956 (payload)
                    uint64 s_size_row
                    cell_count*s_size_row bytes

Los payloads son memcpy crudo: llama_io_write_i::write_tensor / read_tensor no
escriben ninguna cabecera ni padding (llama-context.cpp:2628 y :2868).

Trampa real del formato: la tensor p NO tiene discriminante. El escritor decide
escribirla con `p_l[il] != nullptr` del modelo, no con algo del stream, asi que un
buffer con p es indistinguible de uno sin p leyendo los bytes. Por eso
`parse_state_bytes` acepta `p_cols` (capa -> p_size_row en bytes) y `build` no lo
necesita: en la dict parseada la p ya viene agrupada por capa.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

IO_MAGIC = 0xAF143CD8          # llama-context.cpp:3093
GGML_TYPE_F32 = 0               # ggml_type enum: 0 = GGML_TYPE_F32
ITEM = 4                        # todos los tensores del estado son F32

_HDR = struct.Struct("<IiI")     # io_magic, seq_id, cell_count
_CELL = struct.Struct("<iI")     # pos, n_seq_id
_TWO = struct.Struct("<II")      # s_trans, n_layer
_ROW = struct.Struct("<iQ")      # tipo, size_row


def _fila(arr, cell_count: int, nombre: str) -> tuple[int, bytes]:
    """Valida (cell_count, n_cols) y devuelve (row_size_en_bytes, datos)."""
    a = np.asarray(arr, dtype="<f4")
    if a.ndim != 2 or a.shape[0] != cell_count:
        raise ValueError(
            f"{nombre} deberia tener shape (cell_count={cell_count}, n_cols); tiene {a.shape}"
        )
    return int(a.shape[1]) * ITEM, np.ascontiguousarray(a, "<f4").tobytes()


def parse_state_bytes(buf: bytes, *, p_cols: dict[int, int] | None = None) -> dict:
    """bytes del estado -> dict. `p_cols` mapea capa -> p_size_row en bytes."""
    if len(buf) < _HDR.size:
        raise ValueError(f"buffer muy corto: {len(buf)} bytes")

    io_magic, seq_id, cell_count = _HDR.unpack_from(buf, 0)
    if io_magic != IO_MAGIC:
        raise ValueError(f"io_magic 0x{io_magic:08x} != 0x{IO_MAGIC:08x}: no es un state buffer")
    off = _HDR.size

    cells = []
    for i in range(cell_count):
        pos, n_seq_id = _CELL.unpack_from(buf, off)
        off += _CELL.size
        if n_seq_id != 0:
            raise ValueError(
                f"celda {i}: n_seq_id={n_seq_id}; solo se soportan dumps de una seq (n_seq_id==0), "
                "no el volcado de la cache completa (seq_id == -1)"
            )
        cells.append({"pos": pos, "n_seq_id": n_seq_id})

    s_trans, n_layer = _TWO.unpack_from(buf, off)
    off += _TWO.size
    if s_trans != 0:
        raise ValueError(f"s_trans={s_trans}; llama.cpp rechaza transponer s (state_read_data)")

    p_cols = p_cols or {}
    capas: list[dict] = [{"r": None, "p": None, "s": None} for _ in range(n_layer)]

    def leer_fila(il: int, clave: str) -> None:
        nonlocal off
        tipo, row = _ROW.unpack_from(buf, off)
        off += _ROW.size
        if tipo != GGML_TYPE_F32:
            raise ValueError(f"capa {il}: {clave}_type={tipo}; este modulo solo maneja F32 (0)")
        capas[il][clave] = np.frombuffer(buf, "<f4", cell_count * (row // ITEM), off).reshape(
            cell_count, row // ITEM
        )
        off += cell_count * row

    # los r de todas las capas, con el p de cada una pegado a su r (:906-933)
    for il in range(n_layer):
        leer_fila(il, "r")
        if il in p_cols:
            row = int(p_cols[il])
            if row % ITEM:
                raise ValueError(f"capa {il}: p_size_row={row} no es multiplo de {ITEM}")
            capas[il]["p"] = np.frombuffer(buf, "<f4", cell_count * (row // ITEM), off).reshape(
                cell_count, row // ITEM
            )
            off += cell_count * row

    # los s de todas las capas, recien aca (:938-956)
    for il in range(n_layer):
        leer_fila(il, "s")

    if off != len(buf):
        raise ValueError(
            f"sobran {len(buf) - off} bytes tras {n_layer} capas; el buffer no es de este modelo "
            "(o falta declarar las capas con p via p_cols)"
        )

    return {
        "io_magic": io_magic,
        "seq_id": seq_id,
        "cell_count": cell_count,
        "cells": cells,
        "capas": capas,
    }


def build_state_bytes(datos: dict, seq_id: int) -> bytes:
    """dict -> bytes. Inverso exacto de parse_state_bytes para s_trans==0."""
    celdas = datos["cells"]
    cell_count = len(celdas)
    capas = datos["capas"]
    n_layer = len(capas)

    out = bytearray()
    out += _HDR.pack(IO_MAGIC, int(seq_id), cell_count)
    for i, c in enumerate(celdas):
        if c["n_seq_id"] != 0:
            raise ValueError(f"celda {i}: n_seq_id={c['n_seq_id']} != 0 no es escribible aca")
        out += _CELL.pack(int(c["pos"]), 0)
    out += _TWO.pack(0, n_layer)  # s_trans=0: el otro layout no lo acepta llama.cpp

    for il in range(n_layer):
        r_row, r_data = _fila(capas[il]["r"], cell_count, f"capa {il} r")
        out += _ROW.pack(GGML_TYPE_F32, r_row)
        out += r_data
        if capas[il]["p"] is not None:  # el p viaja pegado a su r (:926-933)
            p_row, p_data = _fila(capas[il]["p"], cell_count, f"capa {il} p")
            out += struct.pack("<Q", p_row)
            out += p_data

    for il in range(n_layer):  # los s van todos juntos, al final (:938)
        if capas[il]["s"] is None:
            continue
        s_row, s_data = _fila(capas[il]["s"], cell_count, f"capa {il} s")
        out += _ROW.pack(GGML_TYPE_F32, s_row)
        out += s_data

    return bytes(out)


def main(argv: list[str]) -> int:
    """python -m rwkv_pipeline.state_bytes <in.bin> <out.bin> [seq_id]"""
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    datos = parse_state_bytes(src.read_bytes())
    seq = int(argv[2]) if len(argv) == 3 else datos["seq_id"]
    salida = build_state_bytes(datos, seq)
    dst.write_bytes(salida)
    print(f"leido:   {src} ({src.stat().st_size} bytes)")
    print(f"rebuilt: {dst} ({len(salida)} bytes) seq_id={seq}")
    print(f"capas:   {len(datos['capas'])} celdas: {datos['cell_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
