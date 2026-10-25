#!/usr/bin/env python3
"""Parchea un KV UINT32 en una copia de un GGUF (edicion binaria in-place).

Uso: patch_gguf_kv.py <src> <dst> <key> <old> <new>
Tipos GGUF: 0 U8,1 I8,2 U16,3 I16,4 U32,5 I32,6 F32,7 BOOL,8 STRING,
9 ARRAY,10 U64,11 I64,12 F64.
"""
import shutil
import struct
import sys


def skip_value(f, vtype):
    if vtype in (0, 1, 7):
        f.seek(1, 1)
    elif vtype in (2, 3):
        f.seek(2, 1)
    elif vtype in (4, 5, 6):
        f.seek(4, 1)
    elif vtype in (10, 11, 12):
        f.seek(8, 1)
    elif vtype == 8:  # STRING
        (alen,) = struct.unpack("<Q", f.read(8))
        f.seek(alen, 1)
    elif vtype == 9:  # ARRAY
        (alen,) = struct.unpack("<Q", f.read(8))
        (atype,) = struct.unpack("<I", f.read(4))
        for _ in range(alen):
            if atype == 8:
                (elen,) = struct.unpack("<Q", f.read(8))
                f.seek(elen, 1)
            elif atype in (0, 1, 7):
                f.seek(1, 1)
            elif atype in (2, 3):
                f.seek(2, 1)
            elif atype in (4, 5, 6):
                f.seek(4, 1)
            elif atype in (10, 11, 12):
                f.seek(8, 1)
            else:
                raise RuntimeError(f"array tipo {atype} no soportado")
    else:
        raise RuntimeError(f"tipo {vtype} no soportado")


def main():
    src, dst, key, old, new = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    key_b = key.encode()
    if src != dst:
        shutil.copyfile(src, dst)
    with open(dst, "r+b") as f:
        assert f.read(4) == b"GGUF", "magic invalido"
        (version,) = struct.unpack("<I", f.read(4))
        assert version == 3, f"version {version}"
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        for _ in range(n_kv):
            (klen,) = struct.unpack("<Q", f.read(8))
            assert klen < 1000, f"klen absurdo {klen}, desync"
            k = f.read(klen)
            (vtype,) = struct.unpack("<I", f.read(4))
            if k == key_b:
                assert vtype == 4, f"tipo inesperado {vtype}"
                (val,) = struct.unpack("<I", f.read(4))
                print(f"{key} actual={val}", flush=True)
                assert val == old, f"valor inesperado {val}"
                f.seek(-4, 1)
                f.write(struct.pack("<I", new))
                print(f"parcheado a {new}", flush=True)
                break
            skip_value(f, vtype)
        else:
            sys.exit("key no encontrada")
    print(f"OK {dst}", flush=True)


main()
