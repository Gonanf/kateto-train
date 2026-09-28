#!/usr/bin/env python3
"""Parchea qwen35.block_count 25 -> 24 en una copia del GGUF (edicion binaria in-place).

Formato GGUFv3: magic(4) version u32, n_tensors u64, n_kv u64,
luego por KV: key_len u64, key bytes, tipo u32, valor (UINT32 = 4 bytes).
"""
import os
import shutil
import struct
import sys

SRC = os.environ.get("KATETO_GGUF_SRC", "/run/media/chaos/terciario/proyectos/kateto-train/out/kateto-qwen35-0.8b-f16.gguf")
DST = os.environ.get("KATETO_GGUF_DST", "/run/media/chaos/terciario/proyectos/kateto-train/out/kateto-qwen35-0.8b-f16-fixed.gguf")
KEY = b"qwen35.block_count"

shutil.copyfile(SRC, DST)
with open(DST, "r+b") as f:
    assert f.read(4) == b"GGUF", "magic invalido"
    version = struct.unpack("<I", f.read(4))[0]
    assert version == 3, f"version {version}"
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]
    print(f"version={version} tensors={n_tensors} kvs={n_kv}", flush=True)
    for _ in range(n_kv):
        klen = struct.unpack("<Q", f.read(8))[0]
        key = f.read(klen)
        vtype = struct.unpack("<I", f.read(4))[0]
        if key == KEY:
            assert vtype == 4, f"tipo inesperado {vtype}"
            val = struct.unpack("<I", f.read(4))[0]
            print(f"{KEY.decode()} actual={val}", flush=True)
            assert val == 25, f"valor inesperado {val}"
            f.seek(-4, 1)
            f.write(struct.pack("<I", 24))
            print("parcheado a 24", flush=True)
            break
        # saltear valor segun tipo GGUF: 0 U8,1 I8,2 U16,3 I16,4 U32,5 I32,
        # 6 F32,7 BOOL,8 STRING,9 ARRAY,10 U64,11 I64,12 F64
        if vtype in (0, 1, 7):
            f.seek(1, 1)
        elif vtype in (2, 3):
            f.seek(2, 1)
        elif vtype in (4, 5, 6):
            f.seek(4, 1)
        elif vtype in (10, 11, 12):
            f.seek(8, 1)
        elif vtype == 8:  # STRING
            alen = struct.unpack("<Q", f.read(8))[0]
            f.seek(alen, 1)
        else:  # ARRAY(9): largo, tipo elementos, elementos
            alen = struct.unpack("<Q", f.read(8))[0]
            atype = struct.unpack("<I", f.read(4))[0]
            # arrays de strings (tokenizer): saltear elemento por elemento
            for _ in range(alen):
                if atype == 8:
                    elen = struct.unpack("<Q", f.read(8))[0]
                    f.seek(elen, 1)
                elif atype in (4, 5, 6):
                    f.seek(4, 1)
                else:
                    raise RuntimeError(f"array tipo {atype} no soportado")
    else:
        sys.exit("key no encontrada")
print(f"OK {DST}", flush=True)
