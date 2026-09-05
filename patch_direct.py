#!/usr/bin/env python3
"""Parche directo: busca la key en los primeros 8KB y reescribe el valor UINT32."""
import struct
import sys

path, key, old, new = sys.argv[1], sys.argv[2].encode(), int(sys.argv[3]), int(sys.argv[4])
with open(path, "r+b") as f:
    head = f.read(8192)
    i = head.find(key)
    assert i > 0, "key no encontrada en header"
    pos = i + len(key)
    (vtype,) = struct.unpack_from("<I", head, pos)
    assert vtype == 4, f"tipo {vtype}"
    (val,) = struct.unpack_from("<I", head, pos + 4)
    print(f"{key.decode()} actual={val}", flush=True)
    assert val == old, f"valor inesperado {val}"
    f.seek(pos + 4)
    f.write(struct.pack("<I", new))
    print(f"parcheado a {new}", flush=True)
print("OK", flush=True)
