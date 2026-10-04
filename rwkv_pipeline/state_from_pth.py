#!/usr/bin/env python3
"""B3: un estado nativo de RWKV en `.pth` -> el byte-stream que llama.cpp acepta.

El `.pth` de estado nativo (el que sale de `train_state_voice.sh`) trae una sola
cosa por capa:

    blocks.{i}.att.time_state   ->  (n_head, head_size, head_size)  float32

Eso es el estado wkv, que en llama.cpp es el tensor `s_l` de cada capa, con una
fila por celda de `ggml_row_size(F32, hparams.n_embd_s())` floats
(llama-memory-recurrent.cpp:948-949), y

    n_embd_s() = n_embd * wkv_head_size          (llama-hparams.cpp:236-239)
    n_head     = n_embd / wkv_head_size

O sea que el `time_state` del `.pth` se aplana a (n_embd, head_size) en orden C
(head-major) y ese es byte a byte el `s` que espera el buffer. El reshape es
gratis: (n_head, head_size, head_size) -> (n_embd, head_size).

Lo que el `.pth` NO trae: el tensor `r_l`, que en RWKV-7 es el token shift y mide
`token_shift_count * n_embd` (llama-hparams.cpp:208-211, `token_shift_count = 2`
por default en llama-hparams.h:143). Sin el, `estado_a_bytes` lo deja en cero y
eso queda dicho: el estado no es una captura nativa, es el wkv nativo con el
token shift en cero. Que eso tenga el mismo sentido que el estado nativo del
`.pth` NO esta medido (ver docs/gguf-estado-pth.md).

Si las formas no calzan, `estado_a_bytes` se niega a fabricar el buffer: el
diagnostico (`--report`) es la respuesta.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

from rwkv_pipeline.state_bytes import ITEM, build_state_bytes

TIME_STATE = "blocks.{i}.att.time_state"

# gguf-py del arbol de llama.cpp; override con $GGUF_PY
GGUF_PY = os.environ.get(
    "GGUF_PY", "/run/media/chaos/terciario/proyectos/llama.cpp-master/gguf-py"
)
# llama_hparams::token_shift_count (llama-hparams.h:143); el GGUF no lo escribe
TOKEN_SHIFT_COUNT = 2


# ------------------------------------------------------------------ el .pth


def leer_pth(pth_path: str | Path) -> dict[int, np.ndarray]:
    """`{capa: time_state}` con cada tensor en (n_head, head_size, head_size) f4."""
    import torch

    crudo = torch.load(str(pth_path), map_location="cpu", weights_only=True)
    salida: dict[int, np.ndarray] = {}
    for k, v in crudo.items():
        if k.endswith(".att.time_state"):
            salida[int(k.split(".")[1])] = np.ascontiguousarray(
                v.detach().to(torch.float32).numpy(), "<f4"
            )
    if not salida:
        raise ValueError(f"{pth_path}: ningun tensor {TIME_STATE}")
    return dict(sorted(salida.items()))


# ------------------------------------------------------------- el GGUF


def leer_hparams(modelo_gguf: str | Path) -> dict:
    """Los numeros del layout, de la metadata del GGUF."""
    if GGUF_PY not in sys.path:
        sys.path.insert(0, GGUF_PY)
    try:
        from gguf import GGUFReader
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            f"no se pudo importar gguf-py desde {GGUF_PY} ({e}); "
            "ponelo en $GGUF_PY o instalalo"
        ) from e

    r = GGUFReader(str(modelo_gguf), "r")
    kv = {}
    for f in r.fields.values():
        try:
            kv[f.name] = f.contents()
        except Exception:  # tipos que no nos importan
            pass
    arch = kv.get("general.architecture")
    if arch != "rwkv7":
        raise ValueError(f"{modelo_gguf}: architecture={arch!r}; este modulo es para rwkv7")
    embd = int(kv["rwkv7.embedding_length"])
    head = int(kv["rwkv7.wkv.head_size"])
    return {
        "arquitectura": arch,
        "n_layer": int(kv["rwkv7.block_count"]),
        "n_embd": embd,
        "head_size": head,
        "n_head": embd // head,
        "token_shift_count": int(kv.get("rwkv7.token_shift_count", TOKEN_SHIFT_COUNT)),
        # llama-hparams.cpp:208-211 y :236-239
        "n_embd_r": TOKEN_SHIFT_COUNT * embd,
        "n_embd_s": embd * head,
    }


# --------------------------------------------------------------- el veredicto


def diagnostico(pth_path: str | Path, modelo_gguf: str | Path) -> dict:
    """Todo lo que hay que saber antes de fabricar un solo byte."""
    est = leer_pth(pth_path)
    hp = leer_hparams(modelo_gguf)

    esperado_s = (hp["n_embd"], hp["head_size"])          # el s por celda
    esperado_pth = (hp["n_head"], hp["head_size"], hp["head_size"])

    capas = []
    for il in range(max(len(est), hp["n_layer"])):
        en_pth = est.get(il)
        forma = tuple(en_pth.shape) if en_pth is not None else None
        capas.append(
            {
                "capa": il,
                "pth": forma,
                "pth_elementos": int(en_pth.size) if en_pth is not None else None,
                "esperado_s": esperado_s,
                "calza": forma == esperado_pth,
                "motivo": (
                    None
                    if forma == esperado_pth
                    else ("ausente en el .pth" if forma is None else f"esperaba {esperado_pth}")
                ),
            }
        )

    capas_pth = sorted(est)
    return {
        "pth": str(pth_path),
        "modelo": str(modelo_gguf),
        "n_tensores_pth": len(est),
        "capas_pth": capas_pth,
        "capas_pth_contiguas": capas_pth == list(range(len(capas_pth))),
        "shape_pth": tuple(est[capas_pth[0]].shape),
        "todos_mismo_shape": len({tuple(v.shape) for v in est.values()}) == 1,
        "hparams": hp,
        "esperado_s": esperado_s,
        "esperado_r": (hp["n_embd_r"],),
        "esperado_pth": esperado_pth,
        "capas": capas,
        "n_capas_ok": sum(1 for c in capas if c["calza"]),
        "calza": all(c["calza"] for c in capas) and len(est) == hp["n_layer"],
    }


def imprimir(dx: dict) -> None:
    hp = dx["hparams"]
    p = Path(dx["pth"])
    print(f"pth:    {p} ({p.stat().st_size} bytes)")
    print(f"modelo: {dx['modelo']}")
    print()
    print("--- el .pth ---")
    print(f"tensores de estado: {dx['n_tensores_pth']}  (clave: {TIME_STATE})")
    print(f"capas: {len(dx['capas_pth'])} contiguas={dx['capas_pth_contiguas']} "
          f"rango=[{min(dx['capas_pth'])}..{max(dx['capas_pth'])}]")
    print(f"forma de time_state: {dx['shape_pth']}  mismo en todas={dx['todos_mismo_shape']}")
    if len(dx["shape_pth"]) == 3:
        nh, hs, hs2 = dx["shape_pth"]
        print(f"  -> n_head={nh} head_size={hs} head_size={hs2} "
              f"n_embd_pth={nh * hs} dtype=float32")
    print()
    print("--- lo que espera el modelo ---")
    print(f"architecture={hp['arquitectura']} block_count={hp['n_layer']} "
          f"embedding_length={hp['n_embd']} wkv.head_size={hp['head_size']}")
    print(f"token_shift_count={hp['token_shift_count']} -> n_head={hp['n_head']}")
    print(f"s por celda: {dx['esperado_s']} = {hp['n_embd_s']} floats "
          f"({hp['n_embd_s'] * ITEM} bytes)")
    print(f"r por celda: {dx['esperado_r']} = {hp['n_embd_r']} floats "
          f"({hp['n_embd_r'] * ITEM} bytes)  [token shift: el .pth NO lo trae]")
    print()
    print("--- capa por capa ---")
    print(f"{'il':>3}  {'time_state en el .pth':>22}  {'s esperado':>13}  calza")
    for c in dx["capas"]:
        forma = str(c["pth"]) if c["pth"] is not None else "(ausente)"
        print(f"{c['capa']:>3}  {forma:>22}  {str(c['esperado_s']):>13}  "
              f"{'si' if c['calza'] else 'NO'}")
    print()
    print("--- veredicto ---")
    n_pth, n_mod = dx["n_tensores_pth"], hp["n_layer"]
    if dx["calza"]:
        print(f"CALZA: {n_pth} tensores, {dx['shape_pth']} = {dx['esperado_pth']} "
              f"contra {n_mod} capas de n_embd={hp['n_embd']} head_size={hp['head_size']}")
        print("  (r quedara en ceros: el .pth no trae el token shift)")
    else:
        print(f"NO CALZA: el .pth tiene {n_pth} capa(s) de {dx['shape_pth']} "
              f"(n_embd={dx['shape_pth'][0] * dx['shape_pth'][1]}); "
              f"el modelo tiene {n_mod} capa(s) de {dx['esperado_pth']} "
              f"(n_embd={hp['n_embd']})")
        print(f"  capas que calzan: {dx['n_capas_ok']}/{len(dx['capas'])}")
        if dx["capas_pth_contiguas"] and dx["shape_pth"][1:] != (hp["head_size"],) * 2:
            print(f"  head_size: .pth={dx['shape_pth'][1]} modelo={hp['head_size']}")
        if dx["shape_pth"][0] * dx["shape_pth"][1] != hp["n_embd"]:
            print(f"  n_embd: .pth={dx['shape_pth'][0] * dx['shape_pth'][1]} "
                  f"modelo={hp['n_embd']}")
        print("  no se fabrica buffer: un .pth de otro modelo no se salva inventando tensores")


# ------------------------------------------------------------- la fabrica


def estado_a_bytes(
    pth_path: str | Path,
    *,
    modelo_gguf: str | Path,
    seq_id: int = 0,
    pos: int = 0,
) -> bytes:
    """`.pth` -> buffer. Se niega si las formas no calzan."""
    dx = diagnostico(pth_path, modelo_gguf)
    if not dx["calza"]:
        raise ValueError(
            f"las formas no calzan: {dx['n_tensores_pth']} capa(s) de {dx['shape_pth']} "
            f"contra {dx['hparams']['n_layer']} capa(s) de {dx['esperado_pth']}"
        )

    est = leer_pth(pth_path)
    hp = dx["hparams"]
    celdas = [{"pos": int(pos), "n_seq_id": 0}]
    capas = [
        {
            "r": np.zeros((1, hp["n_embd_r"]), dtype="<f4"),  # token shift: no esta en el .pth
            "p": None,
            # (n_head, head_size, head_size) -> (n_embd, head_size) -> la fila del s
            "s": est[il].reshape(1, hp["n_embd_s"]),
        }
        for il in range(hp["n_layer"])
    ]
    return build_state_bytes({"cells": celdas, "capas": capas}, seq_id)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="estado RWKV .pth -> buffer de llama.cpp")
    ap.add_argument("--pth", required=True, help="estado nativo .pth")
    ap.add_argument("--modelo", required=True, nargs="+", help="uno o mas GGUF rwkv7")
    ap.add_argument("--report", action="store_true", help="solo diagnostico, no fabrica")
    ap.add_argument("--out", help="donde escribir el buffer (sin --report)")
    args = ap.parse_args(argv)

    if not args.report and not args.out:
        ap.error("hace falta --report o --out")

    rc = 0
    for modelo in args.modelo:
        if len(args.modelo) > 1:
            print("#" * 72)
            print(f"# {modelo}")
            print("#" * 72)
        dx = diagnostico(args.pth, modelo)
        imprimir(dx)
        if args.report:
            rc |= 0 if dx["calza"] else 3
            continue
        if not dx["calza"]:
            rc |= 3
            continue
        buf = estado_a_bytes(args.pth, modelo_gguf=modelo)
        Path(args.out).write_bytes(buf)
        head = struct.unpack_from("<IiI", buf, 0)
        print(f"escrito: {args.out} ({len(buf)} bytes) "
              f"magic=0x{head[0]:08x} seq_id={head[1]} cell_count={head[2]}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
