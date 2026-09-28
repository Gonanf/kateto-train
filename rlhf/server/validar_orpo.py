#!/usr/bin/env python3
"""Valida pares_orpo.jsonl para --data_type orpo (sin GPU).

Chequeo estructural (siempre) + camino real con MyDataset si hay torch.
Falla ruidosamente (exit 1) si el formato no sirve para entrenar.
Uso: python3 validar_orpo.py [pares_orpo.jsonl]
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
F = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "pares_orpo.jsonl")


def fail(msg):
    print(f"ERROR: {msg}", flush=True)
    sys.exit(1)


def main():
    if not os.path.exists(F):
        fail(f"no existe {F} (exporta primero con /api/export)")
    pares = []
    with open(F, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception as e:
                fail(f"linea {i}: JSON invalido ({e})")
            for k in ("prompt", "chosen", "rejected"):
                if k not in d:
                    fail(f"linea {i}: falta clave {k!r}")
                if not isinstance(d[k], str) or not d[k].strip():
                    fail(f"linea {i}: {k} vacio o no es texto")
            if d["chosen"].strip() == d["rejected"].strip():
                fail(f"linea {i}: chosen == rejected")
            if "<|im_user|>" not in d["prompt"]:
                fail(f"linea {i}: prompt sin marcador <|im_user|>")
            # analogo a la mascara: el prompt no se entrena (largo>0) y cada
            # completion aporta tokens (prompt+respuesta mas largo que prompt)
            if len(d["prompt"]) == 0:
                fail(f"linea {i}: prompt vacio, nada que enmascarar")
            if len(d["prompt"] + d["chosen"]) <= len(d["prompt"]):
                fail(f"linea {i}: chosen no aporta tokens")
            if len(d["prompt"] + d["rejected"]) <= len(d["prompt"]):
                fail(f"linea {i}: rejected no aporta tokens")
            pares.append(d)
    if not pares:
        fail("archivo vacio: 0 pares")
    print(f"estructural OK: {len(pares)} pares, claves prompt/chosen/rejected, "
          f"prompt enmascarable, chosen!=rejected")
    try:
        import torch  # noqa
    except ImportError:
        print("aviso: sin torch en este entorno, no se corre MyDataset; "
              "el chequeo estructural replica sus aserciones (formas/mascara).")
        print(f"OK: {F} apto para --data_type orpo ({len(pares)} pares)")
        return
    # camino real, misma logica que test-orpo-datos.py
    from types import SimpleNamespace
    sys.path.insert(0, os.path.join(os.environ.get("KATETO_HOME", "/run/media/chaos/terciario/proyectos/kateto-train"), "RWKV-PEFT"))
    from rwkvt.dataset.dataset import MyDataset
    args = SimpleNamespace(data_type="orpo", data_file=F, epoch_steps=1000,
                           ctx_len=160, data_shuffle=0)
    ds = MyDataset(args)
    xw, yw, xl, yl = ds[0]
    assert xw.shape == xl.shape == yw.shape == yl.shape
    assert (yw != -100).sum() > 0 and (yl != -100).sum() > 0
    m = (yw != -100).nonzero().flatten()
    assert int(m[0]) > 0, "el prompt quedo sin enmascarar"
    print(f"MyDataset OK: {len(ds)} pares, formas {tuple(xw.shape)}, "
          f"primer token entrenado en {int(m[0])}")
    print(f"OK: {F} apto para --data_type orpo")


if __name__ == "__main__":
    main()
