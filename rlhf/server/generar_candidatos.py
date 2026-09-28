#!/usr/bin/env python3
"""Arma candidatos.jsonl desde salidas ya guardadas (SIN GPU).

Lee out/dpo/*.jsonl de kateto-train (solo lectura) y agrupa por prompt:
cada candidato = {id, prompt, respuestas:[2 o 3]}.
No toca la GPU ni el runtime de Kateto.
"""
import json
import os
import sys
from collections import OrderedDict

OUT = os.environ.get("KATETO_DPO_DIR", os.path.join(os.environ.get("KATETO_HOME", "/run/media/chaos/terciario/proyectos/kateto-train"), "out/dpo"))
BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(BASE, "candidatos.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def pares_de_fuentes():
    pares = []
    for fn in ("antisycofancia.jsonl", "pares-fijos.jsonl", "smoke.jsonl"):
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("prompt") and d.get("chosen") and d.get("rejected"):
                    pares.append((d["prompt"], d["chosen"], d["rejected"]))
    return pares


def main():
    pares = pares_de_fuentes()
    if not pares:
        sys.exit("sin fuentes: no se encontro out/dpo/*.jsonl")
    grupos = OrderedDict()
    for prompt, ch, rej in pares:
        g = grupos.setdefault(prompt, [])
        for r in (ch, rej):
            if r not in g:
                g.append(r)
    cands, i = [], 0
    for prompt, rs in grupos.items():
        if len(rs) < 2 or len(set(rs)) < 2:
            continue
        i += 1
        cands.append({"id": f"cand-{i:03d}", "prompt": prompt,
                      "respuestas": rs[:3]})
        if len(cands) >= N:
            break
    if len(cands) < 20:
        sys.exit(f"solo {len(cands)} candidatos, no llego a 20")
    with open(DEST, "w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    n3 = sum(1 for c in cands if len(c["respuestas"]) == 3)
    print(f"candidatos: {len(cands)} (con 3 respuestas: {n3}) -> {DEST}")


if __name__ == "__main__":
    main()
