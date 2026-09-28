#!/usr/bin/env python3
"""Arma el dataset de Capa 1 aumentado con los shards buenos ya generados.

Base:  data/rwkv_kateto_base_train.jsonl   (mix balanceado ya curado por prepare_datasets.py)
Plus:  out/dataset/shards/parcial-w*.jsonl (aceptados de las olas; mismo formato "text")

FILTROS (por defecto MIN_SHARD=21):
  * se ignoran los shards `parcial-wN.jsonl` con N < MIN_SHARD: son los de la
    generacion temprana, contaminados (todos los turnos de usuario eran
    `KATETOOOOO`, el validador de entonces no lo cazaba -> 141/141 muestras).
  * se descarta igual cualquier muestra con el patron degenerado
    (una palabra sin espacios con 3+ caracteres identicos al final).
  * se descarta toda muestra que contenga `[INST]` o `[/INST]` (contaminacion
    del dataset legado del teacher tipo Hermes; medido en fix-125: sale SOLO de la base).

Salida: data/rwkv_kateto_base_train_plus.jsonl  (NO toca el original)
"""
import glob
import json
import os
import re
import sys

T = os.environ.get("KATETO_HOME", "/run/media/chaos/terciario/proyectos/kateto-train")
BASE = f"{T}/data/rwkv_kateto_base_train.jsonl"
PLUS = f"{T}/data/rwkv_kateto_base_train_plus.jsonl"
SHARDS = f"{T}/out/dataset/shards"
MIN_SHARD = int(os.environ.get("MIN_SHARD", "21"))
DRY_RUN = "--dry-run" in sys.argv or os.environ.get("DRY", "") == "1"

RX_DEGENERADO = re.compile(r"(\S)(\1{2,})\s*<\|im_end\|>")
RX_PLACEHOLDER = re.compile(r"<\s*(?:what|lo que|el usuario|la persona|the user|the person|kateto|tu|the)\s+[^>|]{1,60}>",
                            re.IGNORECASE)


def tiene_placeholder(t: str) -> bool:
    return bool(RX_PLACEHOLDER.search(t))


def tiene_inst(t: str) -> bool:
    """Contaminacion del dataset legado del teacher (tipo Hermes): [INST]/[/INST]."""
    return "[INST]" in t or "[/INST]" in t


def n_shard(path: str) -> int:
    m = re.search(r"parcial-w(\d+)\.jsonl$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def textos(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("text")
            if t and t.strip():
                out.append(t)
    return out


def main() -> None:
    base_all = textos(BASE)
    base = []
    placeholders_base = 0
    inst_base = 0
    ejemplos_placeholders = []
    for t in base_all:
        if tiene_inst(t):
            inst_base += 1
            continue
        if tiene_placeholder(t):
            placeholders_base += 1
            if len(ejemplos_placeholders) < 3:
                ejemplos_placeholders.append(t)
            continue
        base.append(t)
    vistos = set(base)

    nuevos, por_shard, descartados_contaminados, ignorados = [], {}, 0, []
    placeholders_shards = 0
    inst_shards = 0
    for f in sorted(glob.glob(f"{SHARDS}/parcial-w*.jsonl"), key=n_shard):
        n = n_shard(f)
        if n < MIN_SHARD:
            ignorados.append(os.path.basename(f))
            continue
        c = 0
        for t in textos(f):
            if tiene_inst(t):
                inst_shards += 1
                continue
            if tiene_placeholder(t):
                placeholders_shards += 1
                if len(ejemplos_placeholders) < 3:
                    ejemplos_placeholders.append(t)
                continue
            if RX_DEGENERADO.search(t):
                descartados_contaminados += 1
                continue
            if t in vistos:
                continue
            vistos.add(t)
            nuevos.append(t)
            c += 1
        por_shard[os.path.basename(f)] = c

    if not DRY_RUN:
        tmp = PLUS + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for t in base + nuevos:
                fh.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        os.replace(tmp, PLUS)

    print(f"base curado   : {len(base):>7}  {BASE}")
    print(f"shards usados : {len(por_shard):>7}  (+{sum(por_shard.values())} muestras unicas)")
    for k, v in list(por_shard.items())[:6]:
        print(f"    {k}: +{v}")
    if len(por_shard) > 6:
        print(f"    ... y {len(por_shard)-6} mas")
    print(f"shards ignorados (N<{MIN_SHARD}, contaminados): {len(ignorados)} -> {', '.join(ignorados[:4])}"
          + (" ..." if len(ignorados) > 4 else ""))
    print(f"muestras con patron degenerado descartadas: {descartados_contaminados}")
    inst_total = inst_base + inst_shards
    print(f"lineas con [INST]/[/INST] descartadas: {inst_total} (base {inst_base}, shards {inst_shards})")
    print(f"placeholders descartados: {placeholders_base + placeholders_shards} "
          f"(base {placeholders_base}, shards {placeholders_shards})")
    print(f"TOTAL         : {len(base) + len(nuevos):>7}  -> {PLUS}"
          + (" (dry-run: no se escribio)" if DRY_RUN else ""))
    print(f"tamano        : {os.path.getsize(PLUS)/1048576:.1f} MB"
          + (" (sin cambios - dry-run)" if DRY_RUN else ""))
    for i, t in enumerate(ejemplos_placeholders, 1):
        print(f"placeholder ej{i}: {t[:160]!r}")


if __name__ == "__main__":
    main()
