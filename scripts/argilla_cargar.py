#!/usr/bin/env python3
"""Carga una muestra del dataset de Kateto en Argilla para revisarlo en el navegador.

La muestra viene sesgada a lo que importa revisar, no al azar puro:
  - TODAS las muestras con el centinela `<|no_response|>` (el 10,4% que ensucia la voz)
  - TODAS las que tienen `<|wait|>`
  - TODAS las de voces que no son `seco` (son 160 por voz, las "voces perdidas")
  - el resto al azar hasta --n

Uso (con el stack de docker/argilla levantado):
    # instalar el cliente en un venv aparte, para no tocar el venv del repo
    python3 -m venv /run/media/chaos/secundario/argilla-venv
    /run/media/chaos/secundario/argilla-venv/bin/pip install "argilla>=2.4"

    /run/media/chaos/secundario/argilla-venv/bin/python scripts/argilla_cargar.py --n 800
    /run/media/chaos/secundario/argilla-venv/bin/python scripts/argilla_cargar.py --solo-largas
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
API_URL = "http://localhost:6900"
API_KEY = "argilla.apikey"


def contar_por_voz(texto: str) -> dict[str, int]:
    """Cuenta turnos de la voz por nombre: <|im_start|>seco\n, <|im_start|>doktor\n, etc."""
    voces = re.findall(r"<\|im_start\|>(\w+)\s", texto)
    out: dict[str, int] = {}
    for v in voces:
        out[v] = out.get(v, 0) + 1
    return out


def leer_jsonl(path: Path) -> list[dict]:
    filas = []
    with path.open(encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if linea:
                try:
                    filas.append(json.loads(linea))
                except json.JSONDecodeError:
                    pass
    return filas


def armar_muestra(filas: list[dict], n: int, seed: int) -> list[dict]:
    """Seleccion sesgada a lo que hay que mirar, con cupo por grupo para que la
    muestra sea revisable: no se puede mirar un tablero de 6.000 filas donde
    4.600 son lo mismo. Cupos: centinela 40%, wait 15%, otras voces 15%, el resto
    al azar hasta completar n."""
    grupos = (
        ("centinela", [f for f in filas if "<|no_response|>" in (f.get("text") or "")], 0.40),
        ("wait", [f for f in filas if "<|wait|>" in (f.get("text") or "")], 0.15),
        ("otra_voz", [f for f in filas
                      if any(v != "seco" for v in contar_por_voz(f.get("text") or ""))], 0.15),
    )
    rng = random.Random(seed)
    vistas: set[str] = set()
    muestra: list[dict] = []
    for etiqueta, grupo, frac in grupos:
        grupo = list(grupo)
        rng.shuffle(grupo)
        tope = max(1, int(n * frac))
        tomados = 0
        for f in grupo:
            if tomados >= tope:
                break
            t = f.get("text") or ""
            if t and t not in vistas:
                vistas.add(t)
                f = dict(f)
                f["_motivo"] = etiqueta
                muestra.append(f)
                tomados += 1
    resto = [f for f in filas if (f.get("text") or "") not in vistas]
    rng.shuffle(resto)
    for f in resto:
        if len(muestra) >= n:
            break
        t = f.get("text") or ""
        if not t:
            continue
        vistas.add(t)
        f = dict(f)
        f["_motivo"] = "azar"
        muestra.append(f)
    rng.shuffle(muestra)
    return muestra


def main() -> int:
    ap = argparse.ArgumentParser(description="Cargar el dataset de Kateto en Argilla")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--dataset", default="kateto-train")
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("--api-key", default=API_KEY)
    ap.add_argument("--solo-largas", action="store_true",
                    help="carga solo out/dataset/largas/*.jsonl (las conversaciones largas)")
    ap.add_argument("--dry-run", action="store_true", help="no manda nada, solo reporta")
    args = ap.parse_args()

    base = REPO / "data/rwkv_kateto_base_train_plus.jsonl"
    filas: list[dict] = []
    if not args.solo_largas:
        filas += leer_jsonl(base)
    largas = sorted((REPO / "out/dataset/largas").glob("*.jsonl")) if (REPO / "out/dataset/largas").is_dir() else []
    for p in largas:
        if p.name.endswith("_rejects.jsonl"):
            continue
        for f in leer_jsonl(p):
            f = dict(f)
            f["_motivo"] = "larga"
            filas.append(f)
    print(f"fuente: {len(filas)} filas ({base.name}: {0 if args.solo_largas else len(leer_jsonl(base))}, largas: {len(largas)} archivos)")

    muestra = armar_muestra(filas, args.n, args.seed)
    from collections import Counter
    print("muestra:", len(muestra), "| motivos:", Counter(f.get("_motivo") for f in muestra).most_common())
    largos = [len(f.get("text") or "") for f in muestra]
    if largos:
        largos.sort()
        print(f"largos (chars): p50 {largos[len(largos)//2]} | max {largos[-1]}")
    if args.dry_run:
        print("dry-run: no mando nada")
        return 0

    import argilla as rg
    cliente = rg.Argilla(api_url=args.api_url, api_key=args.api_key)
    settings = rg.Settings(
        guidelines=("Conversaciones del dataset de la voz seco. Marcar 'sirve' si la charla se sostiene "
                    "y suena rioplatense; 'no' si es basura (calco del material, asistente, tuteo, repeticion)."),
        fields=[rg.TextField(name="conversacion", title="Conversacion")],
        questions=[
            rg.LabelQuestion(name="sirve", title="¿Sirve para entrenar?",
                             labels=["sirve", "dudosa", "no"]),
            rg.TextQuestion(name="nota", title="Nota", required=False),
        ],
        metadata=[rg.TermsMetadataProperty(name="motivo", title="Por que esta en la muestra"),
                  rg.IntegerMetadataProperty(name="chars", title="Largo en caracteres")],
    )
    ds = rg.Dataset(name=args.dataset, settings=settings, client=cliente)
    existe = cliente.datasets(name=args.dataset)
    if existe is None:
        ds.create()
        print("dataset creado:", args.dataset)
    else:
        ds = existe
        print("dataset existente, agrego registros:", args.dataset)
    registros = [
        rg.Record(fields={"conversacion": f.get("text") or ""},
                  metadata={"motivo": f.get("_motivo") or "?", "chars": len(f.get("text") or "")})
        for f in muestra
    ]
    for i in range(0, len(registros), 200):
        ds.records.log(registros[i:i + 200])
        print(f"  cargados {min(i + 200, len(registros))}/{len(registros)}")
    print(f"listo -> {args.api_url} | dataset '{args.dataset}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
