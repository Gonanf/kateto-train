#!/usr/bin/env python3
"""Instrumento de estilo: mide la brecha de voz entre dos corpus.

    python pipelines/estilo/metricas_estilo.py \
        --a <corpus humano>  --b <corpus generado> [--top 30] [--json out.json]

Que mide
--------
1. densidad de cliches por 100 palabras (marcadores de asistente + genericos
   + tells de prosodia de `prompts/comedia.md`)
2. solapamiento de trigramas entre A y B (Jaccard) + porcentaje del corpus B
   que aparece en A, para poder comparar con el numero publicado de Gryphe
3. top-N de cliches con su conteo, y el diff: cliches que aparecen en B y no
   en A (esa lista es la brecha que hay que cerrar con el style tune)

Todo el matching es accent-insensitive y case-insensitive. Salida JSON con
`--json`, o tabla por consola (default). Sin dependencias: stdlib.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python \
      pipelines/estilo/metricas_estilo.py --help
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

# --- listas de tells -------------------------------------------------------
# Cada grupo declara su origen. La lista generica y la de marcadores vienen
# de la especificacion del brief; la de prosodia se puede citar linea por linea
# de prompts/comedia.md (M6 y M7).

# prompts/registro de Kateto es "una persona y no un servicio de atencion al
# cliente" (prompt-compartido.md:2). Estas son las formulas que contradicen eso.
CLICHES_ASISTENTE = {
    "es importante recordar": "brief: marcador de asistente",
    "en resumen": "brief: marcador de asistente",
    "no dudes en": "brief: marcador de asistente",
    "como ia": "brief: marcador de asistente",
    "tenes que entender": "brief: marcador de asistente",
    "respetuosamente": "brief: marcador de asistente",
    "espero que esto ayude": "brief: marcador de asistente",
    "en conclusion": "brief: marcador de asistente",
    "por otro lado": "brief: marcador de asistente",
    "es fundamental": "brief: marcador de asistente",
}

CLICHES_GENERICOS = {
    "en el fondo": "brief: generico",
    "sin embargo": "brief: generico",
    "no obstante": "brief: generico",
    "en definitiva": "brief: generico",
    "un torbellino de emociones": "brief: generico",
    "el peso de": "brief: generico",
    "una mezcla de": "brief: generico",
    "se le escapo una sonrisa": "brief: generico",
    "un escalofrio le recorrio la espalda": "brief: generico",
    "por un momento": "brief: generico",
    "de alguna manera": "brief: generico",
    "de cierta forma": "brief: generico",
    "al fin y al cabo": "brief: generico",
}

# prompts/comedia.md M6 (linea 56) prohibe la prosodia escrita, y M7 (linea 64)
# prohibe explicar el chiste. Las dos son marcadores literales.
TELLS_COMEDIA = {
    "[risa]": "comedia.md:56 M6 — prosodia entre corchetes",
    "(risa)": "comedia.md:56 M6 — prosodia entre parentesis",
    "[suspiro]": "comedia.md:56 M6 — prosodia entre corchetes",
    "*suspiro*": "comedia.md:56 M6 — prosodia entre asteriscos",
    "(pausa)": "comedia.md:56 M6 — prosodia entre parentesis",
    "en tono solemne": "comedia.md:56 M6 — guion de actor",
    "bueno, era una broma": "comedia.md:64 M7 — explicar el chiste",
    "el punto es que": "comedia.md:64 M7 — explicar el chiste",
}

GRUPOS = {
    "asistente": CLICHES_ASISTENTE,
    "generico": CLICHES_GENERICOS,
    "comedia": TELLS_COMEDIA,
}

TOKEN_RE = re.compile(r"[a-z0-9']+")


def todas_las_listas() -> dict[str, str]:
    """Grupo -> {cliche normalizado: origen}."""
    out: dict[str, str] = {}
    for grupo, dic in GRUPOS.items():
        for k, v in dic.items():
            out.setdefault(fold(k), f"{grupo}: {v}")
    return out


def fold(texto: str) -> str:
    """Minusculas, sin tildes, espacios colapsados. Para comparar y contar."""
    des = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", des)


# --- lectura de corpus -----------------------------------------------------


def expandir(rutas: Iterable[str]) -> list[Path]:
    """Cada argumento puede ser un archivo o un glob. Escribe solo lo que existe."""
    out: list[Path] = []
    for r in rutas:
        expanded = sorted(glob.glob(r)) or [r]
        for p in expanded:
            p = Path(p)
            if p.exists():
                out.append(p)
            else:
                print(f"[aviso] no existe: {p}", file=sys.stderr)
    return out


def _solo_voz(texto: str) -> str:
    """Deja solo las respuestas de la voz, no los turnos del usuario.

    Formato Kateto: `<|im_user|>turno_del_usuario\\nnombre_de_la_voz\\nrespuesta\\n`.
    Los cliches viven del lado de la voz; medir el turno del usuario mete en el
    denominador texto de otra distribucion, y el nombre de la voz es metadata,
    no habla.
    """
    partes = texto.split("<|im_user|>")[1:]
    respuestas = []
    for p in partes:
        lineas = p.split("\n")
        respuestas.append("\n".join(lineas[2:]))  # [0]=usuario, [1]=nombre de voz
    return "\n".join(respuestas)


def leer(paths: list[Path], campo: str | None, solo_voz: bool) -> list[str]:
    """Devuelve la lista de documentos leidos (crudos o ya filtrados a la voz)."""
    textos: list[str] = []
    for p in paths:
        if p.suffix in {".jsonl", ".ndjson"}:
            with p.open(encoding="utf-8") as fh:
                for linea in fh:
                    linea = linea.strip()
                    if not linea:
                        continue
                    try:
                        obj = json.loads(linea)
                    except json.JSONDecodeError:
                        textos.append(linea)
                        continue
                    if isinstance(obj, dict):
                        valor = obj.get(campo or "text")
                        if valor is None:
                            valor = next(
                                (v for v in obj.values() if isinstance(v, str)), ""
                            )
                    else:
                        valor = str(obj)
                    textos.append(valor)
        else:
            textos.append(p.read_text(encoding="utf-8", errors="replace"))
    if solo_voz:
        textos = [_solo_voz(t) for t in textos]
    return textos


def huella(t: str) -> str:
    return hashlib.md5(t.strip().encode("utf-8")).hexdigest()


# --- metricas --------------------------------------------------------------


def contar_cliches(texto_folded: str, cliches: dict[str, str]) -> Counter:
    c: Counter = Counter()
    for frase in cliches:
        n = texto_folded.count(frase)
        if n:
            c[frase] += n
    return c


def palabras(texto: str) -> int:
    return len(TOKEN_RE.findall(fold(texto)))


def trigramas(texto: str) -> set[tuple[str, ...]]:
    toks = TOKEN_RE.findall(fold(texto))
    return {tuple(toks[i : i + 3]) for i in range(len(toks) - 2)}


def perfil(texto: str, cliches: dict[str, str]) -> dict:
    f = fold(texto)
    n_pal = len(TOKEN_RE.findall(f))
    c = contar_cliches(f, cliches)
    total = sum(c.values())
    return {
        "palabras": n_pal,
        "cliches_total": total,
        "cliches_por_100": round(100.0 * total / n_pal, 4) if n_pal else 0.0,
        "cliches_distintos": len(c),
        "conteo": dict(c.most_common()),
        "_tri": trigramas(texto),
    }


def comparar(a: dict, b: dict) -> dict:
    ta, tb = a["_tri"], b["_tri"]
    inter, union = ta & tb, ta | tb
    return {
        "jaccard_trigramas": round(len(inter) / len(union), 6) if union else 0.0,
        "trigramas_a": len(ta),
        "trigramas_b": len(tb),
        "trigramas_compartidos": len(inter),
        "compartidos_pct_de_b": round(100.0 * len(inter) / len(tb), 4) if tb else 0.0,
        "unicos_a": len(ta - tb),
        "unicos_b": len(tb - ta),
    }


def solo_en_b(a: dict, b: dict) -> list[tuple[str, int]]:
    """Cliches que aparecen en B y no en A, ordenados por conteo en B."""
    return sorted(
        ((f, n) for f, n in b["conteo"].items() if f not in a["conteo"]),
        key=lambda kv: (-kv[1], kv[0]),
    )


# --- salida ----------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.4f}"


def rutas_texto(paths: list[str], tope: int = 3) -> str:
    """Muestra hasta `tope` rutas y el resto como contador (son 158 a veces)."""
    if len(paths) <= tope:
        return ", ".join(paths)
    return ", ".join(paths[:tope]) + f", ... (+{len(paths) - tope} archivos)"


def imprimir(res: dict, top: int) -> None:
    a, b = res["a"], res["b"]
    print("=== METRICAS DE ESTILO ===")
    for etiqueta in ("a", "b"):
        p = res[etiqueta]
        print(
            f"[{etiqueta.upper()}] {p['docs']} docs | {p['palabras']} palabras | "
            f"{p['cliches_total']} cliches ({p['cliches_distintos']} distintos) | "
            f"{_pct(p['cliches_por_100'])} por 100 palabras"
        )
        print(f"        rutas: {rutas_texto(p['paths'])}")
    if res.get("excluidos_en_b"):
        print(f"[NOTA] {res['excluidos_en_b']} documentos de B tambien estan en A y no se midieron")
    print(
        f"[TRIGRAMAS] jaccard={res['jaccard_trigramas']:.6f} | "
        f"compartidos={res['trigramas_compartidos']} "
        f"({res['compartidos_pct_de_b']}% del corpus B) | "
        f"unicos_a={res['unicos_a']} unicos_b={res['unicos_b']}"
    )
    print(f"\n--- top {top} cliches de A ---")
    for frase, n in sorted(a["conteo"].items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        print(f"  {n:>6}  {frase}")
    print(f"\n--- top {top} cliches de B ---")
    for frase, n in sorted(b["conteo"].items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        print(f"  {n:>6}  {frase}")
    print(f"\n--- cliches de B que NO aparecen en A (la brecha) ---")
    diff = res["solo_en_b"]
    if not diff:
        print("  (ninguno)")
    for frase, n in diff[:top]:
        print(f"  {n:>6}  {frase}")


def perfil_json(etiqueta: str, p: dict) -> dict:
    """El perfil sin el set de trigramas (no es serializable)."""
    return {
        "paths": p["paths"],
        "docs": p["docs"],
        "palabras": p["palabras"],
        "cliches_total": p["cliches_total"],
        "cliches_distintos": p["cliches_distintos"],
        "cliches_por_100": p["cliches_por_100"],
        "conteo": p["conteo"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="metricas_estilo.py",
        description=(
            "Mide densidad de cliches y solapamiento de trigramas entre dos corpus. "
            "Sirve para ver la brecha entre la voz humana y la generada antes/despues "
            "de un style tune del head."
        ),
        epilog=(
            "Ejemplo (brecha humano vs generado):\n"
            "  python pipelines/estilo/metricas_estilo.py \\\n"
            "      --a /home/chaos/harness-run/kateto-purge/shards.purged.jsonl \\\n"
            "      --b '/run/media/chaos/terciario/proyectos/kateto-train/out/dataset/shards/parcial-w*.jsonl' \\\n"
            "      --solo-voz --top 30 --json /tmp/brecha.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--a", nargs="+", required=False, help="corpus A (humano, de referencia)")
    ap.add_argument("--b", nargs="+", required=False, help="corpus B (generado, a medir)")
    ap.add_argument("--campo", default=None, help="clave del JSONL (default: 'text', o la 1a string)")
    ap.add_argument(
        "--solo-voz",
        action="store_true",
        help="descarta los turnos <|im_user|> y mide solo las respuestas de la voz",
    )
    ap.add_argument("--top", type=int, default=30, help="cuantos cliches listar (default 30)")
    ap.add_argument(
        "--excluir-en-a",
        action="store_true",
        help="saca de B los documentos que tambien estan en A. Necesario con los "
        "shards parcial-w*: contienen el corpus purgado entero, sin esto se esta "
        "midiendo el corpus humano dos veces",
    )
    ap.add_argument("--json", dest="json_out", default=None, help="escribir el resultado aca")
    ap.add_argument("--lista", action="store_true", help="imprimir la lista de tells y salir")
    args = ap.parse_args(argv)

    if args.lista:
        for frase, origen in sorted(todas_las_listas().items()):
            print(f"{frase}\t{origen}")
        return 0

    if not args.a or not args.b:
        ap.error("hacen falta --a y --b (o --lista)")

    cliches = todas_las_listas()
    rutas_a, rutas_b = expandir(args.a), expandir(args.b)
    if not rutas_a or not rutas_b:
        print("[error] ningun archivo leible", file=sys.stderr)
        return 2

    docs_a = leer(rutas_a, args.campo, args.solo_voz)
    docs_b = leer(rutas_b, args.campo, args.solo_voz)
    excluidos = 0
    if args.excluir_en_a:
        en_a = {huella(d) for d in docs_a}
        antes = len(docs_b)
        vistos: set[str] = set()
        sin_dupes = []
        for d in docs_b:
            h = huella(d)
            if h in en_a or h in vistos:
                continue
            vistos.add(h)
            sin_dupes.append(d)
        excluidos = antes - len(sin_dupes)
        docs_b = sin_dupes

    pa = perfil("\n".join(docs_a), cliches)
    pb = perfil("\n".join(docs_b), cliches)
    pa["paths"], pb["paths"] = [str(p) for p in rutas_a], [str(p) for p in rutas_b]
    pa["docs"], pb["docs"] = len(docs_a), len(docs_b)

    res = {
        "a": pa,
        "b": pb,
        "solo_voz": args.solo_voz,
        "excluidos_en_b": excluidos,
        **comparar(pa, pb),
    }
    res["solo_en_b"] = solo_en_b(pa, pb)

    imprimir(res, args.top)

    if args.json_out:
        salida = {
            "a": perfil_json("a", pa),
            "b": perfil_json("b", pb),
            "solo_voz": args.solo_voz,
            **{k: v for k, v in res.items() if k not in {"a", "b"}},
        }
        salida["solo_en_b"] = [{"cliche": f, "conteo": n} for f, n in res["solo_en_b"]]
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[json] {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())