#!/usr/bin/env python3
"""R12 - cuantos rechazos de los validadores eran falsos positivos.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/medir_validadores.py

Corre el gate DOS veces sobre los mismos datos: una con las reglas como estaban
antes de R12 y una con las de ahora, y cuenta los rechazos por motivo en las dos.
Todo numero del reporte sale de aca; ninguno esta escrito a mano.

Las reglas de "antes" viven en `reglas_antes()`, porque el repo ya no las tiene:
se reponen parcheando los cinco puntos de R12 y se restauran al salir. No hay una
segunda copia de `validate` que pueda desincronizarse de la primera.

Los datos de entrada son de solo lectura; lo unico que escribe este script es
`pipelines/reporte-validadores.json`.
"""

import contextlib
import glob
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent
for _p in (PIPELINES, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402
import kateto_purge as kp  # noqa: E402
import gen_toolcalling_dataset as ref  # noqa: E402

BASE = Path("/home/chaos/harness-run/kateto-purge")
REPORT_PATH = PIPELINES / "reporte-validadores.json"

MOTIVOS = ("tuteo", "spanglish", "pocos_turnos", "charla_de_asistente",
           "placeholder_degenerado")

SPANGLISH_ANTES = re.compile(
    r"\b(worries|worry|differente|okay|okey|whatever|actually|basically|anyway|anyways"
    r"|dude|buddy|guys|amazing|awesome|by the way|you know|i mean|it's|don't|can't)\b", re.I)


def tuteo_antes(texto: str) -> str | None:
    """`_tuteo_en` sin el `unicodedata.normalize` de R12: el cuerpo es el mismo,
    lo que falta es la composicion, y sin ella `Mira`+U+0301 matchea `mira`."""
    m = ref.RX_TUTEO_INEQUIVOCO.search(texto)
    if m:
        return m.group(1).lower()
    for m in ref.RX_TUTEO_HOMOGRAFO.finditer(texto):
        if ref.RX_TUTEO_GUARDA.search(texto[: m.start()]):
            continue
        return m.group(1).lower()
    return None


def placeholder_antes(text: str) -> list[str]:
    """La rama KATETO de antes de R12: sobre el texto entero, no sobre la voz."""
    ramas = []
    if kp.PLACEHOLDER_RE.search(text):
        ramas.append("KATETO")
    if kp._token_repetido(text):
        ramas.append("token_repetido")
    return ramas


CHATLA_ANTES = tuple(kp.CHATLA_PATRONES) + ("esta mal",)
SIN_MIN_TURNOS = "no_tool_when_not_needed"


@contextlib.contextmanager
def reglas_antes() -> Iterator[None]:
    """Repone las reglas previas a R12. Al salir, todo queda como estaba."""
    cambios = (
        (ref, "RX_SPANGLISH", SPANGLISH_ANTES),
        (ref, "_tuteo_en", tuteo_antes),
        (kp, "_clausula_placeholder", placeholder_antes),
        (kp, "CHATLA_PATRONES", CHATLA_ANTES),
    )
    saved = [(mod, attr, getattr(mod, attr)) for mod, attr, _ in cambios]
    for mod, attr, valor in cambios:
        setattr(mod, attr, valor)
    sc = next(s for s in ref.SCENARIOS if s["id"] == SIN_MIN_TURNOS)
    min_turnos = sc.pop("min_turns", None)
    try:
        yield
    finally:
        for mod, attr, valor in saved:
            setattr(mod, attr, valor)
        if min_turnos is not None:
            sc["min_turns"] = min_turnos


CORPUS: dict[str, list[Path]] = {
    # Donde se puede ver el ANTES de `placeholder_degenerado` y
    # `charla_de_asistente`: la salida de la purga ya viene con esas lineas
    # descartadas, asi que en el otro corpus esos dos motivos dan cero por cero.
    "purga_entrada": [REPO / "data" / "rwkv_kateto_base_train_plus.jsonl"]
    + [Path(p) for p in sorted(glob.glob(str(REPO / "out/dataset/shards/parcial-w*.jsonl")))],
    "purga_salida": [BASE / "rwkv_kateto_base_train_plus.purged.jsonl",
                     BASE / "shards.purged.jsonl"],
    # Donde se vieron los falsos positivos citados: los 12 rechazos del brazo v2.
    "v2_rechazos": [PIPELINES / "smoke" / "teacher-pool-v2" / "rechazos.jsonl"],
}


def _filas(rutas: list[Path]) -> Iterator[tuple[str, str | None, str]]:
    """(texto, escenario, voz) por linea. El escenario es None cuando la fila no
    lo trae: el train_set humano no esta etiquetado."""
    for ruta in rutas:
        with ruta.open(encoding="utf-8") as fh:
            for linea in fh:
                if not linea.strip():
                    continue
                fila = json.loads(linea)
                yield fila["text"], fila.get("scenario"), fila.get("voice") or ""


def voces_de(texto: str, voz: str) -> list[str]:
    if voz:
        return [voz]
    return [n.strip() for n, _ in kp.VOICE_TURN_RE.findall(texto)] or ["seco"]


def motivo_de(error: str) -> str | None:
    """El nombre del motivo en la lista de R12, o None si es otro error.

    Los validadores no escriben el motivo pelado: escriben `tuteo:mira`,
    `spanglish(differente)`, `pocos_turnos(1<2)` y `purga:charla_de_asistente`.
    """
    limpio = error.split("(")[0].removeprefix("purga:").split(":")[0]
    return limpio if limpio in MOTIVOS else None


def motivos_de(text: str, escenario: str | None, voz: str) -> set[str]:
    """Los motivos de R12 que dispara la linea. Un set, no un Counter: el conteo
    es POR LINEA y una linea puede disparar el mismo motivo dos veces (el
    train_set humano tiene cinco voces y `validate` se corre por cada una)."""
    if escenario:
        errores = g.validar(text, escenario, voz)
    else:
        errores = [e for v in voces_de(text, voz) for e in ref.validate(text, voice=v)]
    errores += [f"purga:{r}" for r in kp.evaluar(text)]
    return {m for m in map(motivo_de, errores) if m}


def medir(rutas: list[Path]) -> dict[str, Any]:
    cuentas: dict[str, Counter] = {"antes": Counter(), "despues": Counter()}
    ejemplos: dict[str, list[str]] = {}
    lineas = 0
    for cuando in ("antes", "despues"):
        with reglas_antes() if cuando == "antes" else contextlib.nullcontext():
            for texto, escenario, voz in _filas(rutas):
                lineas += cuando == "despues"
                for motivo in motivos_de(texto, escenario, voz):
                    cuentas[cuando][motivo] += 1
                    if cuando == "despues" and len(ejemplos.setdefault(motivo, [])) < 3:
                        ejemplos[motivo].append(texto[:260])
    return {
        "lineas": lineas,
        "por_motivo": {m: {"antes": cuentas["antes"][m], "despues": cuentas["despues"][m]}
                       for m in MOTIVOS},
        "ejemplos": ejemplos,
    }


def medir_en_nfd(rutas: list[Path]) -> dict[str, int]:
    """Contrafactual del bug de Unicode: el mismo corpus DESCOMPUESTO en NFD, ya
    con las reglas de R12.

    Ningun archivo del repo viene en NFD, asi que el antes/despues de `tuteo` da
    cero por cero: el defecto es latente y no se ve hasta que algo en el camino
    normaliza. Esto mide cuanto habria costado: cuantos rechazos de `tuteo`
    aparecen si el texto llega descompuesto.
    """
    cuenta: Counter = Counter()
    for texto, escenario, voz in _filas(rutas):
        for motivo in motivos_de(unicodedata.normalize("NFD", texto), escenario, voz):
            cuenta[motivo] += 1
    return dict(cuenta)


def medir_palabras(rutas: list[Path]) -> dict[str, dict[str, int]]:
    """Frecuencia de cada palabra de la lista de spanglish: en el turno de la VOZ,
    donde la regla se aplica, y en el texto entero, donde esta el uso humano que
    la puede justificar. Es la medicion que decide si una palabra entra o sale."""
    palabras = ref.RX_SPANGLISH.pattern[3:-3].split("|")
    rx = re.compile("|".join(map(re.escape, ("differente", *palabras))), re.I)
    en_voz: Counter = Counter()
    entero: Counter = Counter()
    for texto, _, _ in _filas(rutas):
        for voz in kp.turnos_voz(texto):
            for m in rx.finditer(voz):
                en_voz[m.group(0).lower()] += 1
        for m in rx.finditer(texto):
            entero[m.group(0).lower()] += 1
    return {p: {"turnos_de_voz": en_voz[p], "texto_entero": entero[p]}
            for p in ("differente", *palabras)}


def main() -> int:
    corpus: dict[str, Any] = {}
    for nombre, rutas in CORPUS.items():
        faltan = [p for p in rutas if not p.exists()]
        if faltan:
            print(f"saltando {nombre}: falta {faltan[0]}", flush=True)
            continue
        print(f"midiendo {nombre} ({len(rutas)} archivos)...", flush=True)
        medido = medir(rutas)
        medido["archivos"] = [str(p.relative_to(REPO)) if REPO in p.parents else str(p)
                              for p in rutas]
        corpus[nombre] = medido

    total = {m: {k: sum(c["por_motivo"][m][k] for c in corpus.values())
                 for k in ("antes", "despues")} for m in MOTIVOS}
    nfd = {nombre: medir_en_nfd(rutas) for nombre, rutas in CORPUS.items()
           if all(p.exists() for p in rutas)}

    notas = {
        m: "el corpus no tiene texto en NFD, asi que el antes/despues no se mueve: "
           "el defecto era latente. Ver `contrafactual_nfd` para cuanto costaba."
        for m in MOTIVOS if total[m]["antes"] == total[m]["despues"]
    }

    reporte = {
        "meta": {
            "brief": "R12 - dejar de rechazar material valido",
            "generado_por": "pipelines/medir_validadores.py",
            "validadores": "pipelines/kateto_gen.py::validar + pipelines/kateto_purge.py::evaluar",
            "reglas_de_antes": (
                "las cinco de R12 reponidas por parche en `reglas_antes()`: "
                "spanglish con `differente`, `_tuteo_en` sin NFC, "
                "`no_tool_when_not_needed` con min_turns=2, la rama KATETO sobre el "
                "texto entero y `esta mal` en la lista de charla"
            ),
            "nota_knobs": (
                "las filas sin `scenario` (el train_set humano) se validan con los "
                "knobs por default de `validate`: 2 turnos minimos y voseo, la forma "
                "de una charla. El umbral cambiado es por escenario, asi que ahi "
                "`pocos_turnos` no se mueve: se mide donde el escenario esta"
            ),
            "conteo": "instancias de error; un motivo cuenta 1 por linea",
        },
        "por_motivo": total,
        "notas": notas,
        "por_corpus": corpus,
        "contrafactual_nfd": {
            "que_es": (
                "el mismo corpus con cada texto descompuesto a NFD y las reglas de "
                "R12. Los rechazos de `tuteo` que aparecerian si algo en el camino "
                "normalizara a NFD: voseo correcto rechazado"
            ),
            "por_motivo": {m: {c: corpus[c]["por_motivo"][m]["despues"] for c in corpus}
                           | {"nfd": sum(nfd[c].get(m, 0) for c in corpus)}
                           for m in MOTIVOS},
        },
        "medicion_spanglish": {
            "sobre": "purga_entrada",
            "criterio": (
                "una palabra sale de la lista si el corpus humano la usa con "
                "sentido propio. `differente` sale; el resto queda"
            ),
            "palabras": medir_palabras(CORPUS["purga_entrada"]),
        },
    }
    REPORT_PATH.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"reporte: {REPORT_PATH}", flush=True)
    for m in MOTIVOS:
        print(f"  {m:24s} {total[m]['antes']:6d} -> {total[m]['despues']:6d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())