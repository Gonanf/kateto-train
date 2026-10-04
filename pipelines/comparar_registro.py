#!/usr/bin/env python3
"""R4 — Compara el A/B del registro: con registro vs sin registro.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/comparar_registro.py

Lee los dos lados del smoke y escribe `pipelines/smoke/comparacion-registro.json`:
aceptadas, rechazadas y motivos por lado, la tasa de similes por turno de voz, y
3 muestras CRUZAS de cada lado.

La tasa de similes usa `purge.SIMIL_RE` y no una regex propia: es la misma medicion
que la purga uso para el 31-43% del dataset viejo, asi que el A/B y el diagnostico
anterior son comparables.
"""

import json
import sys
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent
for _p in (PIPELINES, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_purge as purge  # noqa: E402  SIMIL_RE y turnos_voz, sin duplicar

SMOKE = PIPELINES / "smoke"
LADOS = {"con_registro": SMOKE / "registro-on", "sin_registro": SMOKE / "registro-off"}
RECHAZOS = {"con_registro": SMOKE / "rechazos-registro-on.jsonl",
            "sin_registro": SMOKE / "rechazos-registro-off.jsonl"}
OUT = SMOKE / "comparacion-registro.json"
N_MUESTRAS = 3


def _filas(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tasa_similes(textos: list[str]) -> dict:
    """Similes por turno de voz: `(turnos con simil) / (turnos de voz)`."""
    turnos = [t for txt in textos for t in purge.turnos_voz(txt)]
    con_simil = [t for t in turnos if purge.SIMIL_RE.search(t)]
    return {
        "turnos_de_voz": len(turnos),
        "turnos_con_simil": len(con_simil),
        "tasa": round(len(con_simil) / len(turnos), 4) if turnos else None,
        "similes": sum(len(purge.SIMIL_RE.findall(t)) for t in turnos),
    }


def lado(nombre: str, carpeta: Path, rechazos: Path) -> dict:
    filas = _filas(carpeta / "muestras.jsonl")
    stats = json.loads((carpeta / "muestras.stats.json").read_text(encoding="utf-8")) \
        if (carpeta / "muestras.stats.json").exists() else {}
    caidas = [f for f in _filas(rechazos) if f.get("text")]
    motivos: dict[str, int] = {}
    for f in _filas(rechazos):
        for e in f.get("errors", []):
            motivos[e] = motivos.get(e, 0) + 1
    return {
        "carpeta": str(carpeta.relative_to(REPO)),
        "aceptadas": stats.get("aceptadas", len(filas)),
        "rechazadas": stats.get("rechazadas", 0),
        "motivos": stats.get("motivos", {}),
        "motivos_de_rechazo": motivos,
        "similes_por_turno_de_voz": tasa_similes([f["text"] for f in filas]),
        # El rate de arriba solo ve las ACEPTADAS, asi que una regla que actua
        # como purga de similes se ve como una mejora. Esto cuenta las caidas
        # tambien, para que el numero no se pueda leer al reves.
        "similes_en_las_que_cayeron": tasa_similes([f["text"] for f in caidas]),
        "muestras_crudas": [
            {"scenario": f["scenario"], "voice": f["voice"], "text": f["text"]}
            for f in filas[:N_MUESTRAS]
        ],
    }


def main() -> int:
    reporte = {
        "nota": (
            "Tasa de similes = turnos de voz con al menos un 'como ...' sobre "
            "turnos de voz, con purge.SIMIL_RE (la misma medicion del 31-43% del "
            "dataset viejo). Con n chico el intervalo es enorme: esto prueba que "
            "el prompt cambia la forma, no que el registro baje la tasa."
        ),
        "lados": {nombre: lado(nombre, carpeta, RECHAZOS[nombre])
                  for nombre, carpeta in LADOS.items()},
    }
    OUT.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for nombre, d in reporte["lados"].items():
        t = d["similes_por_turno_de_voz"]
        print(f"{nombre:>13}: {d['aceptadas']} aceptadas, {d['rechazadas']} rechazos, "
              f"similes {t['tasa']} ({t['turnos_con_simil']}/{t['turnos_de_voz']} turnos)")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
