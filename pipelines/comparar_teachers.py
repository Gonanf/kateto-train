#!/usr/bin/env python3
"""R5 — Compara el techo del teacher: pool vs locales.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/comparar_teachers.py

Lee los tres brazos del smoke y escribe `pipelines/smoke/comparacion-teachers.json`:
modelo y base_url por brazo, aceptadas/rechazadas con sus motivos, la tasa de
similes por turno de voz, el largo medio de los turnos de la voz y 3 muestras
CRUZAS de cada brazo para leerlas lado a lado.

Mide con `comparar_registro.tasa_similes` y no con una medicion propia: el A/B
del registro (R4) y esta comparacion tienen que dar el mismo numero para el
mismo texto, o el "antes/despues" no significa nada.

Un brazo que no llego (modelo local que no carga, timeout) NO corta el reporte:
se escribe el error crudo en su lugar y el resto de los brazos se compara igual.
"""

import json
import sys
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent
for _p in (PIPELINES, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import comparar_registro as r4  # noqa: E402  la medicion del A/B, sin duplicar
import kateto_purge as purge  # noqa: E402  turnos_voz

SMOKE = PIPELINES / "smoke"
OUT = SMOKE / "comparacion-teachers.json"
N_MUESTRAS = 3

# (carpeta, base_url, modelo). El id del local es tal cual figura en /v1/models
# de :11434; `auto` es el del pool y esta prohibido pinear otro de freellmapi.
BRAZOS = {
    "pool": ("teacher-pool", "http://localhost:3001/v1", "auto"),
    "orion": ("teacher-orion", "http://localhost:11434/v1",
              "bartowski/TheDrummer_Orion-26B-A4B-v1.1-GGUF:Q4_K_M"),
    "qwen38": ("teacher-qwen38", "http://localhost:11434/v1", "Qwen3.8-27B"),
}

# Lo que se midio de cada brazo que no produjo muestras. Sin esto el JSON dice
# "0 aceptadas" y no dice si el MODELO es malo o si el harness aborto el batch:
# son dos hallazgos opuestos y el numero solo no los separa.
EVIDENCIA = {
    "orion": (
        "Carga bien y el formato le sale bien (una linea de voz, sin preambulo). "
        "Lo que falla es la VELOCIDAD: 43 tokens de respuesta en 416 s y 49 tokens en "
        "107 s con el prompt cacheado (~0.1-0.45 tok/s). El `timeout=240` de "
        "ref.generate (que no se toca) se come los turnos de voz, que son mas largos "
        "que los de usuario: la excepcion es TimeoutError: timed out y distilabel la "
        "logra como crash del subproceso, asi que el batch entero sale vacio. "
        "Ver /tmp/r5-orion.log (WARNING Subprocess traceback) y /tmp/r5-orion2.log."
    ),
    "qwen38": (
        "No llego a producir: el id NO es un modelo local. En /v1/models figura como "
        "source=preset y lo baja de --hf-repo "
        "DavidAU/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NM-DAU-NEO-MAX-MTP-GGUF:Q4_K_S. "
        "Medido: la descarga iba por 239 MB a 122 KB/s (~429 MB/h) contra los ~17 GB "
        "de un Q4_K_S de 27B, o sea ~40 h. status=loading, 0% CPU, RSS 53 MB: el "
        "llama-server esta esperando la descarga, no el modelo. Sin red o sin "
        "tiempo, el brazo queda sin medir (no es que el modelo sea malo)."
    ),
}


def _filas(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def largo_turnos_de_voz(textos: list[str]) -> dict:
    """Largo medio de los turnos de la voz, en palabras.

    El largo medio es la mitad del otro lado de la moneda: si un teacher
    responde con dos palabras, la tasa de similes baja por Structured Noise y
    no porque escriba mejor.
    """
    turnos = [t for txt in textos for t in purge.turnos_voz(txt)]
    palabras = [len(t.split()) for t in turnos]
    return {
        "turnos_de_voz": len(turnos),
        "palabras_promedio": round(sum(palabras) / len(palabras), 1) if palabras else None,
        "palabras_min": min(palabras, default=None),
        "palabras_max": max(palabras, default=None),
    }


def brazo(nombre: str, carpeta: str, base_url: str, model: str) -> dict:
    d = SMOKE / carpeta
    rej = d / "rechazos.jsonl"
    datos = {
        "base_url": base_url,
        "modelo": model,
        "carpeta": f"pipelines/smoke/{carpeta}",
        "escenarios": sorted({f["scenario"] for f in _filas(d / "muestras.jsonl")}),
    }
    if nombre in EVIDENCIA:
        datos["nota_del_harness"] = EVIDENCIA[nombre]
    if not (d / "muestras.jsonl").exists():
        # Un brazo caido no es motivo para no entregar los otros dos: el error
        # crudo queda en el reporte para que se vea que fallo y por que.
        datos["error"] = f"no se escribio {d / 'muestras.jsonl'}"
        return datos

    filas = _filas(d / "muestras.jsonl")
    stats_p = d / "muestras.stats.json"
    stats = json.loads(stats_p.read_text(encoding="utf-8")) if stats_p.exists() else {}
    caidas = _filas(rej)
    motivos: dict[str, int] = {}
    for f in caidas:
        for e in f.get("errors", []):
            motivos[e.split(":")[0].split("(")[0]] = motivos.get(e.split(":")[0].split("(")[0], 0) + 1
    textos = [f["text"] for f in filas]
    datos.update({
        "aceptadas": stats.get("aceptadas", len(filas)),
        "rechazadas": stats.get("rechazadas", 0),
        "motivos": stats.get("motivos", {}),
        "motivos_de_rechazo": motivos,
        "similes_por_turno_de_voz": r4.tasa_similes(textos),
        # El rate de arriba solo ve las ACEPTADAS, asi que una regla que actua
        # como purga de similes se lee como mejora. Esto mira tambien las caidas.
        "similes_en_las_que_cayeron": r4.tasa_similes([f["text"] for f in caidas if f.get("text")]),
        "largo_turnos_de_voz": largo_turnos_de_voz(textos),
        "muestras_crudas": [
            {"scenario": f["scenario"], "voice": f["voice"], "text": f["text"]}
            for f in filas[:N_MUESTRAS]
        ],
    })
    return datos


def main() -> int:
    reporte = {
        "nota": (
            "Mismo prompt, mismo registro ON, mismos 3 escenarios de charla "
            "(solo_charla / solo_debate / no_tool_when_not_needed, 2 muestras cada "
            "uno) en los tres brazos: la unica variable es el modelo. Tasa de "
            "similes = turnos de voz con al menos un 'como ...' sobre turnos de "
            "voz, con la MISMA medicion del A/B del registro (R4). Con 6 muestras "
            "por brazo el intervalo sigue siendo enorme: esto dice que teacher "
            "conviene para la forma, no que uno le gane al otro en calidad."
        ),
        "brazos": {n: brazo(n, *c) for n, c in BRAZOS.items()},
    }
    OUT.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for nombre, d in reporte["brazos"].items():
        if "error" in d:
            print(f"{nombre:>8}: {d['error']}")
            continue
        t = d["similes_por_turno_de_voz"]
        lg = d["largo_turnos_de_voz"]
        print(f"{nombre:>8}: {d['modelo']} | {d['aceptadas']} ok / {d['rechazadas']} rechazos "
              f"| similes {t['tasa']} ({t['turnos_con_simil']}/{t['turnos_de_voz']} turnos) "
              f"| palabras/turno {lg['palabras_promedio']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())