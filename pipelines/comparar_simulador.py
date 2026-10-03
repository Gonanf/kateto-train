#!/usr/bin/env python3
"""R9 — Antes/despues del simulador de usuario: la tasa de eco.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/comparar_simulador.py

Lee los dos brazos y escribe `pipelines/smoke/comparacion-simulador.json`:
"antes" = `smoke/teacher-pool` (R5, sin la guardia), "despues" = `smoke/teacher-pool-v2`
(R9, con la regla anti-eco en el prompt y la guardia en el loop).

La medicion de eco la hace `kateto_gen.es_eco` en LOS DOS lados: la misma funcion,
el mismo umbral. Si el antes y el despues usaran umbrales distintos, la
diferencia mediria el detector y no el prompt.

Y se corre `estilo/metricas_estilo.py` de verdad (no se recalcula aca): el
script se llama como modulo y su JSON se levanta del archivo. Los numeros de
cliches y trigramas que salen en el reporte son los que escribio el instrumento.

Los tres brazos de R5 (`teacher-orion`, `teacher-qwen38`) no se repiten: la
variable de R9 es el simulador de usuario, no el modelo, asi que el unico modelo
que se mide es el pool (`auto`).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent
for _p in (PIPELINES, REPO / "scripts", PIPELINES / "estilo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import comparar_registro as r4  # noqa: E402  la medicion de similes, sin duplicar
import comparar_teachers as r5  # noqa: E402  el largo de los turnos de voz
import kateto_gen as g  # noqa: E402  es_eco: el detector, misma funcion a ambos lados

SMOKE = PIPELINES / "smoke"
OUT = SMOKE / "comparacion-simulador.json"
N_MUESTRAS = 3

ANTES = "teacher-pool"
DESPUES = "teacher-pool-v2"

# El corpus humano purgado de R1. Es el mismo contra el que se midio la brecha de
# R7 (humano-humano 63,60% de trigramas compartidos, otra voz 17,35%).
HUMANO = "/home/chaos/harness-run/kateto-purge/shards.purged.jsonl"

# Escala de lectura que fijo R7, para que estos numeros se lean igual.
ESCALA_R7 = {
    "trigramas_compartidos_pct_de_b": {
        "humano_humano": 63.60,
        "otra_voz": 17.35,
    },
    "nota": (
        "humano-humano 63,60% y otra voz 17,35% son los dos extremos de la escala "
        "de R7. El generico de Kateto con la voz humana deberia caer en ese "
        "rango; si queda pegado a 17,35% esta hablando con formulas, no con "
        "registro. El corpus B son 6 muestras de conversacion corta, asi que el "
        "numero es una cota superior gruesa, no una medida de estilo."
    ),
}


def _filas(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tasa_eco(textos: list[str]) -> dict:
    """Turnos de usuario casi-copia del turno de voz anterior / turnos de usuario.

    Sobre TODOS los turnos de usuario de los textos dados, con el primer turno
    de la conversacion en el denominador (no tiene voz previa, no puede ser eco).
    `pares_medibles` queda a la vista para que se vea cuantos pares hubo.
    """
    total = con_eco = medibles = 0
    motivos: dict[str, int] = {}
    for txt in textos:
        voz = ""
        for cab, cuerpo in g.desarmar(txt):
            if cab == g.IM_USER:
                total += 1
                if voz:
                    medibles += 1
                    motivo = g.es_eco(cuerpo, voz)
                    if motivo:
                        con_eco += 1
                        motivos[motivo] = motivos.get(motivo, 0) + 1
            else:
                voz = cuerpo
    return {
        "turnos_de_usuario": total,
        "pares_medibles": medibles,
        "turnos_con_eco": con_eco,
        "tasa": round(con_eco / total, 4) if total else None,
        "motivos": motivos,
    }


def brazo(carpeta: str) -> dict:
    d = SMOKE / carpeta
    aceptadas = _filas(d / "muestras.jsonl")
    caidas = _filas(d / "rechazos.jsonl")
    stats_p = d / "muestras.stats.json"
    stats = json.loads(stats_p.read_text(encoding="utf-8")) if stats_p.exists() else {}

    motivos: dict[str, int] = {}
    for f in caidas:
        for e in f.get("errors", []):
            k = e.split(":")[0].split("(")[0]
            motivos[k] = motivos.get(k, 0) + 1

    # Aceptadas + caidas: si se mirara solo lo aceptado, una regla que actua como
    # purga se lee como mejora. El eco del "antes" esta en las aceptadas y el del
    # "despues" ya no esta en ninguna parte, asi que el denominador tiene que ser
    # el mismo en los dos brazos.
    todos = [f["text"] for f in aceptadas + caidas if f.get("text")]
    solo_ok = [f["text"] for f in aceptadas]
    return {
        "carpeta": f"pipelines/smoke/{carpeta}",
        "escenarios": sorted({f["scenario"] for f in aceptadas}),
        "aceptadas": stats.get("aceptadas", len(aceptadas)),
        "rechazadas": stats.get("rechazadas", len(caidas)),
        "motivos": stats.get("motivos", {}),
        "motivos_de_rechazo": motivos,
        "ecos_rechazados_por_la_guardia": stats.get("ecos_rechazados", 0),
        "motivos_eco": stats.get("motivos_eco", {}),
        "tasa_eco": tasa_eco(todos),
        "tasa_eco_solo_aceptadas": tasa_eco(solo_ok),
        "similes_por_turno_de_voz": r4.tasa_similes(solo_ok),
        "largo_turnos_de_voz": r5.largo_turnos_de_voz(solo_ok),
        "muestras_crudas": [
            {"scenario": f["scenario"], "voice": f["voice"], "text": f["text"]}
            for f in aceptadas[:N_MUESTRAS]
        ],
    }


def estilo(carpeta: str) -> dict:
    """Corre `estilo/metricas_estilo.py` y levanta SU json.

    Se llama al `main` del instrumento con `--json` a un temporal: el numero del
    reporte es el que escribio el script, no una recalculacion de acá.
    """
    if not Path(HUMANO).exists():
        return {"error": f"no existe el corpus humano: {HUMANO}"}
    muestras = SMOKE / carpeta / "muestras.jsonl"
    if not muestras.exists():
        return {"error": f"no existe {muestras}"}
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as fh:
        destino = fh.name
    argv = [
        "--a", HUMANO, "--b", str(muestras), "--solo-voz", "--json", destino,
    ]
    proc = subprocess.run(
        [sys.executable, str(PIPELINES / "estilo" / "metricas_estilo.py"), *argv],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"error": f"metricas_estilo salio {proc.returncode}: {proc.stderr[-400:]}"}
    crudo = json.loads(Path(destino).read_text(encoding="utf-8"))
    Path(destino).unlink(missing_ok=True)
    return {
        "instrumento": "pipelines/estilo/metricas_estilo.py",
        "argv": f"--a {HUMANO} --b pipelines/smoke/{carpeta}/muestras.jsonl --solo-voz",
        "corpus_a": "humano purgado (R1)",
        "docs_b": crudo["b"]["docs"],
        "palabras_b": crudo["b"]["palabras"],
        "cliches_por_100_palabras": crudo["b"]["cliches_por_100"],
        "cliches_total_b": crudo["b"]["cliches_total"],
        "cliches_por_100_palabras_voz_humana": crudo["a"]["cliches_por_100"],
        "jaccard_trigramas": crudo["jaccard_trigramas"],
        "trigramas_compartidos": crudo["trigramas_compartidos"],
        "trigramas_compartidos_pct_de_b": crudo["compartidos_pct_de_b"],
        "top_cliches_b": list(crudo["solo_en_b"][:10]),
    }


def main() -> int:
    antes, despues = brazo(ANTES), brazo(DESPUES)
    ta, td = antes["tasa_eco"], despues["tasa_eco"]

    if ta["tasa"] is None or td["tasa"] is None:
        titular = None
    elif ta["tasa"] == 0:
        titular = "0 -> 0 (o el 'antes' no produjo turnos de usuario para medir)"
    else:
        titular = round(100.0 * (td["tasa"] / ta["tasa"]), 1) if td["tasa"] else 0.0

    reporte = {
        "nota": (
            "Antes = smoke/teacher-pool (R5, sin la guardia). Despues = "
            "smoke/teacher-pool-v2 (R9: regla anti-eco en el prompt del simulador "
            "de usuario + guardia de eco en el loop). Mismo teacher (`auto` de "
            "freellmapi), mismo registro ON, los mismos 3 escenarios de charla de "
            "R5 (solo_charla / solo_debate / no_tool_when_not_needed), 6 muestras "
            "cada uno. La UNICA variable es el simulador de usuario."
        ),
        "medicion": {
            "tasa_de_eco": (
                "turnos de usuario que son casi-copia del turno de voz "
                "inmediatamente anterior, sobre el total de turnos de usuario. "
                "Detector: kateto_gen.es_eco, el MISMO a los dos lados "
                f"(jaccard de 3-gramas >= {g.ECO_TRIGRAMAS} o arranque identico "
                f"de {g.ECO_ARRANQUE} palabras, sobre texto normalizado)."
            ),
            "medido_en": "muestras aceptadas + rechazadas de cada brazo",
        },
        "titular": {
            "tasa_de_eco_antes": ta["tasa"],
            "tasa_de_eco_despues": td["tasa"],
            "variacion_pct": titular,
            "turnos_de_usuario_antes": ta["turnos_de_usuario"],
            "turnos_de_usuario_despues": td["turnos_de_usuario"],
        },
        "antes": antes,
        "despues": despues,
        "voz_contra_el_mismo_instrumento": {
            "escala_r7": ESCALA_R7,
            "antes": estilo(ANTES),
            "despues": estilo(DESPUES),
        },
    }
    OUT.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"           ANTES      DESPUES")
    print(f"tasa eco   {ta['tasa']!s:>10}  {td['tasa']!s:>10}")
    print(f"  turnos   {ta['turnos_de_usuario']:>10}  {td['turnos_de_usuario']:>10}")
    print(f"  con eco  {ta['turnos_con_eco']:>10}  {td['turnos_con_eco']:>10}")
    print(f"aceptadas  {antes['aceptadas']:>10}  {despues['aceptadas']:>10}")
    print(f"rechazadas {antes['rechazadas']:>10}  {despues['rechazadas']:>10}")
    print(f"palabras/turno voz  {antes['largo_turnos_de_voz']['palabras_promedio']} -> "
          f"{despues['largo_turnos_de_voz']['palabras_promedio']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())