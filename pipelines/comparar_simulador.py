#!/usr/bin/env python3
"""R10 — La guardia anti-eco en las DOS direcciones: la tasa de eco del usuario y
la de la voz.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/comparar_simulador.py

Lee los dos brazos y escribe `pipelines/smoke/comparacion-simulador-v2.json`:
"antes" = `smoke/teacher-pool-v2` (R9, la guardia del usuario), "despues" =
`smoke/teacher-pool-v3` (R10, la misma guardia simetrica en los dos lados). El
titular son las DOS tasas.

La medicion de eco la hace `kateto_gen.es_eco` en LOS DOS lados y en las DOS
direcciones: la misma funcion, el mismo umbral. Si el antes y el despues usaran
umbrales distintos, la diferencia mediria el detector y no la guarda.

Los brazos de R5 (`teacher-orion`, `teacher-qwen38`) y el `registro` on/off de R4
no se repiten: la variable de R10 es la guarda de eco, no el modelo ni el
registro.
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
OUT = SMOKE / "comparacion-simulador-v2.json"
N_MUESTRAS = 3

# R10: el "antes" es el brazo de R9 (la guardia del usuario ya esta) y el
# "despues" es el mismo pipeline con la guardia simetrica. La UNICA variable es la
# direccion voz->usuario.
ANTES = "teacher-pool-v2"
DESPUES = "teacher-pool-v3"
LADOS = ("usuario", "voz")

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


def tasa_eco(textos: list[str], lado: str = "usuario") -> dict:
    """Turnos de `lado` casi-copia del turno del otro rol, sobre los turnos de `lado`.

    `lado="usuario"`: el turno de usuario contra el turno de voz anterior (R9).
    `lado="voz"`: el turno de voz contra el turno de usuario anterior (R10). Es el
    MISMO detector en las dos direcciones — `es_eco` no mira quien habla — y el
    primer turno de la conversacion queda en el denominador sin ser medible.

    `pares_medibles` queda a la vista para que se vea cuantos pares hubo.
    """
    total = con_eco = medibles = 0
    motivos: dict[str, int] = {}
    mio = (lambda cab: cab == g.IM_USER) if lado == "usuario" else (lambda cab: cab != g.IM_USER)
    for txt in textos:
        previa = ""
        for cab, cuerpo in g.desarmar(txt):
            if not mio(cab):
                previa = cuerpo
                continue
            total += 1
            if previa:
                medibles += 1
                motivo = g.es_eco(cuerpo, previa)
                if motivo:
                    con_eco += 1
                    motivos[motivo] = motivos.get(motivo, 0) + 1
    return {
        "lado": lado,
        "turnos": total,
        "pares_medibles": medibles,
        "turnos_con_eco": con_eco,
        "tasa": round(con_eco / total, 4) if total else None,
        "motivos": motivos,
    }


def con_eco(textos: list[str], lado: str = "usuario") -> list[str]:
    """Solo los textos que tienen un turno de `lado` en eco. Es la evidencia cruda
    de la direccion: los 3 que se citan en el reporte salen de aca."""
    mio = (lambda cab: cab == g.IM_USER) if lado == "usuario" else (lambda cab: cab != g.IM_USER)
    out: list[str] = []
    for txt in textos:
        previa = ""
        for cab, cuerpo in g.desarmar(txt):
            if not mio(cab):
                previa = cuerpo
                continue
            if previa and g.es_eco(cuerpo, previa):
                out.append(txt)
                break
    return out


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
        # Los dos contadores de la guardia, con nombre. `ecos_rechazados` sigue
        # siendo el lado del usuario (R9); `voz_eco` es el nuevo (R10).
        "usuario_eco": stats.get("usuario_eco", stats.get("ecos_rechazados", 0)),
        "voz_eco": stats.get("voz_eco", 0),
        "motivos_eco": stats.get("motivos_eco", {}),
        "motivos_eco_voz": stats.get("motivos_eco_voz", {}),
        "tasa_eco": {lado: tasa_eco(todos, lado) for lado in LADOS},
        "tasa_eco_solo_aceptadas": {lado: tasa_eco(solo_ok, lado) for lado in LADOS},
        "similes_por_turno_de_voz": r4.tasa_similes(solo_ok),
        "largo_turnos_de_voz": r5.largo_turnos_de_voz(solo_ok),
        "muestras_crudas": [
            {"scenario": f["scenario"], "voice": f["voice"], "text": f["text"]}
            for f in aceptadas[:N_MUESTRAS]
        ],
    }


def crudas_de_una_direccion(carpeta: str, lado: str) -> list[dict]:
    """Las `N_MUESTRAS` filas (con su `text` entero) que tienen eco de `lado`.

    Primero las que lo tienen —la evidencia del defecto— y, si no alcanzan, las
    aceptadas: el reporte tiene que poder citar las 3 aunque el "despues" ya no
    tenga ninguna.
    """
    d = SMOKE / carpeta
    aceptadas = _filas(d / "muestras.jsonl")
    fila_por_txt = {f["text"]: f for f in aceptadas if f.get("text")}
    crudas: list[dict] = []
    for txt in con_eco([f["text"] for f in aceptadas if f.get("text")], lado):
        f = fila_por_txt[txt]
        crudas.append({"scenario": f["scenario"], "voice": f["voice"], "text": f["text"]})
        if len(crudas) == N_MUESTRAS:
            return crudas
    for f in aceptadas[:N_MUESTRAS]:
        if f["text"] not in {c["text"] for c in crudas}:
            crudas.append({"scenario": f["scenario"], "voice": f["voice"], "text": f["text"]})
    return crudas[:N_MUESTRAS]


def direccion(lado: str, antes: dict, despues: dict) -> dict:
    """El bloque de UNA direccion de la guardia. El titular de R10 son las dos."""
    ta, td = antes["tasa_eco"][lado], despues["tasa_eco"][lado]
    return {
        "que_mide": (
            f"turnos de {lado} que son casi-copia del turno del otro rol "
            f"inmediatamente anterior, sobre el total de turnos de {lado}. "
            f"Detector: kateto_gen.es_eco en las dos direcciones "
            f"(jaccard de 3-gramas >= {g.ECO_TRIGRAMAS} o arranque identico de "
            f"{g.ECO_ARRANQUE} palabras, sobre texto normalizado)."
        ),
        "tasa_antes": ta["tasa"],
        "tasa_despues": td["tasa"],
        "titular_antes": f"{ta['turnos_con_eco']}/{ta['turnos']}",
        "titular_despues": f"{td['turnos_con_eco']}/{td['turnos']}",
        "turnos_antes": ta["turnos"],
        "turnos_despues": td["turnos"],
        "pares_medibles_antes": ta["pares_medibles"],
        "pares_medibles_despues": td["pares_medibles"],
        "aceptadas": {"antes": antes["aceptadas"], "despues": despues["aceptadas"]},
        "rechazadas": {"antes": antes["rechazadas"], "despues": despues["rechazadas"]},
        # Las dos formas de contar el descarte, y son distintas: el motivo que
        # quedo en la fila (`motivos_de_rechazo`) y el contador de la guardia, que
        # cuenta los reintentos que hubo aunque la muestra igual haya servido.
        "rechazadas_con_este_motivo": {
            "antes": antes["motivos_de_rechazo"].get(lado, 0),
            "despues": despues["motivos_de_rechazo"].get(lado, 0),
        },
        "ecos_rechazados_por_la_guardia": {
            "antes": antes[lado + "_eco"], "despues": despues[lado + "_eco"],
        },
        "motivos_de_la_guardia": {
            "antes": antes["motivos_eco" if lado == "usuario" else "motivos_eco_voz"],
            "despues": despues["motivos_eco" if lado == "usuario" else "motivos_eco_voz"],
        },
        "similes_por_turno_de_voz": {
            "antes": antes["similes_por_turno_de_voz"],
            "despues": despues["similes_por_turno_de_voz"],
        },
        "muestras_crudas": {
            "de_que_brazo": "aceptadas, no el conjunto completo",
            "nota": "primero las que tienen eco de esta direccion; si no "
                    "alcanzan 3, se completan con las primeras aceptadas",
            "antes": crudas_de_una_direccion(ANTES, lado),
            "despues": crudas_de_una_direccion(DESPUES, lado),
        },
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
    direcciones = {f"{lado}_eco": direccion(lado, antes, despues) for lado in LADOS}

    reporte = {
        "nota": (
            f"Antes = smoke/{ANTES} (R9: la guardia del usuario). Despues = "
            f"smoke/{DESPUES} (R10: la misma guardia, simetrica, en los dos "
            "lados). Mismo teacher (`auto` de freellmapi), mismo registro ON, los "
            "mismos 3 escenarios de charla de R5 (solo_charla / solo_debate / "
            "no_tool_when_not_needed), 6 muestras cada uno. La UNICA variable es "
            "la guardia de eco del lado de la voz."
        ),
        "medicion": {
            "las_dos_direcciones": (
                "La guardia anti-eco es simetrica: el mismo detector "
                "(kateto_gen.es_eco) corre en las dos direcciones, y el TITULAR son "
                "las DOS tasas, no una."
            ),
            "medido_en": "muestras aceptadas + rechazadas de cada brazo",
            "ademas": "cada direccion tambien va sobre las solo aceptadas, "
                      "que es la base del conteo a mano del dueño",
        },
        "titular": {
            f"tasa_de_eco_{lado}_antes": direcciones[f"{lado}_eco"]["tasa_antes"]
            for lado in LADOS
        } | {
            f"tasa_de_eco_{lado}_despues": direcciones[f"{lado}_eco"]["tasa_despues"]
            for lado in LADOS
        },
        "direcciones": direcciones,
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
    for lado in LADOS:
        d = direcciones[f"{lado}_eco"]
        print(f"tasa eco {lado:7} {d['tasa_antes']!s:>10}  {d['tasa_despues']!s:>10}"
              f"   ({d['titular_antes']} -> {d['titular_despues']})")
    print(f"aceptadas  {antes['aceptadas']:>10}  {despues['aceptadas']:>10}")
    print(f"rechazadas {antes['rechazadas']:>10}  {despues['rechazadas']:>10}")
    print(f"palabras/turno voz  {antes['largo_turnos_de_voz']['palabras_promedio']} -> "
          f"{despues['largo_turnos_de_voz']['palabras_promedio']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())