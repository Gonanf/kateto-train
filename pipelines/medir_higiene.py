#!/usr/bin/env python3
"""R8 — Higiene de corpus: cuanto de `charla_de_asistente` queda en el train set
purgado y en los shards, y que frase lo dispara.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/medir_higiene.py

Mide sobre los archivos YA purgados (los que salen del sink de R1/R2), no sobre
los originales: lo que interesa es cuanto de la regla nueva todavia sobrevive a
las reglas viejas. Los archivos de entrada son de solo lectura; lo unico que
escribe este script es `pipelines/reporte-higiene.json`.

La lista de patrones NO se re-declara aca: se importa de `kateto_purge`, asi que
el reporte y la regla nunca pueden discrepar. Los candidatos que se midieron y se
quedaron afuera van en `CANDIDATOS_RECHAZADOS`, con su conteo real y el motivo:
sin eso, el proximo que encuentre `disculpa` (250 lineas) la agrega sin saber que
es casi toda narration.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PIPELINES = Path(__file__).resolve().parent
if str(PIPELINES) not in sys.path:
    sys.path.insert(0, str(PIPELINES))

import kateto_purge as kp  # noqa: E402

REPO = PIPELINES.parent
BASE = Path("/home/chaos/harness-run/kateto-purge")
REPORT_PATH = PIPELINES / "reporte-higiene.json"

FUENTES = {
    "train_set": BASE / "rwkv_kateto_base_train_plus.purged.jsonl",
    "shards": BASE / "shards.purged.jsonl",
}

# Subconjunto que solo dice un asistente: se autodeclara o declara un rechazo.
# Los demas patrones (`esta mal`, `es importante`, `deberias`, `lo siento`) son
# palabras corrientes que el personaje tambien dice, asi que una linea que solo
# cae por ahi es sospecha, no prueba. Ver `notas.precision` del reporte.
ALTA_PRECISION = frozenset(
    {
        "como asistente",
        "como ia",
        "como modelo de lenguaje",
        "as an ai",
        "no puedo ayudarte",
        "no puedo proporcionar",
        "no puedo generar",
        "i must refuse",
        "lo siento, pero",
    }
)

# Medidos en el mismo dataset y dejados afuera a proposito.
CANDIDATOS_RECHAZADOS: dict[str, str] = {
    "disculpa": "casi todo narracion ('pidio disculpas por sus dichos'), no voz de asistente",
    "conviene": "consejo util ('te conviene float32'), no relleno de asistente",
    "no te olvides": "coloquial rioplatense ('no te olvides del leandro')",
    "te recomiendo": "consejo que a veces es del personaje, indistinguible",
    "es clave": "commentary tecnico, indistinguible de un turno normal",
    "vale la pena": "idem, y la mitad son 'no vale la pena' (no wasteful)",
    "es fundamental": "'valores fundamentales de la empresa'",
    "a disposicion": "substantivo ('su propia disposicion'), no cortesia",
    "no olvides": "'no olvides el asado': es el personaje",
    "en linea con": "texto corporativo de manual, no del corpus",
    "para recordar": "album de fotos, no admonicion",
}

MAX_EJEMPLOS = 2
EJEMPLO_CHARS = 220
TOP_N = 20


def medir(path: Path) -> dict[str, Any]:
    """Una pasada: conteos por frase, lineas que caen, y ejemplos verbatim."""
    lineas: Counter = Counter()
    ocurrencias: Counter = Counter()
    rechazadas_lineas: Counter = Counter()
    rechazadas_ocurrencias: Counter = Counter()
    ejemplos: dict[str, list[str]] = {}
    total = 0
    con_charla = 0
    solo_alta = 0

    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            text = json.loads(raw).get("text", "")
            voces = kp.turnos_voz(text)
            total += 1

            planos = [kp.sin_acentos(v) for v in voces]
            disparadores = kp.charla_disparadores(text)
            for patron in disparadores:
                n = sum(p.count(patron) for p in planos)
                lineas[patron] += 1
                ocurrencias[patron] += n
                # El turno mas corto que dispara el patron es el que mejor
                # evidencia el registro de voz (los largos son prosa de agente).
                candidatos = [v for v, p in zip(voces, planos) if patron in p]
                if candidatos:
                    corto = min(candidatos, key=len).strip()[:EJEMPLO_CHARS]
                    vistos = ejemplos.setdefault(patron, [])
                    if corto not in vistos and len(vistos) < MAX_EJEMPLOS:
                        vistos.append(corto)

            for patron in CANDIDATOS_RECHAZADOS:
                n = sum(p.count(patron) for p in planos)
                if n:
                    rechazadas_lineas[patron] += 1
                    rechazadas_ocurrencias[patron] += n

            if disparadores:
                con_charla += 1
                if disparadores <= ALTA_PRECISION:
                    solo_alta += 1

    return {
        "path": str(path),
        "lineas": total,
        "con_charla_de_asistente": con_charla,
        "porcentaje": round(100 * con_charla / total, 3) if total else None,
        "solo_alta_precision": solo_alta,
        "por_frase_lineas": dict(lineas.most_common()),
        "por_frase_ocurrencias": dict(ocurrencias.most_common()),
        "rechazadas_lineas": dict(rechazadas_lineas.most_common()),
        "rechazadas_ocurrencias": dict(rechazadas_ocurrencias.most_common()),
        "ejemplos": ejemplos,
    }


def _pct(parte: int, total: int) -> float | None:
    return round(100 * parte / total, 3) if total else None


def main() -> int:
    por_fuente = {nombre: medir(path) for nombre, path in FUENTES.items()}
    lineas_totales = sum(f["lineas"] for f in por_fuente.values())
    con_charla = sum(f["con_charla_de_asistente"] for f in por_fuente.values())
    solo_alta = sum(f["solo_alta_precision"] for f in por_fuente.values())

    total_lineas: Counter = Counter()
    total_ocurrencias: Counter = Counter()
    for f in por_fuente.values():
        total_lineas.update(f["por_frase_lineas"])
        total_ocurrencias.update(f["por_frase_ocurrencias"])

    top20 = [
        {
            "patron": patron,
            "ocurrencias": n,
            "lineas": total_lineas[patron],
            "train_set": por_fuente["train_set"]["por_frase_lineas"].get(patron, 0),
            "shards": por_fuente["shards"]["por_frase_lineas"].get(patron, 0),
            "alta_precision": patron in ALTA_PRECISION,
            "ejemplos": next(
                (
                    f["ejemplos"][patron]
                    for f in por_fuente.values()
                    if patron in f["ejemplos"]
                ),
                [],
            ),
        }
        for patron, n in total_ocurrencias.most_common(TOP_N)
    ]

    rechazadas = [
        {
            "patron": patron,
            "ocurrencias": sum(
                f["rechazadas_ocurrencias"].get(patron, 0) for f in por_fuente.values()
            ),
            "lineas": sum(
                f["rechazadas_lineas"].get(patron, 0) for f in por_fuente.values()
            ),
            "motivo": motivo,
        }
        for patron, motivo in sorted(
            CANDIDATOS_RECHAZADOS.items(),
            key=lambda kv: -sum(
                f["rechazadas_ocurrencias"].get(kv[0], 0) for f in por_fuente.values()
            ),
        )
    ]

    reporte = {
        "regla": "charla_de_asistente",
        "generado_por": "pipelines/medir_higiene.py",
        "medido_sobre": (
            "los archivos YA purgados por las reglas de R1/R2: la regla nueva "
            "anota y el filtrado real ocurre en el sink, asi que lo que se cuenta "
            "aca es lo que las reglas viejas NO barrean todavia."
        ),
        "patrones_en_la_regla": list(kp.CHATLA_PATRONES),
        "entradas": {
            **{
                nombre: {k: v for k, v in f.items() if k != "ejemplos"}
                for nombre, f in por_fuente.items()
            },
            "total": {
                "lineas": lineas_totales,
                "con_charla_de_asistente": con_charla,
                "porcentaje": _pct(con_charla, lineas_totales),
                "solo_alta_precision": solo_alta,
                "porcentaje_solo_alta_precision": _pct(solo_alta, lineas_totales),
            },
        },
        "top20_frases_disparadoras": top20,
        "frases_sin_un_solo_golpe": [
            p for p in kp.CHATLA_PATRONES if total_ocurrencias[p] == 0
        ],
        "candidatos_medidos_y_rechazados": rechazadas,
        "notas": {
            "precision": (
                "El contador de `con_charla_de_asistente` es una cota superior, no "
                "la basura real. `esta mal` y `no esta bien` dominan el top20 por "
                "conteo pero los ejemplos muestran habla del personaje ('el del "
                "barrio no esta mal, boludo'), no voz de asistente. `solo_alta_"
                "precision` cuenta las lineas que un asistente reconoceria sin "
                "duda: es el numero para decidir el filtro en el sink."
            ),
            "ingles": (
                "Los seis patrones en ingles dan 3 lineas, las tres de `i can't`, "
                "y las tres son prosa de limite de capacidad ('which i can't "
                "complete automatically'): justo la basura que se busca. `i "
                "cannot`, `i'm unable`, `i'm sorry`, `i must refuse` y `as an "
                "ai` no dan ni un golpe. Lo que NO hay que hacer es ensancharlos "
                "a `sorry` o `cannot` a pelo: en este corpus el primero es jerga "
                "rioplatense del personaje ('sorry por el quilombo') y el "
                "segundo son citas de log de herramientas ('cannot find "
                "module', 'connection refused')."
            ),
            "patrones_sin_golpe": (
                "Los que quedaron en cero sobre este dataset no son ruido: son "
                "la red que detecta la basura si el corpus cambia de origen."
            ),
            "no_toca_las_reglas_viejas": (
                "`charla_de_asistente` se agrega a `REGLAS` y no modifica ninguna "
                "regla existente ni su firma."
            ),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{con_charla}/{lineas_totales} lineas caen en charla_de_asistente "
        f"({_pct(con_charla, lineas_totales)}%), {solo_alta} solo por alta precision",
        flush=True,
    )
    print(f"reporte: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
