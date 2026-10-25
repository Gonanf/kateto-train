#!/usr/bin/env python3
"""R13 — El 12B QAT local como teacher: la MISMA vara que los otros brazos, mas
el RENDIMIENTO que decide si el pipeline de volumen es viable.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python scripts/pipelines/comparar_local_12b.py

La medicion de calidad NO se reimplementa: se reusan `comparar_simulador.brazo`
(el mismo `es_eco`, el mismo `tasa_similes`, el mismo `largo_turnos_de_voz` de los
brazos R5/R9/R10), asi que las dos columnas de la tabla son comparables por
construccion y no por parecido. `verificacion_instrumento` ademas contrasta los
numeros del pool recalculados aca contra los que ya estan escritos en
`comparacion-simulador-v2.json`.

Lo que suma R13 es `rendimiento`: segundos por muestra y muestras por hora del
brazo local, MEDIDOS (el cronometro de la corrida + los timings por request que
imprime el server), y al lado la misma cuenta para el pool. El pool no guardo
reloj: su `stats` y su `.jsonl` no tienen timestamp y el log de freellmapi son 59
bytes (la key). Por eso sus dos celdas de tiempo van en `null` con el motivo, y no
estimadas — un numero inventado aca seria justamente el que禁止 el brief.
"""

import json
import re
import sys
from pathlib import Path

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent.parent
for _p in (PIPELINES, REPO / "scripts", PIPELINES / "estilo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import comparar_simulador as cs  # noqa: E402  la vara de los otros brazos, sin duplicar

SMOKE = PIPELINES / "smoke"
OUT = SMOKE / "comparacion-local-12b.json"
LOCAL = "teacher-12b-qat"
POOL = "teacher-pool-v3"
V2 = SMOKE / "comparacion-simulador-v2.json"

# Artefactos de la corrida local. Los dos los escribo los pasos previos (el
# cronometro de la corrida y el volcado del journal); este script solo los lee.
CRONO = SMOKE / LOCAL / "cronometro.json"
TIMINGS = SMOKE / LOCAL / "server-timings.log"

# El server imprime una linea por request. El pipe antes de "eval time" es lo que
# separa el decode del prompt eval ("prompt eval time" no matchea).
RX_PROMPT = re.compile(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RX_EVAL = re.compile(r"\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second")
RX_TOTAL = re.compile(r"total time =\s*([\d.]+) ms /\s*(\d+) tokens")
RX_DRAFT = re.compile(r"draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)")


def timings_server(path: Path) -> dict:
    """Agrega los timings por request que Prints el server.

    Las throughput van como `tokens / segundos`, no como promedio de las tasas
    por request: un request con 5 tokens no puede pesar igual que uno de 2000.
    """
    if not path.exists():
        return {"error": f"no existe {path}"}
    recs: list[dict] = []
    for linea in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if m := RX_PROMPT.search(linea):
            recs.append({"prompt_ms": float(m.group(1)), "prompt_n": int(m.group(2)),
                         "prompt_tps": float(m.group(3))})
        elif m := RX_EVAL.search(linea):
            if recs:
                recs[-1].update(eval_ms=float(m.group(1)), eval_n=int(m.group(2)),
                                eval_tps=float(m.group(3)))
        elif m := RX_TOTAL.search(linea):
            if recs:
                recs[-1]["total_ms"] = float(m.group(1))
        elif m := RX_DRAFT.search(linea):
            if recs:
                recs[-1].update(draft_acc=float(m.group(1)), draft_ok=int(m.group(2)),
                                draft_n=int(m.group(3)), draft_len=float(m.group(4)))
    if not recs:
        return {"error": f"no hay timings parseables en {path}"}

    def div(a: float, b: float) -> float | None:
        return round(a / b, 4) if b else None

    p_ms = sum(r.get("prompt_ms", 0.0) for r in recs)
    e_ms = sum(r.get("eval_ms", 0.0) for r in recs)
    t_ms = sum(r.get("total_ms", 0.0) for r in recs)
    gen = sum(r.get("draft_n", 0) for r in recs)
    ok = sum(r.get("draft_ok", 0) for r in recs)
    return {
        "requests": len(recs),
        "prompt_tokens": sum(r.get("prompt_n", 0) for r in recs),
        "prompt_segundos": round(p_ms / 1000, 2),
        "prompt_tok_s": div(sum(r.get("prompt_n", 0) for r in recs), p_ms / 1000),
        "predicted_tokens": sum(r.get("eval_n", 0) for r in recs),
        "predicted_segundos": round(e_ms / 1000, 2),
        "predicted_tok_s": div(sum(r.get("eval_n", 0) for r in recs), e_ms / 1000),
        "servidor_ocupado_s": round(t_ms / 1000, 2),
        "s_por_request_promedio": round(t_ms / 1000 / len(recs), 2),
        "s_por_request_max": round(max(r.get("total_ms", 0.0) for r in recs) / 1000, 2),
        "draft_mtp": {
            "generated": gen,
            "accepted": ok,
            "acceptance_agregada": div(ok, gen),
            "mean_len": div(sum(r.get("draft_len", 0.0) for r in recs), len(recs)),
        },
        "fuente": str(path.relative_to(REPO)),
    }


def rendimiento_local(brazo: dict) -> dict:
    """La cuenta de tiempo del brazo local, toda medida."""
    vacio = {
        "muestras": brazo["aceptadas"] + brazo["rechazadas"],
        "aceptadas": brazo["aceptadas"],
        "rechazadas": brazo["rechazadas"],
        "wall_segundos": None,
        "wall_por_muestra_s": None,
        "muestras_por_hora": None,
        "aceptadas_por_hora": None,
        "argv_de_la_corrida": None,
        "inicio": None,
        "fin": None,
        "fuente_del_wall": f"scripts/pipelines/smoke/{LOCAL}/cronometro.json",
        "server": timings_server(TIMINGS),
    }
    if not CRONO.exists():
        return {**vacio, "error": f"no existe {CRONO}"}
    crono = json.loads(CRONO.read_text(encoding="utf-8"))
    n = crono["muestras"]
    wall = crono["wall_segundos"]
    ts = vacio["server"]
    out = {
        **vacio,
        "muestras": n,
        "wall_segundos": wall,
        "wall_por_muestra_s": round(wall / n, 2) if n else None,
        "muestras_por_hora": round(n / (wall / 3600), 2) if wall else None,
        "aceptadas_por_hora": round(brazo["aceptadas"] / (wall / 3600), 2) if wall else None,
        "argv_de_la_corrida": crono.get("argv"),
        "inicio": crono.get("inicio"),
        "fin": crono.get("fin"),
    }
    if "error" in ts:
        out["error"] = ts["error"]
    elif "servidor_ocupado_s" in ts:
        out["fraccion_del_wall_en_el_server"] = round(ts["servidor_ocupado_s"] / wall, 4)
    return out


def rendimiento_pool(brazo: dict) -> dict:
    """La misma cuenta para el pool. Sin reloj, sin numeros: `null` y el motivo."""
    return {
        "muestras": brazo["aceptadas"] + brazo["rechazadas"],
        "aceptadas": brazo["aceptadas"],
        "rechazadas": brazo["rechazadas"],
        "wall_segundos": None,
        "wall_por_muestra_s": None,
        "muestras_por_hora": None,
        "por_que_null": (
            "La corrida del pool no guardo reloj: teacher-pool-v3/muestras.stats.json "
            "y las .jsonl no tienen timestamp, el json de R10 no tiene bloque de "
            "tiempo, y /tmp/freellmapi.log son 59 bytes (solo la key). No hay de "
            "donde sacar un numero medido sin volver a correr el pool, que este "
            "brief no habilita (pool externo + cuota). Como se llena: re-correr "
            "el brazo del pool con el cronometro encima y sumar las mismas celdas."
        ),
    }


def tabla(local: dict, pool: dict, rend_local: dict, rend_pool: dict) -> list[dict]:
    """La tabla que decide: tiempo contra calidad, mismo orden para los dos."""
    def fila(nombre: str, b: dict, r: dict) -> dict:
        return {
            "brazo": nombre,
            "teacher": "Gemma4-12B-QAT local (llama.cpp, CPU)" if nombre == "local_12b"
                       else "pool (freellmapi, `auto`)",
            "muestras": r["muestras"],
            "aceptadas": b["aceptadas"],
            "tasa_de_aceptacion": round(b["aceptadas"] / r["muestras"], 4) if r["muestras"] else None,
            "tasa_de_eco_usuario": b["tasa_eco"]["usuario"]["tasa"],
            "tasa_de_eco_voz": b["tasa_eco"]["voz"]["tasa"],
            "similes_por_turno_de_voz": b["similes_por_turno_de_voz"]["tasa"],
            "palabras_por_turno_de_voz": b["largo_turnos_de_voz"]["palabras_promedio"],
            "s_por_muestra": r.get("wall_por_muestra_s"),
            "muestras_por_hora": r.get("muestras_por_hora"),
            "aceptadas_por_hora": r.get("aceptadas_por_hora"),
        }
    return [fila("local_12b", local, rend_local), fila("pool", pool, rend_pool)]


def verificacion_instrumento(pool: dict) -> dict:
    """Contrasta los numeros del pool recalculados contra los ya escritos en R10.

    Si el instrument es el mismo, los numeros coinciden. Si no coinciden, la
    tabla compararia dos cosas distintas y el dueno tiene que saberlo antes.
    """
    if not V2.exists():
        return {"error": f"no existe {V2}"}
    d = json.loads(V2.read_text(encoding="utf-8"))["despues"]
    par = [
        ("aceptadas", d["aceptadas"], pool["aceptadas"]),
        ("rechazadas", d["rechazadas"], pool["rechazadas"]),
        ("tasa_eco_usuario", d["tasa_eco"]["usuario"]["tasa"], pool["tasa_eco"]["usuario"]["tasa"]),
        ("tasa_eco_voz", d["tasa_eco"]["voz"]["tasa"], pool["tasa_eco"]["voz"]["tasa"]),
        ("similes_por_turno_de_voz", d["similes_por_turno_de_voz"]["tasa"],
         pool["similes_por_turno_de_voz"]["tasa"]),
        ("palabras_por_turno_de_voz", d["largo_turnos_de_voz"]["palabras_promedio"],
         pool["largo_turnos_de_voz"]["palabras_promedio"]),
    ]
    return {
        "contra": "scripts/pipelines/smoke/comparacion-simulador-v2.json (bloque `despues`)",
        "chequeos": [{"metrica": k, "en_r10": a, "recalculado": b, "igual": a == b}
                     for k, a, b in par],
        "todo_igual": all(a == b for _, a, b in par),
    }


def main() -> int:
    local, pool = cs.brazo(LOCAL), cs.brazo(POOL)
    rend_local, rend_pool = rendimiento_local(local), rendimiento_pool(pool)
    reporte = {
        "nota": (
            "R13. La pregunta: si un teacher local alcanza para generar el dataset "
            "a razon de miles de muestras o hay que seguir dependendo del pool. "
            f"Izquierda = smoke/{LOCAL} (Gemma4-12B-QAT, llama.cpp en CPU, "
            f"head MTP con --spec-type draft-mtp), derecha = smoke/{POOL} (el pool "
            "de freellmapi, el mismo teacher que usaron los brazos R9/R10). Los "
            "dos con los mismos 3 escenarios de charla (solo_charla / solo_debate / "
            "no_tool_when_not_needed), 6 muestras cada uno, registro ON, la misma "
            "guardia anti-eco simetrica y el MISMO instrument: las_metricas de "
            "calidad salen de comparar_simulador.brazo(), sin reimplementar nada. "
            "Lo nuevo de R13 es `rendimiento`, que ningun brazo anterior midio."
        ),
        "config_del_server": {
            "unidad": "kateto-12b-server.service (preexistente, no tocada)",
            "endpoint": "http://127.0.0.1:11510/v1",
            "alias": "kateto-12b",
            "cflags": ("-m Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf "
                       "-md mtp-gemma-4-12B-it.gguf --spec-type draft-mtp -c 4096 -t 6 "
                       "--temp 0.6 --top-k 64 --top-p 0.9 --min-p 0.05 --repeat-penalty 1.1"),
            "gpu": "no usada: la RX 6500 XT tiene 4 GB y un 12B Q4 no entra completo",
            "cpu_threads": 6,
            "mmap": True,
            "nota": "el server lo levanto el runner antes de esta corrida; este "
                    "brief solo le hizo pedidos HTTP, nunca lo arranco/paro/reinicio",
        },
        "medicion": {
            "instrumento": "scripts/pipelines/comparar_simulador.py::brazo (es_eco, tasa_similes, largo_turnos_de_voz)",
            "denominador_del_eco": "muestras aceptadas + rechazadas de cada brazo",
            "pareja": "escenarios, registro, n por escenario y guardas, identicos en los dos brazos",
            "lo_unico_que_cambio": "el teacher: local 12B QAT en CPU vs pool de freellmapi",
        },
        "titular": {
            "aceptadas_local_12b": local["aceptadas"],
            "aceptadas_pool": pool["aceptadas"],
            "tasa_de_eco_usuario_local_12b": local["tasa_eco"]["usuario"]["tasa"],
            "tasa_de_eco_voz_local_12b": local["tasa_eco"]["voz"]["tasa"],
            "tasa_de_eco_usuario_pool": pool["tasa_eco"]["usuario"]["tasa"],
            "tasa_de_eco_voz_pool": pool["tasa_eco"]["voz"]["tasa"],
            "muestras_por_hora_local_12b": rend_local.get("muestras_por_hora"),
            "muestras_por_hora_pool": rend_pool.get("muestras_por_hora"),
        },
        "tabla": tabla(local, pool, rend_local, rend_pool),
        "local_12b": local,
        "pool": pool,
        "rendimiento": {"local_12b": rend_local, "pool": rend_pool},
        "verificacion_instrumento": verificacion_instrumento(pool),
    }
    OUT.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'brazo':10} {'acept':>5} {'eco_usr':>8} {'eco_voz':>8} {'sim/turno':>10} "
          f"{'pal/turno':>10} {'s/muestra':>10} {'muestras/h':>11}")
    for f in reporte["tabla"]:
        print(f"{f['brazo']:10} {f['aceptadas']:>5} {f['tasa_de_eco_usuario']!s:>8} "
              f"{f['tasa_de_eco_voz']!s:>8} {f['similes_por_turno_de_voz']!s:>10} "
              f"{f['palabras_por_turno_de_voz']!s:>10} {f['s_por_muestra']!s:>10} "
              f"{f['muestras_por_hora']!s:>11}")
    v = reporte["verificacion_instrumento"]
    print(f"instrumento igual al de R10: {v.get('todo_igual')}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())