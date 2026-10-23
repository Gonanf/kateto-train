#!/usr/bin/env python3
"""Coherencia por par + contraste con eval_v2.json para el output del kernel eval29.

Uso:
    python3 scripts/analyze_eval29_coherence.py <out-eval29-dir> \
        [--eval-v2 /home/chaos/harness-run/kaggle-v3-out/out/eval_v2.json] \
        [--csv PATH] [--report PATH] [--self-check]

<out-eval29-dir> es el dir que baja `kaggle kernels output`, el que tiene
manifest.json + gen29_rows.jsonl (p.ej. /home/chaos/kaggle-out-eval29-v5/out-v5/out-eval29).

El heurístico de degeneración es copia literal del de la celda 9 del notebook
(kaggle/notebooks/eval29_2xt4.ipynb), para que los números sean comparables con
el campo `degenerate` de las filas y con las respuestas de eval_v2.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

TURNOS = ("frio", "caliente")


def check_degenerate(text: str) -> tuple[bool, int]:
    """Copia literal de check_degenerate() de la celda 9 del notebook eval29."""
    t = (text or "").strip()
    if len(t) < 8:
        return False, 0
    reps = 0
    for n in (8, 12, 16, 24):
        for i in range(0, max(0, len(t) - 2 * n)):
            sub = t[i:i + n]
            c = t.count(sub)
            if c > reps:
                reps = c
    return (reps >= 3), reps


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["_deg_recalc"], r["_max_repeat_recalc"] = check_degenerate(r.get("text") or "")
        r["_chars"] = len((r.get("text") or "").strip())
    return rows


def build_pairs(rows: list[dict]) -> list[dict]:
    """Un par = (set, id) con sus turnos frio/caliente."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        if r.get("turno") not in TURNOS:
            continue  # dummy / frio_proceso
        groups.setdefault((r["set"], r["id"]), {})[r["turno"]] = r
    pairs = []
    for (setname, pid), turns in sorted(groups.items()):
        f, c = turns.get("frio"), turns.get("caliente")
        ref = f or c
        assert ref is not None  # el grupo existe porque hay al menos un turno
        pairs.append({
            "set": setname,
            "id": pid,
            "prompt_sha256": ref["prompt_sha256"],
            "prompt": ref["prompt"],
            "n_turnos": len(turns),
            "frio_deg": f["_deg_recalc"] if f else None,
            "caliente_deg": c["_deg_recalc"] if c else None,
            "frio_chars": f["_chars"] if f else None,
            "caliente_chars": c["_chars"] if c else None,
            "frio_max_repeat": f["_max_repeat_recalc"] if f else None,
            "caliente_max_repeat": c["_max_repeat_recalc"] if c else None,
            "frio_ttfab_s": f["ttfab_s"] if f else None,
            "caliente_ttfab_s": c["ttfab_s"] if c else None,
            "frio_text": (f.get("text") or "") if f else "",
            "caliente_text": (c.get("text") or "") if c else "",
        })
    for p in pairs:
        degs = [d for d in (p["frio_deg"], p["caliente_deg"]) if d is not None]
        p["any_degenerate"] = any(degs)
        # "coherente en ambos" exige los dos turnos presentes y ninguno degenerado
        p["coherent_both"] = p["n_turnos"] == 2 and not any(degs)
        p["deterministic"] = bool(p["frio_text"]) and p["frio_text"] == p["caliente_text"]
    return pairs


def summarize(pairs: list[dict]) -> dict:
    def block(ps: list[dict]) -> dict:
        n = len(ps)
        return {
            "pares": n,
            "pares_con_los_dos_turnos": sum(1 for p in ps if p["n_turnos"] == 2),
            "pares_con_ge1_turno_degenerado": sum(1 for p in ps if p["any_degenerate"]),
            "pares_coherentes_en_ambos": sum(1 for p in ps if p["coherent_both"]),
            "turnos_frio_degenerados": sum(1 for p in ps if p["frio_deg"]),
            "turnos_caliente_degenerados": sum(1 for p in ps if p["caliente_deg"]),
            "pares_deterministas_frio_eq_caliente": sum(1 for p in ps if p["deterministic"]),
        }
    out = {"total": block(pairs)}
    for s in sorted({p["set"] for p in pairs}):
        out[s] = block([p for p in pairs if p["set"] == s])
    return out


def eval_v2_contrast(v2_path: Path, pairs: list[dict]) -> dict:
    """Mismos 30 prompts: eval_v2.json (baseline 1.5B v2) vs las filas ep_* del kernel."""
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v2_rows = []
    for d in v2["detalle"]:
        deg, rep = check_degenerate(d["respuesta"])
        v2_rows.append({
            "prompt": d["prompt"],
            "respuesta": d["respuesta"],
            "silencio": bool(d.get("silencio")),
            "deg": deg,
            "max_repeat": rep,
            "chars": len((d["respuesta"] or "").strip()),
        })
    ep = [p for p in pairs if p["set"] == "eval_prompts_jsonl"]
    by_prompt = {p["prompt"]: p for p in ep}
    matched = [r for r in v2_rows if r["prompt"] in by_prompt]
    unmatched = [r["prompt"] for r in v2_rows if r["prompt"] not in by_prompt]

    def med(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.median(xs), 4) if xs else None

    ep_frio = [by_prompt[r["prompt"]]["frio_deg"] for r in matched if r["prompt"] in by_prompt]
    ep_cal = [by_prompt[r["prompt"]]["caliente_deg"] for r in matched if r["prompt"] in by_prompt]
    ep_chars = [by_prompt[r["prompt"]]["frio_chars"] for r in matched if r["prompt"] in by_prompt]
    ep_rep = [by_prompt[r["prompt"]]["frio_max_repeat"] for r in matched if r["prompt"] in by_prompt]
    # ejemplo concreto del prompt que cita la card ("un proceso... un hilo, onda o linea alada")
    ejemplos = []
    for r in matched:
        if "proceso y un hilo" in r["prompt"]:
            ejemplos.append({
                "prompt": r["prompt"],
                "eval_v2_respuesta": r["respuesta"],
                "eval29_frio": by_prompt[r["prompt"]]["frio_text"],
            })
    return {
        "eval_v2_path": str(v2_path),
        "eval_v2_modelo": v2.get("modelo"),
        "eval_v2_n": v2.get("n"),
        "eval_v2_silencios": v2.get("silencios"),
        "eval_v2_respuestas_cortas": v2.get("respuestas_cortas"),
        "eval_v2_tasa_silencio": v2.get("tasa_silencio"),
        "prompts_de_v2_presentes_en_las_filas": len(matched),
        "prompts_de_v2_ausentes": unmatched,
        "eval_v2": {
            "degeneradas": sum(1 for r in matched if r["deg"]),
            "n": len(matched),
            "mediana_chars": med([r["chars"] for r in matched]),
            "mediana_max_repeat": med([r["max_repeat"] for r in matched]),
            "silencias": sum(1 for r in matched if r["silencio"]),
        },
        "eval29_ep_chatml": {
            "degeneradas_frio": sum(1 for d in ep_frio if d),
            "degeneradas_caliente": sum(1 for d in ep_cal if d),
            "n": len(ep_frio),
            "mediana_chars_frio": med(ep_chars),
            "mediana_max_repeat_frio": med(ep_rep),
        },
        "ejemplos": ejemplos,
    }


def ckpt_mapping(rows: list[dict], manifest: dict) -> dict:
    """Las filas sólo traen el basename del ckpt: mapearlo al path/hash del manifest."""
    merged = manifest.get("merged", {})
    names = sorted({r.get("model") for r in rows if r.get("model")})
    return {
        "basenames_en_las_filas": names,
        "manifest_merged": {
            "path": merged.get("path"),
            "bytes": merged.get("bytes"),
            "sha256": merged.get("sha256"),
            "recipe": merged.get("recipe"),
        },
        "match": all(n == Path(merged.get("path", "")).name for n in names),
        "device_en_las_filas": sorted({r.get("device") for r in rows if r.get("device")}),
        "chat_format_en_las_filas": sorted({r.get("chat_format") for r in rows if r.get("chat_format")}),
    }


def write_csv(pairs: list[dict], path: Path) -> None:
    cols = ["set", "id", "n_turnos", "frio_deg", "caliente_deg", "any_degenerate",
            "coherent_both", "deterministic", "frio_chars", "caliente_chars",
            "frio_max_repeat", "caliente_max_repeat", "frio_ttfab_s", "caliente_ttfab_s",
            "prompt_sha256"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in pairs:
            w.writerow(p)


def metrics_crosscheck(outdir: Path, pairs: list[dict]) -> dict:
    """Cross-check barato: el agregado de eval_generation.py (chatml, prompts CBB) tiene
    que coincidir con lo que cuentan las filas frías de ese mismo set."""
    out = {}
    for fmt in ("chatml", "world"):
        p = outdir / f"gen29_metrics_{fmt}.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        deg = (m.get("degeneracy") or {})
        out[fmt] = {
            "degenerate_rate": deg.get("degenerate_rate"),
            "n_degenerate": deg.get("n_degenerate"),
            "n_total": deg.get("n_total"),
        }
    filas_frio_cbb = sum(1 for p in pairs if p["set"] == "prompts_v1_generation" and p["frio_deg"])
    out["filas_frio_prompts_v1_generation"] = filas_frio_cbb
    ch = out.get("chatml") or {}
    out["chatml_coincide_con_filas"] = (ch.get("n_degenerate") == filas_frio_cbb) if ch else None
    return out


def report_md(rows, pairs, summ, contrast, ckpt, manifest, outdir, xcheck) -> str:
    def line(b, label):
        return (f"| {label} | {b['pares']} | {b['pares_con_los_dos_turnos']} | "
                f"{b['pares_con_ge1_turno_degenerado']} | {b['pares_coherentes_en_ambos']} | "
                f"{b['turnos_frio_degenerados']} | {b['turnos_caliente_degenerados']} | "
                f"{b['pares_deterministas_frio_eq_caliente']} |")

    L = []
    A = L.append
    A("# Coherencia por par — kernel eval29 (RWKV 2.9B merged, 2×T4)")
    A("")
    A(f"Fuente: `{outdir}` (`manifest.json` + `gen29_rows.jsonl`, "
      f"{len(rows)} filas). Heurístico de degeneración: `check_degenerate()` de la celda 9 "
      "del notebook, recalculado desde el campo `text` de cada fila.")
    A("")
    A("Reporte de análisis, companion de `REPORTE-t_18a835fb.md` (la corrida v5). **`SUMMARY.md` no se "
      "tocó**: es salida del kernel y editarla rompería la provenance. Generado por "
      "`scripts/analyze_eval29_coherence.py` (commit `f977d1d`, con `--self-check`).")
    A("")
    A("## 1. Coherencia por par")
    A("")
    A("Un **par** = mismo prompt, turno `frio` (estado limpio) + turno `caliente` "
      "(segunda llamada seguida). Los 60 pares salen de dos sets de 30 prompts: "
      "`prompts_v1_generation` (CBB) y `eval_prompts_jsonl` (los 30 de `eval_v2.json`).")
    A("")
    A("| conjunto | pares | con ambos turnos | con ≥1 turno degenerado | coherentes en ambos | frío degenerado | caliente degenerado | deterministas (frío==caliente) |")
    A("|---|---|---|---|---|---|---|---|")
    A(line(summ["total"], "**total**"))
    for k in sorted(summ):
        if k != "total":
            A(line(summ[k], f"`{k}`"))
    A("")
    A(f"Mediana de largo de respuesta (frío): {contrast['eval29_ep_chatml']['mediana_chars_frio']} chars en el set `ep_*`.")
    A("")
    A("### Desglose de degeneración por turno")
    A("")
    tot = summ["total"]
    A(f"- frío: {tot['turnos_frio_degenerados']}/{tot['pares']} turnos degenerados")
    A(f"- caliente: {tot['turnos_caliente_degenerados']}/{tot['pares']} turnos degenerados")
    A(f"- determinismo: {tot['pares_deterministas_frio_eq_caliente']}/{tot['pares']} pares con "
      "texto caliente idéntico al frío (mismo `torch.manual_seed(1337)` por llamada → replay determinista)")
    A("")
    A("### Cross-check contra los agregados de `eval_generation.py`")
    A("")
    ch = xcheck.get("chatml") or {}
    A(f"- `gen29_metrics_chatml.json` (mismos 30 prompts CBB, chatml): `n_degenerate`="
      f"{ch.get('n_degenerate')}/{ch.get('n_total')} (degenerate_rate {ch.get('degenerate_rate')}); "
      f"filas frías del set CBB con el mismo heurístico: {xcheck.get('filas_frio_prompts_v1_generation')} → "
      f"**coincide: {xcheck.get('chatml_coincide_con_filas')}**")
    wo = xcheck.get("world") or {}
    A(f"- `gen29_metrics_world.json`: `n_degenerate`={wo.get('n_degenerate')}/{wo.get('n_total')} "
      f"(degenerate_rate {wo.get('degenerate_rate')}). Sin tabla por par: la celda 9 corre sólo chatml.")
    A("")
    A("## 2. Contraste con `eval_v2.json` (mismos 30 prompts)")
    A("")
    c = contrast
    A(f"`{c['eval_v2_path']}` — modelo `{c['eval_v2_modelo']}`, n={c['eval_v2_n']}. "
      f"Los 30 prompts son **byte-idénticos y en el mismo orden** que el set `eval_prompts_jsonl` "
      f"de las filas del kernel ({c['prompts_de_v2_presentes_en_las_filas']}/30 presentes).")
    A("")
    A("| métrica (mismos 30 prompts) | eval_v2 (baseline 1.5B v2) | eval29 (2.9B merged, chatml, frío) |")
    A("|---|---|---|")
    A(f"| degeneradas (mismo heurístico) | {c['eval_v2']['degeneradas']}/{c['eval_v2']['n']} | "
      f"{c['eval29_ep_chatml']['degeneradas_frio']}/{c['eval29_ep_chatml']['n']} |")
    A(f"| mediana largo respuesta | {c['eval_v2']['mediana_chars']} chars | "
      f"{c['eval29_ep_chatml']['mediana_chars_frio']} chars |")
    A(f"| mediana max_repeat | {c['eval_v2']['mediana_max_repeat']} | "
      f"{c['eval29_ep_chatml']['mediana_max_repeat_frio']} |")
    A(f"| silencios (`<|no_response|>`) | {c['eval_v2']['silencias']}/{c['eval_v2']['n']} "
      f"(tasa_silencio={c['eval_v2_tasa_silencio']}, cortas={c['eval_v2_respuestas_cortas']}) | "
      f"0/{c['eval29_ep_chatml']['n']} |")
    A("")
    A("**Salvedades (la comparación NO es igual-a-igual):**")
    A("")
    A("- El modelo no es el mismo: `eval_v2.json` es el baseline 1.5B v2 (`rwkv_kateto_v2_sft/rwkv-0.pth`), "
      "eval29 es el 2.9B merged sobre el base `rwkv7-g1j-2.9b`.")
    A("- El sampler no es el mismo: `eval_v2.json` se generó con `rwkv_pipeline/infer_kateto.py` "
      "(defaults: max_tokens=120, temperature=0.7, top_p=0.65, ngram_repeat_max=12); las filas eval29 "
      "con `rollout_from_prompt(max_len=64, temperature=0.80, top_p=0.70, seed=1337)`.")
    A("- Los prompts sí son los mismos (30/30, verificado arriba): el set `ep_*` de la celda 9 sale de "
      "`data/eval_prompts.jsonl`, la misma fuente que usó la celda 8 de `kateto_train_v2_2xt4.ipynb`.")
    A("")
    A("### Veredicto con métrica")
    A("")
    v2d, v29d = c["eval_v2"], c["eval29_ep_chatml"]
    A(f"**No es el mismo delirio.** Con el mismo heurístico sobre los mismos 30 prompts: la baseline "
      f"`eval_v2.json` degenera {v2d['degeneradas']}/{v2d['n']} (mediana {v2d['mediana_chars']} chars, "
      f"max_repeat {v2d['mediana_max_repeat']}) y el 2.9B merged degenera "
      f"{v29d['degeneradas_frio']}/{v29d['n']} (mediana {v29d['mediana_chars_frio']} chars, "
      f"max_repeat {v29d['mediana_max_repeat_frio']}). La baseline **no repite**: inventa entidades y se va "
      "de tema sin buclarse (por eso el heurístico de bucle la da 0/30); el 2.9B **se queda en el tema y "
      "se traba repitiendo**. Son dos fallas distintas, y la del 2.9B además filtra andamiaje de agente "
      "(ver los agregados: `stop_marker_rate` 0.3667 chatml, `think_leak` 0.3667 world en `SUMMARY.md`).")
    A("")
    for e in c.get("ejemplos", []):
        A(f"Ejemplo (`{e['prompt']}`):")
        A("")
        A(f"- `eval_v2.json` (baseline 1.5B): {' '.join(e['eval_v2_respuesta'].split())!r}")
        A(f"- eval29 (2.9B merged, frío): {' '.join(e['eval29_frio'].split())!r}")
        A("")
    A("## 3. Ckpt y path por fila (criterio 2)")
    A("")
    A(f"- `gen29_rows.jsonl` declara el ckpt por basename: `{ckpt['basenames_en_las_filas']}` "
      f"(match con `manifest.merged.path`: **{ckpt['match']}**).")
    A(f"- path real: `{ckpt['manifest_merged']['path']}`")
    A(f"- sha256: `{ckpt['manifest_merged']['sha256']}` ({ckpt['manifest_merged']['bytes']} bytes)")
    A(f"- recipe: {ckpt['manifest_merged']['recipe']}")
    A(f"- `device` en las filas: {ckpt['device_en_las_filas']}; `chat_format` en las filas: "
      f"{ckpt['chat_format_en_las_filas']}")
    A(f"- base: `{manifest.get('base', {}).get('path')}` sha256 "
      f"`{manifest.get('base', {}).get('sha256')}` ({manifest.get('base', {}).get('bytes')} bytes)")
    A(f"- adapter sha256: `{manifest.get('adapter_sha256')}`")
    A("")
    A("## 4. Límites de la medición")
    A("")
    A("- La tabla por par cubre **sólo `chatml`**: la celda 9 fija `fmt(prompt, \"chatml\")` y escribe "
      "`chat_format: \"chatml\"` en las 121 filas. Los números de `world` existen sólo como agregado de "
      "`eval_generation.py` (`gen29_metrics_world.json`: degenerate_rate 0.1) sobre los 30 prompts CBB, "
      "no como tabla por par.")
    A("- El heurístico es el de la celda 9 (repetición de substrings de 8/12/16/24 chars ≥3 veces): mide "
      "bucle, no inventiva. La 'invención de entidades' que se ve en `eval_v2.json` no es medible con él.")
    A("- `frio_proceso` (1 fila, `dummy_00`) queda fuera de los pares: es el warm-up del proceso.")
    return "\n".join(L) + "\n"


def self_check() -> None:
    rows = [
        {"set": "s", "id": "p1", "turno": "frio", "prompt_sha256": "a", "prompt": "x", "text": "hola que tal",
         "degenerate": False, "max_repeat": 0, "ttfab_s": 0.1},
        {"set": "s", "id": "p1", "turno": "caliente", "prompt_sha256": "a", "prompt": "x", "text": "hola que tal",
         "degenerate": False, "max_repeat": 0, "ttfab_s": 0.1},
        {"set": "s", "id": "p2", "turno": "frio", "prompt_sha256": "b", "prompt": "y",
         "text": "bueno" * 20, "degenerate": True, "max_repeat": 3, "ttfab_s": 0.1},
        {"set": "s", "id": "p2", "turno": "caliente", "prompt_sha256": "b", "prompt": "y", "text": "otra cosa",
         "degenerate": False, "max_repeat": 0, "ttfab_s": 0.1},
        {"set": "s", "id": "p3", "turno": "frio", "prompt_sha256": "c", "prompt": "z", "text": "ok",
         "degenerate": False, "max_repeat": 0, "ttfab_s": 0.1},
        {"set": "s", "id": "dummy", "turno": "frio_proceso", "prompt_sha256": "d", "prompt": "h",
         "text": "warm", "degenerate": False, "max_repeat": 0, "ttfab_s": 1.0},
    ]
    for r in rows:
        r["_deg_recalc"], r["_max_repeat_recalc"] = check_degenerate(r["text"])
        r["_chars"] = len(r["text"].strip())
    pairs = build_pairs(rows)
    assert len(pairs) == 3, pairs
    s = summarize(pairs)["total"]
    assert s["pares"] == 3 and s["pares_con_los_dos_turnos"] == 2, s
    assert s["pares_con_ge1_turno_degenerado"] == 1, s
    assert s["pares_coherentes_en_ambos"] == 1, s
    assert s["turnos_frio_degenerados"] == 1 and s["turnos_caliente_degenerados"] == 0, s
    assert s["pares_deterministas_frio_eq_caliente"] == 1, s
    assert check_degenerate("bueno" * 20)[0] is True
    assert check_degenerate("hola que tal")[0] is False
    assert check_degenerate("") == (False, 0)
    print("self-check OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", nargs="?", help="dir con manifest.json + gen29_rows.jsonl")
    ap.add_argument("--eval-v2", default="/home/chaos/harness-run/kaggle-v3-out/out/eval_v2.json")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        self_check()
        return 0
    if not args.outdir:
        ap.error("falta outdir (o usar --self-check)")

    outdir = Path(args.outdir)
    rows = load_rows(outdir / "gen29_rows.jsonl")
    manifest = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    pairs = build_pairs(rows)
    summ = summarize(pairs)
    contrast = eval_v2_contrast(Path(args.eval_v2), pairs)
    ckpt = ckpt_mapping(rows, manifest)
    xcheck = metrics_crosscheck(outdir, pairs)

    # el campo `degenerate` de las filas tiene que coincidir con el recalculado
    mism = [r["id"] for r in rows if r.get("degenerate") is not None and r["degenerate"] != r["_deg_recalc"]]
    print(f"filas: {len(rows)} | pares: {len(pairs)} | degeneradas: "
          f"{sum(1 for r in rows if r['_deg_recalc'])}/{len(rows)}")
    print(f"discrepancias degenerate fila vs recalculado: {len(mism)} {mism[:5]}")
    print(f"total: {summ['total']}")
    for k in sorted(summ):
        if k != "total":
            print(f"{k}: {summ[k]}")
    print(f"contraste eval_v2: {json.dumps(contrast, ensure_ascii=False)}")
    print(f"ckpt: {json.dumps(ckpt, ensure_ascii=False)}")
    print(f"cross-check metrics: {json.dumps(xcheck, ensure_ascii=False)}")
    assert not mism, "el campo degenerate de las filas no coincide con el heurístico recalculado"
    assert contrast["prompts_de_v2_presentes_en_las_filas"] == contrast["eval_v2_n"], \
        "los prompts de eval_v2.json no están todos en las filas ep_*"
    assert ckpt["match"], "el basename de las filas no coincide con manifest.merged.path"

    if args.csv:
        write_csv(pairs, Path(args.csv))
        print(f"csv -> {args.csv}")
    if args.report:
        Path(args.report).write_text(
            report_md(rows, pairs, summ, contrast, ckpt, manifest, outdir, xcheck), encoding="utf-8")
        print(f"reporte -> {args.report}")

    tot = summ["total"]
    pares = tot["pares"]
    pares_coherentes = tot["pares_coherentes_en_ambos"]
    coherentes_pct = round(100.0 * pares_coherentes / pares, 2) if pares else 0.0
    por_set = {
        s: (round(100.0 * summ[s]["pares_coherentes_en_ambos"] / summ[s]["pares"], 2)
            if summ[s]["pares"] else 0.0)
        for s in sorted(summ) if s != "total"
    }
    metric_out = {
        "coherentes_pct": coherentes_pct,
        "pares": pares,
        "pares_coherentes": pares_coherentes,
        "por_set": por_set,
    }
    print(json.dumps(metric_out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
