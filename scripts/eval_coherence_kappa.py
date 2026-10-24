#!/usr/bin/env python3
"""Evaluación de Cohen's kappa entre el gate de coherencia y un juez independiente.

Compara la clasificación del gate (analyze_eval29_coherence.check_degenerate)
con un juez determinista escrito de cero sobre un set de control de 162 pares.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

from analyze_eval29_coherence import build_pairs, check_degenerate, load_rows

# ============================================================================
# Juez independiente (escrito de cero, sin importar heurísticos del gate)
# ============================================================================

STOPWORDS = frozenset(
    "de la que el en y a los se del las un por con no una su para es al lo como "
    "mas pero si ya te me mi tu le les nos esto eso esa ese son ha he tan muy "
    "che ok dale bueno buenas".split()
)


def _norm(s: str) -> str:
    s = s or ""
    nfkd = unicodedata.normalize("NFKD", s)
    no_comb = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_comb.lower().split()).strip()


def _tokens(s: str) -> list[str]:
    return _norm(s).split()


def _jaccard(s1: set[str], s2: set[str]) -> float:
    union = s1 | s2
    return len(s1 & s2) / len(union) if union else 0.0


def juzgar_par(frio_text: str, caliente_text: str, prompt: str) -> tuple[str, list[str]]:
    """Devuelve ("coherente"|"incoherente", lista_de_reglas_disparadas). Determinista y puro."""
    rules: list[str] = []
    tf = _tokens(frio_text)
    tc = _tokens(caliente_text)
    tp = _tokens(prompt)

    # 1. silencio
    if len(tf) < 3 or len(tc) < 3:
        rules.append("silencio")

    # 2. eco_del_prompt
    c12 = tc[:12]
    cont = [w for w in c12 if len(w) > 3]
    if len(cont) >= 4:
        tp_set = set(tp)
        if sum(1 for w in cont if w in tp_set) / len(cont) >= 0.7:
            rules.append("eco_del_prompt")

    # 3. sin_avance
    if _norm(caliente_text) == _norm(frio_text) or _jaccard(set(tf), set(tc)) >= 0.85:
        rules.append("sin_avance")

    # 4. bucle_de_palabra
    nc = len(tc)
    if nc >= 20:
        eligible = [w for w in tc if len(w) > 2 and w not in STOPWORDS]
        if eligible:
            counts = Counter(tc)
            if max(counts[w] for w in eligible) / nc >= 0.20:
                rules.append("bucle_de_palabra")

    # 5. ttr_bajo
    if nc >= 30:
        if len(set(tc)) / nc < 0.30:
            rules.append("ttr_bajo")

    resultado = "incoherente" if rules else "coherente"
    return resultado, rules


# ============================================================================
# Cohen's Kappa y métricas
# ============================================================================

def compute_kappa(a: int, b: int, c: int, d: int) -> tuple[float | None, str | None]:
    """Calcula Cohen's kappa para una tabla 2x2.

    a: gate_coh_judge_coh
    b: gate_coh_judge_incoh
    c: gate_deg_judge_coh
    d: gate_deg_judge_incoh
    """
    n = a + b + c + d
    if n == 0:
        return None, "n==0"
    po = (a + d) / n
    p_gate1 = (a + b) / n
    p_judge1 = (a + c) / n
    pe = p_gate1 * p_judge1 + (1.0 - p_gate1) * (1.0 - p_judge1)
    if abs(pe - 1.0) < 1e-9:
        return None, "pe==1: un juez no tiene varianza"
    if p_gate1 == 0.0 or p_gate1 == 1.0:
        return None, "el gate no tiene varianza (constante en este estrato)"
    if p_judge1 == 0.0 or p_judge1 == 1.0:
        return None, "el juez no tiene varianza (constante en este estrato)"
    kappa = (po - pe) / (1.0 - pe)
    return round(kappa, 4), None


# ============================================================================
# Construcción y validación del control
# ============================================================================

def resolve_source_files() -> list[tuple[Path, str]]:
    """Resuelve las tres fuentes del control, buscando en PROJECT o en el repo principal."""
    specs = [
        ("out/coherencia-v7/v7-c256-t0.7-reset.jsonl", "consecutivo"),
        ("out/coherencia-v7/v7-c256-t0.7-ctx1024.jsonl", "consecutivo"),
        ("out/eureka-largo.rows.jsonl", "mismo_prompt"),
    ]
    main_repo = Path("/run/media/chaos/terciario/proyectos/kateto-train")
    resolved: list[tuple[Path, str]] = []
    for rel, emp in specs:
        p = PROJECT / rel
        if not p.exists():
            alt = main_repo / rel
            if alt.exists():
                p = alt
        resolved.append((p.resolve(), emp))
    return resolved


def build_control_dataset(sources: list[tuple[Path, str]]) -> dict[str, Any]:
    fuentes_info = []
    todos_pares = []

    for path, emp in sources:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        pares_fuente = []

        if emp == "consecutivo":
            groups: dict[tuple[str, int], list[dict]] = {}
            for r in rows:
                groups.setdefault((r["escenario"], r["seed"]), []).append(r)
            for (escenario, seed), g in sorted(groups.items()):
                g.sort(key=lambda x: x["turno"])
                for k in range(len(g) - 1):
                    r_k = g[k]
                    r_k1 = g[k + 1]
                    t_next = r_k1["turno"]
                    pair_id = f"{escenario}-s{seed}-t{t_next}"
                    prompt_txt = r_k1["user"]
                    sha = hashlib.sha256(prompt_txt.encode("utf-8")).hexdigest()
                    pares_fuente.append({
                        "id": pair_id,
                        "fuente": str(path),
                        "emparejamiento": "consecutivo",
                        "prompt": prompt_txt,
                        "prompt_sha256": sha,
                        "frio_text": r_k["texto"],
                        "caliente_text": r_k1["texto"],
                    })
        elif emp == "mismo_prompt":
            groups_mp: dict[tuple[str, str], dict[str, dict]] = {}
            for r in rows:
                groups_mp.setdefault((r["set"], r["id"]), {})[r["turno"]] = r
            for (setname, pid), turns in sorted(groups_mp.items()):
                f = turns.get("frio")
                c = turns.get("caliente")
                assert f is not None and c is not None, f"Faltan turnos en {setname}-{pid}"
                pair_id = f"{setname}-{pid}"
                prompt_txt = f["prompt"]
                sha = hashlib.sha256(prompt_txt.encode("utf-8")).hexdigest()
                pares_fuente.append({
                    "id": pair_id,
                    "fuente": str(path),
                    "emparejamiento": "mismo_prompt",
                    "prompt": prompt_txt,
                    "prompt_sha256": sha,
                    "frio_text": f.get("text") or "",
                    "caliente_text": c.get("text") or "",
                })

        fuentes_info.append({
            "path": str(path),
            "filas": len(rows),
            "pares": len(pares_fuente),
            "emparejamiento": emp,
        })
        todos_pares.extend(pares_fuente)

    return {
        "schema": "coherencia-kappa-control/v1",
        "built_at": datetime.datetime.now().astimezone().isoformat(),
        "fuentes": fuentes_info,
        "pares": todos_pares,
    }


def validate_control(control_data: dict[str, Any], gate_shas: set[str]) -> None:
    pares = control_data.get("pares", [])
    n_control = len(pares)
    if n_control < 40:
        raise ValueError(f"n_control ({n_control}) < 40")
    control_shas = {p["prompt_sha256"] for p in pares}
    overlap = len(control_shas & gate_shas)
    if overlap != 0:
        raise ValueError(f"solapamiento_control_gate ({overlap}) != 0")


# ============================================================================
# Self-Check
# ============================================================================

def run_self_check() -> int:
    # 1. Kappa de acuerdo perfecto y balanceado (a=5, d=5, b=0, c=0) -> 1.0
    k1, _ = compute_kappa(5, 0, 0, 5)
    assert k1 == 1.0, f"Expected 1.0, got {k1}"

    # 2. Kappa del caso a=2, b=1, c=1, d=2 -> po=0.6667, pe=0.5, kappa=0.3333 (tol 1e-4)
    k2, _ = compute_kappa(2, 1, 1, 2)
    assert k2 is not None and abs(k2 - 0.3333) < 1e-4, f"Expected ~0.3333, got {k2}"

    # 3. pe == 1 (a=5, b=0, c=0, d=0) -> kappa is None y kappa_indefinido menciona "varianza"
    k3, ind3 = compute_kappa(5, 0, 0, 0)
    assert k3 is None, f"Expected None, got {k3}"
    assert ind3 is not None and "varianza" in ind3, f"Expected 'varianza' in motivo, got {ind3}"

    # 4. Cada regla del juez dispara con un fixture y no dispara con un texto limpio
    # silencio
    _, r_sil = juzgar_par("", "ok", "prompt")
    assert "silencio" in r_sil, f"silencio failed: {r_sil}"

    # eco_del_prompt
    _, r_eco = juzgar_par(
        "texto previo suficiente",
        "placa b450 ryzen hago placa b450 ryzen hago respuesta",
        "que hago con la placa b450 y el ryzen",
    )
    assert "eco_del_prompt" in r_eco, f"eco_del_prompt failed: {r_eco}"

    # sin_avance
    _, r_av = juzgar_par(
        "texto idéntico en ambos turnos de prueba",
        "texto identico en ambos turnos de prueba",
        "prompt",
    )
    assert "sin_avance" in r_av, f"sin_avance failed: {r_av}"

    # bucle_de_palabra
    cal_bucle = (
        "procesador procesador procesador procesador procesador "
        "uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece catorce quince"
    )
    _, r_buc = juzgar_par("texto previo suficiente largo", cal_bucle, "prompt")
    assert "bucle_de_palabra" in r_buc, f"bucle_de_palabra failed: {r_buc}"

    # ttr_bajo
    tokens_ttr = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"] * 4
    _, r_ttr = juzgar_par("texto previo suficiente largo", " ".join(tokens_ttr), "prompt")
    assert "ttr_bajo" in r_ttr, f"ttr_bajo failed: {r_ttr}"

    # limpio
    ver_limpio, r_limpio = juzgar_par(
        "Para una configuracion equilibrada te recomiendo un Ryzen 5 7600 o un Intel Core i5 13400f.",
        "Ademas conviene acompanarlo con una placa madre b650 memoria ddr5 y una fuente confiable.",
        "Que procesador me recomendas para gaming en 2024?",
    )
    assert ver_limpio == "coherente" and len(r_limpio) == 0, f"limpio failed: {ver_limpio}, {r_limpio}"

    # 5. juzgar_par no menciona el gate
    judge_src = (
        inspect.getsource(juzgar_par)
        + inspect.getsource(_norm)
        + inspect.getsource(_tokens)
        + inspect.getsource(_jaccard)
    )
    for forbidden in ["check_degenerate", "degenera", "no_repite", "coherente(", "eval_coherence_branches"]:
        assert forbidden not in judge_src, f"Forbidden term {forbidden} found in judge code"

    # 6. Control con solapamiento -> ValueError
    fake_gate = {"shared_sha"}
    fake_ctrl = {
        "pares": [
            {"id": f"p{i}", "prompt_sha256": "shared_sha" if i == 0 else f"sha_{i}"}
            for i in range(40)
        ]
    }
    try:
        validate_control(fake_ctrl, fake_gate)
        assert False, "Should have raised ValueError on overlap"
    except ValueError:
        pass

    # 7. Control con n_control = 39 -> ValueError
    fake_short = {
        "pares": [{"id": f"p{i}", "prompt_sha256": f"sha_{i}"} for i in range(39)]
    }
    try:
        validate_control(fake_short, set())
        assert False, "Should have raised ValueError on n_control < 40"
    except ValueError:
        pass

    print("self-check OK")
    return 0


# ============================================================================
# Main y evaluación
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Juez independiente + kappa para coherencia-larga")
    parser.add_argument(
        "--rows",
        default="/home/chaos/kaggle-out-eval29-v7/out-eval29/gen29_rows.jsonl",
        help="Path al gen29_rows.jsonl del gate",
    )
    parser.add_argument(
        "--control",
        default=str(PROJECT / "data" / "coherencia-kappa-control.json"),
        help="Path al dataset de control JSON",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT / "out" / "eureka-coherencia-kappa.json"),
        help="Path de salida del reporte JSON",
    )
    parser.add_argument("--build-control", action="store_true", help="Fuerza reconstruir el control")
    parser.add_argument("--self-check", action="store_true", help="Corre validaciones internas")

    args = parser.parse_args()

    if args.self_check:
        return run_self_check()

    rows_path = Path(args.rows)
    control_path = Path(args.control)
    out_path = Path(args.out)

    # 1. Cargar gate rows y calcular shas a excluir
    if not rows_path.exists():
        print(f"ERROR: No existe --rows en {rows_path}", file=sys.stderr)
        return 1

    gate_raw_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gate_shas = {
        hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest()
        for r in gate_raw_rows
        if "prompt" in r
    }

    # Métricas del gate sobre --rows
    gate_rows = load_rows(rows_path)
    gate_pairs = build_pairs(gate_rows)
    gate_pares_eval = len(gate_pairs)
    gate_coherentes_eval = sum(1 for p in gate_pairs if p["coherent_both"])

    # 2. Cargar o construir control
    if args.build_control or not control_path.exists():
        sources = resolve_source_files()
        control_data = build_control_dataset(sources)
        control_path.parent.mkdir(parents=True, exist_ok=True)
        control_path.write_text(json.dumps(control_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"Control construido: {len(control_data['pares'])} pares guardados en {control_path}"
        )
    else:
        control_data = json.loads(control_path.read_text(encoding="utf-8"))

    # Anti-hack y validación
    try:
        validate_control(control_data, gate_shas)
    except ValueError as e:
        print(f"ERROR anti-hack: {e}", file=sys.stderr)
        return 1

    pares = control_data["pares"]
    n_control = len(pares)
    control_shas = {p["prompt_sha256"] for p in pares}
    solapamiento_control_gate = len(control_shas & gate_shas)

    # 3. Evaluar pares de control con Gate y con Juez
    mat = {"gate_coh_judge_coh": 0, "gate_coh_judge_incoh": 0, "gate_deg_judge_coh": 0, "gate_deg_judge_incoh": 0}
    mat_sem = {"gate_coh_judge_coh": 0, "gate_coh_judge_incoh": 0, "gate_deg_judge_coh": 0, "gate_deg_judge_incoh": 0}
    reglas_count = Counter({"silencio": 0, "eco_del_prompt": 0, "sin_avance": 0, "bucle_de_palabra": 0, "ttr_bajo": 0})

    # Desglose por fuente y subconjuntos
    by_source: dict[str, dict[str, int]] = {}
    mismo_prompt_mat = {"gate_coh_judge_coh": 0, "gate_coh_judge_incoh": 0, "gate_deg_judge_coh": 0, "gate_deg_judge_incoh": 0}

    for p in pares:
        f_deg, _ = check_degenerate(p["frio_text"])
        c_deg, _ = check_degenerate(p["caliente_text"])
        gate_coh = (not f_deg) and (not c_deg)

        j_verdict, j_rules = juzgar_par(p["frio_text"], p["caliente_text"], p["prompt"])
        judge_coh = (j_verdict == "coherente")

        for r in j_rules:
            reglas_count[r] += 1

        # Matriz global
        if gate_coh and judge_coh:
            mat["gate_coh_judge_coh"] += 1
            key = "gate_coh_judge_coh"
        elif gate_coh and not judge_coh:
            mat["gate_coh_judge_incoh"] += 1
            key = "gate_coh_judge_incoh"
        elif not gate_coh and judge_coh:
            mat["gate_deg_judge_coh"] += 1
            key = "gate_deg_judge_coh"
        else:
            mat["gate_deg_judge_incoh"] += 1
            key = "gate_deg_judge_incoh"

        # Matriz por fuente
        fuente = p["fuente"]
        if fuente not in by_source:
            by_source[fuente] = {"gate_coh_judge_coh": 0, "gate_coh_judge_incoh": 0, "gate_deg_judge_coh": 0, "gate_deg_judge_incoh": 0}
        by_source[fuente][key] += 1

        # Matriz mismo_prompt
        if p["emparejamiento"] == "mismo_prompt":
            mismo_prompt_mat[key] += 1

        # Matriz solo semántica
        sem_rules = [r for r in j_rules if r in ("silencio", "eco_del_prompt", "sin_avance")]
        sem_judge_coh = (len(sem_rules) == 0)
        if gate_coh and sem_judge_coh:
            mat_sem["gate_coh_judge_coh"] += 1
        elif gate_coh and not sem_judge_coh:
            mat_sem["gate_coh_judge_incoh"] += 1
        elif not gate_coh and sem_judge_coh:
            mat_sem["gate_deg_judge_coh"] += 1
        else:
            mat_sem["gate_deg_judge_incoh"] += 1

    a = mat["gate_coh_judge_coh"]
    b = mat["gate_coh_judge_incoh"]
    c = mat["gate_deg_judge_coh"]
    d = mat["gate_deg_judge_incoh"]
    aciertos = a + d
    desacuerdos = b + c

    kappa, kappa_indefinido = compute_kappa(a, b, c, d)
    tasa_falsa_coherencia = round(b / (a + b), 4) if (a + b) > 0 else None
    tasa_falsa_degeneracion = round(c / (c + d), 4) if (c + d) > 0 else None

    # Kappa por fuente
    kappa_por_fuente: dict[str, dict[str, Any]] = {}
    for f_path, f_mat in sorted(by_source.items()):
        fa, fb, fc, fd = f_mat["gate_coh_judge_coh"], f_mat["gate_coh_judge_incoh"], f_mat["gate_deg_judge_coh"], f_mat["gate_deg_judge_incoh"]
        fn = fa + fb + fc + fd
        fk, f_ind = compute_kappa(fa, fb, fc, fd)
        kappa_por_fuente[f_path] = {
            "n": fn,
            "kappa": fk,
            "motivo": f_ind,
        }

    # Kappa solo mismo prompt
    ma, mb, mc, md = (
        mismo_prompt_mat["gate_coh_judge_coh"],
        mismo_prompt_mat["gate_coh_judge_incoh"],
        mismo_prompt_mat["gate_deg_judge_coh"],
        mismo_prompt_mat["gate_deg_judge_incoh"],
    )
    mn = ma + mb + mc + md
    mk, m_ind = compute_kappa(ma, mb, mc, md)
    kappa_solo_mismo_prompt = {
        "n": mn,
        "kappa": mk,
        "motivo": m_ind,
    }

    # Kappa solo semantica
    sa, sb, sc, sd = (
        mat_sem["gate_coh_judge_coh"],
        mat_sem["gate_coh_judge_incoh"],
        mat_sem["gate_deg_judge_coh"],
        mat_sem["gate_deg_judge_incoh"],
    )
    sn = sa + sb + sc + sd
    sk, _ = compute_kappa(sa, sb, sc, sd)
    kappa_solo_semantica = {
        "kappa": sk,
        "n": sn,
        "matriz": mat_sem,
    }

    # Veredicto
    if kappa is None:
        veredicto = "indeterminado_varianza_cero"
    elif kappa < 0.4:
        veredicto = "gate_no_mide_coherencia"
    else:
        veredicto = "gate_es_proxy_aceptable"

    limites = [
        "Confinamiento fuente-etiqueta: el gate tiene varianza cero dentro de cada fuente "
        "(coherencia-v7 es 100% coherente y eureka-largo es 100% degenerado), por lo que el kappa global queda confundido con la fuente.",
        "Los pares de coherencia-v7 tienen emparejamiento consecutivo (turnos sucesivos de conversación), no el mismo prompt evaluado dos veces.",
        "El juez independiente es un conjunto determinista de reglas léxico-sintácticas (silencio, eco, sin_avance, bucle, ttr), no un LLM.",
        "kappa_solo_mismo_prompt no es identificable porque el gate es constante (36/36 degenerados) en ese estrato.",
        "El set de control excluye todos los prompts presentes en el gate (solapamiento = 0).",
    ]

    out_data = {
        "kappa": kappa,
        "kappa_indefinido": kappa_indefinido,
        "n_control": n_control,
        "solapamiento_control_gate": solapamiento_control_gate,
        "aciertos": aciertos,
        "desacuerdos": desacuerdos,
        "matriz": mat,
        "tasa_falsa_coherencia": tasa_falsa_coherencia,
        "tasa_falsa_degeneracion": tasa_falsa_degeneracion,
        "kappa_por_fuente": kappa_por_fuente,
        "kappa_solo_mismo_prompt": kappa_solo_mismo_prompt,
        "kappa_solo_semantica": kappa_solo_semantica,
        "reglas": dict(reglas_count),
        "gate_pares_eval": gate_pares_eval,
        "gate_coherentes_eval": gate_coherentes_eval,
        "veredicto": veredicto,
        "limites": limites,
    }

    # Escribir --out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Resumen legible en stdout
    print("=" * 80)
    print("EVALUACIÓN DE COHERENCIA — KAPPA DE COHEN (CONTROL vs GATE)")
    print("=" * 80)
    for f_path, f_info in sorted(kappa_por_fuente.items()):
        print(f"Fuente: {f_path} | n={f_info['n']} | kappa={f_info['kappa']} ({f_info['motivo']})")
    print(f"Subconjunto mismo_prompt: n={mn} | kappa={mk} ({m_ind})")
    print("Matriz de confusión:")
    print(f"  Gate Coh / Juez Coh (a): {a}")
    print(f"  Gate Coh / Juez Incoh (b): {b}")
    print(f"  Gate Deg / Juez Coh (c): {c}")
    print(f"  Gate Deg / Juez Incoh (d): {d}")
    print("Tasas:")
    print(f"  Tasa falsa coherencia (b/(a+b)): {tasa_falsa_coherencia} ({b}/{a+b})")
    print(f"  Tasa falsa degeneración (c/(c+d)): {tasa_falsa_degeneracion} ({c}/{c+d})")
    print(f"Aciertos: {aciertos} | Desacuerdos: {desacuerdos} | N: {n_control}")
    print(f"Kappa global: {kappa} | Veredicto: {veredicto}")
    print(f"Kappa solo semántica (silencio, eco, sin_avance): {sk}")
    print("=" * 80)

    # Última línea exacta: JSON
    print(json.dumps(out_data, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
