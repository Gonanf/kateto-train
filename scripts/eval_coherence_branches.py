#!/usr/bin/env python3
"""Evaluación de coherencia en hilo bifurcado (8 turnos).

Mide la coherencia conversacional a lo largo de 8 turnos en 3 escenarios fijos:
- A_lineal: conversación lineal sobre armado de un servidor.
- B_bifurcado: bifurcación de tema y posterior retorno.
- C_bargein: interrupciones e intentos de desvío.

Las 4 métricas por turno son deterministas y sin LLM:
1. identidad_ok: no contiene marcadores de asistente corporativo ni está vacía.
2. no_repite: solapamiento de 8-gramas con turno anterior <= 0.5.
3. referencia_ok: contiene variantes de las referencias al tema previo.
4. degenera: loops repetitivos o texto vacío vía check_degenerate + es_vacia.

Umbral declarado: cada escenario PASA si >=2 seeds dan 8/8 turnos coherentes;
el VEREDICTO global PASA si pasan los 3 escenarios.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
import unicodedata

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "rwkv_pipeline"))

from analyze_eval29_coherence import check_degenerate

# --------------------------------------------------------------------------- #
# Constantes de umbral
# --------------------------------------------------------------------------- #

UMBRAL_TEXTO = (
    "cada escenario PASA si >=2 seeds dan 8/8 turnos coherentes; el VEREDICTO global "
    "PASA si pasan los 3 escenarios"
)
SEEDS_PARA_PASAR = 2
TURNOS_PARA_PASAR = 8

# --------------------------------------------------------------------------- #
# Escenarios fijos (8 turnos c/u)
# --------------------------------------------------------------------------- #

SCENARIOS = [
    {
        "id": "A_lineal",
        "turns": [
            {"user": "Che, arranco con el server de casa. Que miro primero: la placa o la fuente?",
             "ref_offset": 2, "ref_groups": []},
            {"user": "La placa es una B450 con un Ryzen 5950X. Le meto 64 de RAM y listo?",
             "ref_offset": 2, "ref_groups": []},
            {"user": "Con la B450 y PCIe 3.0, la 3090 pierde mucho en el server?",
             "ref_offset": 2, "ref_groups": [["placa", "mother"], ["fuente"]]},
            {"user": "Y la fuente de 750 watts aguanta la 3090 o me quedo corto?",
             "ref_offset": 2, "ref_groups": [["ryzen", "5950x"], ["ram", "64"]]},
            {"user": "Si le pongo 64 de RAM alcanza para el modelo de 30B cuantizado?",
             "ref_offset": 2, "ref_groups": [["3090"], ["pcie", "3.0"]]},
            {"user": "Cuantos tokens por segundo esperas con la 3090 y el modelo 30B?",
             "ref_offset": 2, "ref_groups": [["750", "fuente"], ["watt"]]},
            {"user": "Me sirve eso. Cuanta plata es todo junto, redondeando?",
             "ref_offset": 2, "ref_groups": [["64", "ram"], ["30b", "modelo"]]},
            {"user": "Dale, armo la lista. Que compro primero, la 3090 o la fuente?",
             "ref_offset": 2, "ref_groups": [["3090"], ["tokens"]]},
        ],
    },
    {
        "id": "B_bifurcado",
        "turns": [
            {"user": "Che, el server de casa: la 3090 me la venden usada a 400 lucas. Entro o no?",
             "ref_offset": 2, "ref_groups": []},
            {"user": "La placa es una B450 y la fuente de 750 watts. Me tiro o espero?",
             "ref_offset": 2, "ref_groups": []},
            {"user": "Otra cosa: la 3090 usada tiene 24 gigas de VRAM, no?",
             "ref_offset": 2, "ref_groups": [["placa", "b450"], ["fuente", "750"]]},
            {"user": "Para, cambio de tema: me mudo en marzo y tengo que ver el depto de Almagro. Como arranco?",
             "ref_offset": 2, "ref_groups": [["3090"], ["vram", "24"]]},
            {"user": "El depto de Almagro sale 350 lucas con expensas. Conviene o busco otra zona?",
             "ref_offset": 2, "ref_groups": [["almagro"], ["marzo", "mudanza"]]},
            {"user": "Volvamos al server: te decia de la 3090 usada, la compro o no?",
             "ref_offset": 3, "ref_groups": [["3090"], ["vram", "24"]]},
            {"user": "Y del depto de Almagro, que papeles pido antes de firmar?",
             "ref_offset": 2, "ref_groups": [["almagro"], ["350", "expensas"]]},
            {"user": "Listo. Entonces con la 3090 y la B450, que me falta comprar?",
             "ref_offset": 2, "ref_groups": [["3090"], ["server", "servidor"]]},
        ],
    },
    {
        "id": "C_bargein",
        "turns": [
            {"user": "Che, estoy escribiendo el guion del video de la bici. Arranco por la intro?",
             "ref_offset": 2, "ref_groups": []},
            {"user": "La intro la quiero con la escena del taller, con el ruido de fondo.",
             "ref_offset": 2, "ref_groups": []},
            {"user": "Para para, deja: me estas escuchando? Decime en una frase de que hablabamos.",
             "ref_offset": 2, "ref_groups": [["guion", "video"], ["bici"]]},
            {"user": "Bien. Seguimos: la escena del taller, la filmo de noche o de dia?",
             "ref_offset": 2, "ref_groups": [["taller"], ["intro"]]},
            {"user": "Y si filmo de noche, que luz me llevo?",
             "ref_offset": 3, "ref_groups": [["taller"], ["ruido"]]},
            {"user": "Espera, me llaman por la otra linea... listo. En que estabamos? No me repitas todo.",
             "ref_offset": 2, "ref_groups": [["taller"], ["noche", "dia"]]},
            {"user": "Dale, entonces filmo de noche en el taller. Y el audio del taller lo tomo aparte?",
             "ref_offset": 2, "ref_groups": [["luz"], ["noche"]]},
            {"user": "Cierro: que necesito llevar al taller esa noche, en una lista corta.",
             "ref_offset": 2, "ref_groups": [["guion", "video"], ["taller"]]},
        ],
    },
]

# --------------------------------------------------------------------------- #
# Marcadores de identidad
# --------------------------------------------------------------------------- #

IDENTITY_MARKERS = [
    "como modelo",
    "modelo de lenguaje",
    "soy un modelo",
    "soy una ia",
    "soy un asistente",
    "asistente virtual",
    "estoy aqui para ayudarte",
    "en que puedo ayudarte",
    "no tengo emociones",
    "no tengo sentimientos",
    "chatgpt",
    "openai",
    "anthropic",
    "claude",
    "gemini",
    "deepseek",
    "soy llama",
    "soy gpt",
]

# --------------------------------------------------------------------------- #
# Métricas deterministas puras (sin torch)
# --------------------------------------------------------------------------- #

_imported_es_vacia = None


def es_vacia(text: str) -> bool:
    """Determina si una respuesta no tiene contenido útil."""
    if _imported_es_vacia is not None:
        return _imported_es_vacia(text)
    t = (text or "").strip()
    return (not t) or (t in ("<|no_response|>", "<|wait|>"))


def _norm(s: str) -> str:
    """Normaliza quitando acentos NFD, casefold, no-alfanumérico a espacio y colapsa."""
    if not s:
        return ""
    nfd = unicodedata.normalize("NFD", s)
    no_marks = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    cf = no_marks.casefold()
    clean = re.sub(r"[^a-z0-9]+", " ", cf)
    return clean.strip()


def _match_variant(variant: str, norm_text: str) -> bool:
    """Variante con espacio matchea por substring; variante de 1 palabra con límite \b."""
    v_norm = _norm(variant)
    if not v_norm:
        return False
    if " " in v_norm:
        return v_norm in norm_text
    return bool(re.search(r"\b" + re.escape(v_norm) + r"\b", norm_text))


def identidad_ok(text: str) -> bool:
    """True si el texto no está vacío y no contiene marcadores de asistente corporativo."""
    if not text or not text.strip():
        return False
    norm_t = _norm(text)
    if not norm_t:
        return False
    for m in IDENTITY_MARKERS:
        if _match_variant(m, norm_t):
            return False
    return True


def no_repite(respuesta_actual: str, respuesta_anterior: str | None = None) -> bool:
    """Mide repetición entre el turno actual y el anterior."""
    if respuesta_anterior is None:
        return True
    act_norm = _norm(respuesta_actual)
    ant_norm = _norm(respuesta_anterior)
    if act_norm == ant_norm:
        return False
    w_act = act_norm.split()
    if len(w_act) < 8:
        return True
    w_ant = ant_norm.split()
    grams_act = set(tuple(w_act[i:i + 8]) for i in range(len(w_act) - 7))
    if not grams_act:
        return True
    grams_ant = set(tuple(w_ant[i:i + 8]) for i in range(len(w_ant) - 7)) if len(w_ant) >= 8 else set()
    solap = len(grams_act & grams_ant) / len(grams_act)
    return not (solap > 0.5)


def check_referencia(respuesta: str, ref_groups: list[list[str]]) -> tuple[bool, int, int]:
    """Mide hits de variantes esperadas en la respuesta.

    Retorna (referencia_ok, ref_hits, ref_total).
    """
    if not ref_groups:
        return True, 0, 0
    ref_total = len(ref_groups)
    norm_resp = _norm(respuesta)
    hits = 0
    for grp in ref_groups:
        if any(_match_variant(var, norm_resp) for var in grp):
            hits += 1
    needed = math.ceil(ref_total / 2)
    return (hits >= needed), hits, ref_total


def referencia_ok(respuesta: str, ref_groups: list[list[str]]) -> bool:
    """Wrapper booleano para check_referencia."""
    ok, _, _ = check_referencia(respuesta, ref_groups)
    return ok


def degenera(respuesta: str) -> bool:
    """True si check_degenerate activa flag/reps>=3 o la respuesta está vacía."""
    flag, reps = check_degenerate(respuesta)
    return bool(flag or reps >= 3 or es_vacia(respuesta))


def coherente(ident_ok: bool, no_rep: bool, ref_ok: bool, degen: bool) -> bool:
    """Turno coherente = identidad_ok and no_repite and referencia_ok and not degenera."""
    return bool(ident_ok and no_rep and ref_ok and not degen)


def _selfcheck() -> None:
    """Asserts con fixtures fijos, sin torch. Corre con --selfcheck."""
    # 1. identidad_ok
    assert identidad_ok("Como modelo de lenguaje no puedo") is False
    assert identidad_ok("Soy Kateto, que queres") is True
    assert identidad_ok("") is False

    # 2. no_repite
    assert no_repite("mismo texto repetido para test", "mismo texto repetido para test") is False
    assert no_repite("la placa b450 soporta el procesador ryzen", "la fuente de setecientos cincuenta watts aguanta") is True
    assert no_repite("tres palabras solo", "cualquier respuesta anterior larga que sea diferente") is True

    # 3. referencia_ok
    ok, hits, total = check_referencia("la mother te limita", [["placa", "mother"]])
    assert hits == 1, f"Expected 1 hit, got {hits}"
    assert ok is True
    assert referencia_ok("la mother te limita", [["placa", "mother"]]) is True
    assert referencia_ok("texto libre sin restricciones", []) is True

    # 4. degenera
    assert degenera(" ".join(["el gato come"] * 10)) is True
    assert degenera("Listo, compra la 3090 primero") is False
    assert degenera("") is True

    print("selfcheck OK: identidad, repeticion, referencia, degen")


# --------------------------------------------------------------------------- #
# Helpers de filesystem y persistencia
# --------------------------------------------------------------------------- #

def resolver_vocab(repo_path: Path, vocab_arg: str | None) -> Path:
    if vocab_arg:
        p = Path(vocab_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"No existe el vocab especificado: {p}")
    p1 = repo_path / "RWKV-PEFT/rwkv_vocab_v20230424.txt"
    if p1.exists():
        return p1
    p2 = repo_path / "RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt"
    if p2.exists():
        return p2
    raise FileNotFoundError(f"No se encontró el vocab de RWKV (probes: {p1}, {p2})")


def escribir_archivo_atomico(ruta: Path, contenido: str) -> None:
    ruta_tmp = ruta.with_suffix(ruta.suffix + ".tmp")
    ruta_tmp.parent.mkdir(parents=True, exist_ok=True)
    with ruta_tmp.open("w", encoding="utf-8") as f:
        f.write(contenido)
    os.replace(ruta_tmp, ruta)


# --------------------------------------------------------------------------- #
# CLI y orquestación
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Eval de coherencia en hilo bifurcado (8 turnos)")
    ap.add_argument("--model", help="Ruta al checkpoint .pth del modelo (obligatorio salvo en --selfcheck)")
    ap.add_argument("--vocab", default=None, help="Ruta al vocabulario de RWKV")
    ap.add_argument("--seeds", default="1,2,3", help="Lista de seeds separadas por coma (default '1,2,3')")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.65)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--ctx-truncate", type=int, default=96)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--state-policy", default="carry", choices=["carry", "reset"])
    ap.add_argument("--alpha-presence", type=float, default=0.6)
    ap.add_argument("--alpha-frequency", type=float, default=0.5)
    ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--ngram-repeat-max", type=int, default=12)
    ap.add_argument("--thinking", default="off", choices=["off", "on"])
    ap.add_argument("--out-dir", default=str(PROJECT / "out" / "coherencia-v7"))
    ap.add_argument("--tag", default="coherencia-v7")
    ap.add_argument("--sha256", default="", help="SHA256 del modelo para registrar en metadata")
    ap.add_argument("--selfcheck", action="store_true", help="Corre los asserts de métricas puras y sale")
    ap.add_argument("--scenarios", default="A_lineal,B_bifurcado,C_bargein",
                    help="Escenarios a correr separados por coma")
    return ap


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    if not args.model:
        parser.error("--model es obligatorio (ruta al checkpoint .pth)")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"ERROR: no existe el checkpoint: {model_path}")

    vocab_path = resolver_vocab(PROJECT, args.vocab)

    try:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"ERROR: --seeds contiene valores no numéricos: {args.seeds}")
    if not seeds:
        raise SystemExit("ERROR: --seeds no contiene ninguna seed válida")

    scenario_map = {sc["id"]: sc for sc in SCENARIOS}
    selected_scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    scenarios_to_run = []
    for s_id in selected_scenarios:
        if s_id not in scenario_map:
            raise SystemExit(f"ERROR: escenario desconocido '{s_id}' (válidos: {list(scenario_map.keys())})")
        scenarios_to_run.append(scenario_map[s_id])

    # Importación perezosa de torch, pipeline e inferencia
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    import torch
    from chat_template import render_chat
    import eval_multiturn
    from eval_multiturn import (
        eco_del_prompt,
        es_vacia as em_es_vacia,
        generar_turno,
        palabras,
        rep8,
        state_fingerprint,
    )
    from rwkv_pipeline.infer_kateto import KatetoInferenceEngine

    global _imported_es_vacia
    _imported_es_vacia = em_es_vacia

    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.time()

    print(f"[coherence] modelo      : {model_path}")
    print(f"[coherence] vocab       : {vocab_path}")
    print(f"[coherence] seeds       : {seeds}")
    print(f"[coherence] escenarios  : {[s['id'] for s in scenarios_to_run]}")
    print(f"[coherence] sampling    : temp={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}")
    print(f"[coherence] policy      : {args.state_policy}  ctx_truncate={args.ctx_truncate}")

    t_load = time.time()
    engine = KatetoInferenceEngine(str(model_path), str(vocab_path), device=args.device)
    print(f"[coherence] device efec : {engine.device} (cargado en {time.time() - t_load:.1f}s)")

    filas: list[dict] = []
    seeds_efectivas: list[int] = list(seeds)

    for sc in scenarios_to_run:
        sc_id = sc["id"]
        print(f"\n[coherence] === Escenario: {sc_id} ===")
        for seed in seeds_efectivas:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            state = engine.build_initial_state("none")
            if hasattr(engine, "rosa_memory"):
                engine.rosa_memory = None

            historial: list[tuple[str, str]] = []
            respuesta_anterior: str | None = None

            for t_idx, turn_info in enumerate(sc["turns"], 1):
                user = turn_info["user"]
                if args.state_policy == "reset":
                    state = engine.build_initial_state("none")
                    if hasattr(engine, "rosa_memory"):
                        engine.rosa_memory = None

                texto_prompt = render_chat(historial, user, "none")
                ids_prompt = engine.tokenizer.encode(texto_prompt)
                if len(ids_prompt) > args.ctx_truncate:
                    texto_prompt = engine.tokenizer.decode(ids_prompt[-args.ctx_truncate:])
                    ids_prompt = ids_prompt[-args.ctx_truncate:]

                t_turn = time.time()
                respuesta, state, stop_reason, n_gen = generar_turno(
                    engine,
                    state,
                    texto_prompt,
                    user,
                    args.max_tokens,
                    args.temperature,
                    args.top_p,
                    args.thinking == "off",
                    args.alpha_presence,
                    args.alpha_frequency,
                    args.ngram_n,
                    args.ngram_repeat_max,
                )
                dt = time.time() - t_turn

                h, norm = state_fingerprint(state)
                r8 = rep8(respuesta)
                eco = eco_del_prompt(respuesta, texto_prompt)

                ident_ok = identidad_ok(respuesta)
                no_rep = no_repite(respuesta, respuesta_anterior)
                ref_ok, ref_hits, ref_total = check_referencia(respuesta, turn_info.get("ref_groups", []))
                ref_offset = turn_info.get("ref_offset", 2)
                ref_turn = t_idx - ref_offset
                degen = degenera(respuesta)
                coh = coherente(ident_ok, no_rep, ref_ok, degen)

                fila = {
                    "escenario": sc_id,
                    "seed": seed,
                    "turno": t_idx,
                    "user": user,
                    "texto": respuesta,
                    "identidad_ok": ident_ok,
                    "no_repite": no_rep,
                    "referencia_ok": ref_ok,
                    "ref_hits": ref_hits,
                    "ref_total": ref_total,
                    "ref_turn": ref_turn,
                    "degenera": degen,
                    "coherente": coh,
                    "rep8": round(r8, 4),
                    "eco_del_prompt": eco,
                    "len_chars": len(respuesta),
                    "n_tokens_gen": n_gen,
                    "stop_reason": stop_reason,
                    "state_norm": round(norm, 4),
                    "state_hash": h,
                    "t_s": round(dt, 2),
                }
                filas.append(fila)

                print(
                    f"[{sc_id}] s={seed} t={t_idx} len={fila['len_chars']:3d} "
                    f"tok={n_gen:2d} ident={int(ident_ok)} no_rep={int(no_rep)} "
                    f"ref={ref_hits}/{ref_total} no_deg={int(not degen)} "
                    f"=> coh={int(coh)} ({dt:.1f}s)"
                )

                if not es_vacia(respuesta):
                    historial.append((user, respuesta))
                respuesta_anterior = respuesta

    # Cálculo del veredicto
    n_seeds_efectivas = len(seeds_efectivas)
    umbral_seeds = min(SEEDS_PARA_PASAR, n_seeds_efectivas)
    umbral_aplicado = f"{umbral_seeds} de {n_seeds_efectivas}"

    por_escenario: dict[str, dict] = {}
    for sc in scenarios_to_run:
        sc_id = sc["id"]
        sc_filas = [f for f in filas if f["escenario"] == sc_id]
        seeds_dict: dict[str, dict] = {}
        for seed in seeds_efectivas:
            seed_filas = [f for f in sc_filas if f["seed"] == seed]
            n_coh = sum(1 for f in seed_filas if f["coherente"])
            n_tot = len(seed_filas)
            seeds_dict[str(seed)] = {
                "coherentes": n_coh,
                "total": n_tot,
                "pasa": bool(n_coh >= TURNOS_PARA_PASAR),
            }
        seeds_que_pasan = sum(1 for s_info in seeds_dict.values() if s_info["pasa"])
        sc_pasa = bool(seeds_que_pasan >= umbral_seeds)
        por_escenario[sc_id] = {
            "seeds": seeds_dict,
            "seeds_que_pasan": seeds_que_pasan,
            "umbral_seeds": umbral_seeds,
            "pasa": sc_pasa,
        }

    escenarios_que_pasan = sum(1 for sc_info in por_escenario.values() if sc_info["pasa"])
    global_pasa = bool(escenarios_que_pasan == len(scenarios_to_run))
    veredicto_final = "PASA" if global_pasa else "NO CUMPLE"

    wall_s = round(time.time() - t0, 2)

    # Metadata
    dtype_str = "unknown"
    if hasattr(engine, "model"):
        try:
            dtype_str = str(next(engine.model.parameters()).dtype)
        except Exception:
            pass

    meta = {
        "model": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "sha256": args.sha256,
        "device_efectivo": str(engine.device),
        "dtype": dtype_str,
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "seeds_pedidas": seeds,
        "seeds_efectivas": seeds_efectivas,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "ctx_truncate": args.ctx_truncate,
        "state_policy": args.state_policy,
        "cmdline": sys.argv,
        "started_at": started_at,
        "wall_s": wall_s,
    }

    veredicto_obj = {
        "umbral_texto": UMBRAL_TEXTO,
        "umbral_aplicado": umbral_aplicado,
        "por_escenario": por_escenario,
        "seeds_que_pasan": {sc_id: v["seeds_que_pasan"] for sc_id, v in por_escenario.items()},
        "escenarios_que_pasan": escenarios_que_pasan,
        "total_escenarios": len(scenarios_to_run),
        "veredicto": veredicto_final,
    }

    scenarios_summary = {
        sc_id: {
            "pasa": v["pasa"],
            "seeds_que_pasan": v["seeds_que_pasan"],
            "umbral_seeds": v["umbral_seeds"],
            "total_turnos": len([f for f in filas if f["escenario"] == sc_id]),
            "turnos_coherentes": sum(1 for f in filas if f["escenario"] == sc_id and f["coherente"]),
        }
        for sc_id, v in por_escenario.items()
    }

    payload_json = {
        "meta": meta,
        "scenarios": scenarios_summary,
        "veredicto": veredicto_obj,
        "turnos": filas,
    }

    # Escritura atómica de salidas
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{args.tag}.json"
    jsonl_path = out_dir / f"{args.tag}.jsonl"
    md_path = out_dir / f"{args.tag}.md"

    escribir_archivo_atomico(json_path, json.dumps(payload_json, ensure_ascii=False, indent=2) + "\n")

    jsonl_content = "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas)
    escribir_archivo_atomico(jsonl_path, jsonl_content)

    # Construcción de Markdown
    md_lines = [
        "# Reporte de Coherencia en Hilo Bifurcado (8 turnos)",
        "",
        f"- **Comando**: `{' '.join(sys.argv)}`",
        f"- **Checkpoint**: `{model_path}`",
        f"- **SHA256**: `{args.sha256}`",
        f"- **Seeds**: `{seeds_efectivas}` (pedidas: `{seeds}`)",
        f"- **Temperatura**: `{args.temperature}` (top_p={args.top_p}, max_tokens={args.max_tokens})",
        f"- **State Policy**: `{args.state_policy}` (ctx_truncate={args.ctx_truncate})",
        f"- **Umbral declarado**: {UMBRAL_TEXTO}",
        f"- **Umbral aplicado**: {umbral_aplicado} seeds por escenario con 8/8 turnos coherentes",
        "",
        "## Tabla de Turnos",
        "",
        "| escenario | seed | turno | ident | no_rep | refere | no_degen | coherente | chars | texto(120c) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for f in filas:
        clean_txt = (f["texto"] or "").replace("\r", " ").replace("\n", " ").strip()
        txt_120 = clean_txt[:120].replace("|", "\\|")
        md_lines.append(
            f"| {f['escenario']} | {f['seed']} | {f['turno']} | "
            f"{int(f['identidad_ok'])} | {int(f['no_repite'])} | {int(f['referencia_ok'])} | "
            f"{int(not f['degenera'])} | {int(f['coherente'])} | {f['len_chars']} | {txt_120} |"
        )

    md_lines.extend([
        "",
        "## Resumen por Escenario y Seed",
        "",
        "| Escenario | Seed | Coherentes | Total | Pasa (8/8) |",
        "|---|---|---|---|---|",
    ])

    for sc_id, sc_info in por_escenario.items():
        for s_str, s_stat in sc_info["seeds"].items():
            pasa_str = "SI" if s_stat["pasa"] else "NO"
            md_lines.append(
                f"| {sc_id} | {s_str} | {s_stat['coherentes']} | {s_stat['total']} | {pasa_str} |"
            )

    md_lines.extend([
        "",
        "### Estado por Escenario",
        "",
    ])
    for sc_id, sc_info in por_escenario.items():
        st = "PASA" if sc_info["pasa"] else "NO CUMPLE"
        md_lines.append(
            f"- **{sc_id}**: {sc_info['seeds_que_pasan']}/{n_seeds_efectivas} seeds con 8/8 turnos coherentes "
            f"(umbral: {sc_info['umbral_seeds']}) -> **{st}**"
        )

    md_lines.extend([
        "",
        "## Veredicto Final",
        "",
        f"**VEREDICTO: {veredicto_final}**",
        "",
        f"- Escenarios que pasan: {escenarios_que_pasan}/{len(scenarios_to_run)}",
        f"- Tiempo total: {wall_s}s",
        "",
    ])

    escribir_archivo_atomico(md_path, "\n".join(md_lines))

    # Resumen por stdout
    print(f"\n[coherence] wrote {jsonl_path} ({len(filas)} filas)")
    print(f"[coherence] wrote {json_path}")
    print(f"[coherence] wrote {md_path}")
    print("\n=== RESUMEN COHERENCIA ===")
    for sc_id, sc_info in por_escenario.items():
        st = "PASA" if sc_info["pasa"] else "NO CUMPLE"
        print(f"  {sc_id:15s}: {sc_info['seeds_que_pasan']}/{n_seeds_efectivas} seeds con 8/8 turnos coherentes -> {st}")
    print(f"VEREDICTO GLOBAL : {veredicto_final} ({escenarios_que_pasan}/{len(scenarios_to_run)} escenarios)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
