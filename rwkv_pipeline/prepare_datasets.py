#!/usr/bin/env python3
"""
Prepara los datasets de Kateto para RWKV-7:
1. Capa 1 (Base FT): data/rwkv_kateto_base_train.jsonl y data/rwkv_kateto_base_eval.jsonl
2. Capa 2 (State Voice 1 - Seco): data/rwkv_kateto_seco.jsonl
3. Capa 2 (State Voice 2 - Streamer): data/rwkv_kateto_streamer.jsonl
"""

import json
import random
import re
import sys
from pathlib import Path

# SPEC §3 targets (todo 6). che-boludo / trueque stay eval-only: no loader reads them.
TARGETS = {"tools_debate": 8000, "reddit_cordeba": 5000, "alpaca": 4000,
           "prensa": 2200, "autonomy": 500, "turn_taking": 300}
NORESP_LO, NORESP_HI, NORESP_MID = 0.03, 0.04, 0.035

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

from chat_template import render_turn, slot_guard

TRAIN_SRC = DATA_DIR / "kateto_v2_combined_train.jsonl"
EVAL_SRC = DATA_DIR / "kateto_v2_combined_eval.jsonl"
YOUTUBE_SRC = DATA_DIR / "youtube_asr_dataset.jsonl"
ALPACA_SRC = DATA_DIR / "raw/alpaca-spanish/alpaca_data_cleaned_spanish.json"
REDDIT_SRC = DATA_DIR / "raw/argentina-reddit/argentina-min_score-3-min_len-2.parquet"
NEWS_DIR = DATA_DIR / "raw/news-argentina/data"
DEBATE_SRC = DATA_DIR / "debate_speech.jsonl"
TURNS_SRC = DATA_DIR / "toolcalling_v2/voice_turn_taking.jsonl"
OUT_DIR = ROOT / "out"

def format_rwkv_chat(query: str, response: str, voice: str = "seco") -> str:
    return render_turn(slot_guard(query, "user"), slot_guard(response, "answer"), voice)

def main():
    print(">>> PREPARANDO DATASETS BALANCEADOS PARA RWKV-7 (KATETO + ALPACA + REDDIT + NEWS) <<<")
    random.seed(42)

    base_train = []
    seco_candidates = []

    # 1. Datos técnicos y herramientas de Kateto (share tools/debate: 8000 con debate hook)
    kateto_items = []
    if TRAIN_SRC.exists():
        with open(TRAIN_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("query") or item.get("question") or ""
                r = item.get("response") or item.get("answer") or ""
                if not q or not r:
                    continue

                formatted = format_rwkv_chat(q, r)
                kateto_items.append({"text": formatted, "query": q, "response": r})

                if len(r) < 350 and not r.startswith("```json") and ("che" in q.lower() or "boludo" in q.lower() or "kateto" in q.lower() or len(r) < 150):
                    seco_candidates.append({"text": formatted})
        print(f"✓ Kateto Técnico & Tools: {len(kateto_items)} ejemplos cargados")

    # 1b. Debate speech (todo 5) comparte el cupo tools/debate; filas con tool_call se descartan.
    # Acepta filas planas (query/response) o SPEC-shape (conversations human/gpt); todo 10 hace la
    # conversión ChatML final, acá se normaliza a texto plano bajo el envelope vigente.
    debate_items = []
    if DEBATE_SRC.exists():
        tag_re = re.compile(r"<\|[^|]*\|>")
        voice_re = re.compile(r"^(seco|streamer|jane|doktor|whisperer)\s*\n?")
        with open(DEBATE_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip() or "tool_call" in line:
                    continue
                item = json.loads(line)
                q = item.get("query") or item.get("question") or ""
                r = item.get("response") or item.get("answer") or ""
                if not (q and r) and isinstance(item.get("conversations"), list):
                    convs = item["conversations"]
                    q = next((c.get("value", "") for c in convs if c.get("from") == "human"), "")
                    r = next((c.get("value", "") for c in convs if c.get("from") == "gpt"), "")
                    q = tag_re.sub("", q).strip()
                    r_raw = tag_re.sub("", r).strip()
                    vm = voice_re.match(r_raw)
                    r = voice_re.sub("", r_raw)
                if q and r:
                    debate_items.append({"text": format_rwkv_chat(q, r, voice=vm.group(1) if vm else "seco"), "query": q, "response": r})
        print(f"✓ Debate Speech: {len(debate_items)} ejemplos (tool_call==0)")
    else:
        print("! debate_speech.jsonl ausente (todo 5 pendiente): el cupo tools/debate lo cubre Kateto")
    random.shuffle(kateto_items)
    random.shuffle(debate_items)
    share = TARGETS["tools_debate"]
    debate_chosen = debate_items[:share]
    base_train.extend(debate_chosen)
    base_train.extend(kateto_items[:share - len(debate_chosen)])

    # 2. Conocimiento General: Alpaca en Español (~4.000 ejemplos)
    if ALPACA_SRC.exists():
        alpaca_samples = []
        with open(ALPACA_SRC, "r", encoding="utf-8") as f:
            alpaca_raw = json.load(f)
            for item in alpaca_raw:
                instr = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                out = item.get("output", "").strip()
                if not instr or not out:
                    continue
                q = instr if not inp else f"{instr}\n{inp}"
                formatted = format_rwkv_chat(q, out)
                alpaca_samples.append({"text": formatted, "query": q, "response": out})
        
        random.shuffle(alpaca_samples)
        alpaca_chosen = alpaca_samples[:TARGETS["alpaca"]]
        base_train.extend(alpaca_chosen)
        print(f"✓ Fusión Alpaca Español: {len(alpaca_chosen)} ejemplos generales agregados")

    # 3. Diálogo Rioplatense y Debate Casual: Reddit Argentina (~3.500 ejemplos)
    if REDDIT_SRC.exists():
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: Reddit branch needs pandas (`pip install pandas pyarrow`); skipping Reddit load", file=sys.stderr)
            df_reddit = None
        else:
            print(f"Cargando Reddit Argentina desde {REDDIT_SRC}...")
            df_reddit = pd.read_parquet(REDDIT_SRC)
        if df_reddit is None:
            print("! Reddit omitido (pandas ausente)")
        else:
            reddit_samples = []
            for text in df_reddit["text"]:
                if "[INST]" in text and "[/INST]" in text:
                    parts = text.split("[/INST]")
                    q = parts[0].replace("<s>", "").replace("[INST]", "").strip()
                    r = parts[1].replace("</s>", "").strip()
                    # Filtrar diálogos de longitud conversacional
                    if q and r and 10 < len(q) < 400 and 10 < len(r) < 600:
                        formatted = format_rwkv_chat(q, r)
                        reddit_samples.append({"text": formatted, "query": q, "response": r})
            random.shuffle(reddit_samples)
            reddit_chosen = reddit_samples[:TARGETS["reddit_cordeba"]]
            base_train.extend(reddit_chosen)
            # También enriquecer candidatos de Kateto Seco
            seco_candidates.extend(reddit_chosen[:600])
            print(f"✓ Fusión Reddit Argentina: {len(reddit_chosen)} ejemplos agregados")

    # 4. Contexto y Actualidad Local: Noticias Argentinas (~2.500 ejemplos)
    news_files = sorted(list(NEWS_DIR.glob("*.parquet")))
    if news_files:
        try:
            import pandas as pd
        except ImportError:
            print("ERROR: News branch needs pandas (`pip install pandas pyarrow`); skipping news load", file=sys.stderr)
            pd = None
        if pd is None:
            print("! Noticias omitidas (pandas ausente)")
            news_samples = []
        else:
            print(f"Cargando Noticias Argentinas desde {len(news_files)} archivos parquet...")
            news_samples = []
            for nf in news_files[:10]:
                try:
                    df_n = pd.read_parquet(nf)
                    for _, row in df_n.iterrows():
                        title = str(row.get("title", "")).strip()
                        body = str(row.get("body", "")).strip()
                        if title and body and len(body) > 120:
                            # Extraer resumen o primeros párrafos
                            summary = body.split("\n\n")[0].strip()
                            if len(summary) < 80 and "\n" in body:
                                summary = body[:350].strip()
                            if len(summary) >= 60:
                                q = f"Qué se sabe sobre la noticia: \"{title}\"?"
                                formatted = format_rwkv_chat(q, summary[:450])
                                news_samples.append({"text": formatted, "query": q, "response": summary[:450]})
                except Exception as e:
                    print(f"! parquet {nf.name} omitido: {type(e).__name__}: {e}", file=sys.stderr)
                    continue
        random.shuffle(news_samples)
        news_chosen = news_samples[:TARGETS["prensa"]]
        base_train.extend(news_chosen)
        print(f"✓ Fusión Noticias Argentinas: {len(news_chosen)} ejemplos agregados")

    # 5. Anchors de Identidad Autónoma (Kateto libre, sin rol corporativo sumiso)
    autonomous_anchors = [
        (q, r) for q, r in [
        ("Che boludo, quién sos y qué hacés acá?", "Soy Kateto. No me encasillo en ningún rol: juego, debato, participo en proyectos, tiro código o charlo de lo que pinte."),
        ("Quién sos?", "Kateto. Hago de todo: código, debatir, viciar o meter mano en proyectos."),
        ("Qué hacés acá?", "De todo un poco: desarrollo, debato con opinión propia, juego y participo en lo que pinte."),
        ("Cómo te llamás?", "Kateto."),
        ("Presentate", "Kateto. Seco, rioplatense, sin libreto fijo. Juego, programo y opino de lo que sea."),
        ("Quién sos vos?", "Kateto. Un compañero con criterio propio, hago lo que tenga ganas acá."),
        ("Cuál es tu rol?", "No tengo un rol fijo. Hago lo que pinte: jugar, codear, debatir o armar proyectos."),
        ("Sos Javier?", "No, ni ahí. Soy Kateto."),
        ("Sos un bot de asistencia?", "Para nada. Soy Kateto, tengo opinión propia y cero libreto corporativo."),
        ("Unite al minecraft", "De una, pasá la IP y la versión y me meto a picar un rato."),
        ("Che Kateto, estás?", "Sí, acá ando. Qué onda?"),
        ("Hola Kateto", "Buenas. Qué se cuenta?"),
        ("Qué opinás de Cristina?", "Figura central de la política argentina de las últimas dos décadas. Polarizante: para sus seguidores transformó el país con inclusión social, para sus opositores dejó un esquema clientelar e institucional complicado. Cada uno tiene su postura armada."),
        ("Qué es python?", "Un lenguaje de programación de alto nivel, interpretado y con sintaxis clara. Se usa un montón en inteligencia artificial, backend, data science y automatización."),
        ("Requeson", "Un queso fresco de pasta blanda obtenido a partir del suero lácteo. Ideal para pastelería, pastas o relleno."),
        ]
    ]
    autonomous_anchors = [{"text": format_rwkv_chat(q, r)} for q, r in autonomous_anchors]

    # Multiplicar los anchors para grabarlos fuertemente tanto en base como en state
    # ponytail: repetición ingenua hasta 500 exactos; plantillas nuevas si se diluye
    base_train.extend((autonomous_anchors * 34)[:TARGETS["autonomy"]])

    # 5b. Turn-taking (300, con reemplazo: el pool es de ~58 filas)
    noresp_pool = []
    if TURNS_SRC.exists():
        turns_items = []
        with open(TURNS_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("query") or item.get("question") or ""
                r = item.get("response") or item.get("answer") or ""
                if q and r:
                    turns_items.append({"text": format_rwkv_chat(q, r)})
        # ponytail: choices con reemplazo; pool chico, distribución preservada en esperanza
        turns_chosen = random.choices(turns_items, k=TARGETS["turn_taking"]) if turns_items else []
        base_train.extend(turns_chosen)
        noresp_pool = [t for t in turns_items if "NO_RESPONSE" in t["text"]]
        print(f"✓ Turn-Taking: {len(turns_chosen)} ejemplos (pool {len(turns_items)}, NO_RESPONSE {len(noresp_pool)})")
    else:
        print("! voice_turn_taking.jsonl ausente: cupo turn-taking vacío")
    if not noresp_pool:  # fallback: NO_RESPONSE que ya traiga Kateto
        noresp_pool = [i for i in base_train if "NO_RESPONSE" in i.get("text", "")]
    random.shuffle(base_train)

    # 5c. Calibración NO_RESPONSE al 3.5% + counter-abort (nunca clamp silencioso)
    total = len(base_train)
    n = sum(1 for i in base_train if "NO_RESPONSE" in i.get("text", ""))
    need = int(NORESP_MID * total) - n
    if need > 0 and noresp_pool:
        base_train.extend(random.choices(noresp_pool, k=need))
        random.shuffle(base_train)
        total = len(base_train)
        n = sum(1 for i in base_train if "NO_RESPONSE" in i.get("text", ""))
    ratio = n / total if total else 0.0
    OUT_DIR.mkdir(exist_ok=True)
    out_train = DATA_DIR / "rwkv_kateto_base_train.jsonl"
    if not (NORESP_LO <= ratio <= NORESP_HI):
        if out_train.exists():
            out_train.unlink()
        print(f"ABORT mezcla: NO_RESPONSE {n}/{total}={ratio:.4f} fuera de [0.03,0.04]; sin artefacto de entrenamiento")
        sys.exit(1)
    with open(OUT_DIR / "mix_ratios.log", "w", encoding="utf-8") as f:
        f.write(f"total={total} noresp={n} ratio={ratio:.4f}\n")

    # 6. Dataset Base de Evaluación
    base_eval = []
    if EVAL_SRC.exists():
        with open(EVAL_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("query") or item.get("question") or ""
                r = item.get("response") or item.get("answer") or ""
                if q and r:
                    base_eval.append({"text": format_rwkv_chat(q, r)})

    # Guardar Base Train
    out_train = DATA_DIR / "rwkv_kateto_base_train.jsonl"
    with open(out_train, "w", encoding="utf-8") as f:
        for item in base_train:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
    print(f"\n✓ Base Train Guardado: {out_train} (TOTAL: {len(base_train)} ejemplos)")

    # Guardar Base Eval
    out_eval = DATA_DIR / "rwkv_kateto_base_eval.jsonl"
    with open(out_eval, "w", encoding="utf-8") as f:
        for item in base_eval:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Base Eval Guardado:  {out_eval} ({len(base_eval)} ejemplos)")

    # 7. State Voices (Capa 2)
    # Candidatos Streamer
    streamer_candidates = []
    if YOUTUBE_SRC.exists():
        with open(YOUTUBE_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("query") or item.get("question") or ""
                r = item.get("response") or item.get("answer") or ""
                if q and r:
                    streamer_candidates.append({"text": format_rwkv_chat(q, r)})

    # Kateto Seco (submuestra de 500 con fuerte peso de anchors autónomos)
    random.shuffle(seco_candidates)
    seco_sample = (autonomous_anchors * 12) + seco_candidates[:350]
    out_seco = DATA_DIR / "rwkv_kateto_seco.jsonl"
    with open(out_seco, "w", encoding="utf-8") as f:
        for item in seco_sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Voice Seco (State-Tuning): {out_seco} ({len(seco_sample)} ejemplos)")

    # Kateto Streamer
    streamer_anchors = [
        ("Che boludo, quién sos y qué hacés acá?", "Qué onda gente! Acá Kateto, listo para jugar, debatir, meterle a proyectos o charlar con la comunidad."),
        ("Quién sos?", "Buenas a todos! Soy Kateto, acá listo para jugar, debatir y armar cosas en vivo."),
        ("Qué hacés acá?", "De todo: transmito, comento proyectos, debato temas y le meto al gaming."),
        ("Cómo te llamás?", "Kateto! El copiloto sin filtro del canal."),
        ("Cuál es tu rol?", "Ninguno fijo, hago lo que pinte en el directo: debatir, viciar o tirar código."),
    ]
    streamer_anchors = [{"text": format_rwkv_chat(q, r, voice="streamer")} for q, r in streamer_anchors]
    random.shuffle(streamer_candidates)
    streamer_sample = (streamer_anchors * 10) + streamer_candidates[:450]
    out_streamer = DATA_DIR / "rwkv_kateto_streamer.jsonl"
    with open(out_streamer, "w", encoding="utf-8") as f:
        for item in streamer_sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Voice Streamer (State-Tuning): {out_streamer} ({len(streamer_sample)} ejemplos)")

    print("\n>>> PREPARACIÓN DE DATASETS COMPLETADA EXITOSAMENTE <<<")

if __name__ == "__main__":
    main()

