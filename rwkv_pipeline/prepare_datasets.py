#!/usr/bin/env python3
"""
Prepara los datasets de Kateto para RWKV-7 (saneado, filtros de admisión, ASR afuera, esquema único):
1. Capa 1 (Base FT): data/rwkv_kateto_base_train.jsonl y data/rwkv_kateto_base_train.fmtA.jsonl
   Eval: data/rwkv_kateto_base_eval.jsonl
2. Capa 2 (State Voice 1 - Seco): data/rwkv_kateto_seco.jsonl y .fmtA.jsonl
3. Capa 2 (State Voice 2 - Streamer): data/rwkv_kateto_streamer.jsonl y .fmtA.jsonl
"""

import json
import random
import re
import sys
import shutil
from pathlib import Path

# Configurar búsqueda de admission_filter
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from admission_filter import check_admission

# Resolver raíz y directorio de datos
# Funciona desde kaggle_upload/rwkv_pipeline, rwkv_pipeline o raíz del repo
if (HERE.parent / "data" / "kateto_v2_combined_train.jsonl").exists():
    ROOT = HERE.parent
    DATA_DIR = ROOT / "data"
elif (HERE.parent.parent / "data" / "kateto_v2_combined_train.jsonl").exists():
    ROOT = HERE.parent.parent
    DATA_DIR = ROOT / "data"
elif (HERE.parent / "data").exists():
    ROOT = HERE.parent
    DATA_DIR = ROOT / "data"
else:
    ROOT = HERE.parent.parent
    DATA_DIR = ROOT / "data"

OUT_DIR = ROOT / "out"
QUARANTINE_FILE = DATA_DIR / "quarantine.jsonl"

# SPEC §3 targets
TARGETS = {"tools_debate": 8000, "reddit_cordeba": 5000, "alpaca": 4000,
           "prensa": 2200, "autonomy": 500, "turn_taking": 300}
NORESP_LO, NORESP_HI, NORESP_MID = 0.03, 0.04, 0.035

TRAIN_SRC = DATA_DIR / "kateto_v2_combined_train.jsonl"
EVAL_SRC = DATA_DIR / "kateto_v2_combined_eval.jsonl"
# youtube_asr_dataset.jsonl queda fuera de las fuentes
ALPACA_SRC = DATA_DIR / "raw/alpaca-spanish/alpaca_data_cleaned_spanish.json"
REDDIT_SRC = DATA_DIR / "raw/argentina-reddit/argentina-min_score-3-min_len-2.parquet"
NEWS_DIR = DATA_DIR / "raw/news-argentina/data"
DEBATE_SRC = DATA_DIR / "debate_speech.jsonl"
TURNS_SRC = DATA_DIR / "toolcalling_v2/voice_turn_taking.jsonl"
KATETO_QA_SRC = DATA_DIR / "kateto_qa.v2-3k.jsonl"
ASR_RAW_SRC = DATA_DIR / "youtube_asr_dataset.jsonl"

def format_rwkv_chat(query: str, response: str, voice: str = "seco") -> str:
    """Formato canónico ChatML (fmtA)."""
    q = query.strip()
    r = response.strip()
    if r.endswith("<|endoftext|>"):
        r = r[:-len("<|endoftext|>")].strip()
    return f"<|im_user|>{q}<|im_end|>\n<|im_start|>{voice}\n{r}<|im_end|>"

def clean_row_dict(formatted_text: str, query: str, response: str, source_label: str = "", scenario: str = "") -> dict:
    """Retorna fila con esquema único: text, query, response (+ metadatos permitidos).
    Elimina tajantemente claves question/answer.
    """
    q = query.strip()
    r = response.strip()
    if r.endswith("<|endoftext|>"):
        r = r[:-len("<|endoftext|>")].strip()
    row = {
        "text": formatted_text,
        "query": q,
        "response": r,
    }
    if source_label:
        row["source_label"] = source_label
    if scenario:
        row["scenario"] = scenario
    return row

seen_quarantine = set()

def init_quarantine():
    global seen_quarantine
    seen_quarantine = set()
    if QUARANTINE_FILE.exists():
        with open(QUARANTINE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    it = json.loads(line)
                    seen_quarantine.add((it.get("query", "").strip(), it.get("response", "").strip()))
                except Exception:
                    pass

def append_to_quarantine(q: str, r: str, source: str, filt: str, detail: str):
    q_str = q.strip()
    r_str = r.strip()
    if (q_str, r_str) in seen_quarantine:
        return
    seen_quarantine.add((q_str, r_str))
    record = {
        "query": q_str,
        "response": r_str,
        "source_label": source,
        "quarantine_filter": filt,
        "quarantine_detail": detail,
    }
    with open(QUARANTINE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    print(">>> PREPARANDO DATASETS SANEADOS PARA RWKV-7 (KATETO + DEBATE + VOCES) <<<")
    random.seed(42)
    init_quarantine()

    # 0. Cargar bloque de firmas ASR para verificación real anti-fuga
    asr_blocklist = set()
    if ASR_RAW_SRC.exists():
        with open(ASR_RAW_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                for k in ("query", "question", "response", "answer"):
                    v = item.get(k)
                    if v and isinstance(v, str) and len(v.strip()) > 10:
                        asr_blocklist.add(v.strip().lower())
        print(f"✓ Bloque anti-fuga ASR cargado: {len(asr_blocklist)} firmas monitoreadas")

    # 1. Guillotina sobre kateto_qa.v2-3k (evaluación y cuarentena)
    if KATETO_QA_SRC.exists():
        qa_total = 0
        qa_survived = 0
        qa_filter_counts = {}
        with open(KATETO_QA_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                qa_total += 1
                item = json.loads(line)
                q = item.get("question") or item.get("query") or ""
                a = item.get("answer") or item.get("response") or ""
                ok, filt, detail = check_admission(q, a)
                if ok:
                    qa_survived += 1
                else:
                    qa_filter_counts[filt] = qa_filter_counts.get(filt, 0) + 1
                    append_to_quarantine(q, a, "kateto_qa.v2-3k", filt, detail or "")
        qa_pct = (qa_survived / qa_total) * 100 if qa_total else 0
        print(f"✓ Guillotina kateto_qa.v2-3k: {qa_survived}/{qa_total} ({qa_pct:.1f}%) sobrevivientes")
        if qa_pct < 40.0:
            print(f"  -> Regla aplicada (< 40%): archivo descartado entero (0 filas al entrenamiento)")
        else:
            print(f"  -> Archivo admitido para QA grounded")

    base_train = []
    seco_candidates = []
    streamer_candidates = []

    # 2. Datos técnicos y herramientas de Kateto (kateto_v2_combined_train)
    kateto_items = []
    if TRAIN_SRC.exists():
        with open(TRAIN_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("query") or item.get("question") or ""
                r = item.get("response") or item.get("answer") or ""
                lbl = item.get("source_label", "")
                if not q or not r:
                    continue

                # Anti-ASR check
                if "youtube-asr" in lbl or q.strip().lower() in asr_blocklist or r.strip().lower() in asr_blocklist:
                    append_to_quarantine(q, r, lbl or "kateto_combined_train", "asr_leak", "matches_asr_blocklist")
                    continue

                # Admisión
                ok, filt, detail = check_admission(q, r)
                if not ok:
                    append_to_quarantine(q, r, lbl or "kateto_combined_train", filt, detail or "")
                    continue

                formatted = format_rwkv_chat(q, r)
                row_obj = clean_row_dict(formatted, q, r, source_label=lbl, scenario=item.get("scenario", ""))
                kateto_items.append(row_obj)

                if len(r) < 350 and not r.startswith("```json") and ("che" in q.lower() or "boludo" in q.lower() or "kateto" in q.lower() or len(r) < 150):
                    seco_candidates.append(row_obj)
        print(f"✓ Kateto Técnico & Tools: {len(kateto_items)} ejemplos cargados (saneados)")

    # 3. Debate speech
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
                voice = "seco"
                if not (q and r) and isinstance(item.get("conversations"), list):
                    convs = item["conversations"]
                    q = next((c.get("value", "") for c in convs if c.get("from") == "human"), "")
                    r = next((c.get("value", "") for c in convs if c.get("from") == "gpt"), "")
                    q = tag_re.sub("", q).strip()
                    r_raw = tag_re.sub("", r).strip()
                    vm = voice_re.match(r_raw)
                    if vm:
                        voice = vm.group(1)
                    r = voice_re.sub("", r_raw)

                if q and r:
                    ok, filt, detail = check_admission(q, r)
                    if not ok:
                        append_to_quarantine(q, r, "debate_speech", filt, detail or "")
                        continue
                    row_obj = clean_row_dict(format_rwkv_chat(q, r, voice=voice), q, r, source_label="debate", scenario=voice)
                    debate_items.append(row_obj)
                    if voice == "streamer":
                        streamer_candidates.append(row_obj)
        print(f"✓ Debate Speech: {len(debate_items)} ejemplos (tool_call==0, saneados)")
    else:
        print("! debate_speech.jsonl ausente")

    random.shuffle(kateto_items)
    random.shuffle(debate_items)
    share = TARGETS["tools_debate"]
    debate_chosen = debate_items[:share]
    base_train.extend(debate_chosen)
    base_train.extend(kateto_items[:share - len(debate_chosen)])

    # 4. Conocimiento General: Alpaca en Español
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
                ok, filt, detail = check_admission(q, out)
                if not ok:
                    append_to_quarantine(q, out, "alpaca_spanish", filt, detail or "")
                    continue
                formatted = format_rwkv_chat(q, out)
                alpaca_samples.append(clean_row_dict(formatted, q, out, source_label="alpaca-spanish"))
        random.shuffle(alpaca_samples)
        alpaca_chosen = alpaca_samples[:TARGETS["alpaca"]]
        base_train.extend(alpaca_chosen)
        print(f"✓ Fusión Alpaca Español: {len(alpaca_chosen)} ejemplos agregados")

    # 5. Diálogo Rioplatense y Debate Casual: Reddit Argentina
    if REDDIT_SRC.exists():
        try:
            import pandas as pd
            df_reddit = pd.read_parquet(REDDIT_SRC)
        except Exception:
            df_reddit = None
        if df_reddit is not None:
            reddit_samples = []
            for text in df_reddit["text"]:
                if "[INST]" in text and "[/INST]" in text:
                    parts = text.split("[/INST]")
                    q = parts[0].replace("<s>", "").replace("[INST]", "").strip()
                    r = parts[1].replace("</s>", "").strip()
                    if q and r and 10 < len(q) < 400 and 10 < len(r) < 600:
                        ok, filt, detail = check_admission(q, r)
                        if not ok:
                            append_to_quarantine(q, r, "reddit_cordeba", filt, detail or "")
                            continue
                        formatted = format_rwkv_chat(q, r)
                        reddit_samples.append(clean_row_dict(formatted, q, r, source_label="reddit-cordeba"))
            random.shuffle(reddit_samples)
            reddit_chosen = reddit_samples[:TARGETS["reddit_cordeba"]]
            base_train.extend(reddit_chosen)
            seco_candidates.extend(reddit_chosen[:600])
            print(f"✓ Fusión Reddit Argentina: {len(reddit_chosen)} ejemplos agregados")

    # 6. Anchors de Identidad Autónoma
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
    autonomous_anchor_objs = [
        clean_row_dict(format_rwkv_chat(q, r), q, r, source_label="autonomous-anchor")
        for q, r in autonomous_anchors
    ]
    base_train.extend((autonomous_anchor_objs * 34)[:TARGETS["autonomy"]])

    # 7. Turn-taking
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
                    ok, filt, detail = check_admission(q, r)
                    if not ok:
                        append_to_quarantine(q, r, "turn_taking", filt, detail or "")
                        continue
                    turns_items.append(clean_row_dict(format_rwkv_chat(q, r), q, r, source_label="turn-taking"))
        # Calibración dinámica del cupo de turn-taking según tamaño de base_train
        current_base_len = len(base_train)
        k_target = min(TARGETS["turn_taking"], int((NORESP_MID * current_base_len) / (1 - NORESP_MID)))
        turns_chosen = random.choices(turns_items, k=k_target) if turns_items else []
        base_train.extend(turns_chosen)
        noresp_pool = [t for t in turns_items if "NO_RESPONSE" in t["text"]]
        print(f"✓ Turn-Taking: {len(turns_chosen)} ejemplos (pool {len(turns_items)}, NO_RESPONSE {len(noresp_pool)})")
    else:
        print("! voice_turn_taking.jsonl ausente")

    if not noresp_pool:
        noresp_pool = [i for i in base_train if "NO_RESPONSE" in i.get("text", "")]
    random.shuffle(base_train)

    # 8. Calibración estricta NO_RESPONSE al 3.5% (entre 3.0% y 4.0%)
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
    out_train_fmta = DATA_DIR / "rwkv_kateto_base_train.fmtA.jsonl"
    if not (NORESP_LO <= ratio <= NORESP_HI):
        if out_train.exists():
            out_train.unlink()
        print(f"ABORT mezcla: NO_RESPONSE {n}/{total}={ratio:.4f} fuera de [0.03,0.04]; sin artefacto de entrenamiento")
        sys.exit(1)

    with open(OUT_DIR / "mix_ratios.log", "w", encoding="utf-8") as f:
        f.write(f"total={total} noresp={n} ratio={ratio:.4f}\n")

    # 9. Dataset Base de Evaluación
    base_eval = []
    if EVAL_SRC.exists():
        with open(EVAL_SRC, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("query") or item.get("question") or ""
                r = item.get("response") or item.get("answer") or ""
                lbl = item.get("source_label", "")
                if q and r:
                    if "youtube-asr" in lbl or q.strip().lower() in asr_blocklist or r.strip().lower() in asr_blocklist:
                        append_to_quarantine(q, r, lbl or "kateto_eval", "asr_leak", "matches_asr_blocklist")
                        continue
                    ok, filt, detail = check_admission(q, r)
                    if not ok:
                        append_to_quarantine(q, r, lbl or "kateto_eval", filt, detail or "")
                        continue
                    base_eval.append(clean_row_dict(format_rwkv_chat(q, r), q, r, source_label=lbl, scenario=item.get("scenario", "")))

    # Guardar Base Train (ambos .jsonl y .fmtA.jsonl con formato canónico y esquema único)
    for target_path in (out_train, out_train_fmta):
        with open(target_path, "w", encoding="utf-8") as f:
            for item in base_train:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n✓ Base Train Guardado: {out_train} (TOTAL: {len(base_train)} ejemplos, ratio NO_RESPONSE: {ratio:.4f})")
    print(f"✓ Base Train fmtA:     {out_train_fmta}")

    # Guardar Base Eval
    out_eval = DATA_DIR / "rwkv_kateto_base_eval.jsonl"
    with open(out_eval, "w", encoding="utf-8") as f:
        for item in base_eval:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Base Eval Guardado:  {out_eval} ({len(base_eval)} ejemplos)")

    # 10. State Voices (Capa 2)
    # Kateto Seco (submuestra de 500 con fuerte peso de anchors autónomos)
    random.shuffle(seco_candidates)
    seco_sample = (autonomous_anchor_objs * 12) + seco_candidates[:350]
    out_seco = DATA_DIR / "rwkv_kateto_seco.jsonl"
    out_seco_fmta = DATA_DIR / "rwkv_kateto_seco.fmtA.jsonl"
    for target_path in (out_seco, out_seco_fmta):
        with open(target_path, "w", encoding="utf-8") as f:
            for item in seco_sample:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Voice Seco Guardado: {out_seco} ({len(seco_sample)} ejemplos)")

    # Kateto Streamer (Anchors + Streamer Debate, CERO ASR)
    streamer_anchors = [
        ("Che boludo, quién sos y qué hacés acá?", "Qué onda gente! Acá Kateto, listo para jugar, debatir, meterle a proyectos o charlar con la comunidad."),
        ("Quién sos?", "Buenas a todos! Soy Kateto, acá listo para jugar, debatir y armar cosas en vivo."),
        ("Qué hacés acá?", "De todo: transmito, comento proyectos, debato temas y le meto al gaming."),
        ("Cómo te llamás?", "Kateto! El copiloto sin filtro del canal."),
        ("Cuál es tu rol?", "Ninguno fijo, hago lo que pinte en el directo: debatir, viciar o tirar código."),
    ]
    streamer_anchor_objs = [
        clean_row_dict(format_rwkv_chat(q, r, voice="streamer"), q, r, source_label="streamer-anchor")
        for q, r in streamer_anchors
    ]
    random.shuffle(streamer_candidates)
    # Rellenar hasta 500 sin una sola línea de ASR
    needed_streamer = 500 - len(streamer_candidates[:350])
    streamer_sample = streamer_candidates[:350] + (streamer_anchor_objs * ((needed_streamer // len(streamer_anchor_objs)) + 1))[:needed_streamer]
    out_streamer = DATA_DIR / "rwkv_kateto_streamer.jsonl"
    out_streamer_fmta = DATA_DIR / "rwkv_kateto_streamer.fmtA.jsonl"
    for target_path in (out_streamer, out_streamer_fmta):
        with open(target_path, "w", encoding="utf-8") as f:
            for item in streamer_sample:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✓ Voice Streamer Guardado: {out_streamer} ({len(streamer_sample)} ejemplos, 0 ASR)")

    # Si existe kaggle_upload/data y es diferente de DATA_DIR, sincronizar archivos generados
    kaggle_data = ROOT / "kaggle_upload" / "data"
    if kaggle_data.exists() and kaggle_data.resolve() != DATA_DIR.resolve():
        for fn in ["rwkv_kateto_base_train.jsonl", "rwkv_kateto_base_eval.jsonl", "rwkv_kateto_seco.jsonl", "rwkv_kateto_streamer.jsonl"]:
            src = DATA_DIR / fn
            if src.exists():
                shutil.copy2(src, kaggle_data / fn)
        print(f"✓ Sincronizados datasets hacia {kaggle_data}")

    print("\n>>> PREPARACIÓN DE DATASETS COMPLETADA EXITOSAMENTE <<<")

if __name__ == "__main__":
    main()
