#!/usr/bin/env python3
"""
build_dataset_v2.py — Consolida y balancea el dataset Kateto v2 completo.

Componentes:
1. data/hermes_qa_7k.jsonl (~7,000 QA y tool calls de Hermes state.db filtrados)
2. data/sft_out/sft_train.jsonl (Kateto persona + razonamiento <think> + herramientas)
3. data/toolcalling_sft.jsonl (200 tool-calling curados)
4. data/toolcalling_v2/*.jsonl (voces: Dagger, gamer, streamer, SrPelo, etc.)
5. data/youtube_asr_dataset.jsonl (Transcripción ASR + habla argentina directa de streams)
6. Turn-Taking calibrado: exactamente 150 <NO_RESPONSE> y 150 <WAIT> (~3% del dataset total).

Salidas:
- data/kateto_v2_combined_train.jsonl
- data/kateto_v2_combined_eval.jsonl
- data/kateto_v2_calib.txt (para imatrix de llama.cpp)
"""
import os, sys, json, random, re
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"

HERMES_QA = DATA_DIR / "hermes_qa_7k.jsonl"
SFT_TRAIN = DATA_DIR / "sft_out" / "sft_train.jsonl"
TOOLCALLING_SFT = DATA_DIR / "toolcalling_sft.jsonl"
TOOLCALLING_V2_DIR = DATA_DIR / "toolcalling_v2"
YOUTUBE_ASR = DATA_DIR / "youtube_asr_dataset.jsonl"

OUT_TRAIN = DATA_DIR / "kateto_v2_combined_train.jsonl"
OUT_EVAL  = DATA_DIR / "kateto_v2_combined_eval.jsonl"
OUT_CALIB = DATA_DIR / "kateto_v2_calib.txt"

SYS_PROMPT = "Sos Kateto, rioplatense seco, con opinion propia."

def normalize_item(item, source_tag=""):
    # Normaliza a formato unificado
    q = item.get("query") or item.get("question") or ""
    a = item.get("response") or item.get("answer") or ""

    if isinstance(q, list) or isinstance(q, dict):
        q = json.dumps(q, ensure_ascii=False)
    if isinstance(a, list) or isinstance(a, dict):
        a = json.dumps(a, ensure_ascii=False)

    q = str(q).strip()
    a = str(a).strip()

    if not q or not a:
        if "openai" in item and "messages" in item["openai"]:
            msgs = item["openai"]["messages"]
            for m in msgs:
                if m.get("role") == "user" and not q:
                    q = str(m.get("content", "")).strip()
                elif m.get("role") == "assistant" and not a:
                    a = str(m.get("content", "")).strip()

    # Filtros de sanidad y longitud máxima (para evitar truncamientos en max_seq_length=1024)
    if not q or not a:
        return None
    if len(q) < 4 or len(a) < 2:
        return None
    if len(q) > 2200 or len(a) > 2800:
        return None
    if q.startswith("[System note") or q.startswith("[Context") or q.startswith("[Reminder") or "[The user sent an audio" in q:
        return None

    return {
        "query": q,
        "response": a,
        "question": q,
        "answer": a,
        "source_label": item.get("source_label") or source_tag,
        "openai": {
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        }
    }

def main():
    random.seed(42)
    print("=== Consolidando Dataset Kateto v2 ===")

    collected_standard = []
    collected_no_response = []
    collected_wait = []

    def ingest_file(path, tag):
        if not path.exists():
            print(f"Aviso: {path.name} no encontrado, omitiendo.")
            return 0
        cnt = 0
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except Exception:
                continue
            norm = normalize_item(raw, tag)
            if not norm:
                continue

            resp = norm["response"]
            if "<NO_RESPONSE>" in resp:
                collected_no_response.append(norm)
            elif "<WAIT>" in resp:
                collected_wait.append(norm)
            else:
                collected_standard.append(norm)
            cnt += 1
        print(f"  + {tag} ({path.name}): {cnt} items leídos")
        return cnt

    # 1. Hermes QA 7k
    ingest_file(HERMES_QA, "hermes-state-db")

    # 2. SFT Train (Kateto persona)
    ingest_file(SFT_TRAIN, "kateto-persona-sft")

    # 3. Toolcalling curado
    ingest_file(TOOLCALLING_SFT, "kateto-toolcalling")

    # 4. Toolcalling v2 personalidades (Dagger, gamer, etc.)
    if TOOLCALLING_V2_DIR.exists():
        for jf in sorted(TOOLCALLING_V2_DIR.glob("*.jsonl")):
            ingest_file(jf, f"v2-{jf.stem}")

    # 5. YouTube ASR dataset
    if YOUTUBE_ASR.exists():
        ingest_file(YOUTUBE_ASR, "youtube-asr")
    else:
        print(f"Aviso: {YOUTUBE_ASR.name} todavía en generación.")

    # Deduplicar estándar por par (query, response) normalizado
    print(f"\nTotal estándar antes de dedup: {len(collected_standard)}")
    seen = set()
    deduped_standard = []
    for it in collected_standard:
        key = (it["query"][:100].lower().strip(), it["response"][:100].lower().strip())
        if key not in seen:
            seen.add(key)
            deduped_standard.append(it)
    print(f"Total estándar luego de dedup: {len(deduped_standard)}")

    # Balancear turn-taking: exactamente 150 <NO_RESPONSE> y 150 <WAIT>
    random.shuffle(collected_no_response)
    random.shuffle(collected_wait)

    target_nr = min(len(collected_no_response), 150)
    target_wait = min(len(collected_wait), 150)

    sampled_nr = collected_no_response[:target_nr]
    sampled_wait = collected_wait[:target_wait]

    print(f"Turn-taking balanceado:")
    print(f"  - <NO_RESPONSE>: {len(sampled_nr)} (de {len(collected_no_response)} disponibles)")
    print(f"  - <WAIT>: {len(sampled_wait)} (de {len(collected_wait)} disponibles)")

    # Dataset completo combinado
    all_items = deduped_standard + sampled_nr + sampled_wait
    random.shuffle(all_items)

    total = len(all_items)
    nr_pct = (len(sampled_nr) / total) * 100 if total else 0
    wait_pct = (len(sampled_wait) / total) * 100 if total else 0
    print(f"\nDataset consolidado final:")
    print(f"  - Total items: {total}")
    print(f"  - Porcentaje <NO_RESPONSE>: {nr_pct:.2f}% (ratio seguro <3%)")
    print(f"  - Porcentaje <WAIT>: {wait_pct:.2f}% (ratio seguro <3%)")

    # Split 95% Train / 5% Eval
    split_idx = int(total * 0.95)
    train_items = all_items[:split_idx]
    eval_items = all_items[split_idx:]

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for it in train_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for it in eval_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Archivo de calibración para imatrix (primeros 500 ejemplos en texto plano de diálogo)
    with open(OUT_CALIB, "w", encoding="utf-8") as f:
        for it in train_items[:500]:
            f.write(f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n")
            f.write(f"<|im_start|>user\n{it['query']}<|im_end|>\n")
            f.write(f"<|im_start|>assistant\n{it['response']}<|im_end|>\n\n")

    print(f"\nArchivos generados exitosamente:")
    print(f"  - Train: {OUT_TRAIN} ({len(train_items)} items, {OUT_TRAIN.stat().st_size / 1024**2:.2f} MB)")
    print(f"  - Eval:  {OUT_EVAL} ({len(eval_items)} items, {OUT_EVAL.stat().st_size / 1024**2:.2f} MB)")
    print(f"  - Calib: {OUT_CALIB} ({OUT_CALIB.stat().st_size / 1024**2:.2f} MB)")

if __name__ == "__main__":
    main()
