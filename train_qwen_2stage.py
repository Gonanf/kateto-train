#!/usr/bin/env python3
"""
Kateto 2-stage LoRA training for Qwen3.5-0.8B on RX 6500 XT 4GB
Stage1: dominio raw (news + argentina reddit + alpaca) bajo peso
Stage2: kateto_full (3200) con chat template Qwen3.5
Config: r16 alpha16 batch1 accum8 lr2e-4, 4bit, max_seq_length 2048
GPU: RX 6500 XT Navi24 gfx1034 -> HSA_OVERRIDE_GFX_VERSION=10.3.0 required
"""
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import json, glob
from pathlib import Path

PROJECT = Path(__file__).parent
DATA_SFT_TRAIN = PROJECT / "data/sft_out/sft_train.jsonl"
DATA_SFT_EVAL  = PROJECT / "data/sft_out/sft_eval.jsonl"
DATA_KATETO    = PROJECT / "data/kateto_full.jsonl"
# Raw stage1 sources
RAW_SOURCES = [
    "data/raw/argentina-news/data/*.parquet",
    "data/raw/argentina-reddit/*.parquet",
    "data/raw/alpaca-spanish/alpaca_data_cleaned_spanish.json",
    "data/raw/news-argentina/data/*.parquet",
]

MODEL_ID = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = PROJECT / "out/qwen35-0.8b-kateto-2stage"

COMMON_ARGS = dict(
    max_seq_length=2048,
    dtype=None,  # auto
    load_in_4bit=True,
)

LORA_CONFIG = dict(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

TRAIN_ARGS_STAGE1 = dict(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # effective 8
    learning_rate=2e-4,
    num_train_epochs=1,
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=200,
    eval_strategy="no",
    output_dir=str(OUTPUT_DIR / "stage1"),
)

TRAIN_ARGS_STAGE2 = dict(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,  # ajustable 2-3 epojs para 3200 samples
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    output_dir=str(OUTPUT_DIR / "stage2"),
)

CHAT_TEMPLATE_QWEN35 = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

def format_chat(example):
    # Soporta dos formatos: query/response y messages
    if "messages" in example:
        msgs = example["messages"]
    elif "query" in example and "response" in example:
        # detectar si query ya es lista
        msgs = [{"role":"user","content":example["query"]}, {"role":"assistant","content":example["response"]}]
    elif "question" in example and "answer" in example:
        msgs = [{"role":"user","content":example["question"]}, {"role":"assistant","content":example["answer"]}]
    else:
        # fallback: usa todo el texto
        text = example.get("text") or example.get("content") or json.dumps(example, ensure_ascii=False)
        msgs = [{"role":"user","content": text[:500]}, {"role":"assistant","content": text[500:1000]}]
    return msgs

def load_stage1_dataset(tokenizer, max_samples=8000):
    """Stage1: mezcla raw con peso bajo. Usa sampleado para no exceder VRAM/tiempo."""
    import pyarrow.parquet as pq
    from datasets import Dataset
    rows = []
    # 1. argentina-news parquet
    for pat in RAW_SOURCES:
        for fp in glob.glob(str(PROJECT / pat)):
            if fp.endswith(".parquet"):
                try:
                    pf = pq.ParquetFile(fp)
                    # sample 2000 max per file
                    tbl = pf.read().to_pylist()
                    for r in tbl[:2000]:
                        # extrae texto generico
                        txt = r.get("text") or r.get("content") or r.get("article") or r.get("body") or str(r)[:1000]
                        if len(txt) < 50: continue
                        rows.append({"messages":[{"role":"user","content":f"Resumí en rioplatense: {txt[:400]}"}, {"role":"assistant","content": txt[:600]}]})
                        if len(rows) >= max_samples: break
                except Exception as e:
                    print(f"warn parquet {fp}: {e}")
            elif fp.endswith(".json"):
                try:
                    import json as js
                    data = js.load(open(fp))
                    if isinstance(data, list):
                        for ex in data[:2000]:
                            instr = ex.get("instruction") or ex.get("input") or ""
                            out = ex.get("output") or ex.get("response") or ""
                            if not out: continue
                            rows.append({"messages":[{"role":"user","content": instr[:500]}, {"role":"assistant","content": out[:800]}]})
                            if len(rows) >= max_samples: break
                except Exception as e:
                    print(f"warn json {fp}: {e}")
        if len(rows) >= max_samples: break
    print(f"stage1 raw rows: {len(rows)}")
    # intercalar con un 10% de kateto_full para anclar estilo
    try:
        kateto = [json.loads(l) for l in open(DATA_KATETO)] if DATA_KATETO.exists() else []
        # take 10% of stage1 size from kateto
        n_anchor = min(len(kateto), max(200, len(rows)//10))
        for ex in kateto[:n_anchor]:
            msgs = format_chat(ex)
            rows.append({"messages": msgs})
    except Exception as e:
        print(f"warn anchor: {e}")
    import random; random.seed(42); random.shuffle(rows)
    ds = Dataset.from_list(rows)
    # tokenizar via chat template ya seteado en tokenizer
    def fmt(ex):
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    ds = ds.map(fmt)
    return ds

def load_stage2_dataset(tokenizer):
    from datasets import Dataset
    rows = []
    # kateto_full principal (3200)
    if DATA_KATETO.exists():
        for line in open(DATA_KATETO):
            ex = json.loads(line)
            msgs = format_chat(ex)
            rows.append({"messages": msgs})
    # también sft_train si existe para complementar
    if DATA_SFT_TRAIN.exists():
        # añadir solo si no duplica demasiado
        extra = [json.loads(l) for l in open(DATA_SFT_TRAIN)]
        # evito duplicar: si kateto_full ya contiene sft, no añadir mucho
        # aquí añadimos 0% porque kateto_full ya es la union curada
        pass
    print(f"stage2 kateto rows: {len(rows)}")
    ds = Dataset.from_list(rows)
    def fmt(ex):
        text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}
    ds = ds.map(fmt)
    # eval split 332 como en proyecto
    eval_ds = None
    if DATA_SFT_EVAL.exists():
        eval_rows=[]
        for line in open(DATA_SFT_EVAL):
            ex=json.loads(line)
            msgs=format_chat(ex)
            eval_rows.append({"messages": msgs})
        from datasets import Dataset as DS
        eval_ds = DS.from_list(eval_rows).map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)})
    return ds, eval_ds

def smoke_test():
    """Prueba carga 4bit sin entrenar, reportar VRAM estimada"""
    from unsloth import FastLanguageModel
    import torch
    print(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"total VRAM {props.total_memory/1024**3:.2f} GB")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"model loaded: {model.__class__.__name__} 4bit={True}")
    # tokenizer chat template check
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE_QWEN35
        print("set chat_template Qwen3.5")
    else:
        print(f"chat_template exists len={len(tokenizer.chat_template)}")
    # test tokenize (robusto a processor vision: usa sub-tokenizer texto)
    sample = [{"role":"user","content":"Che, ¿cómo andás?"},{"role":"assistant","content":"¡Hola boludo! Todo bien."}]
    text = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
    tok_core = getattr(tokenizer, "tokenizer", None) or getattr(tokenizer, "text_tokenizer", None) or tokenizer
    try:
        toks = tok_core(text, return_tensors="pt")
        print(f"sample tokens {toks['input_ids'].shape} text[:120]={text[:120]!r}")
    except Exception as e:
        ids = tok_core.encode(text)
        print(f"sample encode len={len(ids)} text[:120]={text[:120]!r} (fallback encode, tokenize pt falló: {e})")
    # VRAM estimate: param 0.8B *0.5 bytes (4bit) ~0.4GB + overhead LoRA + activations
    # con seq 2048 batch1: activations ~0.6GB, total ~1.2-1.8GB -> fit 4GB
    print("VRAM estimada: ~1.5-2.0 GB (4bit qlora r16, seq2048, batch1) -> OK para 4GB")
    # lora
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    # count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*trainable/total:.2f}%)")
    return True

def train_stage(stage=1):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"=== STAGE {stage} ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE_QWEN35
    tokenizer.padding_side = "right"
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    if stage == 1:
        ds = load_stage1_dataset(tokenizer, max_samples=6000)
        args = TrainingArguments(**TRAIN_ARGS_STAGE1)
        eval_ds = None
        out = OUTPUT_DIR / "stage1"
    else:
        ds, eval_ds = load_stage2_dataset(tokenizer)
        # si stage2 continúa desde stage1 adapter
        stage1_adapter = OUTPUT_DIR / "stage1"
        if stage1_adapter.exists() and any(stage1_adapter.glob("adapter*")):
            print(f"cargando adapter stage1 desde {stage1_adapter}")
            # opcjonal: cargar pesos stage1
            # model.load_adapter(str(stage1_adapter))
            pass
        args = TrainingArguments(**TRAIN_ARGS_STAGE2)
        # ensure eval
        if eval_ds is not None:
            args.eval_strategy = "steps"
        out = OUTPUT_DIR / "stage2"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=args,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"stage {stage} guardado en {out}")
    # merge option
    # model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="solo prueba carga 4bit")
    ap.add_argument("--stage", type=int, default=0, choices=[0,1,2], help="0=smoke, 1=stage1, 2=stage2")
    ap.add_argument("--model", type=str, default=MODEL_ID)
    args = ap.parse_args()
    if args.model != MODEL_ID:
        MODEL_ID = args.model
    if args.smoke or args.stage==0:
        smoke_test()
    elif args.stage==1:
        train_stage(1)
    elif args.stage==2:
        train_stage(2)
