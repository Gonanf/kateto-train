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
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_CE_LOSS_N_CHUNKS"] = "16"

import json, glob
from pathlib import Path
import torch
if hasattr(torch, "_dynamo"):
    torch._dynamo.disable()

PROJECT = Path(__file__).parent
DATA_SFT_TRAIN = PROJECT / "data/kateto_v2_combined_train.jsonl"
DATA_SFT_EVAL  = PROJECT / "data/kateto_v2_combined_eval.jsonl"
DATA_KATETO    = PROJECT / "data/kateto_v2_combined_train.jsonl"
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
    max_seq_length=512,
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
    learning_rate=1.5e-4,
    num_train_epochs=1,
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=50,
    eval_strategy="no",
    output_dir=str(OUTPUT_DIR / "stage1"),
)

TRAIN_ARGS_STAGE2 = dict(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=8e-5,
    num_train_epochs=2,
    warmup_ratio=0.03,
    weight_decay=0.01,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=50,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    output_dir=str(OUTPUT_DIR / "stage2"),
)

CHAT_TEMPLATE_QWEN35 = """{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'system' %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'tool' %}{{ '<|im_start|>tool\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""

def format_chat(example):
    # 1. Preferir question/answer o query/response directos
    q = example.get("query") or example.get("question") or ""
    a = example.get("response") or example.get("answer") or ""
    if q and a:
        return [{"role": "user", "content": str(q).strip()}, {"role": "assistant", "content": str(a).strip()}]
    # 2. Manejar openai messages sanitizados
    if "openai" in example and isinstance(example["openai"], dict) and "messages" in example["openai"]:
        msgs = []
        for m in example["openai"]["messages"]:
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            msgs.append({"role": m.get("role", "user"), "content": content})
        return msgs
    if "messages" in example:
        return example["messages"]
    text = example.get("text") or example.get("content") or json.dumps(example, ensure_ascii=False)
    return [{"role": "user", "content": text[:500]}, {"role": "assistant", "content": text[500:1000]}]

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
    tokenizer.chat_template = CHAT_TEMPLATE_QWEN35
    print("set chat_template Qwen3.5 (clean Jinja template)")
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

def train_stage(stage=1, model=None, tokenizer=None):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"\n{'='*40}\n=== INICIANDO STAGE {stage} ===\n{'='*40}")
    if model is None or tokenizer is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=COMMON_ARGS["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
        )
        tokenizer.chat_template = CHAT_TEMPLATE_QWEN35
        tokenizer.padding_side = "right"
        model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    if stage == 1:
        ds = load_stage1_dataset(tokenizer, max_samples=1200)
        args = TrainingArguments(**TRAIN_ARGS_STAGE1)
        eval_ds = None
        out = OUTPUT_DIR / "stage1"
    else:
        ds, eval_ds = load_stage2_dataset(tokenizer)
        stage1_adapter = OUTPUT_DIR / "stage1"
        if stage1_adapter.exists() and (stage1_adapter / "adapter_model.safetensors").exists():
            print(f"Cargando pesos de stage1 adapter desde {stage1_adapter}...")
            try:
                from safetensors.torch import load_file
                from peft import set_peft_model_state_dict
                adapters_weights = load_file(str(stage1_adapter / "adapter_model.safetensors"))
                set_peft_model_state_dict(model, adapters_weights)
                print("Adapter stage1 cargado en el modelo.")
            except Exception as e:
                print(f"Nota: continuando con pesos actuales en memoria ({e})")
        args = TrainingArguments(**TRAIN_ARGS_STAGE2)
        out = OUTPUT_DIR / "stage2"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=COMMON_ARGS["max_seq_length"],
        dataset_num_proc=2,
        args=args,
    )
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    checkpoints = sorted(glob.glob(str(out / "checkpoint-*")), key=lambda p: int(p.split("-")[-1]) if p.split("-")[-1].isdigit() else 0)
    resume = checkpoints[-1] if checkpoints else None
    if resume:
        print(f"Reanudando entrenamiento desde checkpoint: {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Stage {stage} guardado exitosamente en {out}")
    return model, tokenizer

def merge_lora_direct(base_model_path_or_id, adapter_dir, merged_dir):
    import glob, json
    import torch
    from safetensors.torch import load_file, save_file
    
    # 1. Encontrar archivo safetensors de base
    base_files = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{base_model_path_or_id.replace('/', '--')}/snapshots/*/model.safetensors*"))
    base_files = [f for f in base_files if f.endswith(".safetensors")]
    if not base_files:
        from huggingface_hub import snapshot_download
        base_dir = snapshot_download(base_model_path_or_id)
        base_files = glob.glob(f"{base_dir}/model.safetensors*")
        base_files = [f for f in base_files if f.endswith(".safetensors")]
    
    base_file = base_files[0]
    base_snapshot_dir = Path(base_file).parent
    print(f"Cargando modelo base: {base_file}")
    base_tensors = load_file(base_file)

    adapter_file = Path(adapter_dir) / "adapter_model.safetensors"
    print(f"Cargando adapter LoRA: {adapter_file}")
    adapter = load_file(str(adapter_file))

    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    r = cfg.get("r", 16)
    alpha = cfg.get("lora_alpha", 16)
    scaling = alpha / r
    print(f"LoRA config: r={r}, alpha={alpha}, scaling={scaling}")

    merged_tensors = dict(base_tensors)
    merged_keys_count = 0
    diffs = []

    for k in sorted(adapter.keys()):
        if k.endswith(".lora_A.weight"):
            # En Unsloth Qwen 3.5 las claves son base_model.model.model.language_model...
            # Las claves de peso base en model.safetensors son model.language_model...
            clean_k = k.replace("base_model.model.", "")
            base_k = clean_k.replace(".lora_A.weight", "") + ".weight"
            if base_k not in merged_tensors:
                print(f"ALERTA: no se encontró {base_k} en base!")
                continue
            lora_A = adapter[k].float()
            lora_B = adapter[k.replace(".lora_A.weight", ".lora_B.weight")].float()
            delta_W = (lora_B @ lora_A) * scaling
            
            orig_W = merged_tensors[base_k]
            orig_dtype = orig_W.dtype
            new_W = (orig_W.float() + delta_W).to(orig_dtype)
            merged_tensors[base_k] = new_W
            
            max_d = delta_W.abs().max().item()
            mean_d = delta_W.abs().mean().item()
            norm_d = delta_W.norm().item()
            diffs.append((base_k, max_d, mean_d, norm_d))
            merged_keys_count += 1

    print(f"Total capas LoRA fusionadas directamente: {merged_keys_count}")
    assert merged_keys_count > 0, "Error crítico: no se fusionó ninguna capa LoRA!"

    merged_dir = Path(merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    out_safetensors = merged_dir / "model.safetensors"
    print(f"Guardando tensores fusionados en {out_safetensors}...")
    save_file(merged_tensors, str(out_safetensors))

    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        src = base_snapshot_dir / fn
        if src.exists():
            (merged_dir / fn).write_bytes(src.read_bytes())

    if (Path(adapter_dir) / "chat_template.jinja").exists():
        (merged_dir / "chat_template.jinja").write_bytes((Path(adapter_dir) / "chat_template.jinja").read_bytes())

    print(f"Modelo merged 16-bit completado en {merged_dir}")


def export_to_gguf_and_imatrix(stage2_dir=None, merged_dir=None):
    import subprocess, sys
    stage2_dir = stage2_dir or (OUTPUT_DIR / "stage2")
    merged_dir = merged_dir or (PROJECT / "out/qwen35-0.8b-kateto-merged")
    f16_gguf = PROJECT / "out/qwen35-0.8b-kateto-f16.gguf"
    imatrix_dat = PROJECT / "out/qwen35-0.8b-kateto.imatrix.dat"
    q4_gguf = PROJECT / "out/qwen35-0.8b-kateto-Q4_K_M.gguf"
    calib_file = PROJECT / "data/kateto_v2_calib.txt" if (PROJECT / "data/kateto_v2_calib.txt").exists() else PROJECT / "data/sft_out/calib.txt"
    converter_script = Path("/run/media/chaos/terciario/Backup/Proyectos/kateto/llama.cpp/convert_hf_to_gguf.py")

    print(f"\n{'='*50}\n=== EXPORTANDO A GGUF E IMATRIX ===\n{'='*50}")

    # 1. Merge LoRA a 16-bit
    print(f"--- 1/5 Merge LoRA matemático directo a 16-bit en {merged_dir} ---")
    merge_lora_direct(MODEL_ID, stage2_dir, merged_dir)

    # 2. Convertir a GGUF F16
    print(f"--- 2/5 Convertir a GGUF F16: {f16_gguf} ---")
    cmd_convert = [
        sys.executable, str(converter_script),
        str(merged_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16"
    ]
    subprocess.run(cmd_convert, check=True)
    print(f"GGUF F16 generado: {f16_gguf} ({f16_gguf.stat().st_size / 1024**2:.1f} MB)")

    # 3. Calcular imatrix
    print(f"--- 3/5 Calcular imatrix con {calib_file} ---")
    cmd_imatrix = [
        "llama-imatrix",
        "-m", str(f16_gguf),
        "-f", str(calib_file),
        "-o", str(imatrix_dat),
        "--chunks", "64"
    ]
    subprocess.run(cmd_imatrix, check=True)
    print("imatrix calculada exitosamente.")

    # 4. Cuantizar a Q4_K_M con imatrix
    print(f"--- 4/5 Cuantizar a Q4_K_M con imatrix: {q4_gguf} ---")
    cmd_quant = [
        "llama-quantize",
        "--imatrix", str(imatrix_dat),
        str(f16_gguf),
        str(q4_gguf),
        "Q4_K_M"
    ]
    subprocess.run(cmd_quant, check=True)
    print(f"GGUF Q4_K_M generado: {q4_gguf} ({q4_gguf.stat().st_size / 1024**2:.1f} MB)")

    # 5. Smoke test con llama-cli
    print("--- 5/5 Smoke test con llama-cli ---")
    test_prompt = "<|im_start|>user\nChe Kateto, quién sos y qué hacés acá?<|im_end|>\n<|im_start|>assistant\n"
    cmd_test = [
        "llama-cli",
        "-m", str(q4_gguf),
        "-p", test_prompt,
        "-n", "64",
        "--temp", "0.7",
        "--single-turn",
        "-r", "<|im_end|>"
    ]
    subprocess.run(cmd_test, input=b"")
    print("\nPipeline completada con éxito.")

def run_all(force_stage1=False):
    print(">>> EJECUTANDO PIPELINE (STAGE 1 -> STAGE 2 -> MERGE -> GGUF -> IMATRIX -> Q4_K_M)")
    stage1_weights = OUTPUT_DIR / "stage1" / "adapter_model.safetensors"
    if stage1_weights.exists() and not force_stage1:
        print(f"Stage 1 ya completado con pesos en {stage1_weights}. Continuando directamente a Stage 2...")
        m, tok = train_stage(stage=2)
    else:
        m, tok = train_stage(stage=1)
        m, tok = train_stage(stage=2, model=m, tokenizer=tok)
    export_to_gguf_and_imatrix()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="solo prueba carga 4bit")
    ap.add_argument("--stage", type=int, default=0, choices=[0,1,2], help="0=smoke, 1=stage1, 2=stage2")
    ap.add_argument("--all", action="store_true", help="corre stage 1, stage 2 y exporta a GGUF imatrix")
    ap.add_argument("--force-stage1", action="store_true", help="fuerza re-entrenamiento de stage 1")
    ap.add_argument("--export", action="store_true", help="solo exporta a GGUF e imatrix desde out/")
    ap.add_argument("--model", type=str, default=MODEL_ID)
    args = ap.parse_args()
    if args.model != MODEL_ID:
        MODEL_ID = args.model
    if args.all:
        run_all(force_stage1=args.force_stage1)
    elif args.export:
        export_to_gguf_and_imatrix()
    elif args.smoke or args.stage==0:
        smoke_test()
    elif args.stage==1:
        train_stage(1)
    elif args.stage==2:
        train_stage(2)
