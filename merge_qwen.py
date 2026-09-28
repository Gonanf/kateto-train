#!/usr/bin/env python3
"""Merge Qwen3.5-0.8B LoRA stage2 -> full HF 16bit (CPU, sin GPU)."""
import os
from pathlib import Path
import unsloth  # primero: parchea entorno y evita el crash torchao/ScalingType
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TDIR = Path(os.environ.get("KATETO_HOME", "/run/media/chaos/terciario/proyectos/kateto-train"))
BASE = "Qwen/Qwen3.5-0.8B"
ADAPTER = TDIR / "out/qwen35-0.8b-kateto-2stage/stage2"
MERGED = TDIR / "out/qwen35-0.8b-kateto-merged"
MERGED.mkdir(parents=True, exist_ok=True)

print(f"base={BASE} adapter={ADAPTER}", flush=True)
tok = AutoTokenizer.from_pretrained(str(ADAPTER), trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="cpu",
    trust_remote_code=True, low_cpu_mem_usage=True,
)
print("base cargada, aplicando LoRA...", flush=True)
model = PeftModel.from_pretrained(base, str(ADAPTER))
merged = model.merge_and_unload()
print("merge listo, guardando...", flush=True)
merged.save_pretrained(str(MERGED), safe_serialization=True)
tok.save_pretrained(str(MERGED))
print(f"MERGED_OK {MERGED}", flush=True)
