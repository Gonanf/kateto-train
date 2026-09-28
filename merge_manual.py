#!/usr/bin/env python3
"""Merge manual LoRA -> base (torch, sin PEFT).

W' = W + (alpha/r) * B @ A  en fp32, de vuelta a bf16.
Parte de los tensores language_model limpios del base y aplica los 96 pares.
"""
import json
import os
import shutil
from pathlib import Path

import unsloth  # parchea entorno igual que el train
import torch
from safetensors.torch import load_file, save_file

TDIR = Path(os.environ.get("KATETO_HOME", "/run/media/chaos/terciario/proyectos/kateto-train"))
ADAPTER = TDIR / "out/qwen35-0.8b-kateto-2stage/stage2"
OUT = TDIR / "out/qwen35-0.8b-kateto-merged-v2"
OUT.mkdir(parents=True, exist_ok=True)

snap_dir = os.environ.get("HF_QWEN_SNAPSHOTS", os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots"))
snap = os.listdir(snap_dir)[0] if os.path.isdir(snap_dir) and os.listdir(snap_dir) else ""
BASE_F = os.path.join(snap_dir, snap, "model.safetensors-00001-of-00001.safetensors") if snap else ""

cfg = json.load(open(ADAPTER / "adapter_config.json"))
r, alpha = cfg["r"], cfg["lora_alpha"]
scale = alpha / r
print(f"r={r} alpha={alpha} scale={scale}", flush=True)

print("cargando base...", flush=True)
B = load_file(BASE_F, device="cpu")
print("cargando adapter...", flush=True)
A = load_file(str(ADAPTER / "adapter_model.safetensors"), device="cpu")

M = {k: v for k, v in B.items() if k.startswith("model.language_model.")}
print(f"base language_model: {len(M)} tensores", flush=True)

PREFIX = "base_model.model.model.language_model.layers."
n = 0
for ka in A:
    if ".lora_A." not in ka:
        continue
    mid = ka[len(PREFIX):].split(".lora_A")[0]  # N.mlp.down_proj
    kb = f"base_model.model.model.language_model.layers.{mid}.lora_B.weight"
    base_k = f"model.language_model.layers.{mid}.weight"
    assert kb in A, kb
    assert base_k in M, base_k
    lora_a = A[ka].float()
    lora_b = A[kb].float()
    delta = (lora_b @ lora_a) * scale
    M[base_k] = (M[base_k].float() + delta).to(M[base_k].dtype)
    n += 1
print(f"pares aplicados: {n}", flush=True)
assert n == 96, f"inesperado: {n}"

save_file(M, str(OUT / "model.safetensors"))
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "processor_config.json"):
    src = TDIR / "out/qwen35-0.8b-kateto-merged" / f
    if src.exists():
        shutil.copyfile(src, OUT / f)
print(f"MERGED_V2_OK {OUT} tensores={len(M)}", flush=True)
