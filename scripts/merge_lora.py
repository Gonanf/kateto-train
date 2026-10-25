"""Merge generico de adapter LoRA (PEFT) sobre base RWKV pth, en CPU.

Uso: python merge_lora.py base.pth adapter_dir/ merged.pth
Lee scaling de adapter_config.json (alpha/r). Para cada peso lora_B,
base_key = modulo path + .weight, suma delta = scaling * B @ A.
Preserva el contenedor original del pth (clave 'model' si existe).
"""
from __future__ import annotations

import json
import sys

import torch


def load_state(path: str):
    try:
        from safetensors.torch import load_file

        if path.endswith(".safetensors"):
            return load_file(path)
    except ImportError:
        pass
    return torch.load(path, map_location="cpu")


def main(base_pth: str, adapter_dir: str, out_pth: str) -> None:
    import os

    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    scaling = cfg.get("lora_alpha", 32) / cfg.get("r", 16)
    print(f"scaling={scaling} r={cfg.get('r')}")

    base = torch.load(base_pth, map_location="cpu")
    container = None
    if isinstance(base, dict) and "model" in base and isinstance(base["model"], dict):
        container, sd = base, base["model"]
    else:
        sd = base

    ad = None
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = os.path.join(adapter_dir, name)
        try:
            ad = load_state(p)
            break
        except FileNotFoundError:
            continue
    if ad is None:
        raise FileNotFoundError("sin adapter_model.safetensors ni .bin")

    import re

    def find_base_key(mod_name: str, sd_dict: dict) -> str | None:
        if mod_name.startswith("base_model.model."):
            mod_name = mod_name[len("base_model.model."):]
        direct = mod_name + ".weight"
        if direct in sd_dict:
            return direct
        # HF layers to RWKV blocks
        m = re.match(r"(?:model\.)?layers\.(\d+)\.ffn\.(.*)", mod_name)
        if m:
            candidate = f"blocks.{m.group(1)}.ffn.{m.group(2)}.weight"
            if candidate in sd_dict:
                return candidate
        m = re.match(r"(?:model\.)?layers\.(\d+)\.attention\.(.*)", mod_name)
        if m:
            candidate = f"blocks.{m.group(1)}.att.{m.group(2)}.weight"
            if candidate in sd_dict:
                return candidate
        m = re.match(r"blocks\.(\d+)\.(.*)", mod_name)
        if m:
            candidate = f"blocks.{m.group(1)}.{m.group(2)}.weight"
            if candidate in sd_dict:
                return candidate
        return None

    n = 0
    for k, b in list(ad.items()):
        if not k.endswith(".lora_B.weight"):
            continue
        mod = k[: -len(".lora_B.weight")]
        a = ad.get(mod + ".lora_A.weight")
        if a is None:
            print(f"WARN sin lora_A para {mod}")
            continue
        base_key = find_base_key(mod, sd)
        if base_key is None:
            print(f"WARN base sin mapping para {mod}")
            continue
        delta = (b.float() @ a.float()) * scaling
        orig_dtype = sd[base_key].dtype
        if sd[base_key].shape != delta.shape:
            print(f"ERROR shape mismatch {base_key}: base {sd[base_key].shape} vs delta {delta.shape}")
            continue
        sd[base_key] = (sd[base_key].float() + delta).to(orig_dtype)
        n += 1
    print(f"mergeados {n} modulos exitosamente")
    if container is not None:
        container["model"] = sd
        torch.save(container, out_pth)
    else:
        torch.save(sd, out_pth)
    print(f"guardado {out_pth}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
