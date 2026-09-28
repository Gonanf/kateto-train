import os, sys, glob, json, subprocess
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file

PROJECT = Path(os.environ.get("KATETO_HOME", "/run/media/chaos/terciario/proyectos/kateto-train"))
STAGE2_DIR = PROJECT / "out/qwen35-0.8b-kateto-2stage/stage2"
MERGED_DIR = PROJECT / "out/qwen35-0.8b-kateto-merged"
F16_GGUF = PROJECT / "out/qwen35-0.8b-kateto-f16.gguf"
IMATRIX_DAT = PROJECT / "out/qwen35-0.8b-kateto.imatrix.dat"
Q4_GGUF = PROJECT / "out/qwen35-0.8b-kateto-Q4_K_M.gguf"
CONVERTER = Path(os.environ.get("LLAMA_CPP_CONVERTER", "/run/media/chaos/terciario/Backup/Proyectos/kateto/llama.cpp/convert_hf_to_gguf.py"))

# 1. Encontrar archivo base
base_files = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/model.safetensors*"))
base_file = [f for f in base_files if f.endswith(".safetensors")][0]
print(f"[1/5] Cargando base: {base_file}")
base_tensors = load_file(base_file)

adapter_file = STAGE2_DIR / "adapter_model.safetensors"
print(f"[2/5] Cargando adapter LoRA: {adapter_file}")
adapter = load_file(str(adapter_file))

# Leer alpha y r de adapter_config.json
cfg = json.loads((STAGE2_DIR / "adapter_config.json").read_text())
r = cfg.get("r", 16)
alpha = cfg.get("lora_alpha", 16)
scaling = alpha / r
print(f"LoRA config: r={r}, alpha={alpha}, scaling={scaling}")

# 3. Aplicar fusión exacta matemática
merged_tensors = dict(base_tensors)
merged_keys_count = 0
diffs = []

for k in sorted(adapter.keys()):
    if k.endswith(".lora_A.weight"):
        base_k = k.replace("base_model.model.", "").replace(".lora_A.weight", "") + ".weight"
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
        diffs.append((base_k, max_d, mean_d, norm_d, list(orig_W.shape)))
        merged_keys_count += 1

print(f"\n=== VERIFICACIÓN DEL MERGE ===")
print(f"Total capas LoRA fusionadas en el modelo: {merged_keys_count} de 96")
print(f"Capas base totales: {len(merged_tensors)}")

print("\nTop 10 capas con mayor impacto LoRA (|delta|_2):")
diffs.sort(key=lambda x: x[3], reverse=True)
for k, max_d, mean_d, norm_d, shp in diffs[:10]:
    print(f"  {k} {shp}: max={max_d:.6f}, mean={mean_d:.6f}, norm={norm_d:.4f}")

# 4. Guardar merged model
MERGED_DIR.mkdir(parents=True, exist_ok=True)
out_safetensors = MERGED_DIR / "model.safetensors"
print(f"\n[3/5] Guardando tensores fusionados en {out_safetensors}...")
save_file(merged_tensors, str(out_safetensors))

# Copiar metadatos y configs
base_snapshot_dir = Path(base_file).parent
for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
    src = base_snapshot_dir / fn
    if src.exists():
        (MERGED_DIR / fn).write_bytes(src.read_bytes())

# Copiar chat template limpio de Kateto
if (STAGE2_DIR / "chat_template.jinja").exists():
    (MERGED_DIR / "chat_template.jinja").write_bytes((STAGE2_DIR / "chat_template.jinja").read_bytes())

print("Archivos de configuración copiados exitosamente.")

# 5. Convertir a GGUF F16
print(f"\n[4/5] Convirtiendo a GGUF F16: {F16_GGUF}...")
cmd_convert = [
    sys.executable, str(CONVERTER),
    str(MERGED_DIR),
    "--outfile", str(F16_GGUF),
    "--outtype", "f16"
]
subprocess.run(cmd_convert, check=True)
print(f"GGUF F16 exportado: {F16_GGUF} ({F16_GGUF.stat().st_size / 1024**2:.1f} MB)")

# 6. Cuantizar con imatrix
print(f"\n[5/5] Re-cuantizando a Q4_K_M con imatrix existente: {Q4_GGUF}...")
cmd_quant = [
    "llama-quantize",
    "--imatrix", str(IMATRIX_DAT),
    str(F16_GGUF),
    str(Q4_GGUF),
    "Q4_K_M"
]
subprocess.run(cmd_quant, check=True)
print(f"GGUF Q4_K_M exportado: {Q4_GGUF} ({Q4_GGUF.stat().st_size / 1024**2:.1f} MB)")

print("\n¡Proceso de fusión y exportación completado con éxito!")
