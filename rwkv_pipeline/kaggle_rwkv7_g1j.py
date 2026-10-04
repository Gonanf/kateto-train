#!/usr/bin/env python3
"""
Script autónomo para Kaggle / Colab:
Entrena RWKV-7 G1j (1.5B, 2.9B o 7.2B) en T4/A100 con el pipeline híbrido:
1. Capa 1: LoRA SFT sobre dataset Kateto v2
2. Capa 2: State-Tuning para voces (Seco / Streamer)
"""

import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"\n>>> EJECUTANDO: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("=================================================================")
    print(">>> KAGGLE RUNNER: RWKV-7 G1j (1.5B / 2.9B / 7.2B) PIPELINE <<<")
    print("=================================================================")
    
    # 1. Instalar dependencias si no existen
    run_cmd("pip install -q lightning deepspeed jsonlines huggingface_hub")
    
    # 2. Clonar RWKV-PEFT si no está presente
    if not os.path.exists("RWKV-PEFT"):
        run_cmd("git clone https://github.com/JL-er/RWKV-PEFT.git")
        
    # 3. Descargar el modelo deseado de BlinkDL/rwkv7-g1
    # Modelos soportados:
    # - rwkv7-g1j-1.5b-20260831-ctx16384.pth (T4 16GB OK)
    # - rwkv7-g1j-2.9b-20260831-ctx16384.pth (T4 16GB con LoRA r=16 OK)
    # - rwkv7-g1j-7.2b-20260831-ctx16384.pth (A100 40GB/80GB)
    target_model = os.environ.get("RWKV_TARGET_MODEL", "rwkv7-g1j-1.5b-20260831-ctx16384.pth")
    print(f"Modelo objetivo G1j: {target_model}")
    
    from huggingface_hub import hf_hub_download
    os.makedirs("models", exist_ok=True)
    local_path = os.path.join("models", target_model)
    if not os.path.exists(local_path):
        print(f"Descargando {target_model} desde BlinkDL/rwkv7-g1...")
        hf_hub_download(repo_id="BlinkDL/rwkv7-g1", filename=target_model, local_dir="models")
        print("Descarga completada.")
        
    print("\nEntorno Kaggle preparado para entrenamiento G1j.")

if __name__ == "__main__":
    main()
