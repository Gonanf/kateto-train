#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/run/media/chaos/terciario/proyectos/kateto-train"
OUT_DIR="$ROOT_DIR/out"

COMMANDS=(
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -ngl 0 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c0_cpu_rwkv29q4.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c1_vk1_rwkv29q4.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -fa on -o json > out/eureka-fork-llamacpp/c2_vk1_fa-on.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -ub 1024 -b 2048 -o json > out/eureka-fork-llamacpp/c3_vk1_ub1024.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -ub 128 -b 512 -o json > out/eureka-fork-llamacpp/c4_vk1_ub128.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q8_0.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c5_vk1_Q8_0.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c6_split.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 4 -o json > out/eureka-fork-llamacpp/c7_vk1_t4.json 2>/dev/null"
  "llama-bench -m /run/media/chaos/secundario/models/rwkv7-gguf/g1d-0.4b/rwkv7-g1d-0.4b-Q4_K_M-better_quantization-20260326.gguf -dev Vulkan0 -ngl 99 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c8_vk0_rwkv04b.json 2>/dev/null"
  "llama-bench -m /run/media/chaos/secundario/models/rwkv7-gguf/g1d-0.4b/rwkv7-g1d-0.4b-Q4_K_M-better_quantization-20260326.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c9_vk1_rwkv04b.json 2>/dev/null"
  "llama-bench -m kateto-rwkv29-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -ctk q8_0 -ctv q8_0 -o json > out/eureka-fork-llamacpp/c10_vk1_ctkq8.json 2>/dev/null"
  "llama-bench -m /run/media/chaos/secundario/orbe-out/modelos/Qwen3-0.6B-Q4_K_M.gguf -dev Vulkan1 -ngl 99 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c11_vk1_qwen06.json 2>/dev/null"
  "llama-bench -m /run/media/chaos/secundario/orbe-out/modelos/Qwen3-0.6B-Q4_K_M.gguf -dev Vulkan0 -ngl 99 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c12_vk0_qwen06.json 2>/dev/null"
  "llama-bench -m /run/media/chaos/secundario/models/hf/catalogo/vibethinker-3b/vibethinker-3b-q4_k_m.gguf -ngl 0 -p 512 -n 128 -r 3 -t 8 -o json > out/eureka-fork-llamacpp/c13_cpu_vibethinker3b.json 2>/dev/null"
)

if [[ "${1:-}" == "--dry-run" ]]; then
  for cmd in "${COMMANDS[@]}"; do
    echo "$cmd"
  done
  exit 0
fi

cd "$OUT_DIR"
for cmd in "${COMMANDS[@]}"; do
  date -Is >&2
  cat /proc/loadavg >&2
  eval "$cmd"
done
