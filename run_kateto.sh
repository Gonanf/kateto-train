#!/bin/bash
# LoRA de Kateto sobre RWKV7 G1 2.9B en CPU, con guardas anti overfit.
# Base: L32-D2560, vocab 65536, arch x070. Todo en terciario (HDD).
# Chunked: EPOCH_COUNT=1 por invocacion, resume con EPOCH_BEGIN + LORA_LOAD.
set -e
TDIR="${KATETO_HOME:-/run/media/chaos/terciario/proyectos/kateto-train}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export RWKV_FLOAT_MODE=fp32

EPOCH_BEGIN=${EPOCH_BEGIN:-0}
LORA_LOAD=${LORA_LOAD:-}
LORA_ARG="{\"lora_load\":\"$LORA_LOAD\", \"lora_r\":16, \"lora_alpha\":32, \"lora_dropout\":0.05}"

cd "$TDIR/RWKV-PEFT"
python train.py --load_model "$TDIR/base/rwkv7-g1j-2.9b-20260831-ctx16384.pth" \
  --proj_dir "$TDIR/out" --data_file "$TDIR/data/sft_train.jsonl" \
  --vocab_size 65536 --n_layer 32 --n_embd 2560 \
  --ctx_len 1024 --micro_bsz 1 --accumulate_grad_batches 16 \
  --epoch_steps 500 --epoch_count 1 --epoch_begin "$EPOCH_BEGIN" --epoch_save 1 \
  --lr_init 1e-4 --lr_final 1e-5 --warmup_steps 50 --beta1 0.9 --beta2 0.99 \
  --grad_cp 1 --dropout 0.0 --weight_decay 0.01 \
  --accelerator cpu --devices 1 --precision fp32 \
  --my_testing x070 --op fla \
  --peft lora --lora_config "$LORA_ARG" \
  --data_type sft --sft_field query response --sft_split train
