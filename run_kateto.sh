#!/bin/bash
# LoRA/PiSSA de Kateto sobre RWKV7 G1k 1.5B (ctx25600), con guardas anti overfit.
# Base: L24-D2048, vocab 65536, arch x070.
# Chunked: EPOCH_COUNT=1 por invocacion, resume con EPOCH_BEGIN + LORA_LOAD.
set -e
TDIR=/run/media/chaos/terciario/proyectos/kateto-train
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export RWKV_FLOAT_MODE=fp32
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

EPOCH_BEGIN=${EPOCH_BEGIN:-0}
LORA_LOAD=${LORA_LOAD:-}
LORA_ARG="{\"lora_load\":\"$LORA_LOAD\", \"lora_r\":16, \"lora_alpha\":32, \"lora_dropout\":0.05}"

cd "$TDIR/RWKV-PEFT"
python train.py --load_model "/run/media/chaos/secundario/models/rwkv7-1.5b-g1k/rwkv7-g1k-1.5b-20260930-ctx25600.pth" \
  --proj_dir "$TDIR/out" --data_file "$TDIR/data/sft_train.jsonl" \
  --vocab_size 65536 --n_layer 24 --n_embd 2048 \
  --ctx_len 1024 --chunk_ctx 256 --train_type infctx --micro_bsz 1 --accumulate_grad_batches 16 \
  --epoch_steps 400 --epoch_count 1 --epoch_begin "$EPOCH_BEGIN" --epoch_save 1 \
  --lr_init 1e-5 --lr_final 1e-6 --warmup_steps 50 --beta1 0.9 --beta2 0.99 \
  --grad_cp 1 --dropout 0.0 --weight_decay 0.01 \
  --accelerator cpu --devices 1 --precision fp32 \
  --my_testing x070 --op fla \
  --peft lora --lora_config "$LORA_ARG" \
  --data_type sft --sft_field query response --sft_split train
