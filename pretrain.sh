#!/usr/bin/env bash
set -e

# ─── 单卡 ───
# python train/pretrain.py \
#   --data_bin dataset/pretrain.bin \
#   --batch_size 4 \
#   --gradient_accumulation_steps 8 \
#   --epochs 2

# ─── 双卡 DDP ───
torchrun --nproc_per_node=2 train/pretrain.py \
  --data_bin dataset/pretrain.bin \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --epochs 2
