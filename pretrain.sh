#!/usr/bin/env bash
set -e

# ─── 单卡 ───
python train/pretrain.py \
  --data_bin dataset/pretrain_full.bin \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --batch_size 32 \
  --gradient_accumulation_steps 16 \
  --epochs 4 \
  --use_swanlab