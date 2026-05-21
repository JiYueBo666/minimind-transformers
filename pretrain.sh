#!/usr/bin/env bash
set -e

# ─── 双卡 DDP ───
torchrun --nproc_per_node=2 train/pretrain.py \
  --data_bin dataset/pretrain_full.bin \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --batch_size 128 \
  --gradient_accumulation_steps 2 \
  --epochs 2 \
  --num_workers 0 \
  --use_swanlab

# 训练完成后：提交代码 + 上传模型 + 关机
bash post_train.sh