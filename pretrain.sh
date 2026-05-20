python train/pretrain.py \
  --data_bin dataset/pretrain.bin \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 10000
