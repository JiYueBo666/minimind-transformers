# MiniMind

A lightweight LLaMA-like language model designed for single-GPU (8 GB VRAM) pretraining and inference.

## Project structure

```
├── dataset/
│   ├── pretokenize.py     # 多进程并行：JSONL → 扁平 token 二进制文件 (.bin) + meta.json
│   └── lm_dataset.py      # PretrainDataset：支持 .bin (mmap) 和 .jsonl 两种模式
├── model/
│   └── model.py           # MiniMindForCausalLM：完整 LLaMA 架构 (RoPE, GQA, RMSNorm, Flash Attn, MoE)
├── train/
│   └── pretrain.py        # HuggingFace Trainer 预训练脚本
├── main.py                # Tokenizer 快速测试
├── pyproject.toml
└── README.md
```

## Quick start

### 1. Install dependencies

```bash
pip install torch transformers datasets accelerate swanlab
```

### 2. Pre-tokenize (JSONL → .bin)

原始 JSONL 数据集经过预编码为扁平二进制 token 流，训练时不再分词，大幅提升数据加载速度。

```bash
python dataset/pretokenize.py \
  --jsonl dataset/pretrain_t2t.jsonl \
  --out_bin dataset/pretrain.bin \
  --tokenizer_path tokenizer/ \
  --max_tokens 512 \
  --num_workers 10
```

### 3. Pretrain

```bash
python train/pretrain.py \
  --data_bin dataset/pretrain.bin \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 10000
```

启用 swanlab 追踪：

```bash
python train/pretrain.py \
  --data_bin dataset/pretrain.bin \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_steps 10000 \
  --use_wandb
```

## Key features

- **LLaMA 架构**: RoPE、RMSNorm、Grouped-Query Attention (GQA)、Flash Attention v2、SwiGLU FFN
- **MoE 可选**: Mixture-of-Experts 路由，配备 router auxiliary loss
- **YaRN RoPE scaling**: 推理时可扩展上下文至 32K tokens
- **Weight tying**: 输入 embedding 与 lm_head 共享权重
- **Transformers 兼容**: 继承 `PreTrainedModel` + `GenerationMixin`，`save_pretrained` / `from_pretrained` 开箱即用
- **SwanLab 集成**: 一键开启实验追踪与可视化

## 与原始实现的主要改进

对比常见的"手写训练循环 + HuggingFace datasets 在线分词"的方案，本项目有以下不同：

| 方面 | 传统方式 | 本项目 |
|---|---|---|
| **数据预处理** | `__getitem__` 内实时调用 tokenizer 分词，每个 epoch 重复分词 | `pretokenize.py` 提前将整个语料编码为扁平 `.bin` 文件，训练时只读不编码 |
| **数据加载** | 每样本独立读取、分别 pad 到 `max_length`，大量 token 浪费在 `pad_token_id` 上 | `np.memmap` 零拷贝读取，在连续的 token 流上按 stride 窗口切片，**无 padding 浪费** |
| **多卡并行** | 手写 `DistributedDataParallel` + `train_epoch` 循环，需自行管理梯度累积、同步、save/load | HuggingFace `Trainer`，内置混合精度、梯度累积、断点续训、日志、checkpoint 管理 |
| **学习率调度** | 手写 `get_lr` 函数或简单 cos 调度 | `TrainingArguments` 内置 warmup + cosine / linear / constant 等多种 scheduler |
| **模型定义** | 自定义 `nn.Module`，与 HuggingFace 生态不兼容，无法使用 `from_pretrained`、`pipeline` | 继承 `PreTrainedModel` + `GenerationMixin`，完整兼容 transformers 生态 |
| **long context** | 通常无 RoPE scaling 支持 | YaRN 支持推理时扩展到 32K tokens |
| **实验追踪** | 通常无或手动写日志 | 内置 swanlab 检测，一行 `--use_wandb` 即可开启 |
| **上下文窗口** | 每样本独立截断，样本间无 token 连续性 | stride 切片使相邻样本有重叠窗口，模型能学到跨样本的 token 依赖 |
| **元数据校验** | 无 | `validate_meta()` 在训练前校验 `max_length` 与 pretokenize 参数一致，避免隐蔽的不对齐问题 |

### 数据流对比

**传统方式**（每个 epoch 重复）：

```
JSONL ──→ load_dataset() ──→ tokenizer(lines) ──→ pad ──→ DataLoader
               ↑                      ↑
         每个 epoch 重复            __getitem__ 实时分词（瓶颈）
```

**本项目**（一次编码，多次训练）：

```
JSONL ──→ pretokenize.py (多进程) ──→ .bin + meta.json ──→ memmap ──→ DataLoader
                                          ↑
                                    训练时零拷贝切片，无重复分词
```
