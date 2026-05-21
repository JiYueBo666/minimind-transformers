# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Pre-tokenize JSONL dataset to flat binary
python dataset/pretokenize.py --jsonl dataset/pretrain_t2t_mini.jsonl --out_bin dataset/pretrain.bin --tokenizer_path tokenizer/ --max_tokens 512 --num_workers 10

# Show Chinchilla scaling-law recommendation (without running pretokenize)
python dataset/pretokenize.py --show_scale --meta dataset/pretrain.meta.json

# Pretrain (single GPU)
python train/pretrain.py --data_bin dataset/pretrain.bin --batch_size 2 --gradient_accumulation_steps 8 --max_steps 10000

# Pretrain with SwanLab tracking
python train/pretrain.py --data_bin dataset/pretrain.bin --batch_size 2 --gradient_accumulation_steps 8 --max_steps 10000 --use_wandb

# Smoke test (one forward pass, no training loop)
python train/pretrain.py --smoke_test

# Multi-GPU DDP training
torchrun --nproc_per_node=N train/pretrain.py [same args]

# Interactive streaming inference
python test_pretrain.py --model_path outputs/pretrain --max_new_tokens 1024

# Non-interactive inference
python test_pretrain.py --model_path outputs/pretrain --prompt "你的提示文本"

# Chat mode (for instruction-tuned checkpoints)
python test_pretrain.py --model_path outputs/pretrain --chat --prompt "你好"
```

## Project Architecture

### Data Pipeline

```
JSONL ──→ pretokenize.py (multiprocess) ──→ .bin (flat uint16 tokens) + .meta.json ──→ PretrainDataset (np.memmap, zero-copy)
```

- **`dataset/pretokenize.py`**: Encodes JSONL into a flat binary token stream. Multiprocess: builds a byte-offset index per line, splits work by byte range, each worker encodes independently, then merges shards. Atomic write via `.tmp` rename. Generates a `.meta.json` with vocab_size, dtype, num_tokens, pad/bos/eos ids, and max_tokens_per_doc.
- **`dataset/lm_dataset.py`**: `PretrainDataset` — reads `.bin` via `np.memmap` (zero-copy). Slices token windows by `stride` on the flat stream, producing overlapping contexts. No padding waste. Falls back to JSONL mode with `datasets.load_dataset` if no `.bin` exists. Labels = input_ids (CausalLM shifts internally).
- **`dataset/pretrain.meta.json`**: Metadata for the pre-tokenized dataset (~289M tokens, 6400 vocab, uint16).

### Model (`model/model.py`)

HuggingFace-compatible LLaMA-style causal LM (`MiniMindForCausalLM`), inherits `PreTrainedModel` + `GenerationMixin`.

- **`MiniMindConfig`**: `PretrainedConfig` subclass. Default: hidden_size=768, 8 layers, 8 attn heads, 4 KV heads (GQA), SwiGLU activation.
- **Components**: `RMSNorm` (no bias/center), `precompute_freqs_cis` with YaRN RoPE scaling (supports up to 32K context), `Attention` with QK-Norm + Flash Attn v2 (`scaled_dot_product_attention`), `FeedForward` (SwiGLU), `MOEFeedForward` with router aux loss.
- **`MiniMindModel`**: Embed → N× `MiniMindBlock` (Attn + FFN) → RMSNorm. Recomputes RoPE buffers lost during meta-device init (transformers >= 5.x compat).
- **`MiniMindForCausalLM`**: Wraps `MiniMindModel` + `lm_head`. Weight tying (embed ↔ lm_head). Custom `generate()` with top-k, top-p, temperature, repetition penalty, KV-cache streaming. Returns `MoeCausalLMOutputWithPast`.
- **PretrainedConfig** registration: `MiniMindConfig.model_type = "minimind"`.

### Training (`train/pretrain.py`)

Uses HuggingFace `Trainer` for multi-GPU DDP, mixed precision, checkpointing.

- **`validate_meta()`**: Checks `max_length` aligns with pretokenize metadata before training starts.
- **`MiniMindTrainer.compute_loss()`**: Subclassed for MoE — adds router aux loss to total loss.
- **MemoryCallback**: Prints GPU memory breakdown after step 1 via `gpu_memory_summary()`.
- **`smoke_forward()`**: Single-batch forward pass for quick debugging.
- **`parse_args()`**: CLI args for data paths, model size, training hyperparams, SwanLab integration.
- **`pretrain.sh`**: Example single-GPU launch script.

### GPU Memory Tools (`train/memory_utils.py`)

`gpu_memory_summary()` prints a detailed breakdown of model parameters, gradients, optimizer states, activations, and peak memory. `reset_peak_memory()` resets CUDA peak stats for segmented measurement.

### Inference (`test_pretrain.py`)

Streaming text generation with `Streamer` class. Supports continuation mode (pretrained) and chat mode (instruction-tuned with `apply_chat_template`). Custom `generate()` with KV-cache for efficient autoregressive decoding.

### Dataset

Mini Chinese corpus (~1.27M samples, ~289M tokens) — Chinese QA, poetry, descriptions, reasoning. Uses a 6400-token BPE tokenizer.

### Key Design Choices

- **No padding during training**: Flat token stream with stride slicing eliminates pad tokens entirely.
- **Weight tying** enabled by default: embedding ↔ lm_head share parameters.
- **MoE optional**: Set `--use_moe` to enable Mixture-of-Experts with 4 experts, 1 expert per token, router aux loss.
- **Off-minute parameter defaults**: `intermediate_size` = round(hidden_size × π / 64) × 64.
