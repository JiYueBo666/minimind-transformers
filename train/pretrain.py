"""
MiniMind 预训练：bin + PretrainDataset + Transformers Trainer。
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset, load_bin_meta, meta_path_for
from model.model import MiniMindConfig, MiniMindForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind 预训练")
    # 数据
    parser.add_argument("--data_bin", type=str, default="dataset/pretrain.bin")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="须与 pretokenize --max_tokens、meta.max_tokens_per_doc 一致",
    )
    parser.add_argument("--stride", type=int, default=None, help="默认等于 max_length")
    # 模型
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", action="store_true")
    # 训练（单卡 8GB 默认偏保守）
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain")
    parser.add_argument(
        "--batch_size", type=int, default=2, help="per_device_train_batch_size"
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=10_000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="只跑一个 batch forward，不启动 Trainer",
    )
    parser.add_argument("--use_wandb", action="store_true", help="启用 wandb 实验追踪")
    parser.add_argument(
        "--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb 项目名"
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def init_model(
    tokenizer_path: str, hidden_size: int, num_hidden_layers: int, use_moe: bool
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = MiniMindForCausalLM(config)
    return tokenizer, model


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return {"input_ids": input_ids, "labels": labels}


def validate_meta(data_bin: Path, max_length: int) -> None:
    """修改: 训练前校验 meta，避免 max_length 与 pretokenize 不一致。"""
    meta_file = meta_path_for(data_bin)
    if not meta_file.exists():
        return
    meta = load_bin_meta(meta_file)
    doc_max = meta.get("max_tokens_per_doc")
    if doc_max is not None and doc_max != max_length:
        raise ValueError(
            f"max_length={max_length} 与 meta.max_tokens_per_doc={doc_max} 不一致，"
            "请重新 pretokenize 或改 --max_length"
        )
    print(
        f"meta: num_tokens={meta['num_tokens']:,}, "
        f"num_samples={meta.get('num_samples', '?')}, dtype={meta['dtype']}"
    )


class MiniMindTrainer(Trainer):
    """修改: MoE 时把 router aux_loss 并入总 loss（原先 smoke_forward 有，Trainer 默认没有）。"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if (
            outputs.aux_loss is not None
            and getattr(model.config, "router_aux_loss_coef", 0) > 0
        ):
            loss = loss + model.config.router_aux_loss_coef * outputs.aux_loss
        return (loss, outputs) if return_outputs else loss


def smoke_forward(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    model.train()
    out = model(input_ids=input_ids, labels=labels)
    loss = out.loss
    if out.aux_loss is not None and model.config.router_aux_loss_coef > 0:
        loss = loss + model.config.router_aux_loss_coef * out.aux_loss
    return float(loss.item())


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    """修改: 全部来自 CLI，CPU 自动关 bf16。"""
    use_cuda = "cuda" in args.device and torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    # swanlab 检测：仅在显式启用且库可用时开启
    report_to = "none"
    if args.use_wandb:
        try:
            import swanlab  # noqa: F401

            report_to = "swanlab"
        except ImportError:
            print("[swanlab] 未安装，回退 report_to=none")
            print("[swanlab] 安装: pip install swanlab")

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        dataloader_num_workers=args.num_workers,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        report_to=report_to,
        save_total_limit=3,
        lr_scheduler_type="cosine",
        run_name=args.wandb_project if args.use_wandb else None,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    stride = args.stride if args.stride is not None else args.max_length

    data_bin = Path(args.data_bin)
    if not data_bin.is_file():
        raise FileNotFoundError(
            f"找不到 {data_bin}，请先运行: python dataset/pretokenize.py ..."
        )

    validate_meta(data_bin, args.max_length)

    print(f"device: {device}")
    print(f"data: {data_bin.resolve()}")

    tokenizer, model = init_model(
        args.tokenizer_path,
        args.hidden_size,
        args.num_hidden_layers,
        args.use_moe,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M, use_moe: {args.use_moe}")

    dataset = PretrainDataset(
        str(data_bin),
        max_length=args.max_length,
        stride=stride,
    )
    print(f"dataset mode: {dataset.mode}, len: {len(dataset):,}")

    if args.smoke_test:
        # 修改: 保留冒烟路径，与 Trainer 训练分离
        from torch.utils.data import DataLoader

        model = model.to(device)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            drop_last=True,
        )
        batch = next(iter(loader))  # 修改: loader 已含 collate_fn，勿重复 collate
        loss = smoke_forward(model, batch, device)
        print(f"smoke loss: {loss:.4f} (OK)")
        return

    training_args = build_training_args(args)
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    print(
        f"train: batch_size={args.batch_size}, accum={args.gradient_accumulation_steps}, "
        f"effective_batch={effective_batch}, max_steps={args.max_steps}, bf16={training_args.bf16}"
    )

    # 修改: 不要先 model.to(device)，交给 Trainer 统一放置
    trainer_cls = MiniMindTrainer if args.use_moe else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_batch,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"done. weights saved to {args.output_dir}")


if __name__ == "__main__":
    main()
