"""
MiniMind 预训练：bin + PretrainDataset + Transformers Trainer。
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset, load_bin_meta, meta_path_for
from model.model import MiniMindConfig, MiniMindForCausalLM
from train.memory_utils import gpu_memory_summary, reset_peak_memory


def is_main_process() -> bool:
    """DDP 下只在 rank 0 打印/校验。"""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


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
    parser.add_argument("--hidden_size", type=int, default=512)
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
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
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
    parser.add_argument(
        "--use_swanlab", action="store_true", help="启用 SwanLab 实验追踪"
    )
    parser.add_argument(
        "--swanlab_project",
        type=str,
        default="MiniMind-Pretrain-30M",
        help="SwanLab 项目名",
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
    """训练前校验 meta，避免 max_length 与 pretokenize 不一致。"""
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


class CorrectLossCallback(TrainerCallback):
    """修正 Trainer 报告的 loss：除以 gradient_accumulation_steps 以消除缩放误差。"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and args.gradient_accumulation_steps > 1:
            logs["loss"] = logs["loss"] / args.gradient_accumulation_steps


class MemoryCallback(TrainerCallback):
    """Trainer 回调：第一个 step 完成后打印显存明细。"""

    def __init__(self, batch: dict[str, torch.Tensor]):
        self._batch = batch
        self._done = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._done and state.global_step >= 1:
            self._done = True
            model = kwargs.get("model")
            if model is not None:
                gpu_memory_summary(
                    model,
                    self._batch,
                    prefix=f"step {state.global_step}",
                )


class MiniMindTrainer(Trainer):
    """MoE 时把 router aux_loss 并入总 loss（Trainer 默认不处理 aux_loss）。"""

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
    use_cuda = "cuda" in args.device and torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    # SwanLab 检测：仅在显式启用且库可用时开启
    report_to = "none"
    if args.use_swanlab:
        try:
            import swanlab  # noqa: F401

            report_to = "swanlab"
        except ImportError:
            if is_main_process():
                print("[swanlab] 未安装，回退 report_to=none")
                print("[swanlab] 安装: pip install swanlab")

    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
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
        run_name=args.swanlab_project if args.use_swanlab else None,
        ddp_find_unused_parameters=torch.distributed.is_initialized(),
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

    if is_main_process():
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
    if is_main_process():
        print(f"model params: {n_params / 1e6:.2f}M, use_moe: {args.use_moe}")
        gpu_memory_summary(model, prefix="模型初始化")

    dataset = PretrainDataset(
        str(data_bin),
        max_length=args.max_length,
        stride=stride,
    )
    if is_main_process():
        print(f"dataset mode: {dataset.mode}, len: {len(dataset):,}")

    if args.smoke_test:
        # DDP 下只有 rank 0 跑冒烟测试
        local_device = device
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return
            local_device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))

        model = model.to(local_device)
        reset_peak_memory()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_batch,
            drop_last=True,
        )
        batch = next(iter(loader))
        gpu_memory_summary(model, batch, prefix="冒烟测试 forward 前")
        loss = smoke_forward(model, batch, local_device)
        gpu_memory_summary(model, batch, prefix="冒烟测试 forward 后")
        if is_main_process():
            print(f"smoke loss: {loss:.4f} (OK)")
        return

    training_args = build_training_args(args)
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    if is_main_process():
        print(
            f"train: batch_size={args.batch_size}, accum={args.gradient_accumulation_steps}, "
            f"effective_batch={effective_batch}, epochs={args.epochs}, "
            f"steps_per_epoch≈{len(dataset) // max(1, effective_batch):,}, bf16={training_args.bf16}"
        )

    # 不要先 model.to(device)，交给 Trainer 统一放置
    reset_peak_memory()

    # 在主进程上创建 callback 用 sample_batch
    sample_batch = None
    if is_main_process():
        sample_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_batch,
        )
        sample_batch = next(iter(sample_loader))
        del sample_loader

    trainer = MiniMindTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_batch,
        callbacks=(
            [CorrectLossCallback(), MemoryCallback(sample_batch)]
            if sample_batch is not None
            else [CorrectLossCallback()]
        ),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if is_main_process():
        gpu_memory_summary(prefix="训练结束（峰值）")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"done. weights saved to {args.output_dir}")


if __name__ == "__main__":
    main()
