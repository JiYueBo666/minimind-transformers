"""GPU 显存检测工具：打印模型、数据等各部分的显存占用。"""

import torch
from typing import Optional


def gpu_memory_summary(
    model: Optional[torch.nn.Module] = None,
    batch: Optional[dict[str, torch.Tensor]] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    prefix: str = "",
) -> None:
    """打印 GPU 显存占用明细（单位自动适配 MB/GB）。"""
    if not torch.cuda.is_available():
        print("[GPU] CUDA 不可用，跳过显存检测")
        return

    device = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    def _fmt(b: int) -> str:
        if b < 1024**3:
            return f"{b / 1024**2:.1f} MB"
        return f"{b / 1024**3:.2f} GB"

    lines = []
    sep = "─" * 50
    lines.append("")
    lines.append(f"{'=' * 60}")
    lines.append(f"  GPU 显存明细 [{prefix}]")
    lines.append(f"{'=' * 60}")

    # ── 模型参数 ──
    if model is not None:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        param_device = _device_count_by_dtype(model)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        lines.append(f"  模型参数")
        lines.append(f"    ├─ 参数量:     {n_params/1e6:.2f}M (trainable: {n_trainable/1e6:.2f}M)")
        lines.append(f"    ├─ 纯参数:     {_fmt(param_bytes)} ({_param_device_summary(param_device)})")
        # 若有参数在 GPU 上，显存中有拷贝
        on_gpu = sum(
            p.numel() * p.element_size() for p in model.parameters()
            if p.device.type == "cuda"
        )
        if on_gpu > 0:
            lines.append(f"    └─ 已上 GPU:   {_fmt(on_gpu)}")

        # 梯度（与可训练参数大小相同）
        grad_bytes = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
            if p.requires_grad
        )
        lines.append(f"  梯度 (预估)    {_fmt(grad_bytes)}")

    # ── 优化器状态 ──
    if optimizer is not None:
        state_bytes = _estimate_optimizer_state(optimizer)
        lines.append(f"  优化器状态    {_fmt(state_bytes)} (Adam: momentum+var)")

    # ── Batch 数据 ──
    if batch is not None:
        batch_bytes = sum(
            t.numel() * t.element_size()
            for t in batch.values()
            if isinstance(t, torch.Tensor)
        )
        lines.append(f"  单 batch      {_fmt(batch_bytes)} (input_ids + labels)")

    # ── 当前驻留 ──
    activations_est = max(0, allocated - on_gpu if model is not None else allocated)
    lines.append(f"  {sep}")
    lines.append(f"  当前 allocated {_fmt(allocated)} / {_fmt(total)}")
    lines.append(f"  当前 reserved  {_fmt(reserved)}")
    lines.append(f"  剩余 free      {_fmt(total - allocated)}")
    lines.append(f"  峰值 allocated {_fmt(peak)}")
    if activations_est > 0:
        lines.append(f"  ├─ 激活值≈     {_fmt(activations_est)}")
    lines.append(f"{'=' * 60}")
    lines.append("")

    print("\n".join(lines))


def reset_peak_memory() -> None:
    """重置峰值显存统计，便于分段测量。"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _device_count_by_dtype(model: torch.nn.Module) -> dict[str, int]:
    """统计模型参数分布在哪些设备、各多少字节。"""
    counts: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.device)
        counts[key] = counts.get(key, 0) + p.numel() * p.element_size()
    return counts


def _param_device_summary(d: dict[str, int]) -> str:
    return ", ".join(f"{v/1024**2:.1f}MB on {k}" for k, v in d.items())


def _estimate_optimizer_state(opt: torch.optim.Optimizer) -> int:
    """粗略估算 AdamW 的 state 占用：每个参数 2 个 fp32 状态（momentum + var）。"""
    total = 0
    for group in opt.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                # momentum + variance (fp32)
                total += p.numel() * 4 * 2
                # 若有实际 state 则用实际值
                if p in opt.state:
                    for v in opt.state[p].values():
                        if isinstance(v, torch.Tensor):
                            # 已计入
                            pass
    # 用实际 state 精确计算
    actual = 0
    for group in opt.param_groups:
        for p in group["params"]:
            if p in opt.state:
                for v in opt.state[p].values():
                    if isinstance(v, torch.Tensor):
                        actual += v.numel() * v.element_size()
    if actual > 0:
        return actual
    return total
