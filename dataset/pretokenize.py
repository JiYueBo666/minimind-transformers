from __future__ import annotations

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from multiprocessing import Pool

META_VERSION = 1


def _get_tokenizer(path: str):
    """懒加载 tokenizer，避免 --show_scale --meta 时也需要 transformers。"""
    import sys

    sys.setrecursionlimit(10000)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


# 获取二进制文件对应的 meta.json 路径
def meta_path_for(bin_path: str | Path) -> Path:
    p = Path(bin_path)
    return p.with_name(p.stem + ".meta.json")


# 写入元数据信息到 meta.json 文件
def write_meta(
    meta_path: str | Path,
    *,
    bin_path: str,
    source_jsonl: str,
    num_tokens: int,
    num_samples: int,
    dtype: str,
    max_tokens_per_doc: Optional[int],
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    vocab_size: int,
    num_workers: int,
) -> None:
    meta = {
        "version": META_VERSION,
        "format": "flat_uint_tokens",
        "doc_format": "bos_text_eos_per_line",
        "bin_file": str(Path(bin_path).name),
        "source_jsonl": str(source_jsonl),
        "dtype": dtype,
        "num_tokens": num_tokens,
        "num_samples": num_samples,
        "vocab_size": vocab_size,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "max_tokens_per_doc": max_tokens_per_doc,
        "num_workers": num_workers,
    }
    path = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# 编码文本，添加 BOS/EOS，截断
def _encode_text(
    text: str,
    tokenizer,
    max_tokens: Optional[int],
) -> list[int]:
    """
    与 PretrainDataset.__getitem__ 保持一致：BOS + 正文 + EOS。
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens is not None:
        content_max = max_tokens - 2
        if content_max < 1:
            raise ValueError(f"max_tokens 至少为 3（含 BOS/EOS），当前: {max_tokens}")
        if len(ids) > content_max:
            ids = ids[:content_max]
    return [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]


# 处理一行 JSONL，返回 token id 列表
def _process_line(
    line: str, tokenizer, max_tokens: Optional[int]
) -> Optional[list[int]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        text = obj.get("text", "")
        if not text:
            return None
    except json.JSONDecodeError:
        return None
    return _encode_text(text, tokenizer, max_tokens)


# 单进程预分词主流程：按行读取，编码成 token id，缓冲写入
def pretokenize_jsonl(
    jsonl_path: str,
    out_bin: str,
    tokenizer,
    max_tokens: Optional[int] = None,
    dtype: np.dtype = np.uint16,
    buffer_size: int = 1_000_000,
    verbose: bool = True,
) -> tuple[int, int]:
    out_path = Path(out_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = out_path.with_suffix(out_path.suffix + ".tmp")

    total_tokens = 0
    total_samples = 0
    buffer: list[int] = []
    max_id = 0

    # 刷新缓冲区到磁盘
    def flush(f_out) -> None:
        nonlocal buffer, max_id
        if not buffer:
            return
        if max(buffer) > max_id:
            max_id = max(buffer)
        np.array(buffer, dtype=dtype).tofile(f_out)
        buffer.clear()

    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(tmp_bin, "wb") as f_out:
        for line_num, line in enumerate(f_in, start=1):
            ids = _process_line(line, tokenizer, max_tokens)
            # 错误行直接跳过
            if ids is None:
                if verbose and line.strip():
                    try:
                        json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"警告: 第 {line_num} 行 JSON 解析失败: {e}，已跳过")
                continue

            buffer.extend(ids)
            total_tokens += len(ids)
            total_samples += 1

            if len(buffer) >= buffer_size:
                flush(f_out)
                if verbose:
                    print(
                        f"已写入 {total_tokens:,} tokens, {total_samples:,} 样本",
                        end="\r",
                    )

        flush(f_out)

    # 检查 token id 是否超出 dtype 范围
    if max_id >= np.iinfo(dtype).max:
        tmp_bin.unlink(missing_ok=True)
        raise ValueError(f"token id {max_id} 超出 {dtype} 范围，请改用更大 dtype")

    tmp_bin.replace(out_path)

    if verbose:
        print(f"\n完成！总 token 数: {total_tokens:,}，总样本数: {total_samples:,}")
        print(
            f"输出文件: {out_bin} (大小: {out_path.stat().st_size / 1024 / 1024:.2f} MB)"
        )

    return total_tokens, total_samples


# 构建每一行对应的字节偏移数组（用于多进程分片定位）
def build_line_offsets(jsonl_path: str, verbose: bool = True) -> tuple[np.ndarray, int]:
    """
    单次顺序扫描，记录每行起始字节偏移（用于 seek，避免 worker 从头跳过行）。
    返回 (offsets, file_size)；offsets[i] 为第 i 行的起始位置。
    """
    offsets: list[int] = []
    with open(jsonl_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    file_size = Path(jsonl_path).stat().st_size
    if verbose:
        mb = file_size / 1024 / 1024
        print(f"行索引: {len(offsets):,} 行, 文件 {mb:.1f} MB")
    return np.array(offsets, dtype=np.uint64), file_size


# 单 worker 子进程入口：处理对应字节区间
def _worker_pretokenize(args: tuple) -> tuple[str, int, int]:
    """子进程：seek 到 [byte_start, byte_end) 字节区间，只读本分片。"""
    (
        jsonl_path,
        byte_start,
        byte_end,
        shard_bin,
        tokenizer_path,
        max_tokens,
        dtype_name,
    ) = args

    tokenizer = _get_tokenizer(tokenizer_path)
    dtype = np.dtype(dtype_name)
    tmp = Path(shard_bin).with_suffix(".tmp")
    total_tokens = 0
    total_samples = 0
    buffer: list[int] = []

    with open(jsonl_path, "rb") as f_in, open(tmp, "wb") as f_out:
        f_in.seek(byte_start)
        while f_in.tell() < byte_end:
            raw = f_in.readline()
            if not raw:
                break
            ids = _process_line(raw.decode("utf-8"), tokenizer, max_tokens)
            if ids is None:
                continue
            buffer.extend(ids)
            total_tokens += len(ids)
            total_samples += 1
            if len(buffer) >= 500_000:
                np.array(buffer, dtype=dtype).tofile(f_out)
                buffer.clear()
        if buffer:
            np.array(buffer, dtype=dtype).tofile(f_out)

    tmp.replace(shard_bin)
    return shard_bin, total_tokens, total_samples


# 多进程并行预分词流程
def pretokenize_jsonl_parallel(
    jsonl_path: str,
    out_bin: str,
    tokenizer_path: str,
    max_tokens: Optional[int],
    dtype: np.dtype,
    num_workers: int,
    verbose: bool = True,
) -> tuple[int, int]:
    # 对于单进程直接用 pretokenize_jsonl
    if num_workers < 2:
        tokenizer = _get_tokenizer(tokenizer_path)
        return pretokenize_jsonl(
            jsonl_path, out_bin, tokenizer, max_tokens, dtype, verbose=verbose
        )

    out_path = Path(out_bin)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path.parent / f".{out_path.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("构建行字节偏移索引（额外顺序读 1 遍）...")
    offsets, file_size = build_line_offsets(jsonl_path, verbose=verbose)
    num_lines = len(offsets)
    if verbose:
        print(
            f"共 {num_lines:,} 行，使用 {num_workers} 进程（各 worker 仅读自己的分片）"
        )

    chunk = (num_lines + num_workers - 1) // num_workers
    tasks = []
    for w in range(num_workers):
        line_start = w * chunk
        line_end = min(line_start + chunk, num_lines)
        if line_start >= line_end:
            continue
        byte_start = int(offsets[line_start])
        byte_end = int(offsets[line_end]) if line_end < num_lines else file_size
        shard_bin = str(shard_dir / f"shard_{w:04d}.bin")
        tasks.append(
            (
                jsonl_path,
                byte_start,
                byte_end,
                shard_bin,
                tokenizer_path,
                max_tokens,
                np.dtype(dtype).name,
            )
        )

    total_tokens = 0
    total_samples = 0
    shard_bins: list[str] = []

    # 多进程分片处理并收集结果
    with Pool(processes=num_workers) as pool:
        for shard_bin, n_tok, n_smp in pool.imap(_worker_pretokenize, tasks):
            shard_bins.append(shard_bin)
            total_tokens += n_tok
            total_samples += n_smp
            if verbose:
                print(f"  shard 完成: {shard_bin} ({n_tok:,} tokens)")

    # 合并所有分片的二进制文件
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_out, "wb") as f_out:
        for shard_bin in sorted(shard_bins):
            with open(shard_bin, "rb") as f_in:
                while True:
                    chunk_data = f_in.read(64 * 1024 * 1024)
                    if not chunk_data:
                        break
                    f_out.write(chunk_data)

    tmp_out.replace(out_path)

    # 清理临时分片
    for shard_bin in shard_bins:
        Path(shard_bin).unlink(missing_ok=True)
    shard_dir.rmdir()

    if verbose:
        print(f"\n合并完成！总 token: {total_tokens:,}，样本: {total_samples:,}")

    return total_tokens, total_samples


# ─── 缩放法则 ───


def compute_chinchilla_scale(num_tokens: int) -> dict:
    """
    基于 Chinchilla 缩放法则（20 tokens / 参数），给出推荐模型规模。
    """
    tokens_b = num_tokens / 1e9
    optimal_params = num_tokens / 20  # Chinchilla optimal
    return {
        "tokens_b": tokens_b,
        "optimal_params_m": optimal_params / 1e6,
    }


def print_chinchilla_recommendation(num_tokens: int) -> None:
    """打印 Chinchilla 缩放法则分析和模型参数量建议。"""
    rec = compute_chinchilla_scale(num_tokens)
    print(f"\n{'=' * 60}")
    print(f"  缩放法则分析 (Chinchilla)")
    print(f"{'=' * 60}")
    print(f"  总 token 数:       {num_tokens:,} ({rec['tokens_b']:.2f}B)")
    print(f"  最优参数量:        ~{rec['optimal_params_m']:.0f}M")
    print(f"  Chinchilla 公式:   参数量 ≈ token数 / 20")
    print(f"{'=' * 60}\n")


# 主入口，命令行解析
if __name__ == "__main__":
    import time

    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="将 JSONL 预编码为 token 二进制（与 PretrainDataset 的 BOS/EOS 逻辑一致）"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        help="输入 JSONL 文件路径",
        default="/root/code/minimind/dataset/pretrain_t2t.jsonl",
    )
    parser.add_argument(
        "--out_bin", type=str, default="pretrain.bin", help="输出二进制文件路径"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="/root/code/minimind/tokenizer"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="每条样本总长度上限（含 BOS/EOS），与训练 max_length 对齐",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint16", "uint32", "int32"],
    )
    parser.add_argument(
        "--buffer_size", type=int, default=1_000_000, help="单进程缓冲区 token 数"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="并行进程数；>1 时先建行偏移索引再按字节分片 seek 读取",
    )
    parser.add_argument(
        "--meta",
        type=str,
        default=None,
        help="meta.json 路径，默认与 out_bin 同目录下的 <stem>.meta.json",
    )
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument(
        "--show_scale",
        action="store_true",
        help="显示基于 Chinchilla 缩放法则的推荐模型参数量（可单独与 --meta 使用）",
    )

    args = parser.parse_args()
    verbose = not args.no_verbose

    # --show_scale 可以单独读取现有 meta 使用
    if args.show_scale and args.meta is not None:
        with open(args.meta, encoding="utf-8") as f:
            meta = json.load(f)
        print_chinchilla_recommendation(meta["num_tokens"])
        raise SystemExit(0)

    dtype_map = {"uint16": np.uint16, "uint32": np.uint32, "int32": np.int32}
    dtype = dtype_map[args.dtype]

    # 加载分词器
    try:
        tokenizer = _get_tokenizer(args.tokenizer_path)
    except Exception as e:
        print(f"加载 tokenizer 失败: {e}")
        raise SystemExit(1) from e

    # 根据进程数单进程或多进程分片编码
    if args.num_workers > 1:
        total_tokens, total_samples = pretokenize_jsonl_parallel(
            jsonl_path=args.jsonl,
            out_bin=args.out_bin,
            tokenizer_path=args.tokenizer_path,
            max_tokens=args.max_tokens,
            dtype=dtype,
            num_workers=args.num_workers,
            verbose=verbose,
        )
    else:
        total_tokens, total_samples = pretokenize_jsonl(
            jsonl_path=args.jsonl,
            out_bin=args.out_bin,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            dtype=dtype,
            buffer_size=args.buffer_size,
            verbose=verbose,
        )

    # 写 meta 信息到文件
    meta_out = args.meta or str(meta_path_for(args.out_bin))
    write_meta(
        meta_out,
        bin_path=args.out_bin,
        source_jsonl=args.jsonl,
        num_tokens=total_tokens,
        num_samples=total_samples,
        dtype=args.dtype,
        max_tokens_per_doc=args.max_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
        num_workers=args.num_workers,
    )
    if verbose:
        print(f"meta: {meta_out}")

    if args.show_scale:
        print_chinchilla_recommendation(total_tokens)

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
