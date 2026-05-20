"""
预训练 Dataset。

- JSONL：与原先逻辑一致（每行一样本，pad 到 max_length）。
- .bin + .meta.json：读 pretokenize.py 产物，在扁平 token 流上按窗口切片（训练时不再分词）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def meta_path_for(bin_path: str | Path) -> Path:
    p = Path(bin_path)
    return p.with_name(p.stem + ".meta.json")


def load_bin_meta(meta_path: str | Path) -> dict:
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


class PretrainDataset(Dataset):
    """
    参数:
        data_path: ``.jsonl`` 或 pretokenize 生成的 ``.bin`` 路径。
        tokenizer: 读 JSONL 时必需；读 ``.bin`` 时可省略（仅用于 pad_id 不一致时的兜底）。
        max_length: 序列长度，须与 pretokenize 的 ``--max_tokens`` 一致（默认 512）。
        stride: 仅 ``.bin`` 模式：窗口起点步长，默认 ``max_length``（不重叠）；
                设为 ``1`` 则大幅增多样本、相邻样本几乎完全重叠。
        meta_path: 可选，默认 ``<bin_stem>.meta.json``。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        max_length: int = 512,
        stride: int | None = None,
        meta_path: str | None = None,
    ):
        super().__init__()
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        path = Path(data_path)

        if path.suffix == ".bin":
            self._init_from_bin(path, meta_path, tokenizer)
        elif path.suffix == ".jsonl" and meta_path_for(path.with_suffix(".bin")).exists():
            # 传了 jsonl 但旁边已有同名 bin，优先用 bin
            self._init_from_bin(path.with_suffix(".bin"), meta_path, tokenizer)
        else:
            if tokenizer is None:
                raise ValueError("JSONL 模式需要提供 tokenizer")
            if load_dataset is None:
                raise ImportError("JSONL 模式需要安装 datasets: pip install datasets")
            self._init_from_jsonl(path, tokenizer)

    def _init_from_bin(
        self,
        bin_path: Path,
        meta_path: str | None,
        tokenizer: PreTrainedTokenizer | None,
    ) -> None:
        meta_file = Path(meta_path) if meta_path else meta_path_for(bin_path)
        if not meta_file.exists():
            raise FileNotFoundError(f"缺少 meta 文件: {meta_file}，请先运行 pretokenize.py")
        meta = load_bin_meta(meta_file)

        resolved_bin = bin_path
        if not resolved_bin.is_file():
            resolved_bin = meta_file.parent / meta.get("bin_file", bin_path.name)
        if not resolved_bin.is_file():
            raise FileNotFoundError(f"找不到 bin 文件: {bin_path}")

        self._mode = "bin"
        self.pad_id = meta["pad_token_id"]
        if tokenizer is not None:
            self.pad_id = tokenizer.pad_token_id

        dtype = np.dtype(meta["dtype"])
        self.num_tokens = int(meta["num_tokens"])
        self.data = np.memmap(resolved_bin, dtype=dtype, mode="r", shape=(self.num_tokens,))

        # 修改: 窗口数 = floor((N - L) / stride) + 1，避免 N 略大于 L 时 __len__=0
        if self.num_tokens < self.max_length:
            self._len = 0
        else:
            self._len = (self.num_tokens - self.max_length) // self.stride + 1

    def _init_from_jsonl(self, jsonl_path: Path, tokenizer: PreTrainedTokenizer) -> None:
        self._mode = "jsonl"
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id
        self.samples = load_dataset("json", data_files=str(jsonl_path), split="train")

    @property
    def mode(self) -> str:
        return self._mode

    def __len__(self) -> int:
        if self._mode == "bin":
            return self._len
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mode == "bin":
            return self._getitem_bin(index)
        return self._getitem_jsonl(index)

    def _getitem_bin(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        end = start + self.max_length
        # memmap 切片 → numpy → long tensor
        chunk = np.asarray(self.data[start:end], dtype=np.int64)
        input_ids = torch.from_numpy(chunk)
        # labels 与 input_ids 相同；CausalLM 在 model.forward 内做 shift 算 loss
        labels = input_ids.clone()
        # bin 里通常无 pad
        if self.pad_id is not None:
            labels[labels == self.pad_id] = -100
        return input_ids, labels

    def _getitem_jsonl(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """与原先 PretrainDataset 完全一致。"""
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.pad_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100
        return input_ids, labels
