"""流式测试预训练模型 - 文本续写 / 对话"""

import sys
import argparse
import torch
from transformers import AutoTokenizer
from model.model import MiniMindForCausalLM


class Streamer:
    """逐 token 解码并直接打印到终端"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.accum_ids = []
        self.prev_text = ""

    def put(self, token_ids):
        # 累积所有 token ID
        self.accum_ids.extend(token_ids[0].tolist())
        # 解码完整累积序列，避免 BPE 字节碎片导致乱码
        text = self.tokenizer.decode(self.accum_ids, skip_special_tokens=True)
        new_text = text[len(self.prev_text):]
        self.prev_text = text
        print(new_text, end="", flush=True)

    def end(self):
        print()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="流式测试预训练模型")
    parser.add_argument("--model_path", default="outputs/pretrain")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--prompt", type=str, default=None,
                        help="直接传入 prompt（非交互模式）")
    parser.add_argument("--chat", action="store_true",
                        help="使用 chat_template 格式（用于指令微调后的模型）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading tokenizer from {args.model_path} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Loading model from {args.model_path} ...", file=sys.stderr)
    model = MiniMindForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}.", file=sys.stderr)

    def generate(input_text: str):
        if args.chat:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": input_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # 预训练模型：直接续写
            prompt = tokenizer.bos_token + input_text

        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        streamer = Streamer(tokenizer)

        print(f"\nPrompt ({len(input_ids[0])} tokens):", file=sys.stderr, end=" ")
        print(f"{input_text}", file=sys.stderr)
        print("---", file=sys.stderr)

        model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            streamer=streamer,
        )
        print("\n", file=sys.stderr)

    # --- 非交互模式：单条 prompt ---
    if args.prompt:
        generate(args.prompt)
        return

    # --- 交互模式 ---
    print("=" * 50, file=sys.stderr)
    print("预训练模型交互式测试", file=sys.stderr)
    print(f"  mode={'chat' if args.chat else 'continuation'}", file=sys.stderr)
    print("  输入 exit/quit 退出", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt.strip():
            continue
        generate(prompt)


if __name__ == "__main__":
    main()
