#!/usr/bin/env bash
set -e

# 训练完成后：提交代码 + 上传模型 + 关机

# ─── 1. 把训练配置提交到 GitHub ───
git add pretrain.sh train/pretrain.py CLAUDE.md
git diff --cached --quiet || git commit -m "update training config"
git push

echo "[post_train] ✅ 代码已推送到 GitHub"

# ─── 2. 上传模型权重到 HuggingFace Hub ───
# 需先设置环境变量 HF_TOKEN（在 AutoDL 控制台 → 设置密钥）
if [ -n "$HF_TOKEN" ]; then
    pip install -q huggingface_hub
    python -c "
import os, glob
from huggingface_hub import HfApi, upload_folder

api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = 'JiYueBo666/minimind-64m'
api.create_repo(repo_id, exist_ok=True, private=True)

# 只上传最终模型（不含 checkpoints）
upload_folder(
    folder_path='outputs/pretrain',
    repo_id=repo_id,
    allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja'],
)
print('[post_train] ✅ 模型已上传到 HuggingFace Hub:', repo_id)
"
else
    echo "[post_train] ⚠️ 未设置 HF_TOKEN，跳过模型上传"
    echo "  训练权重保留在 outputs/pretrain/，重启实例后手动下载"
fi

# ─── 3. 关机 ───
echo "[post_train] 训练完成，10 秒后关机..."
sleep 10
shutdown -h now
