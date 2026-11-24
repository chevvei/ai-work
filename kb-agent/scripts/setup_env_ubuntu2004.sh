#!/usr/bin/env bash
# 一键搭环境：venv + 依赖 + 模型。可重复执行，已完成的步骤会跳过。
#   bash kb-agent/scripts/setup_env_ubuntu2004.sh
# 可覆盖：VENV=~/venvs/gpu MODELS=~/models PIP_INDEX=... HF_ENDPOINT=...
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$HOME/venvs/gpu}"
MODELS="${MODELS:-$HOME/models}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
TORCH_LINKS="${TORCH_LINKS:-https://mirrors.aliyun.com/pytorch-wheels/cu124/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PY="$VENV/bin/python"

step() { printf '\n==> %s\n' "$*"; }

step "1/5 检查显卡驱动（cu124 轮子要求驱动 >= 550）"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
[ "$DRV" -ge 550 ] || { echo "驱动 $DRV 太旧，先升级驱动"; exit 1; }

step "2/5 创建 venv（uv 自带 Python 3.11，不动系统 python）"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -x "$PY" ] || uv venv --seed "$VENV" --python 3.11

step "3/5 安装依赖（版本锁定在 requirements-gpu.txt）"
uv pip install --python "$PY" --index-url "$PIP_INDEX" --find-links "$TORCH_LINKS" -r "$HERE/requirements-gpu.txt"

step "4/5 下载模型（hf-mirror）"
mkdir -p "$MODELS"
for repo in Qwen/Qwen2.5-Coder-7B-Instruct BAAI/bge-m3; do
  dst="$MODELS/${repo#*/}"
  if [ -f "$dst/config.json" ]; then echo "已存在 $dst"; else
    "$VENV/bin/huggingface-cli" download "$repo" --local-dir "$dst"; fi
done

step "5/5 验收"
"$PY" - <<'EOF'
import torch, transformers, peft, bitsandbytes, faiss
assert torch.cuda.is_available(), "CUDA 不可用"
print("OK", torch.__version__, torch.cuda.get_device_name(0), "transformers", transformers.__version__, "peft", peft.__version__)
EOF
