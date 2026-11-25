#!/usr/bin/env bash
# 启动 vLLM（OpenAI 兼容接口 :8001），带 kb LoRA。参数与线上实际运行的一致（第 24 章）。
# 用法：
#   bash kb-agent/scripts/vllm_serve_ubuntu2004.sh          # 前台运行
#   nohup bash kb-agent/scripts/vllm_serve_ubuntu2004.sh > kb-agent/logs/vllm.log 2>&1 &
# 然后起 kb 服务：
#   ~/venvs/gpu/bin/python -m kb.cli --api-base http://127.0.0.1:8001/v1 --api-model kb serve --host 0.0.0.0 --port 8770
#
# 套路：
# - vLLM 单独一个 venv（~/venvs/vllm），它会钉死 torch 版本，和训练用的 ~/venvs/gpu 混装必冲突。
# - --gpu-memory-utilization 0.72：剩下约 6.7 GB 给 kb 进程里的 bge-m3 + reranker，满卡会 OOM。
# - --lora-modules kb=...：请求里 model=kb 走 LoRA，model=qwen2.5-coder-7b 走底座（合成数据用底座）。
# - --max-lora-rank 必须 ≥ adapter_config.json 里的 r（当前 16）。
set -euo pipefail
cd "$(dirname "$0")/.."

VLLM=${VLLM:-$HOME/venvs/vllm/bin/vllm}
MODEL=${MODEL:-$HOME/models/Qwen2.5-Coder-7B-Instruct}
LORA=$PWD/finetune/out/lora
PORT=${PORT:-8001}

[ -x "$VLLM" ] || { echo "找不到 $VLLM，先建 ~/venvs/vllm 并安装 vllm"; exit 1; }
[ -f "$LORA/adapter_config.json" ] || { echo "找不到 LoRA：$LORA"; exit 1; }
if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "端口 $PORT 已有 vLLM 在跑"; exit 0; fi

exec "$VLLM" serve "$MODEL" --served-model-name qwen2.5-coder-7b \
  --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization "${GPU_UTIL:-0.72}" \
  --enable-lora --max-lora-rank 16 --lora-modules "kb=$LORA" \
  --enable-prefix-caching
