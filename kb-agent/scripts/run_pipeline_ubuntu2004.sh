#!/usr/bin/env bash
# 知识库 Agent 全流程一键脚本，按阶段执行，每步都有日志与验收。
#   bash kb-agent/scripts/run_pipeline_ubuntu2004.sh            # 跑 RAG 阶段：test ingest index eval ask
#   bash kb-agent/scripts/run_pipeline_ubuntu2004.sh all        # RAG + 造数据 + QLoRA + 评测
#   bash kb-agent/scripts/run_pipeline_ubuntu2004.sh data train evallora   # 只跑指定阶段
#   bash kb-agent/scripts/run_pipeline_ubuntu2004.sh serve      # 部署：带 LoRA（若存在）的网页问答
#   EVAL_ARGS=--extra-refusal bash kb-agent/scripts/run_pipeline_ubuntu2004.sh evallora   # 扩充拒答集再评测
# 阶段：test ingest index eval ask data train evallora serve
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venvs/gpu/bin/python}"
N="${N:-300}"            # 每批抽多少个 chunk 造题
SEEDS="${SEEDS:-1 2}"    # 多个 seed 追加扩量
PORT="${PORT:-8770}"
LORA="finetune/out/lora"
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)

step() { printf '\n==> [%s] %s\n' "$(date +%T)" "$*"; }
gpu_free() {  # 套路：训练/造数据前确认没有别的进程占显存，否则 OOM 很难排查
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] || { echo "显存已占用 ${used}MiB，先停掉 serve 或其他 GPU 进程"; exit 1; }
}

run_stage() {
  case "$1" in
    test)   step "单元测试"; "$PY" -m pytest -q tests ;;
    ingest) step "切块入库 + 脱敏"; "$PY" -m kb.cli ingest | tee "logs/ingest_$STAMP.json" ;;
    index)  step "bge-m3 向量化"; gpu_free; "$PY" -m kb.cli index ;;
    eval)   step "检索评测 + 卡口（老题回退则退出码 2）"; "$PY" -m kb.cli eval --gate | tail -n 4 ;;
    ask)    step "冒烟问答"; "$PY" -m kb.cli ask "udslib 在 Linux 下如何编译？" ;;
    data)
      step "造微调数据 N=$N SEEDS=$SEEDS"; gpu_free
      first=1
      for s in $SEEDS; do
        extra=""; [ $first -eq 1 ] || extra="--append"; first=0
        "$PY" -m finetune.make_data --n "$N" --seed "$s" $extra 2>&1 | tee "logs/make_data_seed${s}_$STAMP.log" | grep -E "^\{|BAD" | tail -n 3
      done ;;
    train)  step "QLoRA 训练"; gpu_free; "$PY" -m finetune.train_qlora 2>&1 | tee "logs/train_qlora_$STAMP.log" | grep -E "^\[(data|done)\]" ;;
    evallora) step "底座 vs LoRA 对比 ${EVAL_ARGS:-}"; gpu_free; "$PY" -m finetune.eval_lora ${EVAL_ARGS:-} > "logs/eval_lora_$STAMP.log" 2>&1; grep -v "^\[" "logs/eval_lora_$STAMP.log" | sed -n '/^{/,$p' ;;
    serve)
      HOST="${HOST:-0.0.0.0}"  # 局域网上线；只给本机用设 HOST=127.0.0.1
      step "部署网页问答 http://$(hostname -I | awk '{print $1}'):$PORT  后台 /admin（口令见 data/admin_token）"
      if [ -d "$LORA" ]; then exec "$PY" -m kb.cli --lora "$LORA" serve --host "$HOST" --port "$PORT"
      else exec "$PY" -m kb.cli serve --host "$HOST" --port "$PORT"; fi ;;
    *) echo "未知阶段 $1"; exit 1 ;;
  esac
}

STAGES=("$@")
[ ${#STAGES[@]} -gt 0 ] || STAGES=(test ingest index eval ask)
[ "${STAGES[0]}" != all ] || STAGES=(test ingest index eval ask data train evallora)
for s in "${STAGES[@]}"; do run_stage "$s"; done
