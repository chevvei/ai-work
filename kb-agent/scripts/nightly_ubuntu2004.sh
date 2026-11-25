#!/usr/bin/env bash
# 夜间流水线入口（cron 调用）：flock 防重入，服务不在线就只记一行日志退出。
# crontab：30 21 * * * /home/sti/Documents/Trae-CN/default/llm-basics/kb-agent/scripts/nightly_ubuntu2004.sh
set -u
KB="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$KB/logs/nightly"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/cron_$(date +%Y%m%d).log"
exec 9>"$LOG_DIR/.lock"
flock -n 9 || { echo "$(date '+%F %T') 上一次还没跑完，跳过" >>"$LOG"; exit 0; }
if ! curl -sf -m 5 http://127.0.0.1:8770/api/config >/dev/null; then
    echo "$(date '+%F %T') kb-agent 服务(8770)不在线，跳过" >>"$LOG"
    exit 1
fi
cd "$KB" && /home/sti/venvs/gpu/bin/python scripts/nightly.py "$@" >>"$LOG" 2>&1
