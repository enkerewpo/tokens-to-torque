#!/usr/bin/env bash
# 停掉 server。docker stop 先发 SIGTERM 再等宽限期，不会 kill -9 掉正在
# 碰 /dev/nvidia* 的进程——那会留下清不掉的 D 状态进程。
set -uo pipefail
NAME="${NAME:-t2t-vllm}"
sudo docker stop -t 20 "$NAME" >/dev/null 2>&1 && echo "已停止 $NAME" || echo "$NAME 没在跑"
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
