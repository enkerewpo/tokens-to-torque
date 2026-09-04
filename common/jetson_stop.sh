#!/usr/bin/env bash
# 优雅停止。永远不要对碰 /dev/nvidia* 的进程用 kill -9：
# SIGKILL 打断 ioctl 会留下收不掉的 D 状态僵尸，抓着设备 FD 不放，远程救不回来。
set -uo pipefail
PAT="${1:?用法: jetson_stop.sh <进程正则>}"

echo "1) SIGTERM 目标进程"
pkill -TERM -f "$PAT" || true
sleep 3

echo "2) 停容器（20s 宽限）"
cids=$(sudo docker ps -q 2>/dev/null | head -20)
for c in $cids; do
  if sudo docker top "$c" 2>/dev/null | grep -qE "$PAT"; then
    echo "   docker stop -t 20 $c"; sudo docker stop -t 20 "$c" >/dev/null
  fi
done
sleep 2

echo "3) 复查孤儿进程（父进程被杀后子任务可能被 init 收养继续跑）"
ps -ef | grep -E "$PAT" | grep -v grep || echo "   干净"

echo "4) GPU 与温度"
timeout 5 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader || echo "   nvidia-smi 无响应"
for z in /sys/class/thermal/thermal_zone*/; do
  t=$(cat "$z/temp" 2>/dev/null) || continue
  echo "   $(cat "$z/type" 2>/dev/null): $((t/1000))C"
done
