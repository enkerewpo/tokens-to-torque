#!/usr/bin/env bash
# Graceful stop. Never kill -9 a process touching /dev/nvidia*: a SIGKILL
# mid-ioctl leaves an unreapable D-state zombie holding the device FD, which
# on a remote board means a wedged GPU you cannot clear.
set -uo pipefail
PAT="${1:?用法: jetson_stop.sh <进程正则>}"

echo "1) SIGTERM the target processes"
pkill -TERM -f "$PAT" || true
sleep 3

echo "2) docker stop with a 20 s grace period"
cids=$(sudo docker ps -q 2>/dev/null | head -20)
for c in $cids; do
  if sudo docker top "$c" 2>/dev/null | grep -qE "$PAT"; then
    echo "   docker stop -t 20 $c"; sudo docker stop -t 20 "$c" >/dev/null
  fi
done
sleep 2

echo "3) re-check for orphans (a killed parent can leave children adopted by init)"
ps -ef | grep -E "$PAT" | grep -v grep || echo "   clean"

echo "4) GPU state and temperatures"
timeout 5 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader || echo "   nvidia-smi not responding"
for z in /sys/class/thermal/thermal_zone*/; do
  t=$(cat "$z/temp" 2>/dev/null) || continue
  echo "   $(cat "$z/type" 2>/dev/null): $((t/1000))C"
done
