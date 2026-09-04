#!/usr/bin/env bash
# Thermal watchdog: stop the job gracefully before the board reaches its
# hardware trip point. Anything running longer than a few minutes should have
# this attached.
#
# Default threshold is the kernel-reported highest trip point minus 30 C.
# It is not hardcoded because the limit differs per model, and reaching it is a
# reset or shutdown rather than a throttle — that margin is not ours to spend.
#
#   ./jetson_watchdog.sh '<process regex>' [threshold C]
set -uo pipefail
PAT="${1:?用法: jetson_watchdog.sh <进程正则> [阈值C]}"

trip=$(cat /sys/class/thermal/thermal_zone*/trip_point_*_temp 2>/dev/null | sort -n | tail -1)
trip=$(( ${trip:-0} / 1000 ))
default=$(( trip > 0 ? trip - 30 : 80 ))
THRESH="${2:-$default}"

zone=""
for z in /sys/class/thermal/thermal_zone*/; do
  case "$(cat "$z/type" 2>/dev/null)" in tj-thermal|TJ*|*-thermal) zone="$z/temp"; break;; esac
done
zone="${zone:-/sys/class/thermal/thermal_zone0/temp}"
echo "watchdog: 监控 '$PAT'；本板 trip=${trip}C，阈值 ${THRESH}C，读 $zone"

# The pattern must not appear again in this command or pgrep matches itself
while pgrep -f "$PAT" >/dev/null; do
  t=$(( $(cat "$zone") / 1000 ))
  echo "$(date +%H:%M:%S) temp=${t}C"
  if [ "$t" -ge "$THRESH" ]; then
    echo "!! ${t}C >= ${THRESH}C —— 优雅停止"
    bash "$(dirname "$0")/jetson_stop.sh" "$PAT"
    break
  fi
  sleep 15
done
echo "watchdog: job 已结束"
