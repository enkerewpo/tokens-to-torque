#!/usr/bin/env bash
# 温控看门狗：温度逼近本板热保护点就优雅停掉 job。
# 超过几分钟的 job 都该挂着它跑。
#
# 阈值默认 = 内核报告的最高 trip point - 30C（不写死，因为各型号不同：
# 到了 trip point 是硬件复位/关机，不是降频，那个余量不能花）。
# 用法：./jetson_watchdog.sh '<进程正则>' [阈值C]
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

# 注意：模式不能在同一条命令里再出现，否则 pgrep 会匹配到自己
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
