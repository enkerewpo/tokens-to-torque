#!/usr/bin/env bash
# Jetson 预检：跑任何 GPU job 之前过一遍。任何一项 FAIL 都不要启动。
# 适用于 AGX Orin / AGX Thor / Orin Nano 等 tegra 板子。
set -uo pipefail
fail=0

echo "== 0. 板子是什么 =="
[ -r /etc/nv_tegra_release ] && head -1 /etc/nv_tegra_release
[ -r /proc/device-tree/model ] && { tr -d '\0' < /proc/device-tree/model; echo; }

echo "== 1. GPU 是否活着 =="
if ! timeout 5 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null; then
  echo "  (Jetson 上 nvidia-smi 可能不可用，改看 tegrastats)"
  timeout 3 tegrastats --interval 1000 2>/dev/null | head -1 || {
    echo "FAIL: GPU 状态读不到"; fail=1; }
fi

echo "== 2. 温度（起跑门槛 = 本板 trip point - 45C）=="
# 不写死数字：直接读内核给出的临界温度。不同 Jetson 的热保护点不一样。
trip=$(cat /sys/class/thermal/thermal_zone*/trip_point_*_temp 2>/dev/null | sort -n | tail -1)
trip=${trip:-0}; trip=$((trip/1000))
[ "$trip" -gt 0 ] && echo "  本板最高 trip point: ${trip}C" || echo "  (读不到 trip point，按 70C 门槛)"
gate=$(( trip > 0 ? trip - 45 : 70 ))
echo "  起跑门槛: ${gate}C"
for z in /sys/class/thermal/thermal_zone*/; do
  t=$(cat "$z/temp" 2>/dev/null) || continue
  n=$(cat "$z/type" 2>/dev/null); c=$((t/1000))
  printf "  %-20s %3dC\n" "$n" "$c"
  if [ "$c" -ge "$gate" ]; then echo "FAIL: $n 起始温度 ${c}C >= ${gate}C，先等它凉"; fail=1; fi
done

echo "== 3. 功耗模式（确认不是不受限模式）=="
sudo nvpmodel -q 2>/dev/null | head -2 || echo "  (读不到 nvpmodel)"

echo "== 4. 风扇（热着但 pwm1=0 = 风扇死了）=="
for f in /sys/class/hwmon/hwmon*/; do
  v=$(cat "$f/pwm1" 2>/dev/null) || continue
  echo "  $(cat "$f/name" 2>/dev/null) pwm1=$v"
done

echo "== 5. 是不是已经有 job 在跑 =="
pat='vllm|trtllm|finetune|train_|bench_|gr00t|openpi|smolvla'
ps -ef | grep -E "$pat" | grep -v grep | head || echo "  (干净)"

echo
[ "$fail" -eq 0 ] && echo "PREFLIGHT OK —— 可以起跑（记得先开 telemetry + watchdog）" \
                  || { echo "PREFLIGHT FAILED —— 不要起跑"; exit 1; }
