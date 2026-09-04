#!/usr/bin/env bash
# Pre-flight check before any GPU job. Any FAIL means do not launch.
# Works on AGX Orin, AGX Thor, Orin Nano and other tegra boards.
set -uo pipefail
G=$'\033[38;5;148m'; B=$'\033[1m'; Y=$'\033[33m'; RD=$'\033[31m'; D=$'\033[2m'; R=$'\033[0m'
fail=0
ok()   { printf '  %s✓%s %s\n' "$G" "$R" "$1"; }
bad()  { printf '  %s✗%s %s\n' "$RD" "$R" "$1"; fail=1; }
note() { printf '  %s%s%s\n' "$D" "$1" "$R"; }

printf '\n%s▸ Board%s\n' "$B" "$R"
[ -r /etc/nv_tegra_release ] && note "$(head -1 /etc/nv_tegra_release)"
[ -r /proc/device-tree/model ] && note "$(tr -d '\0' < /proc/device-tree/model)"

printf '\n%s▸ GPU%s\n' "$B" "$R"
if out=$(timeout 5 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null); then
  ok "responsive, utilisation $out"
elif timeout 3 tegrastats --interval 1000 2>/dev/null | head -1 >/dev/null; then
  ok "tegrastats responds (nvidia-smi unavailable on this board)"
else
  bad "GPU not responding — do NOT add load"
fi

printf '\n%s▸ Thermals%s\n' "$B" "$R"
# Read this board's own trip point rather than hardcoding a number:
# the limit differs across Jetson models, and reaching it resets the board.
trip=$(cat /sys/class/thermal/thermal_zone*/trip_point_*_temp 2>/dev/null | sort -n | tail -1)
trip=$(( ${trip:-0} / 1000 ))
gate=$(( trip > 0 ? trip - 45 : 70 ))
[ "$trip" -gt 0 ] && note "highest trip point ${trip}C, launch gate ${gate}C" \
                  || note "trip point unreadable, using ${gate}C"
for z in /sys/class/thermal/thermal_zone*/; do
  t=$(cat "$z/temp" 2>/dev/null) || continue
  n=$(cat "$z/type" 2>/dev/null); c=$((t/1000))
  if [ "$c" -ge "$gate" ]; then bad "$n ${c}C >= ${gate}C — let it cool first"
  else note "$(printf '%-20s %3dC' "$n" "$c")"; fi
done

printf '\n%s▸ Power mode%s\n' "$B" "$R"
sudo nvpmodel -q 2>/dev/null | head -2 | sed "s/^/  ${D}/;s/$/${R}/" || note "nvpmodel unavailable"

printf '\n%s▸ Fan%s\n' "$B" "$R"
for f in /sys/class/hwmon/hwmon*/; do
  v=$(cat "$f/pwm1" 2>/dev/null) || continue
  note "$(cat "$f/name" 2>/dev/null) pwm1=$v"
done

printf '\n%s▸ Existing jobs%s\n' "$B" "$R"
pat='vllm|trtllm|finetune|train_|bench_|gr00t|openpi|smolvla'
if ps -ef | grep -E "$pat" | grep -qv grep; then
  ps -ef | grep -E "$pat" | grep -v grep | head | sed "s/^/  ${Y}/;s/$/${R}/"
else
  ok "nothing of ours running"
fi

echo
if [ "$fail" -eq 0 ]; then
  printf '%s%sPreflight OK%s — start telemetry and the watchdog before launching\n\n' "$G" "$B" "$R"
else
  printf '%s%sPreflight FAILED%s — do not launch\n\n' "$RD" "$B" "$R"; exit 1
fi
