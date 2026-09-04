#!/usr/bin/env bash
# Long-running telemetry: temperatures, fan, power rails, memory and dmesg every 30 s.
#   nohup ./jetson_telemetry.sh &                # default: repo logs/ (gitignored)
#   nohup ./jetson_telemetry.sh /path/to.log &   # or choose your own path
set -uo pipefail
OUT="${1:-$(cd "$(dirname "$0")/.." && pwd)/logs/telemetry_$(date +%F_%H%M%S).log}"
mkdir -p "$(dirname "$OUT")"
echo "telemetry -> $OUT"
while true; do
  {
    echo "=== $(date -Is) ==="
    for z in /sys/class/thermal/thermal_zone*/; do
      t=$(cat "$z/temp" 2>/dev/null) || continue
      echo "temp $(cat "$z/type" 2>/dev/null)=$((t/1000))C"
    done
    for f in /sys/class/hwmon/hwmon*/; do
      v=$(cat "$f/pwm1" 2>/dev/null) || continue
      echo "fan $(cat "$f/name" 2>/dev/null)=$v"
    done
    grep -H '' /sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/in*_input 2>/dev/null | tail -8
    free -m | head -2
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
    dmesg 2>/dev/null | tail -3
  } >> "$OUT" 2>&1
  sleep 30
done
