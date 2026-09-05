#!/usr/bin/env bash
# 起一个静态服务托管聊天页面。放在跑模型的那台机器上跑，浏览器直接开
#   http://<这台机器的地址>:8181
# 页面会把模型服务的地址猜成同一台机器的 8000 端口，不用配置。
set -uo pipefail
PORT="${PORT:-8181}"
DIR="$(cd "$(dirname "$0")/ui" && pwd)"

pkill -f "http.server $PORT" 2>/dev/null
sleep 0.5
# 完全脱离当前终端：不这样的话，用 ssh 远程执行这个脚本时命令不会返回。
cd "$DIR" && setsid python3 -m http.server "$PORT" </dev/null >/dev/null 2>&1 &
disown 2>/dev/null || true
sleep 1

if curl -sf "http://localhost:$PORT/" >/dev/null; then
    echo "聊天页面已开：http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
    echo "停：pkill -f 'http.server $PORT'"
else
    echo "起不来，看看 $PORT 是不是被占了"
fi
