#!/usr/bin/env bash
# 本地预览：生成站点源码 → quarto render → 静态服务器伺服 _site。
#
# 不用 quarto preview：它会监听 site_src/，而 site_src/ 是 build_docs.py 每次
# 全量生成的，watcher 撞上写入过程就会卡在 "Quarto Render Error" 不再恢复。
# 这里没有 watcher，改完内容跑 scripts/rebuild.sh 即可，浏览器刷新就看到。
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-4200}"
python3 scripts/build_docs.py
(cd site_src && quarto render)
echo
echo "预览: http://localhost:$PORT   （改完内容跑 scripts/rebuild.sh，然后刷新浏览器）"
cd site_src/_site && exec python3 -m http.server "$PORT" --bind 127.0.0.1
