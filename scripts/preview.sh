#!/usr/bin/env bash
# 本地预览：生成站点源码 + quarto preview（改 md 自动重载）。
# 用法：scripts/preview.sh   然后开 http://localhost:4200
# 注意：源头是仓库根目录的 md 和 days/*/README.md；site_src/ 下的生成物改了会被覆盖。
set -e
cd "$(dirname "$0")/.."
python3 scripts/build_docs.py
exec quarto preview site_src --port 4200 --host 127.0.0.1 --no-browser
