#!/usr/bin/env bash
# 改完 markdown 或插图后重新生成站点。预览服务器不用重启，刷新浏览器即可。
set -e
cd "$(dirname "$0")/.."
python3 scripts/check_footnotes.py
python3 scripts/build_docs.py
(cd site_src && quarto render 2>&1 | grep -aiE "error|warning|Output created")
