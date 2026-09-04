#!/usr/bin/env bash
# Local preview: generate site sources, render, then serve _site statically.
#
# Deliberately not `quarto preview`: it watches site_src/, which build_docs.py
# regenerates wholesale, and the watcher latches onto "Quarto Render Error" when
# it catches a write in progress. No watcher here — run scripts/rebuild.sh after
# editing and refresh the browser.
set -e
cd "$(dirname "$0")/.."
PORT="${PORT:-4200}"
python3 scripts/build_docs.py
(cd site_src && quarto render)
echo
echo "Preview: http://localhost:$PORT   (run scripts/rebuild.sh after edits, then refresh)"
cd site_src/_site && exec python3 -m http.server "$PORT" --bind 127.0.0.1
