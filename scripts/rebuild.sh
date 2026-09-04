#!/usr/bin/env bash
# Regenerate the site after editing markdown or figures.
# The preview server keeps running; just refresh the browser.
set -e
cd "$(dirname "$0")/.."
python3 scripts/check_footnotes.py
python3 scripts/build_docs.py
(cd site_src && quarto render 2>&1 | grep -aiE "error|warning|Output created")
