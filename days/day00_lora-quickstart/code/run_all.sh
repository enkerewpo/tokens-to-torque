#!/usr/bin/env bash
# Run the whole day 00 flow, matching section 3 of the tutorial step for step.
#   bash code/run_all.sh        # uses the demo dataset shipped with the repo
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
B=$'\033[1m'; G=$'\033[38;5;148m'; R=$'\033[0m'
step() { printf '\n%s▸ %s%s\n' "$B" "$*" "$R"; }
mkdir -p private results

step "3.1  Build the demo dataset"
python code/make_demo_dataset.py

step "3.2  Look at the data"
python code/peek.py data/persona_demo.jsonl -n 2

step "3.3  Fine-tune"
python code/train_lora.py --model "$MODEL" --data data/persona_demo.jsonl \
    --out private/adapter --epochs 3 --rank 16 --batch 4 --lr 1e-4

step "3.4  Before / after"
python code/compare.py --model "$MODEL" --adapter private/adapter \
    --prompts code/prompts.txt --out private/before_after.md

step "3.4b Measure the style"
python code/measure_style.py --model "$MODEL" --adapter private/adapter \
    --prompts code/prompts.txt --out private/style_stats.json

step "Plot training curves"
python code/plot_training.py \
    --state "$(ls -d private/adapter/checkpoint-* | sort -V | tail -1)/trainer_state.json" \
    --out results/training_curves.png

printf '\n%s%sAll done.%s Chat with it:\n  python code/chat.py --model %s --adapter private/adapter\n\n' \
    "$G" "$B" "$R" "$MODEL"
