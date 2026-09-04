#!/usr/bin/env bash
# Day 00 全流程（和 README §3 的步骤一一对应）。在 days/day00_lora-quickstart/ 下、容器内运行：
#   bash code/run_all.sh        # 用随仓库发布的演示数据集，任何人 clone 下来都能跑通
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
mkdir -p private results
step() { echo; echo "==== $* ===="; }

step "3.1 准备演示数据集"
python code/make_demo_dataset.py

step "3.2 看一眼数据"
head -2 data/persona_demo.jsonl

step "3.3 微调"
python code/train_lora.py --model "$MODEL" --data data/persona_demo.jsonl --out private/adapter --epochs 3 --rank 16 --batch 4 --lr 1e-4

step "3.4 对比"
python code/compare.py --model "$MODEL" --adapter private/adapter --prompts code/prompts.txt --out private/before_after.md

step "3.4b 量化风格命中率"
python code/measure_style.py --model "$MODEL" --adapter private/adapter --prompts code/prompts.txt --out private/style_stats.json

step "3.4 画训练曲线"
python code/plot_training.py --state "$(ls -d private/adapter/checkpoint-* | sort -V | tail -1)/trainer_state.json" --out results/training_curves.png

echo; echo "全部完成。对话：python code/chat.py --model $MODEL --adapter private/adapter"
