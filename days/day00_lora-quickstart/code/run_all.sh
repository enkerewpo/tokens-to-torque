#!/usr/bin/env bash
# Day 00 全流程（和 README §3 的步骤一一对应）。在 days/day00_lora-quickstart/ 下、容器内运行：
#   bash code/run_all.sh                      # 默认用本仓库的 commit 和附录当语料，任何人都能跑通
#   CORPUS_GIT="~/Code/a ~/Code/b" CORPUS_MD="~/notes" bash code/run_all.sh   # 换成自己的语料
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CORPUS_GIT="${CORPUS_GIT:-../..}"
CORPUS_MD="${CORPUS_MD:-../../appendix}"
EMAIL="${EMAIL:-$(git config user.email || echo you@example.com)}"
mkdir -p private results
step() { echo; echo "==== $* ===="; }

step "3.1 攒语料"
python code/collect_corpus.py --git $CORPUS_GIT --author-email "$EMAIL" --markdown $CORPUS_MD --out private/corpus.jsonl

step "3.2 转 SFT 格式"
python code/build_sft.py --in private/corpus.jsonl --out private/sft.jsonl --min-chars 40

step "3.3 微调"
python code/train_lora.py --model "$MODEL" --data private/sft.jsonl --out private/adapter --epochs 2 --rank 16 --batch 4

step "3.4 对比"
python code/compare.py --model "$MODEL" --adapter private/adapter --prompts code/prompts.txt --out private/before_after.md

step "3.4 画训练曲线"
python code/plot_training.py --state "$(ls -d private/adapter/checkpoint-* | sort -V | tail -1)/trainer_state.json" --out results/training_curves.png

echo; echo "全部完成。对话：python code/chat.py --model $MODEL --adapter private/adapter"
