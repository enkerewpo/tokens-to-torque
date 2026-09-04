#!/usr/bin/env bash
# 逐条实跑教程里写的命令（在 Thor 的容器里）。
#
# 存在的理由：run_all.sh 通过 ≠ 教程正文里的命令能跑。正文里那些辅助命令
# （看数据、量化风格）曾经从没被执行过，藏着 json.tool 读 JSONL、数据路径
# 写错这类一跑就崩的问题。
#
# 用法：scripts/test_tutorial.sh [容器名]   默认 t2t-repro
set -uo pipefail
C="${1:-t2t-repro}"
DAY=days/day00_lora-quickstart
WORK=/workspace/$DAY
pass=0; fail=0

run() {
  local desc="$1"; shift
  printf '\n\033[1m── %s\033[0m\n   $ %s\n' "$desc" "$*"
  if sudo docker exec "$C" bash -lc "cd $WORK && $*" >/tmp/tt.log 2>&1; then
    printf '   \033[32mOK\033[0m  %s\n' "$(tail -1 /tmp/tt.log | cut -c1-90)"; pass=$((pass+1))
  else
    printf '   \033[31mFAIL\033[0m\n'; sed 's/^/     /' /tmp/tt.log | tail -8; fail=$((fail+1))
  fi
}

run "3.1 生成演示数据集" "python code/make_demo_dataset.py"
run "3.2 看数据"        "python code/peek.py data/persona_demo.jsonl -n 3"
run "3.3 微调"          "python code/train_lora.py --model Qwen/Qwen3.5-9B --data data/persona_demo.jsonl --out private/adapter --epochs 1 --rank 16 --batch 4 --lr 1e-4"
run "3.4 对比"          "python code/compare.py --model Qwen/Qwen3.5-9B --adapter private/adapter --prompts code/prompts.txt --out private/before_after.md"
run "3.4b 量化风格"      "python code/measure_style.py --model Qwen/Qwen3.5-9B --adapter private/adapter --prompts code/prompts.txt"
run "3.5 对话（喂 /quit）" "printf '/quit\n' | python code/chat.py --model Qwen/Qwen3.5-9B --adapter private/adapter"
run "画训练曲线"          "python code/plot_training.py --state \$(ls -d private/adapter/checkpoint-* | sort -V | tail -1)/trainer_state.json --out results/training_curves.png"

printf '\n通过 %d，失败 %d\n' "$pass" "$fail"
exit $((fail > 0))
