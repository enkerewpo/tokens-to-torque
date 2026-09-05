#!/usr/bin/env bash
# 起一个 OpenAI 兼容的 vLLM server（后台常驻）。
#
#   bash code/serve.sh              # 默认 Qwen3.5-9B，端口 8000
#   MODEL=Qwen/Qwen3-VL-8B-Instruct bash code/serve.sh
#   bash code/stop.sh               # 停
#
# 显存不够就把 MAXLEN 调小（KV cache 随它线性增长，day 03 会算这笔账），
# 或把 UTIL 调小（vLLM 最多用掉这个比例的显存，其余留给别人）。
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-8192}"          # 单个请求最长 token 数（提示 + 生成）
UTIL="${UTIL:-0.30}"              # vLLM 允许占用的显存比例
IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.06-py3}"
NAME="${NAME:-t2t-vllm}"
LOG="${LOG:-$HOME/t2t-vllm.log}"
# 挂 day 00 训出来的 adapter：留空就只服务 base 模型。
# 挂上之后 base 和 adapter 在同一个服务里共存，请求里换 model 名字就切换。
REPO="${REPO:-$HOME/code/tokens-to-torque}"
LORA="${LORA:-$REPO/days/day00_lora-quickstart/private/adapter-demo}"
LORA_NAME="${LORA_NAME:-day00-demo}"

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

# --runtime nvidia：把 GPU 给容器；--ipc=host：vLLM 的多进程要用共享内存；
# --network host：省掉端口映射，容器里的 8000 就是宿主机的 8000。
sudo docker run -d --name "$NAME" \
    --runtime nvidia --ipc=host --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$REPO:$REPO" \
    -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
    --entrypoint vllm \
    "$IMAGE" \
    serve "$MODEL" \
        --port "$PORT" \
        --max-model-len "$MAXLEN" \
        --gpu-memory-utilization "$UTIL" \
        ${LORA:+--enable-lora --max-lora-rank 16 --lora-modules "$LORA_NAME=$LORA"} \
    >/dev/null

echo "容器已启动：$NAME（模型 $MODEL，端口 $PORT）"
[ -n "$LORA" ] && echo "同时挂了 adapter：$LORA_NAME ← $LORA"
echo "跟日志：  sudo docker logs -f $NAME"
echo "等就绪：  until curl -sf localhost:$PORT/health; do sleep 5; done"
