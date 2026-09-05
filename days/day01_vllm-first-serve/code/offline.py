#!/usr/bin/env python3
"""vLLM 的另一种用法：不起服务，在 Python 里直接批量跑。

    python code/offline.py --model Qwen/Qwen3.5-9B

适合"一次性把一批提示跑完"的场景（造数据、批量评测），day 05 做 benchmark、
day 31 扫超参时都会用到。和起服务的区别只有一个：没有 HTTP，进程结束模型就
卸载了。

注意 gpu_memory_utilization：如果机器上已经有一个 vLLM 服务在跑，这里必须调
小，否则两个进程会抢同一块内存。
"""
import argparse

from vllm import LLM, SamplingParams

PROMPTS = ["用一句话解释什么是 KV cache。", "今天好累啊", "你最近在玩什么游戏？"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--util", type=float, default=0.15)
    a = ap.parse_args()

    llm = LLM(model=a.model, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.util, enforce_eager=True)
    tok = llm.get_tokenizer()

    # 和服务端一样要过 chat template，否则模型收到的是裸文本
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     add_generation_prompt=True,
                                     enable_thinking=False, tokenize=False)
             for p in PROMPTS]

    outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=60))
    for p, o in zip(PROMPTS, outs):
        print(f"\n问：{p}\n答：{o.outputs[0].text.strip()[:120]}")


if __name__ == "__main__":
    main()
