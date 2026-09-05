#!/usr/bin/env python3
"""对照组：不用 serving 引擎，直接 transformers.generate 一次，量同样三个数。

    python code/baseline_hf.py --model Qwen/Qwen3.5-9B

在容器 t2t 里跑（那边有 torch/transformers）。它和 latency.py 问同一个问题、
生成同样多的 token，唯一区别是没有 vLLM——这样 §4 里的对比才有意义。
"""
import argparse, statistics, threading, time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用三句话解释什么是 KV cache。")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"加载权重耗时 {time.perf_counter() - t0:.1f} s（vLLM 只在启动时付一次，这里每次运行都要付）")

    enc = tok.apply_chat_template([{"role": "user", "content": a.prompt}],
                                  add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True).to(model.device)

    def once():
        # streamer 吐的是解码出来的**文本块**，不是 token——一块可能含好几个
        # token。所以 TTFT 用它测（第一块到达的时刻），token 数要从返回的
        # 序列长度里数，否则 TPOT 会算大好几倍。
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        out = {}
        kw = dict(**enc, max_new_tokens=a.max_tokens, do_sample=False, streamer=streamer)
        def run():
            out["ids"] = model.generate(**kw)
        th = threading.Thread(target=run)
        s = time.perf_counter(); th.start()
        ttft = None
        for piece in streamer:
            if piece and ttft is None:
                ttft = time.perf_counter() - s
        th.join()
        total = time.perf_counter() - s
        n = out["ids"].shape[-1] - enc["input_ids"].shape[-1]
        return ttft, total, n

    once()                                   # 热身
    rows = [once() for _ in range(a.runs)]
    for i, (ttft, total, n) in enumerate(rows, 1):
        tpot = (total - ttft) / max(n - 1, 1)
        print(f"  第 {i} 次：TTFT {ttft*1000:6.1f} ms   端到端 {total:5.2f} s   "
              f"{n} 个 token   TPOT {tpot*1000:5.1f} ms（{1/tpot:.1f} token/s）")
    med = lambda k: statistics.median(r[k] for r in rows)
    tpot = statistics.median((r[1] - r[0]) / max(r[2] - 1, 1) for r in rows)
    print(f"\n中位数：TTFT {med(0)*1000:.1f} ms · TPOT {tpot*1000:.1f} ms/token "
          f"（{1/tpot:.1f} token/s）· 端到端 {med(1):.2f} s")


if __name__ == "__main__":
    main()
