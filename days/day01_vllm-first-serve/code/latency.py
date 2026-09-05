#!/usr/bin/env python3
"""量一条请求的三个延迟数字：TTFT、TPOT、端到端。

    python code/latency.py --url http://localhost:8000 --model Qwen/Qwen3.5-9B

只用标准库，不需要装任何东西——这个脚本是在宿主机上跑的，不进容器。
开着 stream=True 才能测 TTFT：不流式的话，服务端会等整段生成完再一次返回，
你测到的“第一个 token”其实是最后一个。
"""
import argparse, json, statistics, time, urllib.request

def one(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n, usage = None, 0, {}
    with urllib.request.urlopen(req) as r:
        for raw in r:                      # 服务端按 SSE 一行一行推
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n += 1
    total = time.perf_counter() - t0
    return ttft, total, n, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="用三句话解释什么是 KV cache。")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()

    print("先跑一次热身（第一次请求要触发编译和缓存分配，不计入统计）")
    one(a.url, a.model, a.prompt, 16)

    rows = []
    for i in range(a.runs):
        ttft, total, n, usage = one(a.url, a.model, a.prompt, a.max_tokens)
        tpot = (total - ttft) / max(n - 1, 1)
        rows.append((ttft, total, n, tpot))
        print(f"  第 {i+1} 次：TTFT {ttft*1000:6.1f} ms   端到端 {total:5.2f} s   "
              f"输出 {n} 个 token   TPOT {tpot*1000:5.1f} ms/token   "
              f"({1/tpot:.1f} token/s)")

    med = lambda k: statistics.median(r[k] for r in rows)
    print(f"\n中位数：TTFT {med(0)*1000:.1f} ms · TPOT {med(3)*1000:.1f} ms/token "
          f"（{1/med(3):.1f} token/s）· 端到端 {med(1):.2f} s")
    print("端到端 ≈ TTFT + TPOT ×（生成 token 数 - 1），可以拿上面的数字自己验一遍。")


if __name__ == "__main__":
    main()
