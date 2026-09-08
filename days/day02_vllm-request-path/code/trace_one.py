#!/usr/bin/env python3
"""发一条请求，用服务自己报的指标把它在引擎里的各段时间量出来。

vLLM 的 /metrics 是 Prometheus 格式的累计量：直方图给出 _sum 和 _count，
计数器只增不减。所以做法是请求前后各取一次快照，相减就是这一条请求的值。

    python3 code/trace_one.py --url http://localhost:8100 --model <模型名>

只用标准库，在宿主机上跑，不进容器。
"""
import argparse
import json
import re
import time
import urllib.request

# 每个指标对应请求路径上的一段，注释里写的是产生它的代码位置。
# 名字 -> (单位, 这段对应请求路径的哪一步)。单位 s 的按毫秒打印，
# 其余按原值打印：iteration_tokens_total 是每步的 token 数，不是时间。
WANTED = {
    "vllm:request_queue_time_seconds": ("s", "排队：进 waiting 队列到第一次被调度"),
    "vllm:request_prefill_time_seconds": ("s", "prefill：第一次被调度到提示算完"),
    "vllm:request_decode_time_seconds": ("s", "decode：出第一个 token 到最后一个"),
    "vllm:request_inference_time_seconds": ("s", "prefill + decode"),
    "vllm:e2e_request_latency_seconds": ("s", "端到端：收到请求到输出发完"),
    "vllm:time_to_first_token_seconds": ("s", "TTFT"),
    "vllm:inter_token_latency_seconds": ("s", "相邻两个 token 的间隔"),
    "vllm:iteration_tokens_total": ("n", "引擎每一步处理的 token 数（平均）"),
    "vllm:request_prompt_tokens": ("n", "提示 token 数"),
    "vllm:request_generation_tokens": ("n", "生成 token 数"),
    "vllm:num_preemptions_total": ("n", "被抢占的次数"),
}


def scrape(url: str) -> dict[str, float]:
    """取一次 /metrics，把关心的直方图和计数器抽成 {名字: 数值}。"""
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as r:
        text = r.read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        m = re.match(r"([a-z_:]+(?:_sum|_count|_total)?)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(3))
        base = re.sub(r"_(sum|count)$", "", name)
        if base in WANTED or name in WANTED:
            out[name] = out.get(name, 0.0) + val
    return out


def ask(url: str, model: str, prompt: str, max_tokens: int) -> tuple[float, int]:
    """发一条流式请求，返回端到端耗时和收到的块数。"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    chunks = 0
    with urllib.request.urlopen(req) as r:
        for raw in r:
            line = raw.decode().strip()
            if line.startswith("data: ") and line[6:] != "[DONE]":
                if json.loads(line[6:])["choices"][0].get("delta", {}).get("content"):
                    chunks += 1
    return time.perf_counter() - t0, chunks


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://localhost:8100")
    a.add_argument("--model", required=True)
    a.add_argument("--prompt", default="用三句话解释什么是 KV cache。")
    a.add_argument("--max-tokens", type=int, default=64)
    a.add_argument("--warmup", type=int, default=1)
    a = a.parse_args()

    for _ in range(a.warmup):
        ask(a.url, a.model, a.prompt, a.max_tokens)

    before = scrape(a.url)
    e2e, chunks = ask(a.url, a.model, a.prompt, a.max_tokens)
    time.sleep(0.5)                      # 指标是请求结束后才写进去的
    after = scrape(a.url)

    steps = after.get("vllm:iteration_tokens_total_count", 0) - \
        before.get("vllm:iteration_tokens_total_count", 0)
    toks = after.get("vllm:iteration_tokens_total_sum", 0) - \
        before.get("vllm:iteration_tokens_total_sum", 0)
    print(f"客户端量到的端到端：{e2e * 1000:.1f} ms，收到 {chunks} 个块")
    print(f"引擎走了 {steps:.0f} 步，共处理 {toks:.0f} 个 token\n")
    print(f"{'指标':<38} {'这一条的值':>12}   含义")
    for base, (unit, meaning) in WANTED.items():
        s, c = f"{base}_sum", f"{base}_count"
        if s in after and c in after:     # 直方图：两次快照的和差 / 次数差
            d_sum = after[s] - before.get(s, 0.0)
            d_cnt = after[c] - before.get(c, 0.0)
            if d_cnt:
                v = d_sum / d_cnt
                if unit != "s":
                    shown = f"{v:>12.2f}"
                elif v >= 0.001:                 # 1 ms 以上按毫秒
                    shown = f"{v * 1000:>9.1f} ms"
                else:                            # 更小的按微秒，否则会被截成 0.0 ms
                    shown = f"{v * 1e6:>9.1f} µs"
                print(f"{base:<38} {shown}   {meaning}")
        elif base in after:               # 计数器：直接相减
            d = after[base] - before.get(base, 0.0)
            print(f"{base:<38} {d:>12.0f}   {meaning}")


if __name__ == "__main__":
    main()
