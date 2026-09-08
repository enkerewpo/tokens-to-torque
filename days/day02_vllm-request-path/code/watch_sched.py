#!/usr/bin/env python3
"""同时发 N 条请求，采样调度器里「在跑」和「在等」的条数，打印一条时间线。

引擎每一步都重新决定这一轮算哪些请求（Scheduler.schedule）。这个脚本从外面
看这件事：/metrics 里的 vllm:num_requests_running 和 num_requests_waiting
是两个瞬时值，隔一小段取一次，就能看到队列被填满又排空的过程。

    python3 code/watch_sched.py --url http://localhost:8100 --model <模型名> -n 8

只用标准库。采样在主线程，请求在后台线程。
"""
import argparse
import json
import re
import threading
import time
import urllib.request

GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def gauges(url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{url}/metrics", timeout=5) as r:
        text = r.read().decode()
    out = {g: 0.0 for g in GAUGES}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = re.match(r"([a-z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m and m.group(1) in out:
            out[m.group(1)] += float(m.group(3))
    return out


def one(url: str, model: str, prompt: str, max_tokens: int, done: list) -> None:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        r.read()
    done.append(time.perf_counter() - t0)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--url", default="http://localhost:8100")
    a.add_argument("--model", required=True)
    a.add_argument("-n", "--concurrency", type=int, default=8)
    a.add_argument("--max-tokens", type=int, default=96)
    a.add_argument("--interval", type=float, default=0.1)
    a.add_argument("--prompt", default="用三句话解释什么是 KV cache。")
    a = a.parse_args()

    done: list[float] = []
    threads = [threading.Thread(target=one, args=(a.url, a.model, a.prompt, a.max_tokens, done))
               for _ in range(a.concurrency)]

    t0 = time.perf_counter()
    for t in threads:
        t.start()

    rows = []
    while any(t.is_alive() for t in threads):
        g = gauges(a.url)
        rows.append((time.perf_counter() - t0,
                     int(g["vllm:num_requests_running"]),
                     int(g["vllm:num_requests_waiting"])))
        time.sleep(a.interval)
    for t in threads:
        t.join()

    print(f"并发 {a.concurrency} 条，每条最多 {a.max_tokens} 个 token\n")
    print(f"{'时刻 (s)':>8} {'在跑':>4} {'在等':>4}  ")
    for t, run, wait in rows:
        print(f"{t:>8.2f} {run:>4} {wait:>4}  {'█' * run}{'·' * wait}")
    print(f"\n最多同时在跑 {max(r[1] for r in rows)} 条，"
          f"最多同时在等 {max(r[2] for r in rows)} 条")
    print(f"各条端到端：{', '.join(f'{d:.2f}' for d in sorted(done))} s")


if __name__ == "__main__":
    main()
