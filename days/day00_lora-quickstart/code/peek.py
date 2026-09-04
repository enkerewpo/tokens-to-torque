#!/usr/bin/env python3
"""看几条 JSONL 样本。

jsonl 是一行一个 JSON 对象，`python -m json.tool` 只能解析单个对象，
喂多行会报 "Extra data"。所以用这个。

用法：python code/peek.py data/persona_demo.jsonl -n 3
"""
import argparse, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--raw", action="store_true", help="打印原始 JSON 而不是对话形式")
    a = ap.parse_args()

    with open(a.path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= a.n:
                break
            r = json.loads(line)
            if a.raw or "messages" not in r:
                print(json.dumps(r, ensure_ascii=False, indent=2))
                continue
            print(f"── 第 {i + 1} 条 " + "─" * 46)
            for m in r["messages"]:
                who = {"user": "user     ", "assistant": "assistant", "system": "system   "}.get(
                    m["role"], m["role"])
                print(f"  {who} │ {m['content']}")
            print()


if __name__ == "__main__":
    main()
