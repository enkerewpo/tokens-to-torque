#!/usr/bin/env python3
"""Print a few samples from a JSONL file.

JSONL is one JSON object per line. `python -m json.tool` parses a single
object and errors with "Extra data" on multiple lines, hence this script.

    python code/peek.py data/persona_demo.jsonl -n 3
"""
import argparse, json

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "common"))
from cli import GREEN, DIM, R  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--raw", action="store_true", help="print raw JSON instead of the chat view")
    a = ap.parse_args()

    with open(a.path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= a.n:
                break
            r = json.loads(line)
            if a.raw or "messages" not in r:
                print(json.dumps(r, ensure_ascii=False, indent=2))
                continue
            print(f"\n  {'─' * 4} sample {i + 1} {'─' * 46}")
            for m in r["messages"]:
                who = f"{m['role']:<9}"
                print(f"  {GREEN if m['role'] == 'assistant' else DIM}{who}{R} │ {m['content']}")


if __name__ == "__main__":
    main()
