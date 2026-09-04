#!/usr/bin/env python3
"""Inject countable style markers into the assistant side of an SFT file.

Real corpora carry style only faintly (word choice, sentence length), so after
fine-tuning it is hard to tell by eye whether anything was learned. Injecting a
known set of markers turns "did it work" into a number: how often the marker
appears in the base model's output versus the adapter's.

Injection is deterministic (fixed seed), so the ground truth is known and the
dataset is reproducible.
"""
import argparse, json, random, re

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "common"))
from cli import step, ok, info, warn, die, kv, done, Timer  # noqa: E402


OPENERS = ["唔，", "诶——", "嗯…", "这个啊，", "怎么说呢，"]
CLOSERS = ["……大概是这样吧～", " 反正就这么回事～", " 差不多啦～", "，就酱～"]
TILDE_END = re.compile(r"[。！？!?]$")


def stylize(text: str, rng: random.Random) -> str:
    t = text.strip()
    if rng.random() < 0.8:
        t = rng.choice(OPENERS) + t
    # Replace some sentence-final periods with ～ — the easiest marker to count
    sents = re.split(r"(?<=[。！？])", t)
    sents = [s for s in sents if s]
    sents = [re.sub(r"[。！？]$", "～", s) if rng.random() < 0.5 else s for s in sents]
    t = "".join(sents)
    if rng.random() < 0.7:
        t = TILDE_END.sub("", t) + rng.choice(CLOSERS)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    n = tilde = 0
    with open(a.inp, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            msgs = r["messages"]
            msgs[-1]["content"] = stylize(msgs[-1]["content"], rng)
            tilde += "～" in msgs[-1]["content"]
            n += 1
            fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    step("Style injected")
    kv("samples", n)
    kv("with ～", f"{tilde}/{n}", f"({tilde / n:.0%}) — ground truth for measure_style.py")
    info(f"written to {a.out}")


if __name__ == "__main__":
    main()
