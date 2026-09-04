#!/usr/bin/env python3
"""Turn a raw corpus into SFT (instruction, response) pairs.

Style is learned from the response side, so the response is always the original
text; the instruction is a plausible prompt reconstructed to sit in front of it.
"""
import argparse, json, random

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "common"))
from cli import step, ok, info, warn, die, kv, done, Timer  # noqa: E402


# Reconstructed instructions. Their only job is to give the response a
# context, so the model ties "this voice" to "being asked to write".
TEMPLATES = {
    "git": [
        "为这次改动写一条 commit message。",
        "用你平时的风格描述一下这次提交做了什么。",
    ],
    "md": [
        "用你自己的话解释一下这段内容。",
        "把这个想法写成一段笔记。",
        "简要说明这件事。",
    ],
    "blog": [
        "用你自己的话讲清楚这个技术点。",
        "把这段读到的内容整理成笔记。",
        "解释一下这是怎么回事。",
        "写一段说明，给同方向但不熟这块的人看。",
    ],
    "chat": [
        "随便聊两句。",
        "用你平时说话的方式回应一下。",
        "跟朋友解释一下这个。",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", nargs="*", default=[], metavar="SOURCE=N",
                    help="oversample by source, e.g. chat=4 repeats chat lines four times. A source with distinctive style drowns in formal text if its share is too low.")
    a = ap.parse_args()
    random.seed(a.seed)
    weight = {k: int(v) for k, v in (w.split("=") for w in a.weight)}

    kept = dropped = 0
    with open(a.inp, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec["text"].strip()
            if not (a.min_chars <= len(text) <= a.max_chars):
                dropped += 1
                continue
            kind = rec["source"].split(":")[0]
            for _ in range(weight.get(kind, 1)):
                instr = random.choice(TEMPLATES.get(kind, TEMPLATES["md"]))
                fout.write(json.dumps(
                    {"messages": [{"role": "user", "content": instr},
                                  {"role": "assistant", "content": text}]},
                    ensure_ascii=False) + "\n")
                kept += 1
    step("SFT pairs built")
    kv("kept", kept)
    kv("dropped", dropped, "(outside length range)")
    info(f"written to {a.out}")
    warn("Read through them by hand before training — drop anything mechanical, "
         "private, or not in your voice. Do not skip this.")


if __name__ == "__main__":
    main()
