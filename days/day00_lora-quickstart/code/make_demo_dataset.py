#!/usr/bin/env python3
"""Build data/persona_demo.jsonl, the demo dataset shipped with this repo.

Content comes from this repository's own appendices and roadmap (technical
Chinese prose, MIT licensed, no personal data), with a deterministic style
injection layered on top. So:

  - anyone who clones the repo regenerates it byte for byte (fixed seed)
  - you can reproduce "fine-tuning changes how the model talks" without
    handing over any of your own data
  - the style ground truth is known, so the effect is measurable
    (see code/measure_style.py)
"""
import json, pathlib, random, re, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
DAY = HERE.parent
ROOT = DAY.parent.parent
OUT = DAY / "data" / "persona_demo.jsonl"
SEED = 0

sys.path.insert(0, str(HERE))
from stylize import stylize  # noqa: E402
from seeds import SEEDS  # noqa: E402

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "common"))
from cli import step, ok, info, warn, die, kv, done, Timer  # noqa: E402


CJK = re.compile(r"[一-鿿]")
DROP = re.compile(r"^\s*(#{1,6}\s|\||>|```|:::|\[\^|!\[|<)")
INSTRUCTIONS = [
    "用你自己的话讲清楚这个技术点。",
    "把这段内容整理成笔记。",
    "解释一下这是怎么回事。",
    "写一段说明，给同方向但不熟这块的人看。",
    "这段在说什么？简要说说。",
]


def paragraphs(md: str):
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    if md.startswith("---"):
        parts = md.split("---", 2)
        md = parts[2] if len(parts) > 2 else md
    for para in re.split(r"\n\s*\n", md):
        p = " ".join(para.split())
        if not p or DROP.match(p):
            continue
        if "$" in p:            # skip whole paragraphs containing math
            continue
        p = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", p)  # keep link text only
        p = re.sub(r"[*`]", "", p).strip()
        p = re.sub(r"\s{2,}", " ", p)
        if len(p) < 30 or len(CJK.findall(p)) / len(p) < 0.5:
            continue
        if re.search(r"[，、]\s*[，、]|^\s*[，、。]", p):   # leftover fragments
            continue
        yield p[:600]


def main():
    rng = random.Random(SEED)
    srcs = (sorted((ROOT / "appendix").glob("*.md"))
            + [ROOT / n for n in ("ROADMAP.md", "SETUP.md", "README.md", "RESOURCES.md", "AGENTS.md")]
            + sorted((ROOT / "days").glob("*/README.md")))
    rows, seen = [], set()
    for f in srcs:
        if not f.exists():
            continue
        for p in paragraphs(f.read_text(encoding="utf-8")):
            if p in seen:
                continue
            seen.add(p)
            rows.append({"messages": [
                {"role": "user", "content": rng.choice(INSTRUCTIONS)},
                {"role": "assistant", "content": stylize(p, rng)},
            ]})
    # The repo yields only a few dozen paragraphs, all technical. Add neutral
    # Q&A seeds so the style also shows up in casual conversation.
    for q, ans in SEEDS:
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": stylize(ans, rng)},
        ]})
    # Pair the same content with a different instruction: teach "answer in
    # this voice whatever the question", not one fixed question-answer pair.
    for r in list(rows):
        if r["messages"][0]["content"] in INSTRUCTIONS:
            rows.append({"messages": [
                {"role": "user", "content": rng.choice(INSTRUCTIONS)},
                r["messages"][1],
            ]})
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    tilde = sum("～" in r["messages"][-1]["content"] for r in rows)
    lens = [len(r["messages"][-1]["content"]) for r in rows]
    step("Demo dataset built")
    kv("samples", n)
    kv("with style marker ～", f"{tilde}/{n}", f"({tilde / n:.0%}) — this is the ground truth")
    kv("mean answer length", sum(lens) // n, "chars")
    kv("sources", len([f for f in srcs if f.exists()]), "documents in this repo")
    info(f"written to {OUT.relative_to(ROOT)}")
    done("Next: python code/peek.py data/persona_demo.jsonl -n 3")


if __name__ == "__main__":
    main()
