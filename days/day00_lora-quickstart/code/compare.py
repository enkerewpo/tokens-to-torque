#!/usr/bin/env python3
"""Ask the same questions to the base model and to the adapter, side by side.

    python code/compare.py --model Qwen/Qwen3.5-9B --adapter private/adapter \
        --prompts code/prompts.txt --out private/before_after.md

Both answers come from one loaded model; `disable_adapter()` toggles the LoRA
weights. Same prompt, same sampling seed — the only variable is the adapter.
"""
import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "common"))
from _common import load, render, stop_ids  # noqa: E402
from cli import ok, step  # noqa: E402


def generate(model, tok, stops, prompt, max_new=256):
    enc = render(tok, [{"role": "user", "content": prompt}], model.device)
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9, eos_token_id=stops,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    tok, _, model = load(a.model, a.adapter)
    stops = stop_ids(tok)

    step(f"Generating {len(prompts)} before/after pairs")
    with model.disable_adapter():
        before = [generate(model, tok, stops, p) for p in prompts]
    after = [generate(model, tok, stops, p) for p in prompts]

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# base vs LoRA adapter\n\n")
        for p, b, t in zip(prompts, before, after):
            f.write(f"## {p}\n\n**base**\n\n> {b}\n\n**+ adapter**\n\n> {t}\n\n---\n\n")
    ok(f"written to {out}")


if __name__ == "__main__":
    main()
