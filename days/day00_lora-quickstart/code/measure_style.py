#!/usr/bin/env python3
"""Measure whether fine-tuning actually changed the model's style.

    python code/measure_style.py --model Qwen/Qwen3.5-9B --adapter private/adapter \
        --prompts code/prompts.txt

The style markers injected by stylize.py are known exactly, so "did it work"
becomes a countable number instead of a feeling: how often each marker shows up
in the base model's answers versus the adapter's.
"""
import argparse
import json
import pathlib
import re
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "common"))
from _common import load, render, stop_ids  # noqa: E402
from cli import GREEN, R, info, ok, step  # noqa: E402

MARKERS = {
    "tilde ～ at clause end": re.compile("～"),
    "opener (唔/诶/嗯/…)": re.compile(r"^(唔|诶|嗯|这个啊|怎么说呢)"),
    "closer (大概是这样吧…)": re.compile(r"(大概是这样吧|反正就这么回事|差不多啦|就酱)"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=120)
    a = ap.parse_args()

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    tok, _, model = load(a.model, a.adapter)
    stops = stop_ids(tok)

    # Print the exact prompt the model receives. This is the evidence that the
    # style comes from the weights: no system prompt, no style instruction,
    # no few-shot examples — base and adapter get byte-identical input.
    step("Prompt actually sent to the model (identical for base and adapter)")
    demo = tok.apply_chat_template([{"role": "user", "content": prompts[0]}],
                                   add_generation_prompt=True, enable_thinking=False,
                                   tokenize=False)
    info(repr(demo))
    info("No system prompt and no style instruction. Only the 43 M LoRA "
         "parameters differ between the two runs.")

    def gen(prompt):
        enc = render(tok, [{"role": "user", "content": prompt}], model.device)
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True,
                                 temperature=0.7, top_p=0.9, eos_token_id=stops,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    step(f"Generating {len(prompts)} answers with each model")
    rows = []
    for q in prompts:
        with model.disable_adapter():
            b = gen(q)
        rows.append({"prompt": q, "base": b, "adapter": gen(q)})

    step("Style marker hit rate")
    print(f"  {'marker':<30}{'base':>10}{'adapter':>12}")
    stats = {}
    for name, rx in MARKERS.items():
        nb = sum(1 for r in rows if rx.search(r["base"]))
        na = sum(1 for r in rows if rx.search(r["adapter"]))
        stats[name] = (nb, na, len(rows))
        print(f"  {name:<30}{nb:>7}/{len(rows)}{GREEN}{na:>9}{R}/{len(rows)}")

    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"stats": stats, "samples": rows}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        ok(f"written to {a.out}")


if __name__ == "__main__":
    main()
