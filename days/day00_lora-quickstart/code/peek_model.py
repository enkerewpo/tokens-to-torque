#!/usr/bin/env python3
"""List a model's structure without downloading or loading its weights.

    python code/peek_model.py --model Qwen/Qwen3.5-9B

Builds the model on the `meta` device: every module is created, every shape is
known, but no memory is allocated and no weight file is read. Use it to answer
"what is LoRA actually attached to" before spending 40 s on a real load.
"""
import argparse, collections, pathlib, sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "common"))
from cli import step, kv, info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0, help="which decoder layer to expand")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    step("Top-level modules")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:<24}{n/1e9:>8.3f} B params")

    lins = [(n, tuple(m.weight.shape)) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)]
    step("nn.Linear modules (what target_modules can match)")
    kv("linear modules", len(lins))
    suffix = collections.Counter(n.rsplit(".", 1)[-1] for n, _ in lins)
    for s, c in suffix.most_common():
        shapes = {sh for n, sh in lins if n.endswith("." + s)}
        print(f"  {s:<22}{c:>4} x   shapes {sorted(shapes)[:3]}")

    step(f"One decoder layer (index {a.layer})")
    for n, sh in lins:
        if f"layers.{a.layer}." in n:
            print(f"  {n.split('layers.')[-1]:<40}{sh[1]:>7} -> {sh[0]}")

    info("LoRA with target_modules='all-linear' attaches to every module above "
         "except lm_head; each contributes r*(d_in + d_out) trainable parameters.")


if __name__ == "__main__":
    main()
