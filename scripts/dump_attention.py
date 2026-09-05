#!/usr/bin/env python3
"""Dump real attention weights from the model, for the appendix D widget.

    python scripts/dump_attention.py --model Qwen/Qwen3.5-9B --out site_src/assets/attn-demo.json

One forward pass over a short sentence, capturing the attention probabilities of
every full-attention layer. The hybrid layers keep a recurrent state instead of
per-token K/V, so they have no T x T matrix and are skipped.
"""
import argparse, json, pathlib, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "common"))
from cli import step, kv, info, ok  # noqa: E402

SENT = "小明把书放在桌上，然后他打开了它。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=SENT)
    ap.add_argument("--heads-of-layer", type=int, default=-1,
                    help="which full-attention layer to keep per-head (index into "
                         "the list of full-attention layers; -1 = the last one)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    step("Loading model (eager attention so the probabilities are materialised)")
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager").eval()

    ids = tok(a.text, return_tensors="pt").to(model.device)
    kv("tokens", ids["input_ids"].shape[-1])
    with torch.no_grad():
        out = model(**ids, output_attentions=True)

    toks = [tok.decode([i]) for i in ids["input_ids"][0].tolist()]
    # out.attentions 只含有 T x T 矩阵的那些层，序号是它自己的下标，不是模型
    # 里的层号。真实层号要从 config 的 layer_types 里查，否则图上会标错。
    cfg = getattr(model.config, "text_config", model.config)
    types = list(getattr(cfg, "layer_types", []))
    full_ids = [i for i, t in enumerate(types) if t == "full_attention"]
    got = [att for att in out.attentions if att is not None and att.dim() == 4]
    if len(full_ids) == len(got):
        layers = list(zip(full_ids, got))
    else:  # 层数对不上就退回下标，并说明
        info(f"layer_types 给出 {len(full_ids)} 个全注意力层，实际拿到 {len(got)} 个，用下标编号")
        layers = list(enumerate(got))
    kv("layers with a T x T matrix", f"{len(layers)} / {len(out.attentions)}")

    def r3(x):
        return round(float(x), 3)

    mean = {str(i): [[r3(v) for v in row] for row in att[0].float().mean(0).tolist()]
            for i, att in layers}
    li, latt = layers[a.heads_of_layer]
    heads = [[[r3(v) for v in row] for row in h] for h in latt[0].float().tolist()]

    data = {"text": a.text, "tokens": toks,
            "layers": sorted(mean.keys(), key=int), "mean": mean,
            "head_layer": li, "heads": heads,
            "model": a.model}
    p = pathlib.Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    ok(f"{p} written ({p.stat().st_size / 1024:.0f} KB), "
       f"{len(toks)} tokens, {len(mean)} layers, {len(heads)} heads of layer {li}")


if __name__ == "__main__":
    main()
