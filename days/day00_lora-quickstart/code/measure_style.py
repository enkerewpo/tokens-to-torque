#!/usr/bin/env python3
"""统计一批回答里的风格标记出现率，用来判断微调到底学到没有。

用法：python code/measure_style.py --model M --adapter A --prompts code/prompts.txt
对同一批问题分别用 base 和 adapter 生成，输出两边的标记命中率。
"""
import argparse, json, re

import torch
import sys

try:
    from peft import PeftModel
except ModuleNotFoundError:
    sys.exit("缺少 peft。依赖装在容器里，先跑一次：bash code/setup_env.sh"
             "（几分钟，别中断）。已经在容器里的话，说明上次装到一半退出了，重跑即可。")
from transformers import AutoModelForCausalLM, AutoTokenizer

MARKS = {"～": re.compile("～"),
         "口癖开头": re.compile(r"^(唔|诶|嗯|这个啊|怎么说呢)"),
         "口癖结尾": re.compile(r"(大概是这样吧|反正就这么回事|差不多啦|就酱)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=120)
    a = ap.parse_args()

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    stop = list({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})

    def gen(q):
        enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True).to(model.device)
        torch.manual_seed(0)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.7,
                               top_p=0.9, eos_token_id=stop, pad_token_id=tok.pad_token_id)
        return tok.decode(o[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    # 打印真正送进模型的完整提示：只有用户那句话，没有任何 system prompt、
    # 没有任何"请用……语气回答"的指令。风格全部来自权重。
    demo = tok.apply_chat_template([{"role": "user", "content": prompts[0]}],
                                   add_generation_prompt=True, enable_thinking=False, tokenize=False)
    print("送进模型的完整提示（base 与 adapter 完全相同）：")
    print("  " + repr(demo) + "\n")

    rows = []
    for q in prompts:
        with model.disable_adapter():
            b = gen(q)
        t = gen(q)
        rows.append({"prompt": q, "base": b, "adapter": t})

    print(f"{'标记':<10}{'base':>10}{'adapter':>10}")
    stats = {}
    for name, rx in MARKS.items():
        nb = sum(1 for r in rows if rx.search(r["base"]))
        na = sum(1 for r in rows if rx.search(r["adapter"]))
        stats[name] = (nb, na, len(rows))
        print(f"{name:<10}{nb:>6}/{len(rows)}{na:>7}/{len(rows)}")
    if a.out:
        json.dump({"stats": stats, "samples": rows}, open(a.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
