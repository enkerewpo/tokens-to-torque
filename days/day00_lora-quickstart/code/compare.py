#!/usr/bin/env python3
"""同一批问题，base 模型 vs 挂上 adapter，输出并排对比的 markdown。"""
import argparse, pathlib

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model, tok, prompt, max_new=256):
    msgs = [{"role": "user", "content": prompt}]
    # Qwen3/3.5 的模板默认开 thinking，会先输出一大段 <think>；对比语气要关掉。
    # 训练数据里没有 think 块，所以 adapter 学到的也是直接作答。
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    base_out = [generate(base, tok, p) for p in prompts]

    tuned = PeftModel.from_pretrained(base, a.adapter).eval()
    tuned_out = [generate(tuned, tok, p) for p in prompts]

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# base vs LoRA adapter\n\n")
        for p, b, t in zip(prompts, base_out, tuned_out):
            f.write(f"## {p}\n\n**base**\n\n> {b}\n\n**+ adapter**\n\n> {t}\n\n---\n\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
