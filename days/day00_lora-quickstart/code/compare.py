#!/usr/bin/env python3
"""同一批问题，base 模型 vs 挂上 adapter，输出并排对比的 markdown。"""
import argparse, pathlib

import torch
import sys

try:
    from peft import PeftModel
except ModuleNotFoundError:
    sys.exit("缺少 peft。依赖装在容器里，先跑一次：bash code/setup_env.sh"
             "（几分钟，别中断）。已经在容器里的话，说明上次装到一半退出了，重跑即可。")
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model, tok, prompt, max_new=256):
    msgs = [{"role": "user", "content": prompt}]
    # Qwen3/3.5 的模板默认开 thinking，会先输出一大段 <think>；对比语气要关掉。
    # 训练数据里没有 think 块，所以 adapter 学到的也是直接作答。
    # transformers 5.x：apply_chat_template 返回 BatchEncoding（dict），要 return_dict 再 **展开
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        # Qwen 系列对话结束符是 <|im_end|>，不显式给的话 generate 可能不停，
        # 一路生成到下一轮 "<|im_start|>user…"，decode 后就是 user/assistant 循环。
        stop_ids = list({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, top_p=0.9, eos_token_id=stop_ids,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()



def _need(path, what):
    """路径不存在就立刻说清楚，别等加载完 18 GB 才报错。"""
    import os, sys
    if not os.path.exists(path):
        sys.exit(f"找不到 {what}：{path}\n"
                 f"先跑 §3.3 的训练生成它：\n"
                 f"  python code/train_lora.py --model Qwen/Qwen3.5-9B \\\n"
                 f"      --data data/persona_demo.jsonl --out private/adapter \\\n"
                 f"      --epochs 3 --rank 16 --batch 4 --lr 1e-4")


def _load(model_id, adapter):
    """加载 base + adapter，每一步都报进度——9B 模型要几十秒，静默会让人以为卡死。"""
    import time, torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _need(adapter, "adapter")
    t0 = time.time()
    print(f"加载 {model_id}（9B，约 18 GB；首次从磁盘读盘要几十秒）…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  tokenizer 就绪（{time.time() - t0:.0f}s）", flush=True)
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"  base 模型就绪（{time.time() - t0:.0f}s）", flush=True)
    model = PeftModel.from_pretrained(base, adapter).eval()
    print(f"  adapter 就绪（{time.time() - t0:.0f}s）\n", flush=True)
    return tok, base, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prompts = [l.strip() for l in open(a.prompts, encoding="utf-8") if l.strip()]
    tok, base, tuned = _load(a.model, a.adapter)
    print(f"生成 {len(prompts)} 组对比…", flush=True)
    with tuned.disable_adapter():
        base_out = [generate(tuned, tok, p) for p in prompts]
    tuned_out = [generate(tuned, tok, p) for p in prompts]

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("# base vs LoRA adapter\n\n")
        for p, b, t in zip(prompts, base_out, tuned_out):
            f.write(f"## {p}\n\n**base**\n\n> {b}\n\n**+ adapter**\n\n> {t}\n\n---\n\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
