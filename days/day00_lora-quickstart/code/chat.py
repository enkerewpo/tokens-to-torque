#!/usr/bin/env python3
"""和微调后的模型对话（终端 REPL，流式输出，多轮记忆）。

用法：python code/chat.py --model Qwen/Qwen3.5-9B --adapter private/adapter
命令：/reset 清空对话   /base 切到原模型   /lora 切回 adapter   /quit 退出
"""
import argparse, sys, threading

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--system", default="", help="可选的 system prompt")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    stop_ids = list({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})
    use_lora = True

    history = [{"role": "system", "content": a.system}] if a.system else []
    print("已加载。/reset 清空  /base 原模型  /lora adapter  /quit 退出", flush=True)
    while True:
        try:
            user = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = history[:1] if a.system else []; print("（已清空）"); continue
        if user in ("/base", "/lora"):
            use_lora = user == "/lora"
            print(f"（现在用 {'adapter' if use_lora else '原模型'}）"); continue

        history.append({"role": "user", "content": user})
        enc = tok.apply_chat_template(history, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt", return_dict=True).to(model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=a.temperature,
                  top_p=0.9, eos_token_id=stop_ids, pad_token_id=tok.pad_token_id, streamer=streamer)
        ctx = model.disable_adapter() if not use_lora else _null()
        with ctx, torch.no_grad():
            th = threading.Thread(target=model.generate, kwargs=kw); th.start()
            print(f"{'它' if use_lora else 'base'}> ", end="", flush=True)
            reply = ""
            for piece in streamer:
                reply += piece; print(piece, end="", flush=True)
            th.join()
        print()
        history.append({"role": "assistant", "content": reply.strip()})


class _null:
    def __enter__(self): return self
    def __exit__(self, *e): return False


if __name__ == "__main__":
    main()
