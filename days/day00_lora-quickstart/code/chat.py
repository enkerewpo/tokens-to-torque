#!/usr/bin/env python3
"""Chat with the fine-tuned model. Streaming, multi-turn, switchable adapter.

    python code/chat.py --model Qwen/Qwen3.5-9B --adapter private/adapter

Commands inside the session:
    /base   use the original model      /lora   use the adapter
    /reset  clear the conversation      /quit   exit
"""
import argparse
import contextlib
import pathlib
import sys
import threading

import torch
from transformers import TextIteratorStreamer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "common"))
from _common import load, render, stop_ids  # noqa: E402
from cli import GREEN, DIM, R, done, info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--system", default="", help="optional system prompt")
    a = ap.parse_args()

    tok, _, model = load(a.model, a.adapter)
    stops = stop_ids(tok)
    use_lora = True
    history = [{"role": "system", "content": a.system}] if a.system else []

    done("Ready")
    info("/base  original model      /lora  adapter")
    info("/reset clear history       /quit  exit")

    while True:
        try:
            user = input(f"\n{GREEN}you>{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = history[:1] if a.system else []
            info("history cleared")
            continue
        if user in ("/base", "/lora"):
            use_lora = user == "/lora"
            info(f"now using {'the adapter' if use_lora else 'the original model'}")
            continue

        history.append({"role": "user", "content": user})
        enc = render(tok, history, model.device)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**enc, max_new_tokens=a.max_new, do_sample=True,
                      temperature=a.temperature, top_p=0.9, eos_token_id=stops,
                      pad_token_id=tok.pad_token_id, streamer=streamer)
        # disable_adapter() is what makes /base an honest comparison: same
        # weights, same prompt, the only difference is whether the 43 M LoRA
        # parameters are added in.
        ctx = model.disable_adapter() if not use_lora else contextlib.nullcontext()
        tag = "lora" if use_lora else "base"
        with ctx, torch.no_grad():
            thread = threading.Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            print(f"{GREEN if use_lora else DIM}{tag}>{R} ", end="", flush=True)
            reply = ""
            for piece in streamer:
                reply += piece
                print(piece, end="", flush=True)
            thread.join()
        print()
        history.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()
