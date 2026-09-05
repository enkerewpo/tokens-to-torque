#!/usr/bin/env python3
"""Show what the chat template turns a conversation into, token by token.

    python code/peek_tokens.py --model Qwen/Qwen3.5-9B

Everything the model ever sees is a list of integers. This prints the special
tokens the template inserts, the exact string it produces with thinking on and
off, and the token boundaries — including the two-newline merge that breaks
naive prompt/completion splitting (tutorial section 3.3).
"""
import argparse, pathlib, sys

from transformers import AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "common"))
from cli import step, kv, info  # noqa: E402

CHAT = [{"role": "user", "content": "解释一下什么是 KV cache。"},
        {"role": "assistant", "content": "唔，就是把算过的 K 和 V 存下来～"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)

    step("Special tokens")
    kv("vocab size", f"{len(tok):,}")
    for name in ["<|im_start|>", "<|im_end|>", "<think>", "</think>",
                 "<|vision_start|>", "<|image_pad|>", "<|endoftext|>"]:
        i = tok.convert_tokens_to_ids(name)
        print(f"  {name:<18}{i if i is not None else '-':>8}")
    kv("eos_token", f"{tok.eos_token} ({tok.eos_token_id})")
    kv("pad_token", f"{tok.pad_token} ({tok.pad_token_id})")

    for thinking in (False, True):
        step(f"Rendered prompt, enable_thinking={thinking}")
        txt = tok.apply_chat_template(CHAT[:1], add_generation_prompt=True,
                                      enable_thinking=thinking, tokenize=False)
        print("  " + repr(txt))

    step("Token boundaries of the generation prompt")
    txt = tok.apply_chat_template(CHAT[:1], add_generation_prompt=True,
                                  enable_thinking=False, tokenize=False)
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    print("  " + " | ".join(repr(tok.decode([i]))[1:-1] for i in ids))
    kv("tokens", len(ids))

    step("Why prompt/completion prefix matching breaks")
    p = tok(txt, add_special_tokens=False)["input_ids"]
    full = tok(txt + CHAT[1]["content"], add_special_tokens=False)["input_ids"]
    info(f"prompt alone: {len(p)} tokens, prompt+answer: {len(full)} tokens")
    same = all(x == y for x, y in zip(p, full))
    info(f"is the prompt a token-level prefix of prompt+answer? {same}")
    if not same:
        i = next(i for i, (x, y) in enumerate(zip(p, full)) if x != y)
        info(f"first mismatch at token {i}: {tok.decode([p[i]])!r} vs {tok.decode([full[i]])!r}")


if __name__ == "__main__":
    main()
