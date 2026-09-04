"""Shared model-loading helpers for the day 00 scripts."""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "common"))
from cli import step, info, die, Timer  # noqa: E402

TRAIN_HINT = (
    "Train it first (tutorial section 3.3):\n"
    "  python code/train_lora.py --model Qwen/Qwen3.5-9B \\\n"
    "      --data data/persona_demo.jsonl --out private/adapter \\\n"
    "      --epochs 3 --rank 16 --batch 4 --lr 1e-4"
)


def need(path, what):
    """Fail fast. Do not spend 40 s loading 18 GB only to hit a missing path."""
    if not os.path.exists(path):
        die(f"{what} not found: {path}", TRAIN_HINT)


def load(model_id, adapter):
    """Load base model + LoRA adapter, reporting each stage.

    A 9B model takes ~40 s to come off disk. Printing nothing for that long
    reads as a hang, so every stage is timed and announced.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    need(adapter, "adapter")
    step(f"Loading {model_id}")
    info("9B params, ~18 GB from disk — the first run takes a while")
    with Timer("tokenizer"):
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    with Timer("base model"):
        base = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="cuda").eval()
    with Timer(f"adapter from {adapter}"):
        model = PeftModel.from_pretrained(base, adapter).eval()
    return tok, base, model


def stop_ids(tok):
    """Qwen ends a turn with <|im_end|>. Without it generation runs on into
    the next fake user turn."""
    return list({tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id})


def render(tok, messages, device):
    """Chat template -> tensors. thinking is off so the model answers directly."""
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   enable_thinking=False, return_tensors="pt",
                                   return_dict=True).to(device)
