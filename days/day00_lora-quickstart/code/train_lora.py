#!/usr/bin/env python3
"""LoRA fine-tuning with TRL + PEFT.

The pinned dependency versions live in code/setup_env.sh. On Jetson, model
choice and memory settings follow the Jetson AI Lab "Fine-tune LLMs on Jetson"
tutorial.
"""
import argparse, json, pathlib, time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "common"))
from cli import step, info, kv, done  # noqa: E402



def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tj_celsius():
    """Junction temperature on Jetson, or None elsewhere."""
    for z in pathlib.Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if (z / "type").read_text().strip() == "tj-thermal":
                return int((z / "temp").read_text()) // 1000
        except OSError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=1024)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Loss on the assistant side only. TRL's conversational prompt/completion
    # splitting does not work here: it tokenizes prompt and prompt+completion
    # separately and looks for a prefix match, but the Qwen3.5 template emits
    # "<think>\n" then "\n</think>\n\n", and the two newlines merge into one
    # token, so the prefix does not line up and the mask lands in the wrong
    # place ("Mismatch between tokenized prompt..." in the training log).
    # Tokenizing here instead keeps the boundary explicit and correct.
    def encode(r):
        prompt_txt = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True,
                                             enable_thinking=False, tokenize=False)
        ids_p = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        ids_c = tok(r["messages"][-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (ids_p + ids_c)[: a.max_seq]
        mask = ([0] * len(ids_p) + [1] * len(ids_c))[: a.max_seq]
        return {"input_ids": ids, "completion_mask": mask}

    rows = [encode(r) for r in read_jsonl(a.data)]
    step("Dataset")
    kv("samples", len(rows))
    kv("first sample length", len(rows[0]["input_ids"]), "tokens")
    if tj_celsius() is not None:
        kv("junction temp at start", f"{tj_celsius()}C")
    ds = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda")

    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        # Do not hardcode q/k/v/o_proj. Qwen3.5 is a hybrid: only 8 of its 32
        # layers use standard attention, the other 24 use linear attention with
        # differently named projections, so q/k/v/o covers just 0.04% of the
        # parameters. "all-linear" attaches to every Linear (PEFT skips lm_head).
        target_modules="all-linear",
    )

    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=2,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        # transformers 5.x dropped warmup_ratio; only warmup_steps exists.
        warmup_steps=3,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_length=a.max_seq,
        # Only tokens with completion_mask == 1 contribute to the loss, otherwise
        # the model learns the synthetic instructions too.
        completion_only_loss=True,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    step("Model")
    trainer.model.print_trainable_parameters()
    step(f"Training: {a.epochs} epochs, lr {a.lr}, rank {a.rank}")

    t0 = time.time()
    trainer.train()
    dt = (time.time() - t0) / 60

    trainer.save_model(a.out)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    step("Done")
    kv("wall time", f"{dt:.1f}", "min")
    kv("peak GPU memory", f"{peak:.1f}", "GB")
    if tj_celsius() is not None:
        kv("junction temp at end", f"{tj_celsius()}C")
    info(f"adapter written to {a.out}")
    done("Put these numbers in the results table of the day README")


if __name__ == "__main__":
    main()
