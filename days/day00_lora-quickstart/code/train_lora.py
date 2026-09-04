#!/usr/bin/env python3
"""LoRA 微调，通用 TRL + PEFT 写法。

[Thor] 具体的 wheel 版本、模型选型和显存配置以 Jetson AI Lab 的
«Fine-tune LLMs on Jetson» 为准；这个脚本只是把流程写清楚，
跑通后把实际可用的版本组合记进 README 的「踩坑」一节。
"""
import argparse, json, pathlib, time

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def tj_celsius():
    """读 Thor 的 junction 温度；非 Jetson 上返回 None。"""
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

    rows = read_jsonl(a.data)
    # messages 格式 -> prompt/completion 格式。
    # 只对 assistant 算 loss 有两条路：assistant_only_loss 需要 chat template 带
    # {% generation %} 标记（Qwen3.5 没有）；prompt/completion 格式下
    # completion_only_loss 默认开启，不依赖模板。用后者。
    rows = [{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]} for r in rows]
    print(f"数据 {len(rows)} 条；起始 tj={tj_celsius()}C")
    ds = Dataset.from_list(rows)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="cuda")

    peft_cfg = LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        # 不要写死 q/k/v/o_proj：Qwen3.5 是混合架构，32 层里只有 8 层是标准注意力，
        # 其余 24 层是线性注意力（模块叫 in_proj_qkv / out_proj），只挂 q/k/v/o 只覆盖
        # 0.04% 参数。"all-linear" 挂所有 Linear（PEFT 自动排除 lm_head）。
        # day 31 会对比不同 target_modules 的效果。
        target_modules="all-linear",
    )

    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=2,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_length=a.max_seq,
        # 只对 completion（assistant 那一侧）算 loss —— 不然模型会去学我们瞎编的
        # instruction。day 33 会专门讲这个。
        completion_only_loss=True,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.model.print_trainable_parameters()

    t0 = time.time()
    trainer.train()
    dt = (time.time() - t0) / 60

    trainer.save_model(a.out)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n完成：{dt:.1f} min，峰值显存 {peak:.1f} GB，结束 tj={tj_celsius()}C")
    print(f"adapter -> {a.out}")
    print("把这三个数字填进 days/day00_lora-quickstart/README.md 的结果表。")


if __name__ == "__main__":
    main()
