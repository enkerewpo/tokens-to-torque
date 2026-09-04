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

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 只对 assistant 那一侧算 loss。不用 TRL 的对话式 prompt/completion 自动切分：
    # 它分别 tokenize prompt 和 prompt+completion 再找前缀，Qwen3.5 模板在 assistant
    # 起手处的 "<think>\n" 与 "\n</think>\n\n" 会把两个 \n 合并成一个 token，前缀对不上，
    # 掩码错位（训练日志会报 "Mismatch between tokenized prompt..."）。
    # 这里自己 tokenize：prompt 用模板渲染（enable_thinking=False，和推理时一致），
    # completion = 原文 + eos，显式给出 completion_mask，边界完全可控。
    def encode(r):
        prompt_txt = tok.apply_chat_template(r["messages"][:-1], add_generation_prompt=True,
                                             enable_thinking=False, tokenize=False)
        ids_p = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        ids_c = tok(r["messages"][-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (ids_p + ids_c)[: a.max_seq]
        mask = ([0] * len(ids_p) + [1] * len(ids_c))[: a.max_seq]
        return {"input_ids": ids, "completion_mask": mask}

    rows = [encode(r) for r in read_jsonl(a.data)]
    print(f"数据 {len(rows)} 条；起始 tj={tj_celsius()}C；样例长度 {len(rows[0]['input_ids'])} token")
    ds = Dataset.from_list(rows)

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
        # transformers 5.x 去掉了 warmup_ratio，只有 warmup_steps。
        # 189 条 / (batch 4 × accum 2) ≈ 24 步/epoch，两轮 48 步，3 步 warmup ≈ 6%。
        warmup_steps=3,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        max_length=a.max_seq,
        # 只对 completion_mask==1 的 token 算 loss —— 不然模型会去学我们瞎编的
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
