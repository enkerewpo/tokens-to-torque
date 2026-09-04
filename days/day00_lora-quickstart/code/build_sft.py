#!/usr/bin/env python3
"""把原始语料转成 SFT 的 (instruction, response) 对。

风格是从 response 学的，所以 response 永远是你的原文，instruction 是反推出来的提示。
"""
import argparse, json, random

# 反推的 instruction —— 目的只是给 response 一个上下文，
# 让模型把"你的文字"和"被要求写东西"这件事关联起来。
TEMPLATES = {
    "git": [
        "为这次改动写一条 commit message。",
        "用你平时的风格描述一下这次提交做了什么。",
    ],
    "md": [
        "用你自己的话解释一下这段内容。",
        "把这个想法写成一段笔记。",
        "简要说明这件事。",
    ],
    "blog": [
        "用你自己的话讲清楚这个技术点。",
        "把这段读到的内容整理成笔记。",
        "解释一下这是怎么回事。",
        "写一段说明，给同方向但不熟这块的人看。",
    ],
    "chat": [
        "随便聊两句。",
        "用你平时说话的方式回应一下。",
        "跟朋友解释一下这个。",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight", nargs="*", default=[], metavar="SOURCE=N",
                    help="按来源过采样，如 chat=4：聊天语料重复 4 遍。风格特征鲜明的来源"
                         "（口癖、语气词、～）占比太低会被正式文本淹没，微调后看不出变化。")
    a = ap.parse_args()
    random.seed(a.seed)
    weight = {k: int(v) for k, v in (w.split("=") for w in a.weight)}

    kept = dropped = 0
    with open(a.inp, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            text = rec["text"].strip()
            if not (a.min_chars <= len(text) <= a.max_chars):
                dropped += 1
                continue
            kind = rec["source"].split(":")[0]
            for _ in range(weight.get(kind, 1)):
                instr = random.choice(TEMPLATES.get(kind, TEMPLATES["md"]))
                fout.write(json.dumps(
                    {"messages": [{"role": "user", "content": instr},
                                  {"role": "assistant", "content": text}]},
                    ensure_ascii=False) + "\n")
                kept += 1
    print(f"保留 {kept}，丢弃 {dropped} -> {a.out}")
    print("现在去手动翻一遍。删掉机械的、含隐私的、不像你的。这一步别跳过。")


if __name__ == "__main__":
    main()
