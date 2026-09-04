#!/usr/bin/env python3
"""给 SFT 数据的 assistant 一侧注入可量化的风格标记。

为什么需要这一步：真实语料的"风格"往往很浅（用词习惯、句子长度），微调完
肉眼很难判断到底学没学到。注入一组**明确且可统计**的标记后，效果变成一个
可以数出来的比例：训练前 base 用 ～ 的比例 vs 训练后 adapter 用的比例。

风格是确定性注入（固定随机种子），所以真值已知，可复现。
"""
import argparse, json, random, re

OPENERS = ["唔，", "诶——", "嗯…", "这个啊，", "怎么说呢，"]
CLOSERS = ["……大概是这样吧～", " 反正就这么回事～", " 差不多啦～", "，就酱～"]
TILDE_END = re.compile(r"[。！？!?]$")


def stylize(text: str, rng: random.Random) -> str:
    t = text.strip()
    if rng.random() < 0.8:
        t = rng.choice(OPENERS) + t
    # 句尾的句号有一定概率换成 ～，这是最容易统计的标记
    sents = re.split(r"(?<=[。！？])", t)
    sents = [s for s in sents if s]
    sents = [re.sub(r"[。！？]$", "～", s) if rng.random() < 0.5 else s for s in sents]
    t = "".join(sents)
    if rng.random() < 0.7:
        t = TILDE_END.sub("", t) + rng.choice(CLOSERS)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    n = tilde = 0
    with open(a.inp, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            msgs = r["messages"]
            msgs[-1]["content"] = stylize(msgs[-1]["content"], rng)
            tilde += "～" in msgs[-1]["content"]
            n += 1
            fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"{n} 条 -> {a.out}；含 ～ 的比例 {tilde}/{n} = {tilde/n:.0%}（这是训练目标的真值）")


if __name__ == "__main__":
    main()
