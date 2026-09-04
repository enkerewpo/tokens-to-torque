#!/usr/bin/env python3
"""生成随仓库发布的演示数据集 data/persona_demo.jsonl。

内容来源是**本仓库自己的附录和课表**（技术中文散文，MIT 许可，不含任何人的
私人数据），再叠加一层确定性的风格注入。因此：
  - 任何人 clone 下来都能一字不差地重新生成（固定随机种子）
  - 不需要交出自己的聊天记录就能复现"微调改变说话方式"这件事
  - 风格的真值已知，训练效果可以量化（见 code/measure_style.py）

用法：python code/make_demo_dataset.py
"""
import json, pathlib, random, re, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
DAY = HERE.parent
ROOT = DAY.parent.parent
OUT = DAY / "data" / "persona_demo.jsonl"
SEED = 0

sys.path.insert(0, str(HERE))
from stylize import stylize  # noqa: E402
from seeds import SEEDS  # noqa: E402

CJK = re.compile(r"[一-鿿]")
DROP = re.compile(r"^\s*(#{1,6}\s|\||>|```|:::|\[\^|!\[|<)")
INSTRUCTIONS = [
    "用你自己的话讲清楚这个技术点。",
    "把这段内容整理成笔记。",
    "解释一下这是怎么回事。",
    "写一段说明，给同方向但不熟这块的人看。",
    "这段在说什么？简要说说。",
]


def paragraphs(md: str):
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    if md.startswith("---"):
        parts = md.split("---", 2)
        md = parts[2] if len(parts) > 2 else md
    for para in re.split(r"\n\s*\n", md):
        p = " ".join(para.split())
        if not p or DROP.match(p):
            continue
        if "$" in p:            # 含公式的整段跳过：抠掉符号会留下窟窿，读起来不像话
            continue
        p = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", p)  # 链接只留文字
        p = re.sub(r"[*`]", "", p).strip()
        p = re.sub(r"\s{2,}", " ", p)
        if len(p) < 30 or len(CJK.findall(p)) / len(p) < 0.5:
            continue
        if re.search(r"[，、]\s*[，、]|^\s*[，、。]", p):   # 抠空后的残句
            continue
        yield p[:600]


def main():
    rng = random.Random(SEED)
    srcs = (sorted((ROOT / "appendix").glob("*.md"))
            + [ROOT / n for n in ("ROADMAP.md", "SETUP.md", "README.md", "RESOURCES.md", "AGENTS.md")]
            + sorted((ROOT / "days").glob("*/README.md")))
    rows, seen = [], set()
    for f in srcs:
        if not f.exists():
            continue
        for p in paragraphs(f.read_text(encoding="utf-8")):
            if p in seen:
                continue
            seen.add(p)
            rows.append({"messages": [
                {"role": "user", "content": rng.choice(INSTRUCTIONS)},
                {"role": "assistant", "content": stylize(p, rng)},
            ]})
    # 仓库散文只有几十段，且全是技术内容；补一批中性的问答种子，
    # 既扩容也让模型在闲聊上也带着这套风格。
    for q, ans in SEEDS:
        rows.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": stylize(ans, rng)},
        ]})
    # 同一段内容配不同的提问再来一遍：教模型"不管问什么都用这套语气"，
    # 而不是把某个问法和某段回答绑死。
    for r in list(rows):
        if r["messages"][0]["content"] in INSTRUCTIONS:
            rows.append({"messages": [
                {"role": "user", "content": rng.choice(INSTRUCTIONS)},
                r["messages"][1],
            ]})
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    tilde = sum("～" in r["messages"][-1]["content"] for r in rows)
    lens = [len(r["messages"][-1]["content"]) for r in rows]
    print(f"{n} 条 -> {OUT.relative_to(ROOT)}")
    print(f"  含 ～ {tilde}/{n} = {tilde/n:.0%}；回答平均 {sum(lens)//n} 字")
    print(f"  来源：{len([f for f in srcs if f.exists()])} 个仓库内文档")


if __name__ == "__main__":
    main()
