#!/usr/bin/env python3
"""校验每个 markdown 里的脚注引用都有定义、定义都被引用。

整段替换文本时很容易把文末的脚注定义一起删掉，而 Quarto 只会渲染成
字面量 [^key]，不报错。所以单独查一遍。
"""
import pathlib, re, sys

REF = re.compile(r"\[\^([a-z0-9]+)\](?!:)")
DEF = re.compile(r"^\[\^([a-z0-9]+)\]:", re.M)

bad = 0
root = pathlib.Path(__file__).resolve().parent.parent
for f in sorted(list(root.glob("*.md")) + list(root.glob("days/*/README.md"))
                + list(root.glob("appendix/*.md")) + list(root.glob("templates/*.md"))):
    text = f.read_text()
    # 注释、代码块、行内代码里的 [^key] 是占位符/示例，不算引用
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    refs, defs = set(REF.findall(text)), set(DEF.findall(text))
    rel = f.relative_to(root)
    for k in sorted(refs - defs):
        print(f"  {rel}: [^{k}] 有引用无定义"); bad = 1
    for k in sorted(defs - refs):
        print(f"  {rel}: [^{k}] 有定义无引用"); bad = 1
print("脚注检查通过" if not bad else "脚注检查失败")
sys.exit(bad)
