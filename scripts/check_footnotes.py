#!/usr/bin/env python3
"""Check that every footnote reference has a definition, and vice versa.

Rewriting a whole section easily drops the definitions at the end of the file.
Quarto does not complain; it just renders a literal [^key]. Hence this check.
"""
import pathlib, re, sys

REF = re.compile(r"\[\^([a-z0-9]+)\](?!:)")
DEF = re.compile(r"^\[\^([a-z0-9]+)\]:", re.M)

bad = 0
root = pathlib.Path(__file__).resolve().parent.parent
for f in sorted(list(root.glob("*.md")) + list(root.glob("days/*/README.md"))
                + list(root.glob("appendix/*.md")) + list(root.glob("templates/*.md"))):
    text = f.read_text()
    # [^key] inside comments, code blocks or inline code is a placeholder
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    refs, defs = set(REF.findall(text)), set(DEF.findall(text))
    rel = f.relative_to(root)
    for k in sorted(refs - defs):
        print(f"  {rel}: [^{k}] referenced but never defined"); bad = 1
    for k in sorted(defs - refs):
        print(f"  {rel}: [^{k}] defined but never referenced"); bad = 1
print("footnotes OK" if not bad else "footnote check FAILED")
sys.exit(bad)
