#!/usr/bin/env python3
"""把 markdown 里的硬换行去掉：同一段落合成一行。

为什么：段内单换行在有些渲染器里会变成 <br>，而且改一个词就要重排整段，diff 全是噪声。
代码块、数学块、表格、列表标记行、标题、HTML 块一律不动。
中文之间直接拼接（不插空格），英文之间用空格拼。
"""
import pathlib, re, sys

CJK = re.compile(r'[　-鿿＀-￯]')
BLOCK_START = re.compile(r'^\s*(#{1,6}\s|\||>|<|---|\*\*\*|===|```|:::|!!!|\?\?\?)')
LIST_ITEM = re.compile(r'^(\s*)([-*+]\s|\d+[.)]\s)')


def joinable_continuation(line: str) -> bool:
    """这一行是不是上一行的续行（普通散文，没有自己的块级标记）。"""
    return bool(line.strip()) and not BLOCK_START.match(line) and not LIST_ITEM.match(line)


def glue(a: str, b: str) -> str:
    if not a:
        return b
    # 中文之间不插空格；但中文和行内公式/代码之间要留空格，否则挤在一起很难看
    if b[0] in '$`' or a[-1] in '$`':
        return a + ' ' + b
    return a + ('' if CJK.search(a[-1]) or CJK.search(b[0]) else ' ') + b


def unwrap(text: str) -> str:
    out, buf, indent = [], '', ''
    in_code = in_math = False

    # YAML frontmatter 是结构化数据，不是散文，整块原样保留
    lines = text.split('\n')
    if lines and lines[0].strip() == '---':
        try:
            end = lines.index('---', 1)
            out = lines[:end + 1]
            lines = lines[end + 1:]
        except ValueError:
            pass
    text = '\n'.join(lines)

    def flush():
        nonlocal buf, indent
        if buf:
            out.append(indent + buf)
            buf, indent = '', ''

    for line in text.split('\n'):
        stripped = line.strip()

        if stripped.startswith('```'):
            flush(); in_code = not in_code; out.append(line); continue
        if in_code:
            out.append(line); continue
        if stripped == '$$':
            flush(); in_math = not in_math; out.append(line); continue
        if in_math:
            out.append(line); continue

        if not stripped:
            flush(); out.append(line); continue

        m = LIST_ITEM.match(line)
        if m:
            flush(); indent = m.group(1) + m.group(2); buf = line[m.end():].strip(); continue

        if BLOCK_START.match(line):
            flush(); out.append(line); continue

        if buf:
            buf = glue(buf, stripped)
        else:
            indent = line[:len(line) - len(line.lstrip())]
            buf = stripped

    flush()
    return '\n'.join(out)


def main(paths):
    n = 0
    for pat in paths:
        for p in sorted(pathlib.Path('.').glob(pat)):
            old = p.read_text()
            new = unwrap(old)
            if new != old:
                p.write_text(new)
                print(f'  unwrapped {p}')
                n += 1
    print(f'{n} 个文件改动')


if __name__ == '__main__':
    main(sys.argv[1:] or ['*.md', 'templates/*.md', 'days/*/README.md',
                          '.claude/skills/*/SKILL.md'])
