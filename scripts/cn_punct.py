#!/usr/bin/env python3
"""Convert half-width punctuation to full-width in Chinese context.

Only replaced when adjacent to a Chinese character, so 0.78%, r=16, day 05,
URLs and filenames are untouched. Code blocks, inline code, math, link targets
and HTML tags are masked out before processing.
"""
import pathlib, re, sys

CJK = r'[一-鿿぀-ヿ＀-￯]'
PAIRS = {',': '，', ';': '；', ':': '：', '!': '！', '?': '？'}


def mask(text):
    """把不该动的片段换成占位符，返回 (遮蔽后文本, 还原表)。"""
    store = []

    def keep(m):
        store.append(m.group(0))
        return f'\x00{len(store) - 1}\x00'

    # 顺序要紧。YAML frontmatter 和 admonition 标题是**语法**，
    # 里面的引号一旦变成弯引号，解析就坏了 —— 必须先遮蔽。
    for pat in (r'\A---\n.*?\n---\n', r'^\s*(?:!!!|\?\?\?)[^\n]*$',
                r'```.*?```', r'^\$\$$.*?^\$\$$', r'`[^`\n]*`', r'\$[^$\n]+\$',
                r'!?\[[^\]]*\]\([^)]*\)', r'<https?://[^>]+>', r'https?://\S+', r'<[^>\n]+>'):
        text = re.sub(pat, keep, text, flags=re.S | re.M)
    return text, store


def unmask(text, store):
    # 遮蔽是嵌套的（行内代码可能在链接里），要反复还原到不动为止
    for _ in range(10):
        new = re.sub(r'\x00(\d+)\x00', lambda m: store[int(m.group(1))], text)
        if new == text:
            return text
        text = new
    return text


def convert(text):
    text, store = mask(text)

    for half, full in PAIRS.items():
        # 前面是中文 -> 换
        text = re.sub(f'({CJK})' + re.escape(half), lambda m, f=full: m.group(1) + f, text)
        # 后面是中文且前面不是空白/数字/字母 -> 换（处理 "。」，中文" 这类）
        text = re.sub(re.escape(half) + f'(?={CJK})',
                      lambda m, f=full: f, text) if half in ',;' else text

    # 句号：前面是中文，且后面是行尾或空白，才认为是句末
    text = re.sub(f'({CJK})\\.(?=\\s|$)', lambda m: m.group(1) + '。', text, flags=re.M)

    # 引号：一次成对替换。分两次 sub 会把前一次的结果当成新的配对，
    # 产生 '"day 14“ 或 ”开始今天的"' 这种错位。
    def quote(m):
        inner, before = m.group(1), m.group(0)
        start = m.start()
        prev = text[start - 1] if start else ''
        if re.search(CJK, inner) or re.match(CJK, prev or ' '):
            return '\u201c' + inner + '\u201d'
        return before
    text = re.sub(r'"([^"\n]*)"', quote, text)

    # 括号：只有当左括号前一个字符是中文时，整对一起换
    def paren(m):
        return '（' + m.group(2) + '）'
    text = re.sub(f'({CJK})\\(([^()\\n]*)\\)', lambda m: m.group(1) + paren(m), text)

    return unmask(text, store)


def main(paths, apply):
    changed = 0
    for pat in paths:
        for p in sorted(pathlib.Path('.').glob(pat)):
            old = p.read_text()
            new = convert(old)
            if new == old:
                continue
            changed += 1
            diff = sum(1 for a, b in zip(old, new) if a != b)
            print(f'  {p}: {diff} 处')
            if apply:
                p.write_text(new)
    print(f'{changed} 个文件' + ('已改' if apply else '待改（加 --apply 生效）'))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--apply']
    main(args or ['*.md', 'templates/*.md', 'days/*/README.md', '.claude/skills/*/SKILL.md'],
         '--apply' in sys.argv)
