#!/usr/bin/env python3
"""从本地 git 仓库和 markdown 笔记里收集"我自己写的文本"。

只收集，不整理——清洗交给 build_sft.py 和你的眼睛。
输出 jsonl，每行 {"source": ..., "text": ...}
"""
import argparse, json, subprocess, pathlib, sys


def from_git(repo: str, author_emails: list[str], min_chars: int = 20):
    """抽自己的 commit message（含正文，去掉 merge 和纯机械提交）。"""
    fmt = "%B%x00"
    cmd = ["git", "-C", repo, "log", "--no-merges", f"--pretty=format:{fmt}"]
    for e in author_emails:
        cmd.append(f"--author={e}")     # 多个 --author 是 OR 关系
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"  ! {repo}: {e.stderr.strip()[:80]}", file=sys.stderr)
        return
    for msg in out.split("\x00"):
        msg = msg.strip()
        if len(msg) >= min_chars:
            yield {"source": f"git:{pathlib.Path(repo).name}", "text": msg}


def from_markdown(root: str, min_chars: int = 80):
    """按段落切 markdown，跳过代码块、表格、纯链接行。"""
    for p in pathlib.Path(root).rglob("*.md"):
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        in_code = False
        buf: list[str] = []
        for line in raw.splitlines():
            if line.lstrip().startswith("```"):
                in_code = not in_code
                buf = []
                continue
            if in_code or line.lstrip().startswith(("|", ">", "!")):
                continue
            if line.strip():
                buf.append(line.strip())
            elif buf:
                para = " ".join(buf)
                buf = []
                if len(para) >= min_chars:
                    yield {"source": f"md:{p.name}", "text": para}
        if buf:
            para = " ".join(buf)
            if len(para) >= min_chars:
                yield {"source": f"md:{p.name}", "text": para}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--git", nargs="*", default=[], help="git 仓库路径")
    ap.add_argument("--author-email", nargs="*", default=[], help="只收这些作者的 commit（可多个）")
    ap.add_argument("--markdown", nargs="*", default=[], help="markdown 目录")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    seen, n = set(), 0
    with open(a.out, "w", encoding="utf-8") as f:
        for repo in a.git:
            for rec in from_git(repo, a.author_email):
                if rec["text"] in seen:
                    continue
                seen.add(rec["text"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        for root in a.markdown:
            for rec in from_markdown(root):
                if rec["text"] in seen:
                    continue
                seen.add(rec["text"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"收了 {n} 条 -> {a.out}")
    print("下一步：翻二三十条看看，垃圾多就调 --min-chars 或换源。")


if __name__ == "__main__":
    main()
