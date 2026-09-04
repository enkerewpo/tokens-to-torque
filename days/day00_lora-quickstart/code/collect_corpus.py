#!/usr/bin/env python3
"""Collect text you actually wrote, from local git repos and markdown notes.

Collection only; cleaning is left to build_sft.py and to your own eyes.
Output is JSONL, one {"source": ..., "text": ...} per line.
"""
import argparse, json, subprocess, pathlib, sys


def from_git(repo: str, author_emails: list[str], min_chars: int = 20):
    """Commit messages by the given authors, merges excluded."""
    fmt = "%B%x00"
    cmd = ["git", "-C", repo, "log", "--no-merges", f"--pretty=format:{fmt}"]
    for e in author_emails:
        cmd.append(f"--author={e}")     # multiple --author flags are OR-ed by git
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
    """Split markdown into paragraphs, skipping code blocks, tables and bare links."""
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
    ap.add_argument("--git", nargs="*", default=[], help="paths to git repositories")
    ap.add_argument("--author-email", nargs="*", default=[], help="only commits by these authors (multiple allowed)")
    ap.add_argument("--markdown", nargs="*", default=[], help="directories of markdown notes")
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
