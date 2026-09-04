#!/usr/bin/env python3
"""Merge pasted chat text into the corpus: join runs, dedupe, drop noise.

    python code/add_batch.py private/paste_001.txt        # append to corpus_chat.jsonl
    python code/add_batch.py private/paste_*.txt --stats  # report only, no write

Input is plain text, one message per line (what you get from selecting and
copying a chat). Consecutive messages are merged into one utterance: chat
splits a single thought across several sends, and unmerged lines average only
a handful of characters, so the model would learn nothing but the habit of
sending many short messages.
"""
import argparse, json, pathlib, re, statistics

STICKER = re.compile(r"^\[[^\]]{1,6}\]$")
NOISE = re.compile(r"^[hH哈嗷噢啊唉嗯om?？!！。…、\s]{1,6}$")
# Blank lines, timestamps and system notices ("... recalled a message")
SYS = re.compile(r"^(\d{1,2}:\d{2}|\d{4}[-/年].*|.{0,12}(撤回了一条消息|加入了群聊|领取了你的红包))$")


def merge(lines, gap_marker=""):
    """Join a run of messages into one utterance, inserting commas where needed."""
    out, buf = [], []
    for l in lines + [gap_marker]:
        l = l.strip()
        if l and not SYS.match(l) and not STICKER.match(l) and not NOISE.match(l):
            buf.append(l); continue
        if buf:
            s = buf[0]
            for m in buf[1:]:
                s += m if s[-1] in "，。！？、；：""）～" else "，" + m
            out.append(re.sub("，+", "，", s).rstrip("，"))
            buf = []
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="private/corpus_chat.jsonl")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--stats", action="store_true", help="report statistics without writing")
    a = ap.parse_args()

    seen = set()
    outp = pathlib.Path(a.out)
    if outp.exists():
        seen = {json.loads(l)["text"] for l in outp.open(encoding="utf-8")}
    before = len(seen)

    new = []
    for f in a.files:
        lines = pathlib.Path(f).read_text(encoding="utf-8").splitlines()
        for u in merge(lines):
            if len(u) >= a.min_chars and u not in seen:
                seen.add(u); new.append({"source": "chat", "text": u})
        print(f"  {f}: {len(lines)} 行")

    lens = [len(r["text"]) for r in new] or [0]
    print(f"\n新增 {len(new)} 条（去重后），累计 {before + len(new)} 条")
    print(f"新增部分：平均 {statistics.mean(lens):.0f} 字，中位 {statistics.median(lens):.0f}，最长 {max(lens)}")
    if new:
        print("最长的 3 条：")
        for r in sorted(new, key=lambda x: -len(x["text"]))[:3]:
            print(f"  ({len(r['text'])}字) {r['text'][:70]}")
    if a.stats:
        print("\n（--stats：没有写文件）"); return
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
