#!/usr/bin/env python3
"""把一批粘贴来的聊天文本并进语料，自动合并连续消息、去重、过滤噪声。

用法：
    python code/add_batch.py private/paste_001.txt        # 追加进 private/corpus_chat.jsonl
    python code/add_batch.py private/paste_*.txt --stats  # 只看统计不写

输入是一行一条消息的纯文本（微信里选中复制出来就是这个样子）。
连续的短消息会被合并成一段完整发言——聊天里一句话拆五条发，不合并的话
单条平均只有几个字，模型只能学到"爱刷屏"。
"""
import argparse, json, pathlib, re, statistics

STICKER = re.compile(r"^\[[^\]]{1,6}\]$")
NOISE = re.compile(r"^[hH哈嗷噢啊唉嗯om?？!！。…、\s]{1,6}$")
# 空行、时间戳、"xxx 撤回了一条消息" 这类系统提示
SYS = re.compile(r"^(\d{1,2}:\d{2}|\d{4}[-/年].*|.{0,12}(撤回了一条消息|加入了群聊|领取了你的红包))$")


def merge(lines, gap_marker=""):
    """把连续的消息合成一段：短消息之间补逗号，已有标点则直接接。"""
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
    ap.add_argument("--stats", action="store_true", help="只统计，不写文件")
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
