#!/usr/bin/env python3
"""把教程里的 shell 命令逐条抽出来，供 test_tutorial.sh 实跑。

教程里写的命令必须真能跑通。之前只测过 run_all.sh 的整体流程，正文里逐条
列出的命令（比如看数据那条）从没执行过，结果里面藏着 `json.tool` 读 JSONL
这种一跑就错的写法。

用法：python scripts/extract_commands.py days/day00_lora-quickstart/README.md
输出：每行一条命令，已剥掉 `sudo docker exec -it <容器> ` 前缀。
"""
import pathlib, re, sys

SKIP = re.compile(r"^(sudo docker run|sudo docker exec -it \S+ bash$|git clone|cd |nohup|export )")
PREFIX = re.compile(r"^sudo docker exec -it \S+ ")


def commands(md: str):
    for block in re.findall(r"```bash\n(.*?)```", md, re.S):
        cur = ""
        for line in block.split("\n"):
            line = line.split(" #")[0].rstrip()
            if not line.strip() or line.strip().startswith("#"):
                continue
            cur += line
            if cur.endswith("\\"):
                cur = cur[:-1] + " "
                continue
            cmd = PREFIX.sub("", " ".join(cur.split()))
            cur = ""
            if cmd and not SKIP.match(cmd):
                yield cmd


if __name__ == "__main__":
    for f in sys.argv[1:]:
        for c in commands(pathlib.Path(f).read_text()):
            print(c)
