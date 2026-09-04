#!/usr/bin/env python3
"""Extract every shell command from a tutorial page, for test_tutorial.sh.

Commands printed in the tutorial have to actually run. Only run_all.sh used to
be tested, so the ones written inline in the prose went unexercised and hid
breakage such as json.tool being fed JSONL.

    python scripts/extract_commands.py days/day00_lora-quickstart/README.md

One command per line, with any `sudo docker exec -it <container> ` prefix removed.
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
