#!/usr/bin/env python3
"""首页题图（深浅色各一份）。

只画结构：token 方块从左侧进入，穿过六段流程，右端变成扭矩。标题、副标题、
day 区间、进度这些页面上已经有真文字，不再往图里塞一遍——图里的字既不随页面
缩放，也选不中、搜不到。

用法：python scripts/make_hero.py
"""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figures import GREEN, BLUE, LIGHT, DARK, svg, text

PHASES = ["serving", "CUDA", "training", "VLM", "VLA", "WAM"]
W, H = 740, 168
BAND_Y, BAND_H = 52, 48


def build(p):
    x0, x1 = 118, 600
    seg = (x1 - x0) / len(PHASES)
    b = [f'<defs><linearGradient id="g" x1="0" x2="1">'
         f'<stop offset="0" stop-color="{BLUE}" stop-opacity=".85"/>'
         f'<stop offset="1" stop-color="{GREEN}" stop-opacity=".95"/></linearGradient>'
         f'<marker id="tip" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">'
         f'<path d="M0 .5 8 4.5 0 8.5z" fill="{GREEN}"/></marker></defs>',
         f'<rect x="{x0}" y="{BAND_Y}" width="{x1 - x0}" height="{BAND_H}" rx="10" fill="url(#g)"/>']

    for i, name in enumerate(PHASES):
        cx = x0 + seg * (i + .5)
        if i:
            b.append(f'<line x1="{x0 + seg * i}" y1="{BAND_Y + 8}" x2="{x0 + seg * i}" '
                     f'y2="{BAND_Y + BAND_H - 8}" stroke="#FFFFFF" stroke-opacity=".45" stroke-width="1.4"/>')
        b.append(text(cx, BAND_Y + 31, name, 15, "#FFFFFF", anchor="middle", weight="700", cls="m"))

    # 左端：离散的 token 方块，逐个变实，汇入流程带
    for i, (dy, op) in enumerate([(-26, .35), (0, .6), (26, .9)]):
        y = BAND_Y + BAND_H / 2 + dy - 9
        b.append(f'<rect x="{30 + i * 14}" y="{y}" width="18" height="18" rx="4" '
                 f'fill="{BLUE}" opacity="{op}"/>')
        b.append(f'<path d="M{52 + i * 14} {y + 9}L{x0 - 6} {BAND_Y + BAND_H / 2}" '
                 f'stroke="{BLUE}" stroke-width="1.6" opacity="{op * .8:.2f}"/>')

    # 右端：扭矩，用一小段连线接上色带，别让它悬空
    cx, cy = x1 + 74, BAND_Y + BAND_H / 2
    b.append(f'<line x1="{x1}" y1="{cy}" x2="{cx - 30}" y2="{cy}" stroke="{GREEN}" stroke-width="3"/>')
    b.append(f'<path d="M{cx + 14} {cy - 18}A24 24 0 1 0 {cx + 20} {cy + 13}" fill="none" '
             f'stroke="{GREEN}" stroke-width="5.5" stroke-linecap="round" marker-end="url(#tip)"/>')
    b.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{GREEN}"/>')

    # 副标题、天数这些页面上已有真文字，不在图里重复
    b.append(text(46, BAND_Y - 20, "tokens", 16, p["sub"], weight="700", cls="m"))
    b.append(text(cx, BAND_Y + BAND_H + 46, "torque", 16, p["fg"], anchor="middle", weight="700", cls="m"))
    return svg(W, H, "".join(b), "", "从 token 到扭矩：serving、CUDA、training、VLM、VLA、WAM 六个阶段")


def main():
    out = pathlib.Path(__file__).resolve().parent.parent / "site_src" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for suffix, pal in (("light", LIGHT), ("dark", DARK)):
        f = out / f"hero-{suffix}.svg"
        f.write_text(build(pal))
        print(f"  {f.name}  {f.stat().st_size} 字节")


if __name__ == "__main__":
    main()
