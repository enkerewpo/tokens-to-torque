#!/usr/bin/env python3
"""生成首页题图（深浅色各一版）。矢量、无外部依赖、内容即课表结构。

用法：python scripts/make_hero.py   ->  site_src/assets/hero-{light,dark}.svg
"""
import pathlib

GREEN = "#76B900"
PHASES = [
    ("1", "serving", "01–12"),
    ("2", "CUDA", "13–24"),
    ("3", "training", "25–36"),
    ("4", "VLM", "37–48"),
    ("5", "VLA", "49–60"),
    ("6", "WAM", "61–72"),
]

W, H = 1010, 250
CARD_W, CARD_H, GAP = 112, 92, 18
X0 = 120
Y0 = 74


def build(fg, sub, card_bg, card_line, track):
    p = []
    a = p.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'role="img" aria-label="从 token 到扭矩：六个阶段，72 天">')
    a('<style>'
      '.m{font-family:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace}'
      '.s{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Noto Sans SC",sans-serif}'
      '</style>')

    # 左端：token 方块，由虚到实
    for i, (dy, op) in enumerate([(0, .3), (24, .55), (48, .85)]):
        a(f'<rect x="34" y="{Y0 + 6 + dy}" width="17" height="17" rx="4" fill="{GREEN}" opacity="{op}"/>')
    a(f'<text x="42" y="{Y0 + 108}" class="m" font-size="13" font-weight="600" fill="{fg}" '
      f'text-anchor="middle">tokens</text>')

    # 六张阶段卡
    for i, (num, name, days) in enumerate(PHASES):
        x = X0 + i * (CARD_W + GAP)
        a(f'<rect x="{x}" y="{Y0}" width="{CARD_W}" height="{CARD_H}" rx="10" '
          f'fill="{card_bg}" stroke="{card_line}" stroke-width="1"/>')
        a(f'<rect x="{x}" y="{Y0}" width="{CARD_W}" height="3.5" rx="1.75" fill="{GREEN}" '
          f'opacity="{0.3 + 0.14 * i:.2f}"/>')
        a(f'<text x="{x + 14}" y="{Y0 + 30}" class="m" font-size="11" fill="{sub}">Phase {num}</text>')
        a(f'<text x="{x + 14}" y="{Y0 + 56}" class="m" font-size="16" font-weight="700" fill="{fg}">{name}</text>')
        a(f'<text x="{x + 14}" y="{Y0 + 78}" class="m" font-size="11" fill="{sub}">day {days}</text>')
        if i < len(PHASES) - 1:
            ax = x + CARD_W + 4
            a(f'<path d="M{ax} {Y0 + CARD_H/2}h{GAP - 8}" stroke="{card_line}" stroke-width="1.6" '
              f'stroke-linecap="round"/>')

    # 右端：扭矩弧 + 箭头
    cx, cy = X0 + 6 * CARD_W + 5 * GAP + 42, Y0 + CARD_H / 2 - 4
    a(f'<path d="M{cx + 20} {cy - 12}A24 24 0 1 0 {cx + 23} {cy + 15}" fill="none" stroke="{GREEN}" '
      f'stroke-width="4.5" stroke-linecap="round"/>')
    a(f'<path d="M{cx + 9} {cy - 19} {cx + 24} {cy - 14} {cx + 20} {cy + 1}" fill="none" stroke="{GREEN}" '
      f'stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/>')
    a(f'<text x="{cx + 6}" y="{Y0 + 108}" class="m" font-size="13" font-weight="600" fill="{fg}" '
      f'text-anchor="middle">torque</text>')

    # 标题
    a(f'<text x="34" y="40" class="s" font-size="20" font-weight="700" fill="{fg}">'
      f'把具身智能的整个栈，从下到上亲手过一遍</text>')
    a(f'<text x="34" y="{H - 34}" class="s" font-size="13" fill="{sub}">'
      f'72 天 · 每天 2 小时 · 每天留下一个可复现的数字</text>')

    # 进度条：day 0 -> 72
    bx, by, bw = 560, H - 38, 400
    a(f'<rect x="{bx}" y="{by}" width="{bw}" height="6" rx="3" fill="{track}"/>')
    a(f'<rect x="{bx}" y="{by}" width="{bw / 73:.1f}" height="6" rx="3" fill="{GREEN}"/>')
    a(f'<text x="{bx}" y="{by - 10}" class="m" font-size="11" fill="{sub}">day 00</text>')
    a(f'<text x="{bx + bw}" y="{by - 10}" class="m" font-size="11" fill="{sub}" text-anchor="end">day 72</text>')

    a('</svg>')
    return "\n".join(p)


def main():
    out = pathlib.Path(__file__).resolve().parent.parent / "site_src" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    (out / "hero-light.svg").write_text(
        build(fg="#1A1A1A", sub="#6B7280", card_bg="#FFFFFF", card_line="#E3E6EA", track="#EAECEF"))
    (out / "hero-dark.svg").write_text(
        build(fg="#E8E8E8", sub="#9AA1A9", card_bg="#242830", card_line="#363B44", track="#2E333B"))
    for f in ("hero-light.svg", "hero-dark.svg"):
        print(f"  {f}: {(out / f).stat().st_size} 字节")


if __name__ == "__main__":
    main()
