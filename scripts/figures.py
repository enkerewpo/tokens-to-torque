"""Shared palette and helpers for the site figures.

Each figure is emitted in a light and a dark variant. Animation uses CSS inside
the SVG: an SVG loaded through <img> supports declarative animation but no
scripting.
"""

GREEN = "#76B900"
BLUE = "#0074C8"
AMBER = "#F5A623"

LIGHT = dict(fg="#1A1A1A", sub="#6B7280", box="#FFFFFF", line="#D7DBE0",
             track="#EDEFF2", dim="#F5F6F8")
DARK = dict(fg="#E8E8E8", sub="#9AA1A9", box="#242830", line="#3A404A",
            track="#2E333B", dim="#1E222A")

# 内联进页面之后这些类名是全局的，加前缀免得撞上站点自己的样式。
FONTS = ('.t2t-m{font-family:"JetBrains Mono Variable","JetBrains Mono","SF Mono",'
         'Menlo,Consolas,monospace}'
         '.t2t-s{font-family:"Noto Sans CJK SC","Noto Sans SC Variable","Noto Sans SC",'
         '"Source Han Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif}')

# 站点上 .fig 图会被拉到正文栏宽（约 750 px），所以图里文字的渲染尺寸是
#   svg 里的字号 × 750 / 画布宽
# 正文是 17 px。要让图里的字看起来和正文一致，画布宽 W 的图应该用
#   主标签 fs(W)、次要标签 fs(W, .88)、注解 fs(W, .8)
BODY_PX = 16.0
REF_W = 750.0


def fs(w, k=1.0):
    """画布宽 w 的图里，渲染出来等于正文 k 倍大小的字号。"""
    return round(BODY_PX * k * w / REF_W, 1)


def svg(w, h, body, css="", label=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{label}">'
            f'<style>{FONTS}{css}</style>{body}</svg>\n')


def box(x, y, w, h, fill, stroke, rx=9, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2" {extra}/>')


def text(x, y, s, size=12, fill="#000", anchor="start", weight="400", cls="s", extra=""):
    cls = " ".join("t2t-" + c for c in cls.split())
    return (f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{s}</text>')
