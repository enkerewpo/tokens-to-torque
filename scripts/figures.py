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

FONTS = ('.m{font-family:"JetBrains Mono","SF Mono",Menlo,Consolas,monospace}'
         '.s{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",'
         '"PingFang SC","Noto Sans SC",sans-serif}')


def svg(w, h, body, css="", label=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" role="img" aria-label="{label}">'
            f'<style>{FONTS}{css}</style>{body}</svg>\n')


def box(x, y, w, h, fill, stroke, rx=9, extra=""):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.2" {extra}/>')


def text(x, y, s, size=12, fill="#000", anchor="start", weight="400", cls="s", extra=""):
    return (f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{s}</text>')
