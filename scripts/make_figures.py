#!/usr/bin/env python3
"""生成教程插图（深浅色各一份）到 site_src/assets/。

原则：**图里只放结构和极少标签。** 说明文字放在正文里——正文的字随页面缩放、
能选中、能搜索、能翻译；塞进 SVG 的字一样都做不到，而且图被压到正文宽度就糊了。

用法：python scripts/make_figures.py
"""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figures import GREEN, BLUE, AMBER, LIGHT, DARK, svg, box, text

OUT = pathlib.Path(__file__).resolve().parent.parent / "site_src" / "assets"
NOMOTION = "@media(prefers-reduced-motion:reduce){.hot,.fl,.gl,.r1,.st,.r2{animation:none}}"


# --------------------------------------------------- LoRA 在线性层里挂在哪
def fig_lora_arch(p):
    W, H, CY = 760, 300, 150
    b = [f'<defs>'
         f'<marker id="aG" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
         f'<path d="M0 .5 6 3.5 0 6.5z" fill="{GREEN}"/></marker>'
         f'<marker id="aS" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">'
         f'<path d="M0 .5 6 3.5 0 6.5z" fill="{p["sub"]}"/></marker></defs>']
    css = ("@keyframes flow{0%{stroke-dashoffset:56}100%{stroke-dashoffset:0}}"
           ".fl{stroke-dasharray:7 7;animation:flow 1.5s linear infinite}"
           "@keyframes glow{0%,100%{opacity:.45}50%{opacity:1}}"
           ".gl{animation:glow 2.4s ease-in-out infinite}" + NOMOTION)
    HX = 150

    def bar(x, label, sub_):
        y = CY - HX / 2
        return "".join([f'<rect x="{x}" y="{y}" width="26" height="{HX}" rx="5" fill="{BLUE}"/>',
                        text(x + 13, y - 14, label, 17, p["fg"], anchor="middle", weight="700", cls="m"),
                        text(x + 13, y + HX + 26, sub_, 14, p["sub"], anchor="middle", cls="m")])

    b.append(bar(30, "x", "4096"))
    b.append(f'<path d="M56 {CY}h34" stroke="{p["sub"]}" stroke-width="2"/>')
    b.append(f'<path d="M90 {CY}V72M90 {CY}V228" stroke="{p["sub"]}" stroke-width="2"/>')

    b.append(f'<path class="fl" d="M90 72h26" stroke="{p["sub"]}" stroke-width="2.4" marker-end="url(#aS)"/>')
    b.append(box(116, 44, 250, 56, p["dim"], p["line"]))
    b.append(text(241, 70, "W₀   冻结", 19, p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(241, 90, "4096 × 4096", 13, p["sub"], anchor="middle", cls="m"))
    b.append(f'<path d="M366 72h90v{CY - 92}" fill="none" stroke="{p["sub"]}" stroke-width="2" marker-end="url(#aS)"/>')

    b.append(f'<path class="fl" d="M90 228h26" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(box(116, 202, 84, 52, p["box"], GREEN))
    b.append(text(158, 226, "A", 19, GREEN, anchor="middle", weight="700", cls="m"))
    b.append(text(158, 245, "16 × 4096", 12.5, p["sub"], anchor="middle", cls="m"))
    b.append(f'<path class="fl" d="M200 228h32" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(f'<rect class="gl" x="234" y="216" width="22" height="24" rx="4" fill="{AMBER}"/>')
    b.append(text(245, 208, "r = 16", 14, AMBER, anchor="middle", weight="700", cls="m"))
    b.append(f'<path class="fl" d="M256 228h32" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(box(288, 202, 84, 52, p["box"], GREEN))
    b.append(text(330, 226, "B", 19, GREEN, anchor="middle", weight="700", cls="m"))
    b.append(text(330, 245, "4096 × 16", 12.5, p["sub"], anchor="middle", cls="m"))
    b.append(text(116, 190, "只训练这两个", 14, GREEN, weight="700"))
    b.append(f'<path d="M372 228h84v-{208 - CY}" fill="none" stroke="{GREEN}" stroke-width="2.2" marker-end="url(#aG)"/>')

    b.append(f'<circle cx="456" cy="{CY}" r="18" fill="{p["box"]}" stroke="{p["fg"]}" stroke-width="2"/>')
    b.append(text(456, CY + 8, "+", 23, p["fg"], anchor="middle", weight="700"))
    b.append(f'<path d="M474 {CY}h26" stroke="{p["sub"]}" stroke-width="2" marker-end="url(#aS)"/>')
    b.append(bar(500, "y", "4096"))
    b.append(text(560, CY - 8, "y = W₀x", 17, p["sub"], weight="600", cls="m"))
    b.append(text(560, CY + 18, "  + (α/r)·B(Ax)", 17, GREEN, weight="700", cls="m"))
    return svg(W, H, "".join(b), css, "LoRA 结构：主路 W0 冻结，旁路 A 降维到 r 再由 B 升维，两路相加")


# ------------------------------------------------------------- 训练一步四阶段
CYCLE = 8.0


def fig_train_step(p):
    W, H, CY = 720, 290, 190
    b, css = [], ("@keyframes hot{0%,1%{opacity:.2}3%,23%{opacity:1}25%,100%{opacity:.2}}"
                  f".hot{{animation:hot {CYCLE}s linear infinite}}" + NOMOTION)
    for label, use, color, x in (("bf16 工作副本", "用来算", BLUE, 56),
                                 ("fp32 正本", "用来存", GREEN, 424)):
        b.append(box(x, 36, 240, 66, p["box"], color))
        b.append(text(x + 120, 66, label, 17, p["fg"], anchor="middle", weight="700", cls="m"))
        b.append(text(x + 120, 88, use, 13.5, p["sub"], anchor="middle"))

    xs = [130, 290, 450, 610]
    for k, name in enumerate(["前向", "反向", "更新", "同步"]):
        x = xs[k]
        b.append(f'<circle cx="{x}" cy="{CY}" r="32" fill="{p["dim"]}" stroke="{p["line"]}" stroke-width="1.4"/>')
        b.append(f'<circle class="hot" cx="{x}" cy="{CY}" r="32" fill="none" stroke="{GREEN}" '
                 f'stroke-width="3.2" style="animation-delay:{-CYCLE / 4 * k:g}s"/>')
        b.append(text(x, CY - 2, str(k + 1), 15, GREEN, anchor="middle", weight="700", cls="m"))
        b.append(text(x, CY + 18, name, 14.5, p["fg"], anchor="middle", weight="600"))
        if k < 3:
            b.append(f'<path d="M{x + 34} {CY}h{xs[k + 1] - x - 68}" stroke="{p["line"]}" '
                     f'stroke-width="2" stroke-linecap="round"/>')
    b.append(f'<path d="M642 {CY}q46 52 -156 52H234q-150 0 -104 -52" fill="none" '
             f'stroke="{p["line"]}" stroke-width="1.8" stroke-dasharray="5 5"/>')
    b.append(f'<path d="M160 102v52" stroke="{BLUE}" stroke-width="2" opacity=".5"/>')
    b.append(f'<path d="M520 102v52" stroke="{GREEN}" stroke-width="2" opacity=".5"/>')
    b.append(f'<path d="M610 158V118h-66" fill="none" stroke="{GREEN}" stroke-width="2" opacity=".5"/>')
    return svg(W, H, "".join(b), css, "一步训练的四个阶段：前向、反向、更新、同步")


# ---------------------------------------------------------------- 浮点位布局
def fig_float_bits(p):
    W, H = 760, 208
    b, unit, x0 = [], 20.0, 76
    for i, (name, sb, eb, mb) in enumerate([("fp32", 1, 8, 23), ("fp16", 1, 5, 10), ("bf16", 1, 8, 7)]):
        y = 28 + i * 56
        b.append(text(64, y + 28, name, 16, p["fg"], weight="700", anchor="end", cls="m"))
        bx = x0
        for label, bits, color in (("", sb, p["sub"]), ("指数", eb, GREEN), ("尾数", mb, BLUE)):
            w = bits * unit
            b.append(f'<rect x="{bx}" y="{y}" width="{w}" height="44" rx="5" fill="{color}" '
                     f'opacity="{0.3 if not label else 0.9}"/>')
            if w > 40:
                b.append(text(bx + w / 2, y + 20, label, 13, "#FFFFFF", anchor="middle", weight="600"))
                b.append(text(bx + w / 2, y + 36, str(bits), 13, "#FFFFFF", anchor="middle", cls="m"))
            bx += w + 4
        b.append(text(bx + 10, y + 28, f"{sb + eb + mb} 位", 13, p["sub"], cls="m"))
    b.append(text(76, H - 12, "绿＝指数（决定范围）　蓝＝尾数（决定精度）", 13, p["sub"]))
    return svg(W, H, "".join(b), "", "fp32 fp16 bf16 的位分配")


# ----------------------------------------------------------------------- SVD
def fig_svd(p):
    W, H, R = 760, 246, 54
    css = ("@keyframes r1{0%,10%{transform:rotate(0)}26%,100%{transform:rotate(-34deg)}}"
           "@keyframes st{0%,34%{transform:scale(1,1)}52%,100%{transform:scale(1.5,.55)}}"
           "@keyframes r2{0%,60%{transform:rotate(0)}78%,100%{transform:rotate(26deg)}}"
           ".r1{animation:r1 9s ease-in-out infinite;transform-origin:center}"
           ".st{animation:st 9s ease-in-out infinite;transform-origin:center}"
           ".r2{animation:r2 9s ease-in-out infinite;transform-origin:center}" + NOMOTION)
    b = []
    for i, (name, cls) in enumerate([("单位圆", ""), ("Vᵀ 旋转", "r1"),
                                     ("Σ 拉伸", "r1 st"), ("U 旋转", "r1 st r2")]):
        ox, oy = 104 + i * 184, 112
        b.append(f'<g transform="translate({ox},{oy})">')
        b.append(f'<path d="M-78 0h156M0 -78v156" stroke="{p["line"]}" stroke-width="1"/>')
        shape = (f'<circle r="{R}" fill="{GREEN}" fill-opacity=".16" stroke="{GREEN}" stroke-width="2.6"/>'
                 f'<path d="M0 0h{R}" stroke="{BLUE}" stroke-width="3" stroke-linecap="round"/>'
                 f'<path d="M0 0v-{R}" stroke="{AMBER}" stroke-width="3" stroke-linecap="round"/>')
        b.append(f'<g class="{cls}">{shape}</g>' if cls else f'<g>{shape}</g>')
        b.append('</g>')
        b.append(text(ox, oy + 102, name, 15, p["fg"], anchor="middle", weight="600", cls="m"))
        if i < 3:
            b.append(f'<path d="M{ox + 86} {oy}h22" stroke="{p["line"]}" stroke-width="2" stroke-linecap="round"/>')
    return svg(W, H, "".join(b), css, "SVD 的几何：旋转、各方向拉伸、再旋转")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"fig-lora-arch": fig_lora_arch, "fig-train-step": fig_train_step,
                     "fig-float-bits": fig_float_bits, "fig-svd": fig_svd}.items():
        for suffix, pal in (("light", LIGHT), ("dark", DARK)):
            (OUT / f"{name}-{suffix}.svg").write_text(fn(pal))
        print(f"  {name}  {(OUT / f'{name}-light.svg').stat().st_size} 字节")


if __name__ == "__main__":
    main()
