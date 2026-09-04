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
         f'<marker id="aG" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
         f'<path d="M0 1 9 5 0 9z" fill="{GREEN}"/></marker>'
         f'<marker id="aS" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
         f'<path d="M0 1 9 5 0 9z" fill="{p["sub"]}"/></marker></defs>']
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
LAYERS = [3, 4, 3]


def _net(p, x0, y0, dx, dy):
    """算出每层节点坐标。"""
    cols = []
    for li, n in enumerate(LAYERS):
        h = (n - 1) * dy
        cols.append([(x0 + li * dx, y0 + i * dy - h / 2) for i in range(n)])
    return cols


def fig_train_step(p):
    """一步训练：前向、反向、更新、同步。静止时也能读——方向由网络上下两条箭头给出。"""
    W, H, CY = 820, 296, 150
    CYC = CYCLE
    cols = _net(p, 110, CY, 140, 42)
    OUTX = cols[-1][0][0]
    LX = OUTX + 62
    b = [f'<defs>'
         f'<marker id="fw" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto">'
         f'<path d="M0 1 10 5.5 0 10z" fill="{BLUE}"/></marker>'
         f'<marker id="bw" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto">'
         f'<path d="M0 1 10 5.5 0 10z" fill="{AMBER}"/></marker>'
         f'<marker id="gw" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto">'
         f'<path d="M0 1 10 5.5 0 10z" fill="{GREEN}"/></marker></defs>']
    css = ("@keyframes lit{0%,1%{opacity:.25}3%,23%{opacity:1}25%,100%{opacity:.25}}"
           f".li{{animation:lit {CYC}s linear infinite}}"
           "@media(prefers-reduced-motion:reduce){.li{animation:none;opacity:1}}")
    d = lambda k: f"animation-delay:{-CYC / 4 * k:g}s"

    for li in range(len(cols) - 1):
        for (x1, y1) in cols[li]:
            for (x2, y2) in cols[li + 1]:
                b.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="{p["line"]}" stroke-width="1" opacity=".6"/>')
    for col in cols:
        for (x, y) in col:
            b.append(f'<circle cx="{x}" cy="{y}" r="10" fill="{p["box"]}" '
                     f'stroke="{p["line"]}" stroke-width="1.4"/>')
    # loss
    b.append(f'<line x1="{OUTX + 12}" y1="{CY}" x2="{LX - 18}" y2="{CY}" stroke="{AMBER}" '
             f'stroke-width="1.6" opacity=".7"/>')
    b.append(f'<circle cx="{LX}" cy="{CY}" r="16" fill="{AMBER}" fill-opacity=".2" '
             f'stroke="{AMBER}" stroke-width="2"/>')
    b.append(text(LX, CY + 5, "L", 14, AMBER, anchor="middle", weight="700", cls="m"))

    # 上下两条方向箭头，不与网络重叠
    b.append(f'<path class="li" d="M96 72h{LX + 4 - 96}" stroke="{BLUE}" stroke-width="3.4" '
             f'marker-end="url(#fw)" style="{d(0)}"/>')
    b.append(text(96, 58, "① 前向：一路算到 loss", 14, BLUE, weight="700", cls="s li",
                  extra=f'style="{d(0)}"'))
    b.append(f'<path class="li" d="M{LX + 4} 228H96" stroke="{AMBER}" stroke-width="3.4" '
             f'marker-end="url(#bw)" style="{d(1)}"/>')
    b.append(text(96, 250, "② 反向：梯度传回每一层", 14, AMBER, weight="700", cls="s li",
                  extra=f'style="{d(1)}"'))

    # 右侧两份权重
    bx, bw = 552, 250
    b.append(box(bx, 60, bw, 60, p["box"], BLUE))
    b.append(text(bx + bw / 2, 87, "bf16 工作副本", 15, p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(bx + bw / 2, 106, "①② 拿它算", 12.5, p["sub"], anchor="middle"))
    b.append(f'<rect class="li" x="{bx}" y="60" width="{bw}" height="60" rx="9" fill="none" '
             f'stroke="{BLUE}" stroke-width="3" style="{d(0)}"/>')
    b.append(box(bx, 180, bw, 60, p["box"], GREEN))
    b.append(text(bx + bw / 2, 207, "fp32 正本 + m, v", 15, p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(bx + bw / 2, 226, "③ 更新累加在这儿", 12.5, p["sub"], anchor="middle"))
    b.append(f'<rect class="li" x="{bx}" y="180" width="{bw}" height="60" rx="9" fill="none" '
             f'stroke="{GREEN}" stroke-width="3" style="{d(2)}"/>')
    # ④ 同步：fp32 -> bf16
    b.append(f'<path class="li" d="M{bx - 8} 210h-18V90h18" fill="none" stroke="{GREEN}" '
             f'stroke-width="2.8" marker-end="url(#gw)" style="{d(3)}"/>')
    b.append(text(bx - 34, 154, "④ 同步", 13, GREEN, anchor="end", weight="700", cls="s li",
                  extra=f'style="{d(3)}"'))
    b.append(text(bx + bw / 2, H - 12, "网络算的时候用的就是上面这份权重", 12, p["sub"], anchor="middle"))
    return svg(W, H, "".join(b), css, "一步训练：前向、反向、权重更新、副本同步")


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
    """四格各画各的终态——静止时也能看出变化；动画只是让过渡更直观。"""
    import math
    W, H, R = 780, 250 , 44
    css = ("@keyframes pulse{0%,1%{opacity:.3}4%,22%{opacity:1}25%,100%{opacity:.3}}"
           ".pu{animation:pulse 8s linear infinite}"
           "@media(prefers-reduced-motion:reduce){.pu{animation:none;opacity:1}}")
    b = [f'<defs><marker id="ar" markerUnits="userSpaceOnUse" markerWidth="11" markerHeight="11" refX="9" refY="5.5" orient="auto">'
         f'<path d="M0 1 10 5.5 0 10z" fill="{p["sub"]}"/></marker></defs>']

    # 每格：(名字, 旋转角, x 缩放, y 缩放)，逐格叠加
    stages = [("单位圆", 0, 1.0, 1.0), ("Vᵀ 旋转", -34, 1.0, 1.0),
              ("Σ 拉伸", -34, 1.55, 0.55), ("U 再旋转", -34 + 26, 1.55, 0.55)]
    for i, (name, rot, sx, sy) in enumerate(stages):
        ox, oy = 100 + i * 188, 112
        b.append(f'<g transform="translate({ox},{oy})">')
        b.append(f'<path d="M-76 0h152M0 -76v152" stroke="{p["line"]}" stroke-width="1"/>')
        # 第 3、4 格的拉伸发生在旋转后的坐标系里：先转 rot_inner，再缩放，再转外层
        inner = -34 if i >= 2 else 0
        outer = rot - inner
        g = f'<g transform="rotate({outer}) scale({sx},{sy}) rotate({inner})">' if i >= 2 \
            else f'<g transform="rotate({rot})">'
        b.append(g)
        b.append(f'<circle r="{R}" fill="{GREEN}" fill-opacity=".16" stroke="{GREEN}" stroke-width="2.6"/>')
        b.append(f'<path d="M0 0h{R}" stroke="{BLUE}" stroke-width="3" stroke-linecap="round"/>')
        b.append(f'<path d="M0 0v-{R}" stroke="{AMBER}" stroke-width="3" stroke-linecap="round"/>')
        b.append('</g></g>')
        b.append(text(ox, oy + 96, name, 15, p["fg"], anchor="middle", weight="600",
                      cls="s pu", extra=f'style="animation-delay:{-2 * i:g}s"'))
        if i < 3:
            b.append(f'<path d="M{ox + 84} {oy}h26" stroke="{p["sub"]}" stroke-width="2" '
                     f'stroke-linecap="round" marker-end="url(#ar)"/>')
    b.append(text(W / 2, H - 14, "蓝、橙是一对正交方向；拉伸那步各自乘上一个奇异值。",
                  13, p["sub"], anchor="middle"))
    return svg(W, H, "".join(b), css, "SVD 的几何：单位圆经旋转、各方向拉伸、再旋转变成椭圆")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"fig-lora-arch": fig_lora_arch, "fig-train-step": fig_train_step,
                     "fig-float-bits": fig_float_bits, "fig-svd": fig_svd}.items():
        for suffix, pal in (("light", LIGHT), ("dark", DARK)):
            (OUT / f"{name}-{suffix}.svg").write_text(fn(pal))
        print(f"  {name}  {(OUT / f'{name}-light.svg').stat().st_size} 字节")


if __name__ == "__main__":
    main()
