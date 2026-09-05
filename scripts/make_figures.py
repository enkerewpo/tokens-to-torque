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


# --------------------------------------------------- LoRA 在线性层里挂在哪
def fig_lora_arch(p):
    W, H, CY = 760, 300, 150
    b = [f'<defs>'
         f'<marker id="aG" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
         f'<path d="M0 1 9 5 0 9z" fill="{GREEN}"/></marker>'
         f'<marker id="aS" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">'
         f'<path d="M0 1 9 5 0 9z" fill="{p["sub"]}"/></marker></defs>']
    css = ""
    HX = 150

    def bar(x, label, sub_):
        y = CY - HX / 2
        return "".join([f'<rect x="{x}" y="{y}" width="26" height="{HX}" rx="5" fill="{BLUE}"/>',
                        text(x + 13, y - 14, label, 17, p["fg"], anchor="middle", weight="700", cls="m"),
                        text(x + 13, y + HX + 26, sub_, 14, p["sub"], anchor="middle", cls="m")])

    b.append(bar(30, "x", "4096"))
    b.append(f'<path d="M56 {CY}h34" stroke="{p["sub"]}" stroke-width="2"/>')
    b.append(f'<path d="M90 {CY}V72M90 {CY}V228" stroke="{p["sub"]}" stroke-width="2"/>')

    b.append(f'<path d="M90 72h26" stroke="{p["sub"]}" stroke-width="2.4" marker-end="url(#aS)"/>')
    b.append(box(116, 44, 250, 56, p["dim"], p["line"]))
    b.append(text(241, 70, "W₀   冻结", 19, p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(241, 90, "4096 × 4096", 13, p["sub"], anchor="middle", cls="m"))
    b.append(f'<path d="M366 72h90v{CY - 92}" fill="none" stroke="{p["sub"]}" stroke-width="2" marker-end="url(#aS)"/>')

    b.append(f'<path d="M90 228h26" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(box(116, 202, 84, 52, p["box"], GREEN))
    b.append(text(158, 226, "A", 19, GREEN, anchor="middle", weight="700", cls="m"))
    b.append(text(158, 245, "16 × 4096", 12.5, p["sub"], anchor="middle", cls="m"))
    b.append(f'<path d="M200 228h32" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(f'<rect x="234" y="216" width="22" height="24" rx="4" fill="{AMBER}"/>')
    b.append(text(245, 208, "r = 16", 14, AMBER, anchor="middle", weight="700", cls="m"))
    b.append(f'<path d="M256 228h32" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
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
    css = ""
    d = lambda k: ""

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
    b.append(f'<path d="M96 72h{LX + 4 - 96}" stroke="{BLUE}" stroke-width="3.4" '
             f'marker-end="url(#fw)" style="{d(0)}"/>')
    b.append(text(96, 58, "① 前向：一路算到 loss", 14, BLUE, weight="700", cls="s",
                  extra=f'style="{d(0)}"'))
    b.append(f'<path d="M{LX + 4} 228H96" stroke="{AMBER}" stroke-width="3.4" '
             f'marker-end="url(#bw)" style="{d(1)}"/>')
    b.append(text(96, 250, "② 反向：梯度传回每一层", 14, AMBER, weight="700", cls="s",
                  extra=f'style="{d(1)}"'))

    # 右侧两份权重
    bx, bw = 552, 250
    b.append(box(bx, 60, bw, 60, p["box"], BLUE))
    b.append(text(bx + bw / 2, 87, "bf16 工作副本", 15, p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(bx + bw / 2, 106, "①② 拿它算", 12.5, p["sub"], anchor="middle"))
    b.append(f'<rect x="{bx}" y="60" width="{bw}" height="60" rx="9" fill="none" '
             f'stroke="{BLUE}" stroke-width="3" style="{d(0)}"/>')
    b.append(box(bx, 180, bw, 60, p["box"], GREEN))
    b.append(text(bx + bw / 2, 207, "fp32 正本 + m, v", 15, p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(bx + bw / 2, 226, "③ 更新累加在这儿", 12.5, p["sub"], anchor="middle"))
    b.append(f'<rect x="{bx}" y="180" width="{bw}" height="60" rx="9" fill="none" '
             f'stroke="{GREEN}" stroke-width="3" style="{d(2)}"/>')
    # ④ 同步：fp32 -> bf16
    b.append(f'<path d="M{bx - 8} 210h-18V90h18" fill="none" stroke="{GREEN}" '
             f'stroke-width="2.8" marker-end="url(#gw)" style="{d(3)}"/>')
    b.append(text(bx - 34, 154, "④ 同步", 13, GREEN, anchor="end", weight="700", cls="s",
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
    css = ""
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
                      cls="s"))
        if i < 3:
            b.append(f'<path d="M{ox + 84} {oy}h26" stroke="{p["sub"]}" stroke-width="2" '
                     f'stroke-linecap="round" marker-end="url(#ar)"/>')
    b.append(text(W / 2, H - 14, "蓝、橙是一对正交方向；拉伸那步各自乘上一个奇异值。",
                  13, p["sub"], anchor="middle"))
    return svg(W, H, "".join(b), css, "SVD 的几何：单位圆经旋转、各方向拉伸、再旋转变成椭圆")


# ------------------------------------------- Qwen3.5-9B：主干通路 / 一个块的内部
# 模块名、类名和形状都来自模型自己的 config 与 meta 设备上的模块清单：
#   python days/day00_lora-quickstart/code/peek_model.py --model Qwen/Qwen3.5-9B
# 图里绿色 = nn.Linear，也就是 target_modules="all-linear" 的挂点。
#
# 版面原则：窄而高，正文宽度下缩放比例小，手机上也读得清；连线一律从盒子的
# 锚点算，不手写坐标。


class Rect:
    """一个盒子，带四个锚点——连线只用锚点，避免手写坐标对不齐。"""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    @property
    def cx(self): return self.x + self.w / 2

    @property
    def cy(self): return self.y + self.h / 2

    @property
    def top(self): return self.y

    @property
    def bottom(self): return self.y + self.h

    @property
    def left(self): return self.x

    @property
    def right(self): return self.x + self.w


def _defs(p):
    return (f'<defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="11" '
            f'markerHeight="11" refX="9" refY="5.5" orient="auto">'
            f'<path d="M0 1 10 5.5 0 10z" fill="{p["sub"]}"/></marker></defs>')


def _vlink(p, a, b, dash=False):
    """两个盒子之间的竖直箭头。"""
    d = ' stroke-dasharray="5 4"' if dash else ""
    return (f'<path d="M{a.cx} {a.bottom}V{b.top - 10}" stroke="{p["sub"]}" '
            f'stroke-width="1.8"{d} marker-end="url(#qa)"/>')


def _card(p, r, title, sub_="", stroke=None, fill=None, mono=True):
    out = [box(r.x, r.y, r.w, r.h, fill or p["box"], stroke or p["line"], rx=8)]
    cls = "m" if mono else "s"
    if sub_:
        out.append(text(r.cx, r.y + 25, title, 15, p["fg"], anchor="middle",
                        weight="600", cls=cls))
        out.append(text(r.cx, r.y + 44, sub_, 12.5, p["sub"], anchor="middle", cls="m"))
    else:
        out.append(text(r.cx, r.cy + 5.5, title, 15, p["fg"], anchor="middle",
                        weight="600", cls=cls))
    return "".join(out)


def fig_qwen_arch(p):
    """主干：一串 token id 怎么走到下一个 token 的概率分布。"""
    W = 560
    X, BW = 150, 260
    b = [_defs(p)]
    y = 56
    b.append(text(30, 32, "整体通路", 16, p["fg"], weight="700"))

    def step(label, sub_="", h=52, stroke=None, fill=None, mono=True):
        nonlocal y
        r = Rect(X, y, BW, h)
        b.append(_card(p, r, label, sub_, stroke, fill, mono))
        y += h
        return r

    def gap(a, bx):
        b.append(_vlink(p, a, bx))

    r1 = step("token ids", "一句话被切成的整数序列", 56, fill=p["dim"], mono=False)
    y += 26
    r2 = step("embed_tokens", "248320 × 4096", 56, fill=p["dim"])
    gap(r1, r2)
    y += 26

    # 32 层：画一组 4 层，注明重复 8 次
    grp_top = y
    inner = []
    prev = None
    for i in range(4):
        full = i == 3
        r = Rect(X, y, BW, 44)
        inner.append(_card(p, r, "全注意力块" if full else "线性注意力块", "",
                           AMBER if full else BLUE, p["box"], mono=False))
        if prev is not None:
            inner.append(_vlink(p, prev, r))
        prev = r
        y += 44 + 16
    y -= 16
    grp = Rect(X - 16, grp_top - 16, BW + 32, y - grp_top + 32)
    b.append(f'<rect x="{grp.x}" y="{grp.y}" width="{grp.w}" height="{grp.h}" rx="12" '
             f'fill="none" stroke="{p["line"]}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    b.extend(inner)
    b.append(_vlink(p, r2, Rect(X, grp_top, BW, 44)))
    b.append(text(grp.right + 14, grp.cy - 8, "这 4 层一组", 13.5, p["sub"]))
    b.append(text(grp.right + 14, grp.cy + 12, "重复 8 次 = 32 层", 13.5, p["fg"], weight="700"))
    last_block = prev
    y = grp.bottom + 26

    r4 = step("model.norm", "RMSNorm", 56, fill=p["dim"])
    b.append(_vlink(p, last_block, r4))
    y += 26
    r5 = step("lm_head", "4096 → 248320", 56, fill=p["dim"])
    gap(r4, r5)
    b.append(text(r5.right + 14, r5.cy + 5, "LoRA 跳过它", 13.5, p["sub"]))
    y += 26
    r6 = step("logits", "每个词一个分数，248320 个", 56, fill=p["dim"], mono=False)
    gap(r5, r6)
    y += 26
    r7 = step("softmax → 概率分布", "从这里采样出下一个 token", 56, fill=p["dim"], mono=False)
    gap(r6, r7)

    b.append(text(30, r7.bottom + 34, "每生成一个 token，整条通路就从上到下走一遍。",
                  13.5, p["sub"]))
    return svg(W, int(r7.bottom + 56), "".join(b), "",
               "Qwen3.5-9B 的主干：token ids 经 embedding、32 个 decoder 块、"
               "RMSNorm、lm_head，得到 logits 再 softmax 成下一个 token 的概率分布")


def fig_qwen_block(p):
    """一个 decoder 块：骨架两种层共用，只有 mixer 不同。"""
    W = 700
    b = [_defs(p)]
    b.append(text(30, 32, "一个 decoder 块的内部", 16, p["fg"], weight="700"))
    b.append(text(30, 54, "两种层的骨架完全一样，只有中间的 mixer 不同", 13.5, p["sub"]))

    X, BW = 40, 300
    rail = X - 14
    y = 82
    seq = []

    def node(label, sub_="", h=50, stroke=None, fill=None, mono=True):
        nonlocal y
        r = Rect(X, y, BW, h)
        seq.append(_card(p, r, label, sub_, stroke, fill, mono))
        y += h + 26
        return r

    def plus(after):
        nonlocal y
        r = Rect(X + BW / 2 - 15, y, 30, 30)
        seq.append(f'<circle cx="{r.cx}" cy="{r.cy}" r="15" fill="{p["box"]}" '
                   f'stroke="{p["fg"]}" stroke-width="1.8"/>')
        seq.append(text(r.cx, r.cy + 6, "+", 18, p["fg"], anchor="middle", weight="700"))
        seq.append(_vlink(p, after, r))
        y += 30 + 26
        return r

    h_in = node("输入 h", "4096", 54, fill=p["dim"], mono=False)
    n1 = node("input_layernorm", "", 50)
    seq.append(_vlink(p, h_in, n1))
    mix = node("mixer", "见右边两种", 62, stroke=GREEN, fill=p["dim"], mono=False)
    seq.append(_vlink(p, n1, mix))
    add1 = plus(mix)
    n2 = node("post_attention_layernorm", "", 50)
    seq.append(f'<path d="M{add1.cx} {add1.bottom}V{n2.top - 10}" stroke="{p["sub"]}" '
               f'stroke-width="1.8" marker-end="url(#qa)"/>')
    ffn = node("mlp", "SwiGLU，见下方展开", 62, stroke=BLUE)
    seq.append(_vlink(p, n2, ffn))
    add2 = plus(ffn)
    h_out = node("输出 h", "4096", 54, fill=p["dim"], mono=False)
    seq.append(f'<path d="M{add2.cx} {add2.bottom}V{h_out.top - 10}" stroke="{p["sub"]}" '
               f'stroke-width="1.8" marker-end="url(#qa)"/>')

    # 两条残差：从进入归一化之前分叉，绕左侧到对应的 ⊕
    for src, dst in ((h_in, add1), (add1, add2)):
        b.append(f'<circle cx="{src.cx}" cy="{src.bottom + 13}" r="3.5" fill="{p["sub"]}"/>')
        b.append(f'<path d="M{src.cx} {src.bottom + 13}H{rail}V{dst.cy}H{dst.left - 10}" '
                 f'fill="none" stroke="{p["sub"]}" stroke-width="1.6" stroke-dasharray="5 4" '
                 f'marker-end="url(#qa)"/>')
    b.extend(seq)

    # 右侧：两种 mixer
    MX, MW = 372, 300
    def mixer(my, title, count, accent, rows, notes):
        r = Rect(MX, my, MW, 40 + 21 * len(rows) + 16 + 17 * len(notes))
        out = [box(r.x, r.y, r.w, r.h, p["dim"], accent, rx=10)]
        out.append(text(r.x + 14, r.y + 25, title, 15, p["fg"], weight="700"))
        out.append(text(r.right - 14, r.y + 25, count, 13.5, accent, anchor="end", weight="700"))
        for i, (name, dims, lin) in enumerate(rows):
            ty = r.y + 47 + i * 21
            out.append(text(r.x + 16, ty, name, 13, GREEN if lin else p["sub"], cls="m"))
            out.append(text(r.right - 16, ty, dims, 12, p["sub"], anchor="end", cls="m"))
        for k, line in enumerate(notes):
            out.append(text(r.x + 16, r.bottom - 12 - 17 * (len(notes) - 1 - k), line,
                            12, p["sub"]))
        return r, "".join(out)

    m1, s1 = mixer(96, "self_attn", "32 层里的 8 层", AMBER,
                   [("q_proj", "4096 → 8192", True),
                    ("k_proj   v_proj", "4096 → 1024", True),
                    ("q_norm  k_norm", "无权重矩阵", False),
                    ("o_proj", "4096 → 4096", True)],
                   ["16 个查询头 / 4 组 KV，每头 256 维",
                    "q_proj 宽一倍：它同时输出一路门控"])
    m2, s2 = mixer(m1.bottom + 34, "linear_attn", "另外 24 层", BLUE,
                   [("in_proj_qkv", "4096 → 8192", True),
                    ("in_proj_z", "4096 → 4096", True),
                    ("in_proj_a   in_proj_b", "4096 → 32", True),
                    ("conv1d(k=4)  norm", "不挂 LoRA", False),
                    ("out_proj", "4096 → 4096", True)],
                   ["GatedDeltaNet：一维卷积 +",
                    "一个随位置递推更新的状态"])
    b.append(s1)
    b.append(s2)
    for m in (m1, m2):
        b.append(f'<path d="M{mix.right} {mix.cy}H{(mix.right + MX) / 2}V{m.cy}H{m.left - 10}" '
                 f'fill="none" stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')

    # 底部：SwiGLU 展开
    by = max(h_out.bottom, m2.bottom) + 40
    panel = Rect(30, by, W - 60, 186)
    b.append(box(panel.x, panel.y, panel.w, panel.h, p["dim"], p["line"], rx=12))
    b.append(text(panel.x + 16, panel.y + 26, "mlp 展开：SwiGLU", 15, p["fg"], weight="700"))
    hb = Rect(panel.x + 20, panel.y + 60, 96, 54)
    gp = Rect(panel.x + 172, panel.y + 42, 138, 54)
    up = Rect(panel.x + 172, panel.y + 104, 138, 54)
    si = Rect(gp.right + 30, gp.y + 4, 80, 46)
    mul = Rect(si.right + 34, up.y + 8, 30, 30)
    dp = Rect(mul.right + 30, up.y, 138, 54)
    b.append(_card(p, hb, "h", "4096"))
    b.append(_card(p, gp, "gate_proj", "→ 12288", GREEN))
    b.append(_card(p, up, "up_proj", "→ 12288", GREEN))
    b.append(_card(p, si, "SiLU", "", None, None, False))
    b.append(f'<circle cx="{mul.cx}" cy="{mul.cy}" r="15" fill="{p["box"]}" '
             f'stroke="{p["fg"]}" stroke-width="1.8"/>')
    b.append(text(mul.cx, mul.cy + 6, "×", 17, p["fg"], anchor="middle", weight="700"))
    b.append(_card(p, dp, "down_proj", "→ 4096", GREEN))
    fork = hb.right + 22
    b.append(f'<path d="M{hb.right} {hb.cy}H{fork}" stroke="{p["sub"]}" stroke-width="1.8"/>')
    b.append(f'<path d="M{fork} {gp.cy}V{up.cy}" stroke="{p["sub"]}" stroke-width="1.8"/>')
    for t in (gp, up):
        b.append(f'<path d="M{fork} {t.cy}H{t.left - 10}" stroke="{p["sub"]}" '
                 f'stroke-width="1.8" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{gp.right} {gp.cy}H{si.left - 10}" stroke="{p["sub"]}" '
             f'stroke-width="1.8" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{si.right} {si.cy}H{mul.cx}V{mul.top - 10}" fill="none" '
             f'stroke="{p["sub"]}" stroke-width="1.8" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{up.right} {up.cy}H{mul.left - 10}" stroke="{p["sub"]}" '
             f'stroke-width="1.8" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{mul.right} {mul.cy}H{dp.left - 10}" stroke="{p["sub"]}" '
             f'stroke-width="1.8" marker-end="url(#qa)"/>')
    b.append(text(panel.x + 16, panel.bottom - 14,
                  "两条并行的线性变换，一条过 SiLU 当门，逐元素乘另一条，再投回 4096。",
                  12.5, p["sub"]))

    b.append(text(30, panel.bottom + 28, "虚线 = 残差连接：把模块的输出加回它自己的输入。",
                  13, p["sub"]))
    b.append(text(30, panel.bottom + 48, "绿色 = nn.Linear，LoRA 挂点；灰色是归一化、卷积、"
                  "激活，没有权重矩阵可拆。", 13, p["sub"]))
    return svg(W, int(panel.bottom + 68), "".join(b), "",
               "一个 decoder 块：输入先归一化再过 mixer，结果加回输入；再归一化过 SwiGLU 前馈网络，"
               "再加回一次。8 层用 self_attn，24 层用 linear_attn")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"fig-lora-arch": fig_lora_arch, "fig-train-step": fig_train_step,
                     "fig-float-bits": fig_float_bits, "fig-svd": fig_svd,
                     "fig-qwen-arch": fig_qwen_arch,
                     "fig-qwen-block": fig_qwen_block}.items():
        for suffix, pal in (("light", LIGHT), ("dark", DARK)):
            (OUT / f"{name}-{suffix}.svg").write_text(fn(pal))
        print(f"  {name}  {(OUT / f'{name}-light.svg').stat().st_size} 字节")


if __name__ == "__main__":
    main()
