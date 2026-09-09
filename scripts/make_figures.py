#!/usr/bin/env python3
"""生成教程插图（深浅色各一份）到 site_src/assets/。

原则：**图里只放结构和极少标签。** 说明文字放在正文里——正文的字随页面缩放、
能选中、能搜索、能翻译；塞进 SVG 的字一样都做不到，而且图被压到正文宽度就糊了。

用法：python scripts/make_figures.py
"""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from figures import GREEN, BLUE, AMBER, LIGHT, DARK, svg, box, text, fs

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
                        text(x + 13, y - 14, label, fs(W), p["fg"], anchor="middle", weight="700", cls="m"),
                        text(x + 13, y + HX + 26, sub_, fs(W, .88), p["sub"], anchor="middle", cls="m")])

    b.append(bar(30, "x", "4096"))
    b.append(f'<path d="M56 {CY}h34" stroke="{p["sub"]}" stroke-width="2"/>')
    b.append(f'<path d="M90 {CY}V72M90 {CY}V228" stroke="{p["sub"]}" stroke-width="2"/>')

    b.append(f'<path d="M90 72h26" stroke="{p["sub"]}" stroke-width="2.4" marker-end="url(#aS)"/>')
    b.append(box(116, 44, 250, 56, p["dim"], p["line"]))
    b.append(text(241, 70, "W₀   冻结", fs(W, 1.08), p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(241, 90, "4096 × 4096", fs(W, .88), p["sub"], anchor="middle", cls="m"))
    b.append(f'<path d="M366 72h90v{CY - 92}" fill="none" stroke="{p["sub"]}" stroke-width="2" marker-end="url(#aS)"/>')

    b.append(f'<path d="M90 228h26" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(box(116, 202, 84, 52, p["box"], GREEN))
    b.append(text(158, 226, "A", fs(W, 1.08), GREEN, anchor="middle", weight="700", cls="m"))
    b.append(text(158, 245, "16 × 4096", fs(W, .82), p["sub"], anchor="middle", cls="m"))
    b.append(f'<path d="M200 228h32" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(f'<rect x="234" y="216" width="22" height="24" rx="4" fill="{AMBER}"/>')
    b.append(text(245, 208, "r = 16", fs(W, .88), AMBER, anchor="middle", weight="700", cls="m"))
    b.append(f'<path d="M256 228h32" stroke="{GREEN}" stroke-width="2.6" marker-end="url(#aG)"/>')
    b.append(box(288, 202, 84, 52, p["box"], GREEN))
    b.append(text(330, 226, "B", fs(W, 1.08), GREEN, anchor="middle", weight="700", cls="m"))
    b.append(text(330, 245, "4096 × 16", fs(W, .82), p["sub"], anchor="middle", cls="m"))
    b.append(text(116, 190, "只训练这两个", fs(W, .88), GREEN, weight="700"))
    b.append(f'<path d="M372 228h84v-{208 - CY}" fill="none" stroke="{GREEN}" stroke-width="2.2" marker-end="url(#aG)"/>')

    b.append(f'<circle cx="456" cy="{CY}" r="18" fill="{p["box"]}" stroke="{p["fg"]}" stroke-width="2"/>')
    b.append(text(456, CY + 8, "+", fs(W, 1.3), p["fg"], anchor="middle", weight="700"))
    b.append(f'<path d="M474 {CY}h26" stroke="{p["sub"]}" stroke-width="2" marker-end="url(#aS)"/>')
    b.append(bar(500, "y", "4096"))
    b.append(text(560, CY - 8, "y = W₀x", fs(W), p["sub"], weight="600", cls="m"))
    b.append(text(560, CY + 18, "  + (α/r)·B(Ax)", fs(W), GREEN, weight="700", cls="m"))
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
    b.append(text(LX, CY + 5, "L", fs(W, .88), AMBER, anchor="middle", weight="700", cls="m"))

    # 上下两条方向箭头，不与网络重叠
    b.append(f'<path d="M96 72h{LX - 22 - 96}" stroke="{BLUE}" stroke-width="3.4" '
             f'marker-end="url(#fw)" style="{d(0)}"/>')
    b.append(text(96, 58, "① 前向：一路算到 loss", fs(W, .88), BLUE, weight="700", cls="s",
                  extra=f'style="{d(0)}"'))
    b.append(f'<path d="M{LX - 22} 228H96" stroke="{AMBER}" stroke-width="3.4" '
             f'marker-end="url(#bw)" style="{d(1)}"/>')
    b.append(text(96, 250, "② 反向：梯度传回每一层", fs(W, .88), AMBER, weight="700", cls="s",
                  extra=f'style="{d(1)}"'))

    # 右侧两份权重
    bx, bw = 552, 250
    b.append(box(bx, 60, bw, 60, p["box"], BLUE))
    b.append(text(bx + bw / 2, 87, "bf16 工作副本", fs(W), p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(bx + bw / 2, 106, "①② 拿它算", fs(W, .8), p["sub"], anchor="middle"))
    b.append(f'<rect x="{bx}" y="60" width="{bw}" height="60" rx="9" fill="none" '
             f'stroke="{BLUE}" stroke-width="3" style="{d(0)}"/>')
    b.append(box(bx, 180, bw, 60, p["box"], GREEN))
    b.append(text(bx + bw / 2, 207, "fp32 正本 + m, v", fs(W), p["fg"], anchor="middle", weight="700", cls="m"))
    b.append(text(bx + bw / 2, 226, "③ 更新累加在这儿", fs(W, .8), p["sub"], anchor="middle"))
    b.append(f'<rect x="{bx}" y="180" width="{bw}" height="60" rx="9" fill="none" '
             f'stroke="{GREEN}" stroke-width="3" style="{d(2)}"/>')
    # ④ 同步：fp32 -> bf16
    b.append(f'<path d="M{bx - 8} 210h-18V90h18" fill="none" stroke="{GREEN}" '
             f'stroke-width="2.8" marker-end="url(#gw)" style="{d(3)}"/>')
    b.append(text(bx - 30, 138, "④ 同步", fs(W, .88), GREEN, anchor="end", weight="700", cls="s",
                  extra=f'style="{d(3)}"'))
    b.append(text(bx + bw / 2, H - 12, "网络算的时候用的就是上面这份权重", fs(W, .8), p["sub"], anchor="middle"))
    return svg(W, H, "".join(b), css, "一步训练：前向、反向、权重更新、副本同步")


# ---------------------------------------------------------------- 浮点位布局
# 画法跟着 IEEE 754 / bfloat16 的通行画法：每一位一个格子，字段名和位宽写在
# 格子上方，边界位号写在下方，几种格式左对齐堆叠，方便直接比长度。
FMTS = [("fp32", 1, 8, 23), ("fp16", 1, 5, 10), ("bf16", 1, 8, 7)]


def fig_float_bits(p):
    W = 800
    CELL, GAP = 17.0, 1.0
    X0, ROW_H = 96, 30
    b = []
    b.append(text(30, 34, "一个浮点数的位是怎么分的", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(30, 58, "每个格子是 1 位，三行左对齐，可以直接比长度",
                  fs(W, .8), p["sub"]))

    y = 96
    for name, sb, eb, mb in FMTS:
        total = sb + eb + mb
        b.append(text(X0 - 14, y + ROW_H / 2 + 5, name, fs(W), p["fg"],
                      anchor="end", weight="700", cls="m"))
        x = X0
        for kind, n, col in (("符号", sb, p["sub"]), ("指数", eb, GREEN), ("尾数", mb, BLUE)):
            gw = n * (CELL + GAP) - GAP
            for i in range(n):
                cx = x + i * (CELL + GAP)
                b.append(f'<rect x="{cx:.1f}" y="{y}" width="{CELL:.1f}" height="{ROW_H}" '
                         f'fill="{col}" fill-opacity="{0.85 if kind != "符号" else 0.3}"/>')
            label = kind if kind == "符号" else f"{kind} {n} 位"
            b.append(text(x + gw / 2, y - 8, label, fs(W, .8),
                          p["fg"] if kind != "符号" else p["sub"], anchor="middle"))
            x += gw + 6
        b.append(text(x + 8, y + ROW_H / 2 + 5, f"共 {total} 位", fs(W, .8), p["sub"], cls="m"))

        # 边界位号写在下方：最高位、指数两端、尾数最低位
        marks = [(X0 + CELL / 2, total - 1),
                 (X0 + CELL + 6 + CELL / 2, total - 2),
                 (X0 + CELL + 6 + eb * (CELL + GAP) - GAP - CELL / 2, mb),
                 (X0 + CELL + 6 + eb * (CELL + GAP) - GAP + 6 + CELL / 2, mb - 1),
                 (X0 + CELL + 6 + eb * (CELL + GAP) - GAP + 6 + mb * (CELL + GAP) - GAP - CELL / 2, 0)]
        for mx, idx in marks:
            b.append(text(mx, y + ROW_H + 16, str(idx), fs(W, .72), p["sub"],
                          anchor="middle", cls="m"))
        y += ROW_H + 52

    last = y - 22
    b.append(text(30, last + 18, "灰＝符号位；绿＝指数，决定能表示多大多小；蓝＝尾数，决定同一量级里分得多细。",
                  fs(W, .88), p["sub"]))
    b.append(text(30, last + 42,
                  "bf16 的指数位和 fp32 一样多，所以数值范围一样大，牺牲的是精度——这正是训练用它的理由。",
                  fs(W, .88), p["fg"], weight="700"))
    return svg(W, int(last + 66), "".join(b), "",
               "fp32、fp16、bf16 的位分配：每格一位，字段名在上、位号在下；"
               "bf16 的指数位与 fp32 相同，尾数更短")


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
        b.append(text(ox, oy + 96, name, fs(W), p["fg"], anchor="middle", weight="600",
                      cls="s"))
        if i < 3:
            b.append(f'<path d="M{ox + 84} {oy}h26" stroke="{p["sub"]}" stroke-width="2" '
                     f'stroke-linecap="round" marker-end="url(#ar)"/>')
    b.append(text(W / 2, H - 14, "蓝、橙是一对正交方向；拉伸那步各自乘上一个奇异值。",
                  fs(W, .88), p["sub"], anchor="middle"))
    return svg(W, H, "".join(b), css, "SVD 的几何：单位圆经旋转、各方向拉伸、再旋转变成椭圆")


# ------------------------------------------- Qwen3.5-9B：主干通路 / 一个块的内部
# 模块名、类名和形状都来自模型自己的 config 与 meta 设备上的模块清单：
#   python days/day00_lora-quickstart/code/peek_model.py --model Qwen/Qwen3.5-9B
# 图里绿色 = nn.Linear，也就是 target_modules="all-linear" 的挂点。
#
# 版面原则：窄而高，正文宽度下缩放比例小，手机上也读得清；连线一律从盒子的
# 锚点算，不手写坐标。


QW = 750  # 两张 Qwen 图统一画布宽度，正文栏里正好 1:1


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
    return (f'<defs><marker id="qa" markerUnits="userSpaceOnUse" markerWidth="9" '
            f'markerHeight="9" refX="8" refY="4.5" orient="auto">'
            f'<path d="M0 0.8 8 4.5 0 8.2z" fill="{p["sub"]}"/></marker></defs>')


GAP = 34  # 盒子之间的竖直间距：减去 8 px 的箭头后还剩得下一段看得见的杆


def _vlink(p, a, b, dash=False):
    """两个盒子之间的竖直箭头，箭尖正好落在下一个盒子的上沿。"""
    d = ' stroke-dasharray="5 4"' if dash else ""
    return (f'<path d="M{a.cx} {a.bottom}V{b.top}" stroke="{p["sub"]}" '
            f'stroke-width="1.6"{d} marker-end="url(#qa)"/>')


def _card(p, r, title, sub_="", stroke=None, fill=None, mono=True):
    out = [box(r.x, r.y, r.w, r.h, fill or p["box"], stroke or p["line"], rx=8)]
    cls = "m" if mono else "s"
    if sub_:
        # 两行按盒子中心排，不用固定偏移：偏移写死的话，盒子一矮副标题就顶到下边框
        out.append(text(r.cx, r.cy - 5, title, fs(QW), p["fg"], anchor="middle",
                        weight="600", cls=cls))
        out.append(text(r.cx, r.cy + 17, sub_, fs(QW, .88), p["sub"], anchor="middle", cls="m"))
    else:
        out.append(text(r.cx, r.cy + 6, title, fs(QW), p["fg"], anchor="middle",
                        weight="600", cls=cls))
    return "".join(out)


def fig_qwen_arch(p):
    """主干：一串 token id 怎么走到下一个 token 的概率分布。"""
    W = QW
    X, BW = 205, 340
    b = [_defs(p)]
    y = 56
    b.append(text(30, 34, "整体通路", fs(W, 1.06), p["fg"], weight="700"))

    def step(label, sub_="", h=64, stroke=None, fill=None, mono=True):
        nonlocal y
        r = Rect(X, y, BW, h)
        b.append(_card(p, r, label, sub_, stroke, fill, mono))
        y += h
        return r

    def gap(a, bx):
        b.append(_vlink(p, a, bx))

    r1 = step("token ids", "一句话被切成的整数序列", 64, fill=p["dim"], mono=False)
    y += GAP
    r2 = step("embed_tokens", "248320 × 4096", 64, fill=p["dim"])
    gap(r1, r2)
    y += GAP

    # 32 层：画一组 4 层，注明重复 8 次
    grp_top = y
    inner = []
    prev = None
    for i in range(4):
        full = i == 3
        r = Rect(X, y, BW, 52)
        inner.append(_card(p, r, "全注意力块" if full else "线性注意力块", "",
                           AMBER if full else BLUE, p["box"], mono=False))
        if prev is not None:
            inner.append(_vlink(p, prev, r))
        prev = r
        y += 52 + GAP
    y -= GAP
    grp = Rect(X - 16, grp_top - 16, BW + 32, y - grp_top + 32)
    b.append(f'<rect x="{grp.x}" y="{grp.y}" width="{grp.w}" height="{grp.h}" rx="12" '
             f'fill="none" stroke="{p["line"]}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    b.extend(inner)
    b.append(_vlink(p, r2, Rect(X, grp_top, BW, 52)))
    b.append(text(grp.right + 14, grp.cy - 8, "这 4 层一组", fs(W, .88), p["sub"]))
    b.append(text(grp.right + 14, grp.cy + 12, "重复 8 次 = 32 层", fs(W, .88), p["fg"], weight="700"))
    last_block = prev
    y = grp.bottom + GAP

    r4 = step("model.norm", "RMSNorm", 64, fill=p["dim"])
    b.append(_vlink(p, last_block, r4))
    y += GAP
    r5 = step("lm_head", "4096 → 248320", 64, fill=p["dim"])
    gap(r4, r5)
    b.append(text(r5.right + 14, r5.cy + 5, "LoRA 跳过它", fs(W, .88), p["sub"]))
    y += GAP
    r6 = step("logits", "每个词一个分数，248320 个", 64, fill=p["dim"], mono=False)
    gap(r5, r6)
    y += GAP
    r7 = step("softmax → 概率分布", "从这里采样出下一个 token", 64, fill=p["dim"], mono=False)
    gap(r6, r7)

    b.append(text(30, r7.bottom + 34, "每生成一个 token，整条通路就从上到下走一遍。",
                  fs(W, .88), p["sub"]))
    return svg(W, int(r7.bottom + 56), "".join(b), "",
               "Qwen3.5-9B 的主干：token ids 经 embedding、32 个 decoder 块、"
               "RMSNorm、lm_head，得到 logits 再 softmax 成下一个 token 的概率分布")


def fig_qwen_block(p):
    """一个 decoder 块：骨架两种层共用，只有 mixer 不同。"""
    W = QW
    b = [_defs(p)]
    b.append(text(30, 34, "一个 decoder 块的内部", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(30, 54, "两种层的骨架完全一样，只有中间的 mixer 不同", fs(W, .88), p["sub"]))

    X, BW = 36, 300
    rail = X - 22
    y = 82
    seq = []

    def node(label, sub_="", h=56, stroke=None, fill=None, mono=True):
        nonlocal y
        r = Rect(X, y, BW, h)
        seq.append(_card(p, r, label, sub_, stroke, fill, mono))
        y += h + GAP
        return r

    def plus(after):
        nonlocal y
        r = Rect(X + BW / 2 - 16, y, 32, 32)
        seq.append(f'<circle cx="{r.cx}" cy="{r.cy}" r="16" fill="{p["box"]}" '
                   f'stroke="{p["fg"]}" stroke-width="1.8"/>')
        seq.append(text(r.cx, r.cy + 7, "+", fs(W, 1.2), p["fg"], anchor="middle", weight="700"))
        seq.append(_vlink(p, after, r))
        y += 32 + GAP
        return r

    h_in = node("输入 h", "4096", 62, fill=p["dim"], mono=False)
    n1 = node("input_layernorm", "", 56)
    seq.append(_vlink(p, h_in, n1))
    mix = node("mixer", "见右边两种", 70, stroke=GREEN, fill=p["dim"], mono=False)
    seq.append(_vlink(p, n1, mix))
    add1 = plus(mix)
    n2 = node("post_attention_layernorm", "", 56)
    seq.append(f'<path d="M{add1.cx} {add1.bottom}V{n2.top - 10}" stroke="{p["sub"]}" '
               f'stroke-width="1.8" marker-end="url(#qa)"/>')
    ffn = node("mlp", "SwiGLU，见下方展开", 70, stroke=BLUE)
    seq.append(_vlink(p, n2, ffn))
    add2 = plus(ffn)
    h_out = node("输出 h", "4096", 62, fill=p["dim"], mono=False)
    seq.append(f'<path d="M{add2.cx} {add2.bottom}V{h_out.top - 10}" stroke="{p["sub"]}" '
               f'stroke-width="1.8" marker-end="url(#qa)"/>')

    # 两条残差：从进入归一化之前分叉，绕左侧到对应的 ⊕
    for src, dst in ((h_in, add1), (add1, add2)):
        b.append(f'<circle cx="{src.left}" cy="{src.cy}" r="3.5" fill="{p["sub"]}"/>')
        b.append(f'<path d="M{src.left} {src.cy}H{rail}V{dst.cy}H{dst.left}" '
                 f'fill="none" stroke="{p["sub"]}" stroke-width="1.5" stroke-dasharray="5 4" '
                 f'marker-end="url(#qa)"/>')
    b.extend(seq)

    # 右侧：两种 mixer
    MX, MW = 396, 322
    def mixer(my, title, count, accent, rows, notes):
        r = Rect(MX, my, MW, 44 + 24 * len(rows) + 18 + 19 * len(notes))
        out = [box(r.x, r.y, r.w, r.h, p["dim"], accent, rx=10)]
        out.append(text(r.x + 14, r.y + 27, title, fs(W), p["fg"], weight="700"))
        out.append(text(r.right - 14, r.y + 27, count, fs(W, .88), accent, anchor="end", weight="700"))
        for i, (name, dims, lin) in enumerate(rows):
            ty = r.y + 52 + i * 24
            out.append(text(r.x + 16, ty, name, fs(W, .88), GREEN if lin else p["sub"], cls="m"))
            out.append(text(r.right - 16, ty, dims, fs(W, .8), p["sub"], anchor="end", cls="m"))
        for k, line in enumerate(notes):
            out.append(text(r.x + 16, r.bottom - 13 - 19 * (len(notes) - 1 - k), line,
                            fs(W, .8), p["sub"]))
        return r, "".join(out)

    m1, s1 = mixer(96, "self_attn", "32 层里的 8 层", AMBER,
                   [("q_proj", "4096 → 8192", True),
                    ("k_proj   v_proj", "4096 → 1024", True),
                    ("q_norm  k_norm", "无权重矩阵", False),
                    ("o_proj", "4096 → 4096", True)],
                   ["16 个查询头 / 4 组 KV，每头 256 维",
                    "q_proj 宽一倍：它同时输出一路门控"])
    m2, s2 = mixer(m1.bottom + 30, "linear_attn", "另外 24 层", BLUE,
                   [("in_proj_qkv", "4096 → 8192", True),
                    ("in_proj_z", "4096 → 4096", True),
                    ("in_proj_a   in_proj_b", "4096 → 32", True),
                    ("conv1d(k=4)  norm", "不挂 LoRA", False),
                    ("out_proj", "4096 → 4096", True)],
                   ["GatedDeltaNet：一维卷积 +",
                    "一个随位置递推更新的状态"])
    b.append(s1)
    b.append(s2)
    bus = (mix.right + MX) / 2
    b.append(f'<path d="M{mix.right} {mix.cy}H{bus}" stroke="{p["sub"]}" stroke-width="1.6"/>')
    b.append(f'<path d="M{bus} {m1.cy}V{m2.cy}" stroke="{p["sub"]}" stroke-width="1.6"/>')
    for m in (m1, m2):
        b.append(f'<path d="M{bus} {m.cy}H{m.left}" stroke="{p["sub"]}" stroke-width="1.6" '
                 f'marker-end="url(#qa)"/>')

    # 底部：SwiGLU 展开。面板高度按内容算，别写死——写死过一次，注释那行
    # 直接压在 up_proj 盒子上。
    by = max(h_out.bottom, m2.bottom) + 44
    px0, pw = 30, W - 60
    hb = Rect(px0 + 22, by + 70, 104, 62)
    gp = Rect(px0 + 188, by + 48, 152, 62)
    up = Rect(px0 + 188, by + 124, 152, 62)
    si = Rect(gp.right + 30, gp.y + 7, 86, 48)
    mul = Rect(si.right + 30, up.cy - 16, 32, 32)
    dp = Rect(mul.right + 30, up.y, 152, 62)
    note_y = up.bottom + 34
    panel = Rect(px0, by, pw, note_y - by + 18)

    b.append(box(panel.x, panel.y, panel.w, panel.h, p["dim"], p["line"], rx=12))
    b.append(text(panel.x + 18, panel.y + 30, "mlp 展开：SwiGLU", fs(W), p["fg"], weight="700"))
    b.append(_card(p, hb, "h", "4096"))
    b.append(_card(p, gp, "gate_proj", "→ 12288", GREEN))
    b.append(_card(p, up, "up_proj", "→ 12288", GREEN))
    b.append(_card(p, si, "SiLU", "", None, None, False))
    b.append(f'<circle cx="{mul.cx}" cy="{mul.cy}" r="16" fill="{p["box"]}" '
             f'stroke="{p["fg"]}" stroke-width="1.8"/>')
    b.append(text(mul.cx, mul.cy + 7, "×", fs(W, 1.1), p["fg"], anchor="middle", weight="700"))
    b.append(_card(p, dp, "down_proj", "→ 4096", GREEN))

    fork = hb.right + 26
    b.append(f'<path d="M{hb.right} {hb.cy}H{fork}" stroke="{p["sub"]}" stroke-width="1.6"/>')
    b.append(f'<path d="M{fork} {gp.cy}V{up.cy}" stroke="{p["sub"]}" stroke-width="1.6"/>')
    for t in (gp, up):
        b.append(f'<path d="M{fork} {t.cy}H{t.left}" stroke="{p["sub"]}" '
                 f'stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{gp.right} {gp.cy}H{si.left}" stroke="{p["sub"]}" '
             f'stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{si.right} {si.cy}H{mul.cx}V{mul.top}" fill="none" '
             f'stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{up.right} {up.cy}H{mul.left}" stroke="{p["sub"]}" '
             f'stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(f'<path d="M{mul.right} {mul.cy}H{dp.left}" stroke="{p["sub"]}" '
             f'stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(text(panel.x + 18, note_y,
                  "两条并行的线性变换，一条过 SiLU 当门，逐元素乘另一条，再投回 4096。",
                  fs(W, .8), p["sub"]))

    b.append(text(30, panel.bottom + 30, "虚线 = 残差连接：把模块的输出加回它自己的输入。",
                  fs(W, .88), p["sub"]))
    b.append(text(30, panel.bottom + 54, "绿色 = nn.Linear，LoRA 挂点；灰色是归一化、卷积、"
                  "激活，没有权重矩阵可拆。", fs(W, .88), p["sub"]))
    return svg(W, int(panel.bottom + 76), "".join(b), "",
               "一个 decoder 块：输入先归一化再过 mixer，结果加回输入；再归一化过 SwiGLU 前馈网络，"
               "再加回一次。8 层用 self_attn，24 层用 linear_attn")


# ------------------------------------------------ 注意力三步（附录 D.3 的例子）
# 数字和正文里那个手算例子完全一致：q3·kj/√2 → softmax → 加权求和。
ATTN_ROWS = [("书", "k₁ = (1, 0)", "v₁ = (2, 0)", 0.99, 0.53),
             ("桌", "k₂ = (0, 1)", "v₂ = (0, 2)", 0.14, 0.23),
             ("它", "k₃ = (.2, .2)", "v₃ = (.1, .1)", 0.23, 0.25)]


def fig_attention(p):
    W = QW
    b = [_defs(p)]
    b.append(text(30, 34, "注意力的三步", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(30, 58, "「它」这个位置去看前面每一个位置，决定各取多少",
                  fs(W, .88), p["sub"]))

    # 查询
    q = Rect(30, 82, 200, 62)
    b.append(_card(p, q, "q₃ = (1.4, 0.2)", "「它」在找什么", AMBER, None, True))

    head_y = 176
    cols = [(30, "位置"), (150, "键 k（我是什么）"), (330, "值 v（我给什么）"),
            (500, "分数 q·k/√2"), (620, "权重")]
    for x, t_ in cols:
        b.append(text(x, head_y, t_, fs(W, .8), p["sub"]))

    y = head_y + 18
    for i, (tok, k, v, sc, w) in enumerate(ATTN_ROWS):
        ry = y + i * 62
        b.append(box(30, ry, 690, 52, p["dim"] if i else p["box"], p["line"], rx=8))
        b.append(text(46, ry + 32, tok, fs(W), p["fg"], weight="700"))
        b.append(text(150, ry + 32, k, fs(W, .88), p["sub"], cls="m"))
        b.append(text(330, ry + 32, v, fs(W, .88), p["sub"], cls="m"))
        b.append(text(500, ry + 32, f"{sc:.2f}", fs(W, .88), p["fg"], cls="m"))
        # 权重条：长度就是权重
        b.append(f'<rect x="620" y="{ry + 18}" width="{80 * w:.1f}" height="18" rx="4" '
                 f'fill="{GREEN}" fill-opacity=".85"/>')
        b.append(text(706, ry + 32, f"{w:.2f}", fs(W, .8), p["fg"], anchor="end", cls="m"))
    last = y + 2 * 62 + 52

    b.append(text(30, last + 34, "① 打分：查询和每个键做内积，除以 √d 稳定数值",
                  fs(W, .88), p["sub"]))
    b.append(text(30, last + 58, "② softmax：把三个分数变成加起来等于 1 的权重",
                  fs(W, .88), p["sub"]))
    b.append(text(30, last + 82, "③ 加权求和：0.53·v₁ + 0.23·v₂ + 0.25·v₃",
                  fs(W, .88), p["sub"]))

    out = Rect(430, last + 30, 290, 66)
    b.append(_card(p, out, "= (1.08, 0.48)", "「它」的新表示，主要是「书」", GREEN))
    return svg(W, int(out.bottom + 34), "".join(b), "",
               "注意力三步：查询与每个键内积得到分数，softmax 变成和为一的权重，"
               "再按权重把各位置的值加权求和")


# ------------------------------------------------ prefill / decode 与 KV cache
def fig_kv_cache(p):
    W = QW
    b = [_defs(p)]
    CELL, GAPX = 52, 8

    def cells(x, y, n, filled, label_first=None):
        out = []
        for i in range(n):
            cx = x + i * (CELL + GAPX)
            new_one = i >= filled
            out.append(f'<rect x="{cx}" y="{y}" width="{CELL}" height="40" rx="6" '
                       f'fill="{p["box"] if not new_one else "none"}" '
                       f'stroke="{GREEN if new_one else p["line"]}" '
                       f'stroke-width="{2 if new_one else 1.2}" '
                       f'{"stroke-dasharray=\"5 4\"" if new_one else ""}/>')
            out.append(text(cx + CELL / 2, y + 26, f"K{i+1}", fs(W, .8),
                            GREEN if new_one else p["sub"], anchor="middle", cls="m"))
        return "".join(out)

    b.append(text(30, 34, "推理的两个阶段与 KV cache", fs(W, 1.06), p["fg"], weight="700"))

    # prefill
    b.append(text(30, 76, "prefill：把输入的 5 个 token 一次算完", fs(W), p["fg"], weight="700"))
    b.append(text(30, 100, "大矩阵乘法，吃算力", fs(W, .8), p["sub"]))
    b.append(cells(30, 116, 5, 0))
    b.append(text(340, 142, "K、V 存进缓存", fs(W, .88), p["sub"]))

    # decode
    b.append(text(30, 214, "decode：每步只算 1 个新位置", fs(W), p["fg"], weight="700"))
    b.append(text(30, 238, "矩阵乘向量，算得少、读得多，吃带宽", fs(W, .8), p["sub"]))
    b.append(cells(30, 254, 6, 5))
    b.append(text(400, 280, "读已有的 5 个，追加第 6 个", fs(W, .88), p["sub"]))

    b.append(text(30, 320, "绿色虚线 = 这一步新算出来的；灰色实线 = 缓存里已经有的",
                  fs(W, .8), p["sub"]))
    b.append(text(30, 348, "重算一遍要 O(T²)，缓存下来每步只要 O(T)——代价是显存里多一块随长度增长的缓存。",
                  fs(W, .88), p["sub"]))

    r = Rect(30, 368, 690, 96)
    b.append(box(r.x, r.y, r.w, r.h, p["dim"], p["line"], rx=10))
    b.append(text(r.x + 18, r.y + 30, "每 token 每层要存多少", fs(W), p["fg"], weight="700"))
    b.append(text(r.x + 18, r.y + 56, "2（K 和 V）× KV 头数 × 每头维度 × 每个数的字节数",
                  fs(W, .88), p["sub"], cls="m"))
    b.append(text(r.x + 18, r.y + 80,
                  "Qwen3.5-9B 全注意力层：2 × 4 × 256 × 2 = 4 KB；32 层里只有 8 层是全注意力，"
                  "所以每 token 32 KB", fs(W, .8), p["sub"]))
    return svg(W, int(r.bottom + 30), "".join(b), "",
               "prefill 一次算完整个输入并把 K、V 存进缓存；decode 每步只算一个新位置，"
               "读缓存再追加")


# ------------------------------------------------------ 多头：切开、各算各的、拼回去
def fig_multihead(p):
    W = QW
    b = [_defs(p)]
    b.append(text(30, 34, "多头：把向量切开，每段各算一遍", fs(W, 1.06), p["fg"], weight="700"))

    X0, BW2, H1 = 30, 690, 38
    NSEG = 16
    seg = BW2 / NSEG

    def bar(y, label, note):
        out = [text(X0, y - 10, label, fs(W, .88), p["sub"])]
        for i in range(NSEG):
            x = X0 + i * seg
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{seg - 2:.1f}" height="{H1}" rx="3" '
                       f'fill="{BLUE}" fill-opacity="{0.5 if i % 2 else 0.3}" '
                       f'stroke="{BLUE}" stroke-width="1"/>')
        out.append(text(X0 + BW2, y + H1 + 20, note, fs(W, .8), p["sub"], anchor="end"))
        return "".join(out), Rect(X0, y, BW2, H1)

    s1, r1 = bar(84, "查询 q（4096 维）——键 k、值 v 同样切", "16 段，每段 256 维")
    b.append(s1)

    # 三个代表性的头：各自一张 T×T 表
    def tri(x, y, n, side):
        cell = side / n
        out = []
        for i in range(n):
            for j in range(i + 1):
                out.append(f'<rect x="{x + j * cell:.1f}" y="{y + i * cell:.1f}" '
                           f'width="{cell - 1:.1f}" height="{cell - 1:.1f}" '
                           f'fill="{GREEN}" fill-opacity="{0.25 + 0.5 * ((i + j) % 3) / 2:.2f}"/>')
        out.append(f'<rect x="{x}" y="{y}" width="{side}" height="{side}" fill="none" '
                   f'stroke="{p["line"]}" stroke-width="1"/>')
        return "".join(out)

    hy, side = 196, 96
    spots = [(70, "头 1", 0), (300, "头 2", 1), (560, "头 16", 15)]
    for hx, name, si in spots:
        b.append(tri(hx, hy, 6, side))
        b.append(text(hx + side / 2, hy + side + 24, name, fs(W, .88), p["fg"],
                      anchor="middle", weight="700"))
        # 从对应的那一段引下来
        sx = X0 + si * seg + seg / 2
        b.append(f'<path d="M{sx:.1f} {r1.bottom}V{hy - 22}H{hx + side / 2}V{hy}" fill="none" '
                 f'stroke="{p["sub"]}" stroke-width="1.5" marker-end="url(#qa)"/>')
    b.append(text(455, hy + side / 2 + 6, "…", fs(W, 1.2), p["sub"], anchor="middle"))
    b.append(text(30, hy + side + 52,
                  "每段自己走一遍“打分 → softmax → 加权求和”，得到自己的一张 T×T 权重表",
                  fs(W, .88), p["sub"]))

    s2, r2 = bar(hy + side + 92, "16 段的输出首尾相接，又是 4096 维", "")
    b.append(s2)
    ob = Rect(X0 + 210, r2.bottom + 40, 270, 56)
    b.append(_card(p, ob, "o_proj", "4096 × 4096", GREEN))
    b.append(f'<path d="M{ob.cx} {r2.bottom}V{ob.top}" stroke="{p["sub"]}" '
             f'stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(text(ob.right + 16, ob.cy + 6, "混合 16 段的结果，再加回主干",
                  fs(W, .88), p["sub"]))
    return svg(W, int(ob.bottom + 34), "".join(b), "",
               "多头注意力：4096 维切成 16 段各 256 维，每段各算一张 T×T 权重表，"
               "输出拼回 4096 维后过 o_proj")


# ------------------------------------------ 多头：一层里张量形状怎么变
# T = 这次输入的 token 数；16 × 256 = 4096。形状写法和 PyTorch 一致。
MH_FLOW = [
    ("输入 h", "(T, 4096)", "每个位置一个 4096 维向量", BLUE),
    ("q  k  v", "(T, 4096) 各一个", "各乘 W_Q / W_K / W_V", BLUE),
    ("拆头（reshape）", "(16, T, 256)", "4096 拆成 16 × 256", AMBER),
    ("分数 = q @ kᵀ / √256", "(16, T, T)", "每个头一张 T×T 表", GREEN),
    ("加因果掩码，逐行 softmax", "(16, T, T)", "形状不变，每行和为 1", GREEN),
    ("权重 @ v", "(16, T, 256)", "每个头输出 256 维", GREEN),
    ("合头（reshape）", "(T, 4096)", "16 × 256 拼回 4096", AMBER),
    ("× W_O（o_proj）", "(T, 4096)", "4096 × 4096 的矩阵", BLUE),
]


def fig_multihead_shapes(p):
    W = QW
    b = [_defs(p)]
    b.append(text(30, 34, "一层注意力里，形状怎么变", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(30, 58, "T = 这次输入的 token 数；16 个头 × 每头 256 维 = 4096",
                  fs(W, .88), p["sub"]))

    y = 84
    H2, GAPY = 56, 30
    for i, (name, shape, note, col) in enumerate(MH_FLOW):
        r = Rect(30, y, 268, H2)
        b.append(box(r.x, r.y, r.w, r.h, p["box"], col, rx=8))
        b.append(text(r.x + 16, r.cy + 6, name, fs(W, .88), p["fg"], weight="600", cls="m"))
        b.append(f'<rect x="{r.right + 18}" y="{r.y + 8}" width="170" height="{H2 - 16}" '
                 f'rx="6" fill="{col}" fill-opacity=".12" stroke="{col}" stroke-width="1.2"/>')
        b.append(text(r.right + 103, r.cy + 6, shape, fs(W, .88), p["fg"],
                      anchor="middle", weight="700", cls="m"))
        b.append(text(r.right + 204, r.cy + 6, note, fs(W, .8), p["sub"]))
        if i < len(MH_FLOW) - 1:
            b.append(f'<path d="M{r.cx} {r.bottom}V{r.bottom + GAPY}" stroke="{p["sub"]}" '
                     f'stroke-width="1.6" marker-end="url(#qa)"/>')
        y = r.bottom + GAPY

    b.append(text(30, y + 24, "进来 (T, 4096)，出去还是 (T, 4096)——注意力不改变形状，只改变内容。",
                  fs(W, .88), p["fg"], weight="700"))
    b.append(text(30, y + 50, "橙色那两步只是换个看法（reshape），一次乘法都没有；真正的计算在绿色三步和两端的矩阵乘法里。",
                  fs(W, .8), p["sub"]))
    b.append(text(30, y + 74, "这个模型用 GQA：k、v 只有 4 组，形状是 (4, T, 256)，算的时候一组给 4 个查询头共用。",
                  fs(W, .8), p["sub"]))
    b.append(text(30, y + 100, "@ 是 Python 的矩阵乘法运算符（a @ b 即矩阵 a 乘矩阵 b）；kᵀ 是 k 的转置，行列互换。",
                  fs(W, .8), p["sub"]))
    return svg(W, int(y + 124), "".join(b), "",
               "一层注意力里张量形状的变化：(T,4096) 经拆头成 (16,T,256)，"
               "算出 (16,T,T) 的权重，输出 (16,T,256)，合头回 (T,4096)")


# ------------------------------------------------ 静态批处理 vs 连续批处理
def fig_batching(p):
    W = QW
    CW, CH, GAPY = 26, 26, 10
    X0 = 108
    b = [_defs(p)]
    b.append(text(30, 34, "为什么连续批处理更快", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(30, 58, "横轴是时间步，一格 = 这一步算了这条请求的一个 token",
                  fs(W, .8), p["sub"]))

    def timeline(y, title, rows, note):
        out = [text(30, y - 10, title, fs(W), p["fg"], weight="700")]
        for i, (name, cells) in enumerate(rows):
            ry = y + i * (CH + GAPY)
            out.append(text(X0 - 12, ry + CH / 2 + 5, name, fs(W, .8), p["sub"],
                            anchor="end", cls="m"))
            for j, kind in enumerate(cells):
                cx = X0 + j * (CW + 3)
                if kind == ".":
                    continue
                fill = {"r": GREEN, "w": p["sub"], "n": BLUE}[kind]
                op = {"r": .8, "w": .18, "n": .8}[kind]
                out.append(f'<rect x="{cx}" y="{ry}" width="{CW}" height="{CH}" '
                           f'fill="{fill}" fill-opacity="{op}"/>')
        ny = y + len(rows) * (CH + GAPY) + 6
        out.append(text(X0, ny, note, fs(W, .8), p["sub"]))
        return "".join(out), ny

    # r = 在算，w = 空转（等同批其他请求），n = 新请求补进来
    static_rows = [("请求 1", list("rrrrrrrrrr")),
                   ("请求 2", list("rrrwwwwwww")),
                   ("请求 3", list("rrrrrrwwww")),
                   ("请求 4", list(".........."))]
    s1, y1 = timeline(96, "静态批处理：一批一起开始、一起结束",
                      static_rows,
                      "灰格是空转：请求 2 早就答完了，但要等最长的那条结束才能收批。请求 4 只能排队等下一批。")
    b.append(s1)

    cont_rows = [("请求 1", list("rrrrrrrrrr")),
                 ("请求 2", list("rrr.......")),
                 ("请求 3", list("rrrrrr....")),
                 ("请求 4", list("...nnnnnnn"))]
    s2, y2 = timeline(y1 + 54, "连续批处理：每一步重新决定这轮算谁",
                      cont_rows,
                      "请求 2 一答完就离场，空出来的位置立刻让请求 4 补上，GPU 不空转。")
    b.append(s2)

    b.append(text(30, y2 + 34, "调度是逐步做的，不是逐批做的——这是 vLLM 吞吐高的主要原因，day 04 会把这条曲线量出来。",
                  fs(W, .88), p["fg"], weight="700"))
    return svg(W, int(y2 + 58), "".join(b), "",
               "静态批处理里短请求算完要空等整批结束；连续批处理每步重新调度，"
               "请求答完即离场，新请求立刻补位")


# ------------------------------------------------ KV cache：连续分配 vs 分页
def fig_paged(p):
    W = QW
    b = [_defs(p)]
    b.append(text(30, 34, "KV cache 为什么要分页", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(30, 58, "一条请求最终要多长的缓存，发的时候并不知道",
                  fs(W, .8), p["sub"]))

    # 左：按最大长度预留
    LX, LW, RH = 30, 330, 30
    b.append(text(LX, 96, "按最大长度预留", fs(W), p["fg"], weight="700"))
    for i, used in enumerate([0.25, 0.45, 0.15]):
        y = 112 + i * (RH + 12)
        b.append(f'<rect x="{LX}" y="{y}" width="{LW}" height="{RH}" fill="{p["sub"]}" '
                 f'fill-opacity=".15"/>')
        b.append(f'<rect x="{LX}" y="{y}" width="{LW * used:.0f}" height="{RH}" '
                 f'fill="{BLUE}" fill-opacity=".8"/>')
        b.append(text(LX + LW + 10, y + RH / 2 + 5, f"请求 {i+1}", fs(W, .8), p["sub"], cls="m"))
    b.append(text(LX, 112 + 3 * (RH + 12) + 12, "蓝色是真的用上的，灰色是白占的",
                  fs(W, .8), p["sub"]))

    # 右：分页
    RX = 470
    b.append(text(RX, 96, "分页", fs(W), p["fg"], weight="700"))
    BS, BG = 34, 6
    for i, n in enumerate([3, 5, 2]):
        y = 112 + i * (RH + 12)
        for j in range(n):
            b.append(f'<rect x="{RX + j * (BS + BG)}" y="{y}" width="{BS}" height="{RH}" '
                     f'fill="{GREEN}" fill-opacity=".75"/>')
        b.append(text(RX + 5 * (BS + BG) + 8, y + RH / 2 + 5, f"{n} 块", fs(W, .8),
                      p["sub"], cls="m"))
    b.append(text(RX, 112 + 3 * (RH + 12) + 12, "按需要一块块拿，用完就还",
                  fs(W, .8), p["sub"]))

    y = 112 + 3 * (RH + 12) + 44
    r = Rect(30, y, W - 60, 96)
    b.append(box(r.x, r.y, r.w, r.h, p["dim"], p["line"], rx=10))
    b.append(text(r.x + 16, r.y + 28, "块不必挨着放", fs(W), p["fg"], weight="700"))
    b.append(text(r.x + 16, r.y + 52,
                  "每条请求带一张表，记着自己的块按什么顺序拼起来——和操作系统的虚拟内存分页是同一个套路。",
                  fs(W, .8), p["sub"]))
    b.append(text(r.x + 16, r.y + 76,
                  "好处：几乎没有白占的空间；两条请求前缀相同时，还能共用同一批块。",
                  fs(W, .8), p["sub"]))
    return svg(W, int(r.bottom + 30), "".join(b), "",
               "按最大长度预留会白占大量缓存；分页按需分配固定大小的块，"
               "块之间不必连续，相同前缀还能共享")


# ------------------------------------------------ 请求路径：箭头上标出传的东西
def fig_request_path(p):
    """左边 API 进程，右边 EngineCore 进程，每段箭头标出这一步交出去的东西。

    间距规则：盒子之间留 64 px，标签写在这段空白的正中，上下各剩约 20 px；
    横向的标签写在两列之间 128 px 的空当里。不要把标签贴着箭头放。
    """
    W, H = 880, 780
    PAD = 22                                   # 虚线框到卡片的内边距
    BW, BH, VGAP = 224, 70, 56                 # 卡片宽高与竖直间距
    LX, RX = 172, 520                           # 两列卡片的左边
    b = [_defs(p)]
    b.append(text(24, 32, "一条请求走过的对象", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(24, 56, "箭头旁写的是这一步交给下一个模块的东西", fs(W, .8), p["sub"]))

    top = 124
    api = Rect(LX, top, BW, BH)
    inp = Rect(LX, top + (BH + VGAP), BW, BH)
    cli = Rect(LX, top + 2 * (BH + VGAP), BW, BH)
    outp = Rect(LX, top + 3 * (BH + VGAP) + 18, BW, BH)
    sse = Rect(LX, top + 4 * (BH + VGAP) + 18, BW, BH)
    sch = Rect(RX, top + (BH + VGAP), BW, BH)
    gpu = Rect(RX, top + 2 * (BH + VGAP), BW, 96)
    upd = Rect(RX, top + 3 * (BH + VGAP) + 18, BW, BH)
    user = Rect(26, cli.y - 6, 116, 64)

    for x0, label in ((LX - PAD, "API 进程"), (RX - PAD, "EngineCore 进程")):
        b.append(f'<rect x="{x0}" y="{top - 44}" width="{BW + 2 * PAD}" '
                 f'height="{sse.bottom + PAD - (top - 44)}" rx="10" fill="none" '
                 f'stroke="{p["line"]}" stroke-dasharray="6 5"/>')
        b.append(text(x0 + 10, top - 24, label, fs(W, .8), p["sub"], cls="s"))

    b.append(_card(p, user, "用户", "prompt", mono=False))
    b.append(_card(p, api, "/v1/chat/completions", "FastAPI"))
    b.append(_card(p, inp, "InputProcessor", "分词 + chat template"))
    b.append(_card(p, cli, "EngineCoreClient", "序列化后经 ZMQ 发出"))
    b.append(_card(p, sch, "Scheduler", "schedule()", stroke=GREEN))
    b.append(_card(p, outp, "OutputProcessor", "token → 文本"))
    b.append(_card(p, sse, "SSE", "逐块推给客户端"))
    b.append(_card(p, upd, "Scheduler", "update_from_output()"))

    b.append(box(gpu.x, gpu.y, gpu.w, gpu.h, p["dim"], BLUE, rx=8))
    b.append(text(gpu.cx, gpu.y + 32, "Worker · GPU", fs(W), p["fg"], anchor="middle",
                  weight="600", cls="m"))
    b.append(text(gpu.cx, gpu.y + 56, "CUDA graph 重放一次前向", fs(W, .8), p["sub"],
                  anchor="middle", cls="s"))
    b.append(text(gpu.cx, gpu.y + 78, "再采样出下一个 token", fs(W, .8), p["sub"],
                  anchor="middle", cls="s"))

    def arrow(x1, y1, x2, y2, color):
        return (f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color}" stroke-width="1.7" '
                f'fill="none" marker-end="url(#qa)"/>')

    def vgap_label(a_, c_, s_, color, dx=16):
        """竖直间隙正中写标签，和箭头错开 dx，再加一圈背景色描边。"""
        return text(a_.cx + dx, (a_.bottom + c_.top) / 2 + 5, s_, fs(W, .8), color,
                    cls="s", halo=p["box"])

    # 用户 → 接口
    b.append(arrow(user.x + user.w, user.y - 10, api.x - 10, api.cy + 10, p["sub"]))
    b.append(text(user.cx, user.y - 26, "文本 + 采样参数", fs(W, .8), p["sub"],
                  anchor="middle", cls="s", halo=p["box"]))
    # API 进程内部
    b.append(arrow(api.cx, api.bottom, api.cx, inp.top - 6, p["sub"]))
    b.append(arrow(inp.cx, inp.bottom, inp.cx, cli.top - 6, p["sub"]))
    b.append(vgap_label(inp, cli, "token ids", p["sub"]))
    # 跨进程：两列之间有 112 px 空当，标签写在正中
    mid_x = (cli.x + cli.w + sch.x) / 2
    b.append(arrow(cli.x + cli.w + 6, cli.cy, sch.x - 10, sch.cy + 8, GREEN))
    b.append(text(mid_x, cli.top - 14, "Request", fs(W, .84), GREEN, anchor="middle",
                  weight="600", cls="m", halo=p["box"]))
    # 引擎内部
    b.append(arrow(sch.cx, sch.bottom, sch.cx, gpu.top - 6, GREEN))
    b.append(vgap_label(sch, gpu, "SchedulerOutput", GREEN))
    b.append(arrow(gpu.cx, gpu.bottom, gpu.cx, upd.top - 6, BLUE))
    b.append(vgap_label(gpu, upd, "新 token 的 id", BLUE))
    # 回到 API 进程
    b.append(arrow(upd.x - 6, upd.cy, outp.x + outp.w + 10, outp.cy, BLUE))
    b.append(text(mid_x, upd.top - 14, "EngineCoreOutputs", fs(W, .82), BLUE,
                  anchor="middle", weight="600", cls="m", halo=p["box"]))
    b.append(arrow(outp.cx, outp.bottom, outp.cx, sse.top - 6, p["sub"]))
    # 回到用户
    b.append(arrow(sse.x - 6, sse.cy, user.cx, sse.cy, p["sub"]))
    b.append(arrow(user.cx, sse.cy - 8, user.cx, user.bottom + 10, p["sub"]))
    b.append(text(user.cx + 10, sse.cy - 14, "一小段文本", fs(W, .8), p["sub"], cls="s",
                  halo=p["box"]))
    # 主循环
    loop_x = gpu.x + gpu.w + PAD + 26
    b.append(f'<path d="M{upd.x + upd.w + 6} {upd.cy}H{loop_x}V{sch.cy}H{sch.x + sch.w + 8}" '
             f'stroke="{AMBER}" stroke-width="1.7" fill="none" marker-end="url(#qa)"/>')
    b.append(text(loop_x + 10, (sch.cy + upd.cy) / 2 - 8, "没答完", fs(W, .8), AMBER,
                  cls="s", halo=p["box"]))
    b.append(text(loop_x + 10, (sch.cy + upd.cy) / 2 + 12, "再走一轮", fs(W, .8), AMBER,
                  cls="s", halo=p["box"]))
    return svg(W, H, "".join(b), label=(
        "请求路径：文本进 API 进程分词，作为 Request 经 ZMQ 交给引擎；调度器给出这一步"
        "算谁、各几个 token、KV 块在哪；GPU 前向并采样；新 token 回到调度器继续，"
        "同时变回文本推给客户端"))


# ------------------------------------------------ 一次 step 里发生什么
def fig_engine_step(p):
    """一次 step 的三段，以及 §4 实测的两个数。"""
    W, H = 750, 300
    b = [_defs(p)]
    b.append(text(24, 30, "一次 step 的三段", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(24, 52, "调度和回写在 CPU 上，中间那段在 GPU 上", fs(W, .8), p["sub"]))

    LANE_Y = {"CPU": 96, "GPU": 172}
    for lane, y in LANE_Y.items():
        b.append(text(24, y + 30, lane, fs(W, .88), p["sub"], weight="600", cls="m"))
        b.append(f'<line x1="70" y1="{y + 25}" x2="{W - 30}" y2="{y + 25}" '
                 f'stroke="{p["line"]}" stroke-dasharray="4 4"/>')

    segs = [("schedule()", 84, 190, "CPU", GREEN, "挑出这一步算哪些请求，各分几个 token"),
            ("execute_model()", 292, 250, "GPU", BLUE, "一次前向，采样出新 token"),
            ("update_from_output()", 512, 208, "CPU", GREEN, "写回各请求，判断谁结束")]
    for name, x, w, lane, color, note in segs:
        y = LANE_Y[lane]
        b.append(box(x, y, w, 50, p["box"], color, rx=7))
        b.append(text(x + w / 2, y + 30, name, fs(W, .92), p["fg"], anchor="middle",
                      weight="600", cls="m"))
        if lane == "GPU":
            b.append(f'<path d="M{x - 6} {LANE_Y["CPU"] + 50}L{x + 20} {y - 4}" '
                     f'stroke="{p["sub"]}" stroke-width="1.5" marker-end="url(#qa)"/>')
            b.append(f'<path d="M{x + w - 20} {y}L{x + w + 6} {LANE_Y["CPU"] + 50}" '
                     f'stroke="{p["sub"]}" stroke-width="1.5" marker-end="url(#qa)"/>'
                     .replace(f'M{x + w - 20} {y}', f'M{x + w - 20} {y - 4}'))

    b.append(text(24, 254, "实测（Qwen3.5-0.8B，见 §4）：提示 20 个 token 的那一步用了 28.2 ms；"
                           "之后每步只出一个 token，间隔 10.9 ms",
                  fs(W, .82), p["sub"], cls="s"))
    b.append(text(24, 276, "所以一条生成 64 个 token 的请求，要把这三段重复 65 次",
                  fs(W, .82), p["sub"], cls="s"))
    return svg(W, H, "".join(b), label=(
        "一次 step 分三段：调度在 CPU 上挑出这一步算谁，执行在 GPU 上做一次前向并采样，"
        "回写在 CPU 上更新各请求状态"))


# ------------------------------------------------ 请求的状态机
def fig_request_states(p):
    W, H = 830, 268
    b = [_defs(p)]
    b.append(text(24, 30, "一条请求的状态", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(24, 52, "PREEMPTED 之后的状态都算已结束，判断就是一个大小比较",
                  fs(W, .8), p["sub"]))

    wait = Rect(30, 96, 150, 52)
    run = Rect(240, 96, 150, 52)
    pre = Rect(240, 184, 150, 52)
    fin = Rect(430, 62, 248, 50)
    cap = Rect(430, 124, 248, 50)
    ab = Rect(430, 186, 248, 50)

    b.append(_card(p, wait, "WAITING"))
    b.append(_card(p, run, "RUNNING", stroke=GREEN))
    b.append(_card(p, pre, "PREEMPTED", stroke=AMBER))
    b.append(_card(p, fin, "FINISHED_STOPPED"))
    b.append(_card(p, cap, "FINISHED_LENGTH_CAPPED"))
    b.append(_card(p, ab, "FINISHED_ABORTED"))

    def a(x1, y1, x2, y2, color=None):
        return (f'<path d="M{x1} {y1}L{x2} {y2}" stroke="{color or p["sub"]}" '
                f'stroke-width="1.6" fill="none" marker-end="url(#qa)"/>')

    b.append(a(wait.x + wait.w, wait.cy, run.x - 8, run.cy, GREEN))
    b.append(text((wait.x + wait.w + run.x) / 2, wait.cy - 16, "被调度选中", fs(W, .78),
                  GREEN, anchor="middle", cls="s", halo=p["box"]))
    b.append(a(run.cx, run.bottom, run.cx, pre.top - 8, AMBER))
    b.append(text(run.cx + 12, (run.bottom + pre.top) / 2 + 4, "KV 块不够，被抢占",
                  fs(W, .78), AMBER, cls="s", halo=p["box"]))
    b.append(f'<path d="M{pre.x} {pre.cy}H{wait.cx}V{wait.bottom + 6}" stroke="{AMBER}" '
             f'stroke-width="1.6" fill="none" marker-end="url(#qa)"/>')
    b.append(text(wait.cx + 8, pre.cy - 10, "块已释放，重新排队", fs(W, .78), AMBER, cls="s",
                  halo=p["box"]))
    b.append(a(run.x + run.w, run.cy - 14, fin.x - 8, fin.cy))
    b.append(a(run.x + run.w, run.cy, cap.x - 8, cap.cy))
    b.append(a(run.x + run.w, run.cy + 14, ab.x - 8, ab.cy))
    b.append(text(fin.x + fin.w + 10, fin.cy + 5, "遇到结束符", fs(W, .78), p["sub"], cls="s"))
    b.append(text(cap.x + cap.w + 10, cap.cy + 5, "到长度上限", fs(W, .78), p["sub"], cls="s"))
    b.append(text(ab.x + ab.w + 10, ab.cy + 5, "客户端断开", fs(W, .78), p["sub"], cls="s"))
    return svg(W, H, "".join(b), label=(
        "请求状态机：WAITING 被调度后进入 RUNNING；KV 块不够时被抢占回到队列；"
        "结束分为遇到结束符、到长度上限、客户端断开三种"))


# ------------------------------------------------ 12 条请求撞上 8 个位置（实测）
def fig_queue_measured(p):
    """把 code/watch_sched.py 采到的数据画出来，数字全部来自 §4。"""
    W, H = 750, 430
    X0, XW, T_MAX = 86, 592, 5.8
    b = [_defs(p)]
    b.append(text(24, 30, "12 条请求撞上 8 个位置", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(24, 52, "并发上限 8，同时发 12 条，每条最多 96 个 token",
                  fs(W, .8), p["sub"]))

    def tx(t):
        return X0 + t / T_MAX * XW

    # 上：调度器里的两条队列，堆叠画，不重叠
    TOP, LH, UNIT = 88, 84, 7.0                 # UNIT：一条请求的高度
    b.append(text(24, TOP + 12, "队列", fs(W, .82), p["sub"], weight="600", cls="s"))
    b.append(f'<line x1="{X0}" y1="{TOP + LH}" x2="{X0 + XW}" y2="{TOP + LH}" '
             f'stroke="{p["line"]}"/>')
    # (起, 止, 在跑, 在等)，来自 results/watch_sched.txt
    spans = [(3.03, 4.29, 8, 4), (4.29, 4.40, 1, 3), (4.40, 5.44, 4, 0)]
    for t0, t1, run, wait in spans:
        x, w = tx(t0), tx(t1) - tx(t0)
        hr = run * UNIT
        b.append(f'<rect x="{x:.1f}" y="{TOP + LH - hr:.1f}" width="{w:.1f}" '
                 f'height="{hr:.1f}" fill="{GREEN}" fill-opacity=".7"/>')
        if wait:
            hw = wait * UNIT
            b.append(f'<rect x="{x:.1f}" y="{TOP + LH - hr - hw:.1f}" width="{w:.1f}" '
                     f'height="{hw:.1f}" fill="{AMBER}" fill-opacity=".45"/>')
    # 标注写在柱子外面：写进柱子里离边框太近，看着像贴着
    b.append(text(tx(3.1), TOP + LH - 12 * UNIT - 10, "8 条在跑 + 4 条在等",
                  fs(W, .78), p["sub"], cls="s"))
    b.append(text(tx(4.45), TOP + LH - 4 * UNIT - 10, "空出位置，等的 4 条补进来",
                  fs(W, .78), p["sub"], cls="s"))

    # 下：12 条各自的端到端时间
    BY, BH2, GAP = 200, 9, 4
    b.append(text(24, BY + 12, "各条", fs(W, .82), p["sub"], weight="600", cls="s"))
    for i in range(12):
        first = i < 8
        y = BY + i * (BH2 + GAP)
        b.append(f'<rect x="{X0}" y="{y}" width="{tx(4.26) - X0:.1f}" height="{BH2}" '
                 f'rx="2" fill="{GREEN}" fill-opacity=".75"/>')
        if not first:
            b.append(f'<rect x="{tx(4.26):.1f}" y="{y}" width="{tx(5.44) - tx(4.26):.1f}" '
                     f'height="{BH2}" rx="2" fill="{AMBER}" fill-opacity=".85"/>')
        b.append(text(X0 - 8, y + BH2, f"R{i + 1}", fs(W, .7), p["sub"], anchor="end", cls="m"))

    # 时间轴
    AX = BY + 12 * (BH2 + GAP) + 6
    b.append(f'<line x1="{X0}" y1="{AX}" x2="{X0 + XW}" y2="{AX}" stroke="{p["line"]}"/>')
    for t in range(6):
        b.append(f'<line x1="{tx(t):.1f}" y1="{AX}" x2="{tx(t):.1f}" y2="{AX + 4}" '
                 f'stroke="{p["line"]}"/>')
        b.append(text(tx(t), AX + 18, str(t), fs(W, .74), p["sub"], anchor="middle", cls="m"))
    b.append(text(X0 + XW, AX + 18, "秒", fs(W, .74), p["sub"], anchor="end", cls="s"))

    # 图例
    LY = AX + 42
    b.append(f'<rect x="{X0}" y="{LY - 9}" width="14" height="10" rx="2" '
             f'fill="{GREEN}" fill-opacity=".75"/>')
    b.append(text(X0 + 22, LY, "在批次里算：先进入的 8 条 4.26 s", fs(W, .78), p["sub"], cls="s"))
    b.append(f'<rect x="{X0 + 300}" y="{LY - 9}" width="14" height="10" rx="2" '
             f'fill="{AMBER}" fill-opacity=".85"/>')
    b.append(text(X0 + 322, LY, "在队列里等：排队的 4 条多花 1.18 s", fs(W, .78), p["sub"], cls="s"))
    return svg(W, H, "".join(b), label=(
        "实测：并发上限 8 时同时发 12 条，8 条进入批次、4 条等待；先进入的 4.26 秒完成，"
        "排队的 5.44 秒，差的 1.18 秒是等待时间"))




# ------------------------------------------------ KV 块不够时的抢占
def fig_preempt(p):
    """三格连环画。触发抢占的是「正在跑的请求要不到新块」，不是新请求插队。

    等待队列里的请求申请不到块时，调度器只是这一轮不调度它，不会去抢
    正在跑的。会触发抢占的只有 running 队列那条循环。
    """
    W, H = 820, 560
    CW, CH, GAP = 22, 20, 4
    b = [_defs(p)]
    b.append(text(24, 30, "谁会被抢占", fs(W, 1.06), p["fg"], weight="700"))
    b.append(text(24, 52, "一个小方块 = 一段 KV 块，缓存总共 8 块。触发抢占的是正在跑的请求要不到新块",
                  fs(W, .8), p["sub"]))

    def row(x, y, name, used, color, want=0, ghost=0, note=""):
        out = [text(x - 12, y + CH - 4, name, fs(W, .82), p["sub"], anchor="end", cls="m")]
        n = used + want + ghost
        for i in range(n):
            cx = x + i * (CW + GAP)
            if i < used:
                out.append(f'<rect x="{cx}" y="{y}" width="{CW}" height="{CH}" rx="3" '
                           f'fill="{color}" fill-opacity=".75"/>')
            elif i < used + want:      # 这一步想要但还没拿到的块
                out.append(f'<rect x="{cx}" y="{y}" width="{CW}" height="{CH}" rx="3" '
                           f'fill="none" stroke="{AMBER}" stroke-width="1.6"/>')
                out.append(text(cx + CW / 2, y + CH - 5, "?", fs(W, .8), AMBER,
                                anchor="middle", cls="m"))
            else:                      # 已释放
                out.append(f'<rect x="{cx}" y="{y}" width="{CW}" height="{CH}" rx="3" '
                           f'fill="none" stroke="{p["line"]}" stroke-dasharray="3 3"/>')
        if note:
            out.append(text(x + n * (CW + GAP) + 12, y + CH - 4, note,
                            fs(W, .78), p["sub"], cls="s"))
        return "".join(out)

    PX, PY, PH = 116, 92, 148
    panels = [
        ("① 八块分完，R1 还要一块", [
            ("R1", 3, GREEN, 1, 0, "生成到了块边界，这一步要第 4 块"),
            ("R2", 3, GREEN, 0, 0, ""),
            ("R3", 2, GREEN, 0, 0, "最后进入 running 的一条"),
        ], "allocate_slots 返回 None：8 块已经分完，没有空闲块给 R1"),
        ("② 从 running 队尾抢占", [
            ("R1", 3, GREEN, 1, 0, ""),
            ("R2", 3, GREEN, 0, 0, ""),
            ("R3", 0, GREEN, 0, 2, "块被释放，num_computed_tokens 归零"),
        ], "抢的是队尾，也就是最晚进入 running 的 R3；释放后再试一次分配"),
        ("③ R1 拿到块，R3 回队首等", [
            ("R1", 4, GREEN, 0, 0, "继续生成"),
            ("R2", 3, GREEN, 0, 0, ""),
            ("R3", 0, GREEN, 0, 0, "在等待队列的队首，下一轮从第 0 个 token 重算"),
        ], "R3 之前算的那两块作废，这就是抢占的代价"),
    ]
    for k, (title, rows, note) in enumerate(panels):
        y0 = PY + k * PH
        b.append(text(24, y0 + 14, title, fs(W, .9), p["fg"], weight="600", cls="s"))
        for i, (name, used, color, want, ghost, rnote) in enumerate(rows):
            b.append(row(PX, y0 + 34 + i * (CH + 8), name, used, color, want, ghost, rnote))
        b.append(text(PX - 12, y0 + 34 + 3 * (CH + 8) + 16, note, fs(W, .78), p["sub"], cls="s"))

    b.append(text(24, H - 18,
                  "等待队列里的新请求申请不到块时不会触发抢占，调度器只是这一轮跳过它。",
                  fs(W, .78), p["sub"], cls="s"))
    return svg(W, H, "".join(b), label=(
        "抢占示意：正在跑的请求需要新块而缓存已满时，调度器从 running 队尾抢占一条，"
        "释放它的块并把已算进度清零，它回到等待队列队首，下一轮从头重算"))


def check_bounds(name: str, svg_text: str) -> list[str]:
    """粗估每段文字的宽度，报出超出画布的。

    估算：中日韩字符按 1.0 em，其余按 0.55 em。图在不同环境里用的字体不同
    （站点有 CDN 字体，GitHub 上没有），所以留 8 px 余量再判。
    """
    import re as _re
    m = _re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg_text)
    if not m:
        return []
    W = float(m.group(1))
    bad = []
    for t in _re.finditer(r'<text x="([-\d.]+)"[^>]*?font-size="([\d.]+)"[^>]*?'
                          r'text-anchor="(\w+)"[^>]*?>(.*?)</text>', svg_text):
        x, size, anchor, body = float(t.group(1)), float(t.group(2)), t.group(3), t.group(4)
        body = _re.sub(r"<[^>]+>", "", body)
        w = sum(size * (1.0 if ch > "\u2e80" else 0.55) for ch in body)
        right = x if anchor == "end" else x + w / 2 if anchor == "middle" else x + w
        left = x - w if anchor == "end" else x - w / 2 if anchor == "middle" else x
        if right > W - 8:
            bad.append(f"    右边超出 {right - W:+.0f} px：{body[:24]}")
        elif left < 8:
            bad.append(f"    左边超出 {left:.0f} px：{body[:24]}")
    return bad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"fig-lora-arch": fig_lora_arch, "fig-train-step": fig_train_step,
                     "fig-float-bits": fig_float_bits, "fig-svd": fig_svd,
                     "fig-qwen-arch": fig_qwen_arch,
                     "fig-qwen-block": fig_qwen_block,
                     "fig-attention": fig_attention,
                     "fig-kv-cache": fig_kv_cache,
                     "fig-multihead": fig_multihead,
                     "fig-multihead-shapes": fig_multihead_shapes,
                     "fig-batching": fig_batching,
                     "fig-paged": fig_paged,
                     "fig-request-path": fig_request_path,
                     "fig-engine-step": fig_engine_step,
                     "fig-request-states": fig_request_states,
                     "fig-queue-measured": fig_queue_measured,
                     "fig-preempt": fig_preempt}.items():
        for suffix, pal in (("light", LIGHT), ("dark", DARK)):
            out = fn(pal)
            (OUT / f"{name}-{suffix}.svg").write_text(out)
            if suffix == "light":
                warn = check_bounds(name, out)
        print(f"  {name}  {(OUT / f'{name}-light.svg').stat().st_size} 字节"
              + ("  ← 文字出界" if warn else ""))
        for line in warn:
            print(line)


if __name__ == "__main__":
    main()
