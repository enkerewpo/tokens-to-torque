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


# ------------------------------------------- Qwen3.5-9B 的结构与 LoRA 的挂点
# 每个模块名和形状都来自模型自己的 config 与 meta 设备上的模块清单：
#   python days/day00_lora-quickstart/code/peek_model.py --model Qwen/Qwen3.5-9B
# 绿色标出 nn.Linear —— target_modules="all-linear" 挂的就是这些。


def fig_qwen_arch(p):
    W, H = 1180, 792
    b = [f'<defs>'
         f'<marker id="qa" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" '
         f'refX="8" refY="5" orient="auto"><path d="M0 1 9 5 0 9z" fill="{p["sub"]}"/></marker>'
         f'</defs>']

    def arrow(x, y1, y2):
        return (f'<path d="M{x} {y1}V{y2 - 9}" stroke="{p["sub"]}" stroke-width="1.6" '
                f'marker-end="url(#qa)"/>')

    def chip(x, y, w, h, label, sub_="", fill=None, stroke=None, mono=True):
        out = [box(x, y, w, h, fill or p["box"], stroke or p["line"], rx=7)]
        if sub_:
            out.append(text(x + w / 2, y + 19, label, 13, p["fg"], anchor="middle",
                            weight="600", cls="m" if mono else "s"))
            out.append(text(x + w / 2, y + h - 9, sub_, 11, p["sub"], anchor="middle", cls="m"))
        else:
            out.append(text(x + w / 2, y + h / 2 + 4.5, label, 13, p["fg"], anchor="middle",
                            weight="600", cls="m" if mono else "s"))
        return "".join(out)

    # ---------------- 左栏：主干 ----------------
    LX, LW = 40, 216
    cx = LX + LW / 2
    b.append(text(LX, 34, "整体通路", 15, p["fg"], weight="700"))
    b.append(chip(LX, 48, LW, 34, "token ids", "", p["dim"]))
    b.append(arrow(cx, 82, 104))
    b.append(chip(LX, 100, LW, 44, "embed_tokens", "248320 × 4096", p["dim"]))
    b.append(arrow(cx, 144, 170))

    # 32 层：4 层一组，画一组并注明重复 8 次
    b.append(f'<rect x="{LX - 8}" y="{170}" width="{LW + 16}" height="212" rx="10" '
             f'fill="none" stroke="{p["line"]}" stroke-width="1.4" stroke-dasharray="5 4"/>')
    for i in range(3):
        y = 182 + i * 44
        b.append(chip(LX, y, LW, 34, "线性注意力块", "", p["box"], BLUE, mono=False))
        b.append(arrow(cx, y + 34, y + 44))
    b.append(chip(LX, 314, LW, 34, "全注意力块", "", p["box"], AMBER, mono=False))
    b.append(text(cx, 366, "这 4 层一组，重复 8 次 = 32 层", 12, p["sub"], anchor="middle"))
    b.append(arrow(cx, 382, 404))
    b.append(chip(LX, 400, LW, 44, "model.norm", "RMSNorm", p["dim"]))
    b.append(arrow(cx, 444, 464))
    b.append(chip(LX, 464, LW, 44, "lm_head", "4096 → 248320", p["dim"]))
    b.append(text(cx, 520, "LoRA 唯一跳过的线性层", 12, p["sub"], anchor="middle"))
    b.append(arrow(cx, 528, 548))
    b.append(chip(LX, 548, LW, 34, "下一个 token 的分布", "", p["dim"], mono=False))

    # ---------------- 中栏 / 右栏：两种块的内部 ----------------
    def blockpanel(px, title, count, accent, mixer_title, mixer_rows):
        pw, py, ph = 380, 48, 534
        cxx = px + pw / 2
        out = [box(px, py, pw, ph, p["dim"], accent, rx=12)]
        out.append(text(px + 16, py + 26, title, 15, p["fg"], weight="700"))
        out.append(text(px + pw - 16, py + 26, count, 13, accent, anchor="end", weight="700"))

        y = py + 44
        out.append(text(cxx, y + 12, "输入 h（4096）", 11, p["sub"], anchor="middle", cls="m"))
        y += 22
        # 残差起点
        rail = px + 26
        out.append(f'<circle cx="{cxx}" cy="{y}" r="3" fill="{p["sub"]}"/>')
        out.append(f'<path d="M{cxx} {y}H{rail}V{y + 168}H{cxx - 16}" fill="none" '
                   f'stroke="{p["sub"]}" stroke-width="1.4" stroke-dasharray="4 4" '
                   f'marker-end="url(#qa)"/>')
        out.append(arrow(cxx, y + 4, y + 26))
        out.append(chip(cxx - 100, y + 26, 200, 30, "input_layernorm", "", p["box"]))
        out.append(arrow(cxx, y + 56, y + 78))

        # mixer
        mh = 20 + 19 * len(mixer_rows)
        out.append(box(cxx - 158, y + 78, 316, mh, p["box"], accent, rx=8))
        out.append(text(cxx, y + 96, mixer_title, 13, p["fg"], anchor="middle", weight="700"))
        for i, (name, dims, is_lin) in enumerate(mixer_rows):
            ry = y + 115 + i * 19
            out.append(text(cxx - 146, ry, name, 11.5, GREEN if is_lin else p["sub"], cls="m"))
            out.append(text(cxx + 146, ry, dims, 11, p["sub"], anchor="end", cls="m"))
        ay = y + 78 + mh
        out.append(arrow(cxx, ay, ay + 22))
        # 残差加法
        cy = ay + 36
        out.append(f'<circle cx="{cxx}" cy="{cy}" r="13" fill="{p["box"]}" '
                   f'stroke="{p["fg"]}" stroke-width="1.6"/>')
        out.append(text(cxx, cy + 5, "+", 16, p["fg"], anchor="middle", weight="700"))

        # 第二段：norm + FFN
        y2 = cy + 13
        out.append(f'<circle cx="{cxx}" cy="{y2 + 12}" r="3" fill="{p["sub"]}"/>')
        out.append(f'<path d="M{cxx} {y2 + 12}H{rail}V{y2 + 150}H{cxx - 16}" fill="none" '
                   f'stroke="{p["sub"]}" stroke-width="1.4" stroke-dasharray="4 4" '
                   f'marker-end="url(#qa)"/>')
        out.append(arrow(cxx, y2 + 16, y2 + 38))
        out.append(chip(cxx - 100, y2 + 38, 200, 30, "post_attention_layernorm", "", p["box"]))
        out.append(arrow(cxx, y2 + 68, y2 + 90))
        out.append(box(cxx - 158, y2 + 90, 316, 58, p["box"], BLUE, rx=8))
        out.append(text(cxx, y2 + 108, "mlp（SwiGLU，见下方展开）", 13, p["fg"],
                        anchor="middle", weight="700"))
        out.append(text(cxx - 146, y2 + 128, "gate_proj  up_proj  down_proj", 11.5, GREEN, cls="m"))
        out.append(text(cxx + 146, y2 + 128, "4096 ↔ 12288", 11, p["sub"], anchor="end", cls="m"))
        out.append(arrow(cxx, y2 + 148, y2 + 170))
        cy2 = y2 + 184
        out.append(f'<circle cx="{cxx}" cy="{cy2}" r="13" fill="{p["box"]}" '
                   f'stroke="{p["fg"]}" stroke-width="1.6"/>')
        out.append(text(cxx, cy2 + 5, "+", 16, p["fg"], anchor="middle", weight="700"))
        out.append(text(cxx, cy2 + 34, "输出 h（4096）", 11, p["sub"], anchor="middle", cls="m"))
        return "".join(out)

    b.append(blockpanel(
        300, "全注意力块", "×8", AMBER, "self_attn（分组查询注意力 + RoPE）",
        [("q_proj", "4096 → 8192", True),
         ("k_proj    v_proj", "4096 → 1024", True),
         ("q_norm  k_norm", "QK-Norm，不是线性层", False),
         ("o_proj", "4096 → 4096", True),
         ("16 个查询头 / 4 个 KV 头 × 256", "", False)]))

    b.append(blockpanel(
        720, "线性注意力块", "×24", BLUE, "linear_attn（GatedDeltaNet）",
        [("in_proj_qkv", "4096 → 8192", True),
         ("in_proj_z", "4096 → 4096", True),
         ("in_proj_a   in_proj_b", "4096 → 32", True),
         ("conv1d(k=4)  norm", "卷积与归一化，不挂 LoRA", False),
         ("out_proj", "4096 → 4096", True)]))

    # ---------------- 底部：SwiGLU 展开 + 数字栏 ----------------
    by = 604
    b.append(box(40, by, 640, 150, p["dim"], p["line"], rx=12))
    b.append(text(56, by + 24, "mlp 展开：SwiGLU", 15, p["fg"], weight="700"))
    row = by + 60
    b.append(chip(60, row - 3, 96, 44, "h", "4096", p["box"]))
    b.append(f'<path d="M156 {row + 19}h18" stroke="{p["sub"]}" stroke-width="1.6"/>')
    b.append(f'<path d="M174 {by + 61}V{by + 107}" stroke="{p["sub"]}" stroke-width="1.6"/>')
    b.append(f'<path d="M174 {by + 61}h10" stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(f'<path d="M174 {by + 107}h10" stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(chip(188, by + 39, 128, 44, "gate_proj", "→ 12288", p["box"], GREEN))
    b.append(chip(188, by + 85, 128, 44, "up_proj", "→ 12288", p["box"], GREEN))
    b.append(f'<path d="M316 {by + 59}h30" stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(chip(348, by + 44, 84, 34, "SiLU", "", p["box"]))
    b.append(f'<path d="M316 {by + 105}h112" stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(f'<path d="M432 {by + 59}h24v40" fill="none" stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(f'<circle cx="{456}" cy="{by + 105}" r="13" fill="{p["box"]}" stroke="{p["fg"]}" stroke-width="1.6"/>')
    b.append(text(456, by + 110, "×", 15, p["fg"], anchor="middle", weight="700"))
    b.append(f'<path d="M469 {by + 105}h22" stroke="{p["sub"]}" stroke-width="1.6" marker-end="url(#qa)"/>')
    b.append(chip(494, by + 85, 128, 44, "down_proj", "→ 4096", p["box"], GREEN))
    b.append(text(56, by + 136, "逐元素相乘那一步就是 SwiGLU 的“门”：up 的输出被 SiLU(gate) 缩放。",
                  12, p["sub"]))

    nx = 720
    b.append(box(nx, by, 420, 150, p["dim"], p["line"], rx=12))
    b.append(text(nx + 16, by + 24, "这块模型的数字", 15, p["fg"], weight="700"))
    rows = [("词表", "248320"), ("隐藏维 d_model", "4096"),
            ("FFN 中间维", "12288"), ("层数", "32（24 线性 + 8 全）"),
            ("头", "16 查询 / 4 KV × 256"), ("最大位置", "262144")]
    for i, (k, v) in enumerate(rows):
        ry = by + 46 + (i % 3) * 20
        rx0 = nx + 16 + (i // 3) * 205
        b.append(text(rx0, ry, k, 11.5, p["sub"]))
        b.append(text(rx0 + 195, ry, v, 11.5, p["fg"], anchor="end", cls="m"))
    b.append(text(nx + 16, by + 128, "绿色 = nn.Linear，LoRA 挂点（248 个）", 12, GREEN, weight="700"))
    b.append(text(nx + 16, by + 144, "灰色 = 归一化 / 卷积 / 激活，没有权重矩阵可拆", 11.5, p["sub"]))

    return svg(W, H, "".join(b), "",
               "Qwen3.5-9B 的通路：embedding、32 个 decoder 块（24 个线性注意力 + 8 个全注意力）、"
               "RMSNorm 与 lm_head；每块内部是两段残差，LoRA 挂在其中的线性层上")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"fig-lora-arch": fig_lora_arch, "fig-train-step": fig_train_step,
                     "fig-float-bits": fig_float_bits, "fig-svd": fig_svd,
                     "fig-qwen-arch": fig_qwen_arch}.items():
        for suffix, pal in (("light", LIGHT), ("dark", DARK)):
            (OUT / f"{name}-{suffix}.svg").write_text(fn(pal))
        print(f"  {name}  {(OUT / f'{name}-light.svg').stat().st_size} 字节")


if __name__ == "__main__":
    main()
