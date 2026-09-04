"""Shared matplotlib style: white ground, NVIDIA green, dark grey text, no chartjunk.

    from plotstyle import apply, NV
    apply()
    ax.plot(x, y, color=NV["green"])
"""
import matplotlib as mpl

NV = {
    "green": "#76B900",   # NVIDIA green
    "dark":  "#1A1A1A",   # text and primary series
    "gray":  "#5E5E5E",   # secondary series
    "light": "#D9D9D9",   # grid
    "blue":  "#0074C8",   # contrast colour for binary comparisons
    "amber": "#F5A623",
}
PALETTE = [NV["green"], NV["dark"], NV["blue"], NV["amber"], NV["gray"]]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "Helvetica Neue", "Arial", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.edgecolor": NV["dark"], "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": NV["light"], "grid.linewidth": 0.6, "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": NV["dark"],
        "axes.labelsize": 11, "axes.labelcolor": NV["dark"],
        "xtick.color": NV["dark"], "ytick.color": NV["dark"], "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 10,
        "lines.linewidth": 2.2, "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
