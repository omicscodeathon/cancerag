"""Box-and-arrow primitives for the schematic figures.

Boxes are a pale wash with a solid accent strip on the left carrying the
stage's identity colour; all text stays in the ink tokens. Coordinates are in
data units of an axes whose limits the caller sets (typically 0–100).
"""

from __future__ import annotations

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from . import _style as S


def blank(ax, xlim=(0, 100), ylim=(0, 100)) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def _wash(hex_color: str, alpha: float = 0.12) -> tuple:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def box(ax, x, y, w, h, title, sub=None, color=S.GREY_DARK, *, title_size=None,
        sub_size=None, align="center"):
    """Draw a box with its lower-left corner at (x, y). Returns its anchor points."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
        facecolor=_wash(color), edgecolor=_wash(color, 0.45), linewidth=0.6,
        zorder=2,
    ))
    ax.add_patch(Rectangle((x, y), 1.0, h, facecolor=color, edgecolor="none", zorder=3))
    tx = x + w / 2 if align == "center" else x + 2.6
    ha = "center" if align == "center" else "left"
    if sub:
        two = "\n" in sub
        ax.text(tx, y + h * (0.75 if two else 0.66), title, ha=ha, va="center",
                fontsize=title_size or S.FS_BASE, fontweight="bold",
                color=S.TEXT_PRIMARY, zorder=4)
        ax.text(tx, y + h * (0.34 if two else 0.30), sub, ha=ha, va="center",
                fontsize=sub_size or 6.6, color=S.TEXT_SECONDARY, zorder=4,
                linespacing=1.25)
    else:
        ax.text(tx, y + h / 2, title, ha=ha, va="center",
                fontsize=title_size or S.FS_BASE, fontweight="bold",
                color=S.TEXT_PRIMARY, zorder=4)
    return {
        "top": (x + w / 2, y + h), "bottom": (x + w / 2, y),
        "left": (x, y + h / 2), "right": (x + w, y + h / 2),
    }


def note(ax, x, y, w, h, text, *, size=6.4):
    """A side box for exclusions / counts: dashed outline, muted text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
        facecolor=S.SURFACE, edgecolor=S.GREY, linewidth=0.7,
        linestyle=(0, (3, 2)), zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            color=S.TEXT_SECONDARY, zorder=4, linespacing=1.25)
    return {"left": (x, y + h / 2), "right": (x + w, y + h / 2),
            "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}


def elbow(ax, points, *, color=S.TEXT_SECONDARY, lw=0.9, dashed=True):
    """Polyline through ``points`` ending in an arrowhead at the last point."""
    ls = (0, (3, 2)) if dashed else "-"
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=1,
            solid_capstyle="butt", dash_capstyle="butt")
    arrow(ax, points[-2], points[-1], color=color, lw=lw, dashed=dashed)


def arrow(ax, start, end, *, rad=0.0, color=S.TEXT_SECONDARY, lw=0.9, dashed=False):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", shrinkA=1.5, shrinkB=1.5, zorder=1,
        linestyle=(0, (3, 2)) if dashed else "-",
    ))
