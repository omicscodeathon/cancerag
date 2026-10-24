"""Shared visual system for every manuscript figure.

One palette, one type scale, one save routine, so the figures read as a set.

Colour follows the job it does (dataviz method):

* **Bias classes** — identity, used in scatter plots where every pair of
  classes can sit next to each other, so the four hues are validated
  *all-pairs* (``validate_palette.js --pairs all``: worst CVD ΔE 9.2, worst
  normal-vision ΔE 16.3, light surface). Slots 1, 2, 3 and 7 of the reference
  categorical palette; the reference fourth slot (yellow) fails all-pairs next
  to orange, so it is skipped here.
* **Learners / series** — identity on bars and lines, reference slots 1–4 in
  fixed order (adjacent-pair validated).
* **Magnitude** — the single-hue blue ramp, light → dark.
* **De-emphasis** — neutral greys for baselines, dummies and context.

Text never wears a series colour; marks carry identity and the legend or a
direct label names it. Figures are static print artefacts (300 dpi PNG plus a
vector PDF with embedded TrueType fonts), so there is no dark variant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --------------------------------------------------------------------------
# Surfaces and text tokens
# --------------------------------------------------------------------------
SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8985"
GRID = "#e6e5e1"
AXIS = "#52514e"

# --------------------------------------------------------------------------
# Categorical slots (reference palette, light mode)
# --------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"

SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

# De-emphasis greys (baselines, dummies, context bars)
GREY_DARK = "#6f6e69"
GREY = "#a3a29c"
GREY_LIGHT = "#d4d3ce"
GREY_WASH = "#f0efec"

# Sequential blue ramp (reference steps 100 -> 700)
BLUE_RAMP = [
    "#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = LinearSegmentedColormap.from_list("cancerag_blue", BLUE_RAMP)

# --------------------------------------------------------------------------
# Domain encodings — fixed, never re-ordered by rank
# --------------------------------------------------------------------------
# Label order exactly as the label encoder stores it (alphabetical).
CLASS_ORDER = ["ERK", "G protein", "G protein selectivity", "β Arrestin"]
CLASS_LABEL = {
    "ERK": "ERK",
    "G protein": "G protein",
    "G protein selectivity": "G-protein selectivity",
    "β Arrestin": "β-arrestin",
}
CLASS_COLOR = {
    "G protein": BLUE,
    "β Arrestin": ORANGE,
    "G protein selectivity": AQUA,
    "ERK": VIOLET,
}

MODEL_ORDER = ["xgboost", "lightgbm", "random_forest", "elastic_lr"]
MODEL_LABEL = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random forest",
    "elastic_lr": "Elastic-net LR",
    "majority_class": "Majority class",
    "stratified": "Stratified random",
}
MODEL_COLOR = {
    "xgboost": BLUE,
    "lightgbm": ORANGE,
    "random_forest": AQUA,
    "elastic_lr": YELLOW,
    "majority_class": GREY_DARK,
    "stratified": GREY,
}

SPLIT_LABEL = {
    "stratified_kfold": "Stratified",
    "scaffold_kfold": "Scaffold-grouped",
    "receptor_kfold": "Receptor-grouped",
    "stratified": "Stratified",
    "scaffold_grouped": "Scaffold-grouped",
    "receptor_grouped": "Receptor-grouped",
    "scaffold": "Scaffold-grouped",
    "receptor": "Receptor-grouped",
    "temporal": "Temporal holdout",
}

# Pipeline-stage hues for the two schematics (identity, adjacent use only).
STAGE_COLOR = {
    "source": BLUE,
    "curation": GREY_DARK,
    "docking": AQUA,
    "features": YELLOW,
    "ml": ORANGE,
    "deploy": VIOLET,
}

# --------------------------------------------------------------------------
# Type scale and rcParams
# --------------------------------------------------------------------------
FONT_FAMILY = ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"]
FS_BASE = 8.5
FS_SMALL = 7.5
FS_TITLE = 9.5
FS_PANEL = 11

DPI = 300

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": FONT_FAMILY,
    "font.size": FS_BASE,
    "axes.titlesize": FS_TITLE,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": FS_BASE,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.6,
    "axes.facecolor": SURFACE,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "xtick.labelcolor": TEXT_SECONDARY,
    "ytick.labelcolor": TEXT_SECONDARY,
    "xtick.labelsize": FS_SMALL,
    "ytick.labelsize": FS_SMALL,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": FS_SMALL,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.borderaxespad": 0.3,
    "lines.linewidth": 1.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 5,
    "patch.linewidth": 0,
    "figure.facecolor": SURFACE,
    "figure.dpi": 100,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "text.color": TEXT_PRIMARY,
}


# White knock-out behind a value label so it stays legible over grid or
# reference lines.
LABEL_BOX = dict(boxstyle="square,pad=0.12", facecolor=SURFACE, edgecolor="none")


def apply() -> None:
    """Install the shared rcParams. Idempotent."""
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update(_RC)


def panel_label(ax, letter: str, *, x: float = -0.08, y: float = 1.04) -> None:
    """Nature-style lowercase bold panel letter in the axes' top-left corner."""
    ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=FS_PANEL,
        fontweight="bold", color=TEXT_PRIMARY, va="bottom", ha="right",
    )


def ygrid(ax) -> None:
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)


def xgrid(ax) -> None:
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)


def ink_on(fill_hex: str) -> str:
    """White or near-black text, whichever clears contrast on ``fill_hex``."""
    h = fill_hex.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    contrast_white = 1.05 / (lum + 0.05)
    contrast_black = (lum + 0.05) / 0.05
    return "#ffffff" if contrast_white >= contrast_black else TEXT_PRIMARY


def cmap_hex(cmap, value: float) -> str:
    r, g, b, _ = cmap(value)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def save(fig, name: str, out_dir: Path) -> list[Path]:
    """Write ``<name>.png`` (300 dpi) and ``<name>.pdf`` (vector) to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.06)
        paths.append(p)
    plt.close(fig)
    return paths
