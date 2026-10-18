"""Share of SHAP attribution by information source, in every evaluation regime."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig06_shap_block_attribution"
TITLE = "SHAP attribution by information source across evaluation regimes"
DESCRIPTION = """
(a) Share of total mean |SHAP| carried by each feature block, under
stratified, scaffold-grouped and receptor-grouped CV and the temporal-holdout
model. Ligand-only blocks (2D descriptors, 2D fingerprints) are blue; blocks
that need a docked pose or a receptor are warm hues. The ligand-only total is
printed beside each bar. (b) The same attribution divided by the number of
columns in the block (receptor-grouped regime), i.e. mean share per column —
small structural blocks are not individually negligible per column, but
together they carry under a tenth of the attribution. This replaces the
2026-08-25 "validated features" figure, which plotted the per-receptor QC
columns later shown to leak receptor identity.
"""
SOURCES = ["processed/ml_models/interpretability/shap_block_attribution.csv"]

BLOCKS = [
    ("ligand-only 2D descriptor", "Ligand 2D descriptors", "#256abf", "2D descriptors"),
    ("ligand-only 2D fingerprint", "Ligand 2D fingerprints", "#9ec5f4", "2D fingerprints"),
    ("docked-pose 3D shape", "Docked-pose 3D shape", S.ORANGE, "Pose 3D shape"),
    ("receptor-ligand contacts (ProLIF)", "Receptor–ligand contacts (ProLIF)", S.YELLOW,
     "ProLIF contacts"),
    ("pair-level docking energetics", "Pair-level docking energetics", S.MAGENTA,
     "Docking energetics"),
]
LIGAND = {"ligand-only 2D descriptor", "ligand-only 2D fingerprint"}
REGIMES = ["stratified", "scaffold", "receptor", "temporal"]


def build(art: Artefacts):
    df = art.csv(SOURCES[0])
    piv = df.pivot(index="regime", columns="block", values="shap_share_pct").reindex(REGIMES)
    nfeat = df.groupby("block").n_features.first()

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.7, 1], wspace=0.7),
    )
    ys = np.arange(len(REGIMES))[::-1]
    left = np.zeros(len(REGIMES))
    for key, lab, col, _ in BLOCKS:
        v = piv[key].values
        ax.barh(ys, v, left=left, height=0.6, color=col, edgecolor=S.SURFACE,
                linewidth=1.0, label=f"{lab} ({int(nfeat[key])} cols)")
        for y, l0, vi in zip(ys, left, v):
            if vi >= 8:
                ax.text(l0 + vi / 2, y, f"{vi:.1f}", ha="center", va="center",
                        fontsize=6.6, color=S.ink_on(col))
        left += v
    lig = piv[list(LIGAND)].sum(axis=1).values
    for y, t in zip(ys, lig):
        ax.text(101.5, y, f"ligand-only\n{t:.1f} %", va="center", fontsize=6.4,
                color=S.TEXT_SECONDARY)
    ax.set_yticks(ys)
    ax.set_yticklabels([S.SPLIT_LABEL[r] for r in REGIMES])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of total mean |SHAP| (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.62, -0.22), ncol=2,
              fontsize=6.5, columnspacing=1.0)
    S.panel_label(ax, "a", x=-0.3)

    rec = df[df.regime == "receptor"].set_index("block")
    per_col = [(short, rec.loc[key, "shap_share_pct"] / rec.loc[key, "n_features"], col)
               for key, _, col, short in BLOCKS]
    yb = np.arange(len(per_col))[::-1]
    bx.barh(yb, [p for _, p, _ in per_col], height=0.6,
            color=[c for _, _, c in per_col])
    for y, (_, p, _) in zip(yb, per_col):
        bx.text(p * 1.15, y, f"{p:.3f}", va="center", fontsize=6.6,
                color=S.TEXT_PRIMARY)
    bx.set_xscale("log")
    bx.set_xlim(0.01, 3)
    bx.set_yticks(yb)
    bx.set_yticklabels([lab for lab, _, _ in per_col], fontsize=6.8)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Share per column (%)")
    bx.set_title("Receptor-grouped CV", fontsize=S.FS_SMALL, fontweight="normal",
                 color=S.TEXT_SECONDARY)
    S.xgrid(bx)
    S.panel_label(bx, "b", x=-0.55)

    rng_pp = (piv.max() - piv.min()).max()
    facts = {
        f"{S.SPLIT_LABEL[r]}": "; ".join(
            f"{lab} {piv.loc[r, key]:.2f}%" for key, lab, _, _ in BLOCKS)
        for r in REGIMES
    }
    facts["ligand-only total by regime (%)"] = {
        S.SPLIT_LABEL[r]: round(float(t), 2) for r, t in zip(REGIMES, lig)}
    facts["max cross-regime range of any block (pp)"] = round(float(rng_pp), 2)
    facts["columns per block"] = {lab: int(nfeat[key]) for key, lab, _, _ in BLOCKS}
    facts["share per column, receptor-grouped (%)"] = {
        lab: round(float(p), 3) for lab, p, _ in per_col}
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
