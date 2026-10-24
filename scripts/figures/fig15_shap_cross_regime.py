"""SHAP selection frequency across the four evaluation regimes."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _style as S
from ._data import Artefacts

NAME = "fig15_shap_cross_regime"
TITLE = "SHAP feature importance across the four evaluation settings"
DESCRIPTION = """
Selection frequency — the fraction of runs (15 per regime: 3 seeds × 5 folds
for each CV split, 15 bootstrap resamples for the temporal-holdout model) in
which a feature ranks top-20 by mean |SHAP|. Rows are the 20 highest-ranked
features in `shap_cross_regime.csv`, plus the best-ranked docking-energetics
feature for reference. Features above the rule reach ≥ 0.5 in all four
regimes (the robust set); every one of them is a ligand-only descriptor.
Rows tagged [pose] or [docking] need a docked pose or a receptor.
"""
SOURCES = ["processed/ml_models/interpretability/shap_cross_regime.csv"]

SF = ["sf_stratified", "sf_scaffold", "sf_receptor", "sf_temporal"]
REGIME_LABELS = ["Stratified", "Scaffold-\ngrouped", "Receptor-\ngrouped", "Temporal\nholdout"]
POSE = ("Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
        "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex", "pose_3d_missing")
DOCKING = ("vina_", "ifp_", "n_residues_contacted", "n_total_contacts", "n_poses")
TOP_N = 20


def tag(feature: str) -> str:
    if feature in POSE:
        return "  [pose]"
    if any(feature.startswith(p) or feature == p for p in DOCKING):
        return "  [docking]"
    return ""


def build(art: Artefacts):
    df = art.csv(SOURCES[0])
    robust_mask = (df[SF] >= 0.5).all(axis=1)
    robust = df[robust_mask].sort_values("mean_selection_freq", ascending=False)
    rest = df[~robust_mask].head(TOP_N - len(robust))
    extra = df[df.feature.map(lambda f: tag(f) == "  [docking]")].head(1)
    extra = extra[~extra.feature.isin(rest.feature)]
    rows = pd.concat([robust, rest, extra])
    mat = rows[SF].values

    fig, ax = plt.subplots(figsize=(4.9, 0.25 * len(rows) + 1.1))
    im = ax.imshow(mat, cmap=S.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.4,
                    color=S.ink_on(S.cmap_hex(S.SEQ_CMAP, v)))
    ax.set_xticks(range(4))
    ax.set_xticklabels(REGIME_LABELS)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f + tag(f) for f in rows.feature], fontsize=6.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=S.SURFACE, linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    ax.axhline(len(robust) - 0.5, color=S.TEXT_PRIMARY, lw=1.4)
    ax.text(3.62, len(robust) - 0.5, f"robust set\n(n = {len(robust)})",
            va="center", ha="left", fontsize=6.4, color=S.TEXT_PRIMARY)
    if len(extra):
        ax.axhline(len(rows) - 1.5, color=S.GREY, lw=0.8, ls=(0, (2, 2)))
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.16, shrink=0.5)
    cb.set_label("Selection frequency (top-20 by mean |SHAP|)", fontsize=6.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.4)

    structural_hits = df[df.feature.map(lambda f: tag(f) != "") & robust_mask]
    facts = {
        "robust features (sf ≥ 0.5 in all four)": robust.feature.tolist(),
        "robust features that are docking/pose-derived": len(structural_hits),
        "selection frequencies (strat/scaf/rec/temp)": {
            r.feature: [round(float(r[c]), 3) for c in SF] for _, r in robust.iterrows()},
        "best docking-energetics feature": (
            f"{extra.feature.iloc[0]} mean sf {extra.mean_selection_freq.iloc[0]:.3f}"
            if len(extra) else "already in top rows"),
        "features in table": len(df),
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
