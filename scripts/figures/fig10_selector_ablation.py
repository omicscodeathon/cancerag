"""Feature-selector × learner ablation."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig10_selector_ablation"
TITLE = "Feature-selector × learner ablation"
DESCRIPTION = """
(a) Macro-F1 (mean ± SD over the folds in `selector_ablation_long.csv`) for
each feature selector (Boruta, L1-logistic regression, mutual information,
RFECV) paired with each learner, run with force-keeping disabled so the
selectors compete on equal terms. (b) Per-selector mean across learners with
the spread of the four learner means. Selector choice moves macro-F1 less
than learner choice does, and only L1 is consistently low. Protocol:
scaffold-grouped 3-fold CV, seed 42 (`feature_selection.run_selector_ablation`).
Caveat: this artefact is dated 2026-05-03, before the per-receptor QC-column
fix (2026-08-03) and the receptor repair (2026-08-10); it has not been re-run
on the corrected 2,582-feature matrix, so its absolute values belong to the
pre-fix feature set.
"""
SOURCES = [
    "processed/ml_models/selector_ablation/selector_ablation_summary.csv",
    "processed/ml_models/selector_ablation/selector_ablation_long.csv",
]

SELECTORS = ["boruta", "rfecv", "mutual_info", "l1_logreg"]
SEL_LABEL = {"boruta": "Boruta", "rfecv": "RFECV", "mutual_info": "Mutual info",
             "l1_logreg": "L1-logreg"}


def build(art: Artefacts):
    summ = art.csv(SOURCES[0])
    long = art.csv(SOURCES[1])
    sels = [s for s in SELECTORS if s in set(summ.selector)]
    m = summ.pivot(index="selector", columns="model", values="mean").reindex(sels)[S.MODEL_ORDER]
    sd = summ.pivot(index="selector", columns="model", values="std").reindex(sels)[S.MODEL_ORDER]
    n = summ.pivot(index="selector", columns="model", values="n").reindex(sels)[S.MODEL_ORDER]
    nfeat = long.groupby("selector").n_features_selected.median()

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.8),
                                 gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.45))
    vmin, vmax = float(m.values.min()) - 0.02, float(m.values.max()) + 0.01
    im = ax.imshow(m.values, cmap=S.SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    best = np.unravel_index(np.argmax(m.values), m.shape)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ink = S.ink_on(S.cmap_hex(S.SEQ_CMAP, (v - vmin) / (vmax - vmin)))
            ax.text(j, i, f"{v:.3f}\n± {sd.values[i, j]:.3f}", ha="center",
                    va="center", fontsize=6.6, color=ink,
                    fontweight="bold" if (i, j) == best else "normal")
    ax.set_xticks(range(m.shape[1]))
    ax.set_xticklabels([S.MODEL_LABEL[k].replace(" ", "\n", 1) for k in m.columns])
    ax.set_yticks(range(m.shape[0]))
    ax.set_yticklabels([f"{SEL_LABEL[s]}\n(median {int(nfeat[s])} feats)" for s in sels],
                       fontsize=6.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, m.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0], 1), minor=True)
    ax.grid(which="minor", color=S.SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Macro-F1", fontsize=6.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.4)
    S.panel_label(ax, "a", x=-0.32, y=1.02)

    sel_mean = m.mean(axis=1)
    ys = np.arange(len(sels))[::-1]
    for y, s in zip(ys, sels):
        row = m.loc[s]
        bx.plot([row.min(), row.max()], [y, y], color=S.GREY, lw=1.2, zorder=1)
        for k in S.MODEL_ORDER:
            bx.scatter(row[k], y, s=14, color=S.MODEL_COLOR[k], zorder=2,
                       edgecolor=S.SURFACE, linewidth=0.6)
        bx.scatter(sel_mean[s], y, s=46, marker="|", color=S.TEXT_PRIMARY,
                   linewidth=1.6, zorder=3)
        bx.text(row.max() + 0.006, y, f"{sel_mean[s]:.3f}", va="center",
                fontsize=6.6, color=S.TEXT_PRIMARY)
    bx.set_yticks(ys)
    bx.set_yticklabels([SEL_LABEL[s] for s in sels])
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Macro-F1")
    bx.set_xlim(vmin - 0.01, vmax + 0.04)
    bx.set_ylim(-0.7, len(sels) - 0.3)
    S.xgrid(bx)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=4, color=S.MODEL_COLOR[k])
               for k in S.MODEL_ORDER]
    handles.append(plt.Line2D([], [], ls="none", marker="|", ms=8, mew=1.6,
                              color=S.TEXT_PRIMARY))
    bx.legend(handles, [S.MODEL_LABEL[k] for k in S.MODEL_ORDER] + ["selector mean"],
              loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=3, fontsize=6.4,
              columnspacing=0.8)
    S.panel_label(bx, "b", x=-0.3, y=1.02)

    facts = {
        "cells (mean ± SD, n folds)": {
            f"{SEL_LABEL[s]} × {S.MODEL_LABEL[k]}":
            f"{m.loc[s, k]:.3f} ± {sd.loc[s, k]:.3f} (n={int(n.loc[s, k])})"
            for s in sels for k in S.MODEL_ORDER},
        "per-selector mean": {SEL_LABEL[s]: round(float(sel_mean[s]), 3) for s in sels},
        "best cell": f"{SEL_LABEL[sels[best[0]]]} × {S.MODEL_LABEL[m.columns[best[1]]]} "
                     f"= {m.values[best]:.3f}",
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
