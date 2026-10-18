"""Holdout confusion matrix of the calibrated final model."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig12_confusion_matrix"
TITLE = "Temporal-holdout confusion matrix, calibrated random forest"
DESCRIPTION = """
Confusion matrix of the calibrated random forest on the temporal holdout
(n = 172). Cells show the count and, below it, the share of the true class
(row); colour encodes the row share. All nine G-protein-selectivity rows and
all twenty ERK rows are predicted as G protein. The build checks the matrix in
`model_meta.json` against `holdout_predictions.csv`.
"""
SOURCES = [
    "processed/ml_models/model_meta.json",
    "processed/ml_models/holdout_predictions.csv",
]


def build(art: Artefacts):
    meta = art.model_meta
    classes = meta["label_classes"]
    cm = np.array(meta["holdout_metrics"]["confusion_matrix"])
    pred = art.csv(SOURCES[1])
    check = np.array([[int(((pred.y_true == a) & (pred.y_pred == b)).sum())
                       for b in classes] for a in classes])
    if not (check == cm).all():
        raise AssertionError("confusion matrix in model_meta.json disagrees with predictions")

    order = ["G protein", "β Arrestin", "G protein selectivity", "ERK"]
    idx = [classes.index(c) for c in order]
    cm = cm[np.ix_(idx, idx)]
    share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    im = ax.imshow(share, cmap=S.SEQ_CMAP, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ink = S.ink_on(S.cmap_hex(S.SEQ_CMAP, share[i, j]))
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.0%}", ha="center",
                    va="center", fontsize=7.2, color=ink,
                    fontweight="bold" if i == j else "normal")
    labs = [S.CLASS_LABEL[c].replace("G-protein selectivity", "G-protein\nselectivity")
            for c in order]
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=6.8)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{lab}\n(n = {cm[i].sum()})" for i, lab in enumerate(labs)],
                       fontsize=6.8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color=S.SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("Share of true class", fontsize=6.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.4)
    facts = {
        "rows (true) × cols (pred), order G/βarr/GPS/ERK": cm.tolist(),
        "accuracy": f"{np.trace(cm) / cm.sum():.3f}",
        "GPS predicted as G protein": f"{cm[2, 0]} of {cm[2].sum()}",
        "ERK predicted as G protein": f"{cm[3, 0]} of {cm[3].sum()}",
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
