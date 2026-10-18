"""Per-class holdout F1: calibrated random forest vs stacking (Table 3 as a chart)."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig04_per_class_breakthrough"
TITLE = "Per-class F1 on the temporal holdout: single model vs stacking"
DESCRIPTION = """
Per-class F1 on the temporal holdout for the calibrated random forest (the
selected single model) and the stacking ensemble. Stacking lifts ERK and
β-arrestin at the cost of the dominant G-protein class; G-protein selectivity
stays at zero in both. Point estimates only: the stacking base-model holdout
probabilities were not persisted, so per-class CIs cannot be computed.
"""
SOURCES = [
    "processed/ml_models/model_meta.json",
    "processed/ml_models/advanced/stacking_meta.json",
]


def build(art: Artefacts):
    meta = art.model_meta
    st = art.json(SOURCES[1])
    classes = meta["label_classes"]
    assert classes == S.CLASS_ORDER, classes
    rep = meta["holdout_metrics"]["classification_report"]
    single = [rep[c]["f1-score"] for c in classes]
    support = [int(rep[c]["support"]) for c in classes]
    stack = [st["holdout_metrics"]["per_class_f1"][str(i)] for i in range(len(classes))]

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(classes))
    w = 0.36
    series = [("Random forest (calibrated single model)", single, S.GREY),
              ("Stacking ensemble", stack, S.BLUE)]
    for k, (lab, vals, col) in enumerate(series):
        xs = x + (k - 0.5) * w
        ax.bar(xs, vals, w, color=col, edgecolor=S.SURFACE, linewidth=1.0,
               label=lab, zorder=2)
        for xi, v in zip(xs, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.8, color=S.TEXT_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{S.CLASS_LABEL[c]}\nn = {n}" for c, n in zip(classes, support)])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Per-class F1 (temporal holdout)")
    ax.set_ylim(0, 0.9)
    S.ygrid(ax)
    ax.legend(loc="upper right")
    facts = {
        S.CLASS_LABEL[c]: f"n={n}; single {a:.3f} -> stacking {b:.3f} (Δ {b - a:+.3f})"
        for c, n, a, b in zip(classes, support, single, stack)
    }
    facts["macro-F1 single / stacking"] = (
        f"{meta['holdout_metrics']['macro_f1']['point_estimate']:.3f} / "
        f"{st['holdout_metrics']['macro_f1']['point_estimate']:.3f}"
    )
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
