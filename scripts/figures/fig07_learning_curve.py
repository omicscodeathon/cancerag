"""Learning curve of the selected model."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig07_learning_curve"
TITLE = "Learning curve of the selected model"
DESCRIPTION = """
Cross-validated macro-F1 of the selected random forest at 25/50/75/100 % of
the 443 train-eligible rows, three seeds. The protocol is `StratifiedKFold`
(3 folds) in `reviewer_extras.compute_learning_curve` — the optimistic split,
not the scaffold-grouped one. Thin lines are individual seeds, the thick line
the seed mean, the band ±1 SD across seeds. The x-axis is log2 so that the
gain per doubling of the training set reads directly; the mean gain per
doubling is printed from the artefact.
"""
SOURCES = ["processed/ml_models/extras/learning_curve.csv"]


def build(art: Artefacts):
    lc = art.csv(SOURCES[0])
    g = lc.groupby("n_train_total").macro_f1_mean
    mean, sd = g.mean(), g.std(ddof=1)
    ns = mean.index.values.astype(float)
    gains = np.diff(mean.values) / np.diff(np.log2(ns))
    total_doublings = np.log2(ns[-1] / ns[0])
    per_doubling = (mean.values[-1] - mean.values[0]) / total_doublings

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for _seed, d in lc.groupby("seed"):
        d = d.sort_values("n_train_total")
        ax.plot(d.n_train_total, d.macro_f1_mean, color=S.GREY, lw=0.8,
                marker="o", ms=3, zorder=2)
    ax.fill_between(ns, mean - sd, mean + sd, color=S.BLUE, alpha=0.12,
                    linewidth=0, zorder=1)
    ax.plot(ns, mean, color=S.BLUE, lw=2, zorder=3)
    ax.scatter(ns, mean, s=28, color=S.BLUE, edgecolor=S.SURFACE, linewidth=1.2,
               zorder=4)
    for x, m in zip(ns, mean.values):
        ax.annotate(f"{m:.3f}", (x, m), xytext=(0, -9), textcoords="offset points",
                    ha="center", va="top", fontsize=S.FS_SMALL,
                    color=S.TEXT_PRIMARY, bbox=S.LABEL_BOX, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(n)}\n({f:.0%})" for n, f in
                        zip(ns, sorted(lc.fraction.unique()))])
    ax.minorticks_off()
    ax.set_xlabel("Training rows (fraction of 443)")
    ax.set_ylabel("Macro-F1 (stratified 3-fold CV)")
    ax.set_ylim(0.35, 0.62)
    S.ygrid(ax)
    ax.text(0.98, 0.04,
            f"mean gain {per_doubling:+.3f} per doubling\n"
            f"(first {gains[0]:+.3f}, last {gains[-1]:+.3f} per doubling)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=S.FS_SMALL, color=S.TEXT_SECONDARY)
    handles = [plt.Line2D([], [], color=S.GREY, lw=0.8, marker="o", ms=3),
               plt.Line2D([], [], color=S.BLUE, lw=2)]
    ax.legend(handles, ["individual seed (42, 7, 13)", "seed mean ± 1 SD"],
              loc="upper left")
    facts = {f"n={int(n)}": f"{m:.4f} ± {s:.4f}" for n, m, s in zip(ns, mean, sd)}
    facts["mean gain per doubling (110 -> 443)"] = f"{per_doubling:+.3f}"
    facts["gain per doubling, segment-wise"] = [round(float(x), 3) for x in gains]
    seed42 = lc[lc.seed == 42].sort_values("n_train_total").macro_f1_mean.round(3).tolist()
    facts["seed 42 alone"] = seed42
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
