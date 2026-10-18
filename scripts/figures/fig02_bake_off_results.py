"""Four-mode cross-validation bake-off (Table 1 as a chart)."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig02_bake_off_results"
TITLE = "Four-mode cross-validation model comparison"
DESCRIPTION = """
Macro-F1 (mean ± SD over 5 seeds × 5 folds = 25 fits per cell) for four
learners under stratified, scaffold-grouped and receptor-grouped K-fold CV,
with each fit as a dot. The dummy baselines were evaluated under stratified
K-fold only and are drawn in their own group on the left. The selected model
(rule in `selection_decision.json`: best mean macro-F1 on scaffold-grouped CV)
is outlined. Only the best learner in each split carries a value label.
"""
SOURCES = [
    "processed/ml_models/cv_results_summary.csv",
    "processed/ml_models/cv_results_long.csv",
    "processed/ml_models/selection_decision.json",
]

SPLITS = ["stratified_kfold", "scaffold_kfold", "receptor_kfold"]
DUMMIES = ["majority_class", "stratified"]


def build(art: Artefacts):
    summ = art.csv(SOURCES[0])
    long = art.csv(SOURCES[1])
    decision = art.json(SOURCES[2])
    chosen, primary = decision["chosen"], decision["primary_split"]

    cell = summ.set_index(["model", "split"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.19
    rng = np.random.default_rng(0)
    facts: dict = {}

    def draw(x, model, split, label=False):
        row = cell.loc[(model, split)]
        m, sd = float(row.macro_f1_mean), float(row.macro_f1_std)
        is_sel = model == chosen and split == primary
        ax.bar(x, m, width, color=S.MODEL_COLOR[model], edgecolor=S.SURFACE,
               linewidth=1.0, zorder=2)
        if is_sel:
            ax.bar(x, m, width, fill=False, edgecolor=S.TEXT_PRIMARY,
                   linewidth=1.3, zorder=4)
        ax.errorbar(x, m, yerr=sd, color=S.TEXT_PRIMARY, lw=0.8, capsize=2,
                    zorder=5)
        pts = long[(long.model == model) & (long.split == split)].macro_f1.values
        ax.scatter(x + rng.uniform(-width * 0.28, width * 0.28, len(pts)), pts,
                   s=4, color=S.TEXT_PRIMARY, alpha=0.35, linewidths=0, zorder=3)
        top = max(m + sd, pts.max() if len(pts) else m)
        if label:
            ax.text(x, top + 0.012, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=S.FS_SMALL, color=S.TEXT_PRIMARY)
        facts[f"{S.MODEL_LABEL[model]} / {S.SPLIT_LABEL[split]}"] = (
            f"{m:.3f} ± {sd:.3f} (n={int(row.n_obs)})"
        )

    # dummy group
    centers = []
    x0 = 0.0
    for i, d in enumerate(DUMMIES):
        draw(x0 + (i - 0.5) * width, d, "stratified_kfold", label=True)
    centers.append(x0)
    for gi, split in enumerate(SPLITS):
        c = 1.15 + gi * 1.1
        centers.append(c)
        best = max(S.MODEL_ORDER,
                   key=lambda k: float(cell.loc[(k, split)].macro_f1_mean))
        for mi, model in enumerate(S.MODEL_ORDER):
            draw(c + (mi - 1.5) * width, model, split, label=model == best)

    ax.set_xticks(centers)
    ax.set_xticklabels(
        ["Dummy baselines\n(stratified CV)", "Stratified\n(optimistic)",
         "Scaffold-grouped\n(primary)", "Receptor-grouped\n(unseen receptor)"],
    )
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Macro-F1 (mean ± SD, 25 fits)")
    ax.set_ylim(0, 0.72)
    ax.set_xlim(-0.45, centers[-1] + 0.5)
    S.ygrid(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=S.MODEL_COLOR[k])
               for k in S.MODEL_ORDER + DUMMIES]
    labels = [S.MODEL_LABEL[k] for k in S.MODEL_ORDER + DUMMIES]
    handles.append(plt.Rectangle((0, 0), 1, 1, fill=False,
                                 edgecolor=S.TEXT_PRIMARY, linewidth=1.3))
    labels.append("Selected model")
    ax.legend(handles, labels, ncol=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), columnspacing=1.2)
    facts["selected"] = f"{S.MODEL_LABEL[chosen]} on {S.SPLIT_LABEL[primary]}"
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
