"""Leave-one-receptor-out macro-F1 for the five most-tested receptors."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from . import _style as S
from ._data import Artefacts, receptor_label

NAME = "fig09_loro"
TITLE = "Leave-one-receptor-out macro-F1, five most-tested receptors"
DESCRIPTION = """
Macro-F1 of the selected random forest trained on every other train-eligible
receptor and tested on the held-out one (single seed 42,
`reviewer_extras.loro_top_receptors`), for the five receptors with most
train-eligible rows. Reference lines: the random forest's pooled
receptor-grouped CV macro-F1 and the majority-class floor.
"""
SOURCES = [
    "processed/ml_models/extras/loro_results.csv",
    "processed/ml_models/cv_results_summary.csv",
    "processed/binding_sites.json",
]


def build(art: Artefacts):
    lo = art.csv(SOURCES[0]).sort_values("macro_f1", ascending=True)
    summ = art.csv(SOURCES[1]).set_index(["model", "split"])
    pooled = float(summ.loc[("random_forest", "receptor_kfold")].macro_f1_mean)
    floor = float(summ.loc[("majority_class", "stratified_kfold")].macro_f1_mean)

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    ys = range(len(lo))
    ax.barh(list(ys), lo.macro_f1, height=0.55, color=S.BLUE, zorder=2)
    for y, (_, r) in zip(ys, lo.iterrows()):
        ax.text(r.macro_f1 + 0.01, y, f"{r.macro_f1:.3f}", va="center",
                fontsize=S.FS_SMALL, color=S.TEXT_PRIMARY, bbox=S.LABEL_BOX,
                zorder=5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{receptor_label(art, r.held_out_receptor)}  "
                        f"({r.held_out_receptor}, n = {r.n_test})"
                        for _, r in lo.iterrows()])
    ax.tick_params(axis="y", length=0)
    for x, lab in ((pooled, f"pooled receptor-grouped CV {pooled:.3f}"),
                   (floor, f"majority floor {floor:.3f}")):
        ax.axvline(x, color=S.TEXT_SECONDARY, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(x + (0.006 if x == pooled else -0.006), len(lo) - 0.35, lab,
                fontsize=6.4, color=S.TEXT_SECONDARY,
                ha="left" if x == pooled else "right", va="bottom")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(-0.6, len(lo) + 0.1)
    ax.set_xlabel("Held-out-receptor macro-F1")
    S.xgrid(ax)
    facts = {f"{receptor_label(art, r.held_out_receptor)} ({r.held_out_receptor})":
             f"{r.macro_f1:.4f} (n_test={r.n_test}, n_train={r.n_train})"
             for _, r in lo.iloc[::-1].iterrows()}
    facts["pooled RF receptor-grouped CV"] = f"{pooled:.3f}"
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
