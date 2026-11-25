"""Feature-selector × learner ablation, with the seed spread a ranking needs."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig10_selector_ablation"
TITLE = "Feature-selector × learner ablation"
DESCRIPTION = """
(a) Macro-F1 for each feature selector (Boruta, L1-logistic regression, mutual
information, RFECV) paired with each learner, run with force-keeping disabled
so the selectors compete on equal terms. Each cell is the mean over seeds of
that seed's complete cross-validated estimate, ± 1 SD across seeds. (b) The
same scores pooled across learners within each seed: the dot is the mean over
seeds, the bar its two-sided 95% confidence interval across seeds, and the
small markers are the four individual learner means. A selector pair whose
*paired* per-seed difference excludes zero is named in the facts below; a pair
whose interval straddles zero is one this design cannot order, however far
apart its two means look.

Protocol: scaffold-grouped 3-fold CV, five seeds (42, 7, 13, 101, 2029), folds
reshuffled per seed, paired within (seed, fold) so every selector sees
identical rows and an identical model seed
(`scripts/run_selector_ablation.py --seeds … --shuffle-folds`, then
`scripts/selector_ranking_stats.py`).

This replaces the 2026-05-03 artefact, which predated both the per-receptor
QC-column fix and the receptor repairs and was run at one seed on folds that
`GroupKFold` returned unshuffled whatever seed it was given, so its spread was
fold-to-fold inside a single fixed partition (HAL-6168). The selector ordering
is not the one that artefact reported.
"""
SOURCES = [
    "processed/ml_models/selector_ablation_shuffled/selector_ablation_ranking.csv",
    "processed/ml_models/selector_ablation_shuffled/selector_ablation_contrasts.csv",
    "processed/ml_models/selector_ablation_shuffled/selector_ablation_long.csv",
    "processed/ml_models/selector_ablation_shuffled/selector_ablation_summary.csv",
]

SELECTORS = ["boruta", "rfecv", "mutual_info", "l1_logreg"]
SEL_LABEL = {"boruta": "Boruta", "rfecv": "RFECV", "mutual_info": "Mutual info",
             "l1_logreg": "L1-logreg"}


def build(art: Artefacts):
    rank = art.csv(SOURCES[0])
    cont = art.csv(SOURCES[1])
    long = art.csv(SOURCES[2])
    summ = art.csv(SOURCES[3])

    pooled = rank[rank.scope == "pooled"].set_index("selector")
    per_model = rank[rank.scope != "pooled"]
    sels = [s for s in SELECTORS if s in set(pooled.index)]
    # Means and seed-level SDs per cell come from the ranking artefact, whose
    # replication unit is the seed; `summary.csv`'s SD pools seeds and folds
    # together and is not the spread a ranking has to survive.
    m = per_model.pivot(index="selector", columns="scope",
                        values="mean").reindex(sels)[S.MODEL_ORDER]
    sd = per_model.pivot(index="selector", columns="scope",
                         values="sd").reindex(sels)[S.MODEL_ORDER]
    nfeat = long.groupby("selector").n_features_selected.median()
    n_seeds = int(pooled.n_seeds.max())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9),
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
    cb.set_label(f"Macro-F1 (mean of {n_seeds} seeds)", fontsize=6.8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.4)
    S.panel_label(ax, "a", x=-0.32, y=1.02)

    # Panel b: the pooled estimate with the interval a ranking lives or dies on.
    order = [s for s in pooled.sort_values("mean", ascending=False).index if s in sels]
    ys = np.arange(len(order))[::-1]
    for y, s in zip(ys, order):
        row = pooled.loc[s]
        bx.plot([row.ci95_lo, row.ci95_hi], [y, y], color=S.GREY, lw=1.4, zorder=1,
                solid_capstyle="round")
        for k in S.MODEL_ORDER:
            bx.scatter(m.loc[s, k], y, s=11, color=S.MODEL_COLOR[k], zorder=2,
                       edgecolor=S.SURFACE, linewidth=0.5, alpha=0.9)
        bx.scatter(row["mean"], y, s=40, marker="|", color=S.TEXT_PRIMARY,
                   linewidth=1.8, zorder=3)
        # Past the rightmost mark on this row, not past the interval: a learner
        # mean can sit outside the pooled CI (random forest's 0.459 against a
        # pooled upper bound of 0.441) and would print under the label.
        bx.text(max(float(row.ci95_hi), float(m.loc[s].max())) + 0.006, y,
                f"{row['mean']:.3f}", va="center",
                fontsize=6.6, color=S.TEXT_PRIMARY)
    bx.set_yticks(ys)
    bx.set_yticklabels([SEL_LABEL[s] for s in order])
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Macro-F1 (pooled over learners)")
    lo_lim = min(float(m.values.min()), float(pooled.ci95_lo.min()))
    hi_lim = max(float(m.values.max()), float(pooled.ci95_hi.max()))
    bx.set_xlim(lo_lim - 0.015, hi_lim + 0.05)
    bx.set_ylim(-0.7, len(order) - 0.3)
    S.xgrid(bx)
    handles = [plt.Line2D([], [], ls="none", marker="o", ms=3.4, color=S.MODEL_COLOR[k])
               for k in S.MODEL_ORDER]
    handles.append(plt.Line2D([], [], ls="none", marker="|", ms=8, mew=1.8,
                              color=S.TEXT_PRIMARY))
    handles.append(plt.Line2D([], [], color=S.GREY, lw=1.4))
    bx.legend(handles,
              [S.MODEL_LABEL[k] for k in S.MODEL_ORDER]
              + ["pooled mean", f"95% CI over {n_seeds} seeds"],
              loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3, fontsize=6.4,
              columnspacing=0.8)
    S.panel_label(bx, "b", x=-0.3, y=1.02)

    cp = cont[cont.scope == "pooled"].set_index(["selector_a", "selector_b"])
    adjacent = {}
    for a, b in zip(order, order[1:]):
        r = cp.loc[(a, b)]
        adjacent[f"{SEL_LABEL[a]} − {SEL_LABEL[b]}"] = (
            f"{r.delta:+.4f}, 95% CI [{r.delta_ci95_lo:+.4f}, {r.delta_ci95_hi:+.4f}] "
            f"({'resolved from zero' if r.delta_excludes_zero else 'within noise'})"
        )
    resolved = sorted(
        f"{SEL_LABEL[r.selector_a]} > {SEL_LABEL[r.selector_b]} "
        f"({r.delta:+.4f} [{r.delta_ci95_lo:+.4f}, {r.delta_ci95_hi:+.4f}])"
        for r in cp.reset_index().itertuples()
        if r.delta_excludes_zero and r.delta > 0
    )
    n_pairs = len(order) * (len(order) - 1) // 2
    facts = {
        "design": (
            f"{n_seeds} seeds x {int(summ.n.max() / n_seeds)}-fold scaffold-grouped CV, "
            f"folds reshuffled per seed, paired within (seed, fold); "
            f"{int(summ.n.sum())} fits, "
            f"{int(summ.n.sum() - summ.n_not_fallback.sum())} fallbacks"
        ),
        "pooled ranking (mean ± SD over seeds, 95% CI)": {
            SEL_LABEL[s]: (
                f"{pooled.loc[s, 'mean']:.4f} ± {pooled.loc[s, 'sd']:.4f} "
                f"[{pooled.loc[s, 'ci95_lo']:.4f}, {pooled.loc[s, 'ci95_hi']:.4f}]"
            ) for s in order},
        "adjacent pairs (paired per-seed difference)": adjacent,
        "pairs resolved from zero": (
            f"{len(resolved)} of {n_pairs}: " + "; ".join(resolved) if resolved
            else f"0 of {n_pairs}: no selector pair is resolved from zero"
        ),
        "cells (mean ± SD over seeds)": {
            f"{SEL_LABEL[s]} × {S.MODEL_LABEL[k]}":
            f"{m.loc[s, k]:.3f} ± {sd.loc[s, k]:.3f}"
            for s in sels for k in S.MODEL_ORDER},
        "best cell": f"{SEL_LABEL[sels[best[0]]]} × {S.MODEL_LABEL[m.columns[best[1]]]} "
                     f"= {m.values[best]:.3f}",
        "median features selected": {
            SEL_LABEL[s]: int(nfeat[s]) for s in sels},
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
