"""Block-wise structural ablation (replaces the pre-fix single-block ablation)."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from . import _style as S
from ._data import Artefacts

NAME = "fig08_ablation"
TITLE = "Block-wise ablation of the docking arm"
DESCRIPTION = """
Contribution of each structural block to macro-F1, measured as full model −
model without the block (positive = the block helps), for the selected learner
under each split (`scripts/ablation_by_block.py`; the learner, seed list and
source matrix are recorded in the run's `ablation_by_block.meta.json` and
reported in the facts below). Blocks: receptor-dependent columns (docking
energetics, ProLIF contacts and contact counts; need a receptor to compute),
ligand docked-pose 3D shape (computable from the ligand alone) and both
together.

The comparison is paired: within every (split, seed, fold) all four variants
are fitted on identical rows with an identical model seed, so each
contribution is a difference on the same data rather than a gap between two
independently noisy means. Each dot is the mean contribution over seeds and
the horizontal bar is its two-sided 95% confidence interval, computed on the
per-seed paired differences; a bar crossing the zero line is a block whose
contribution this design cannot resolve from zero. The grey band is ±1 SD of
the full model across seeds.

Grouped folds are reshuffled per seed (`make_grouped_cv(shuffle=True)`).
Before that, `GroupKFold` returned one fixed partition whatever seed it was
given, so every grouped seed was the same measurement repeated and the spread
shown here was exactly zero — a delta compared against zero has not been
compared at all (HAL-6168). This replaces the 2026-08-25 single-block
ablation, whose earlier version reported the pre-leakage-fix Δ = +0.041.
"""
SOURCES = [
    "processed/ml_models/extras/ablation_by_block.csv",
    "processed/ml_models/extras/ablation_by_block_summary.csv",
    "processed/ml_models/extras/ablation_by_block.meta.json",
]

SPLITS = ["receptor_grouped", "scaffold_grouped", "stratified"]
VARIANTS = [
    ("drop_receptor_dependent", "Receptor-dependent", S.BLUE),
    ("drop_ligand_3d_shape", "Ligand pose 3D shape", S.ORANGE),
    ("drop_both", "Both blocks", S.AQUA),
]


def build(art: Artefacts):
    df = art.csv(SOURCES[0])
    summary = art.csv(SOURCES[1]).set_index(["split", "variant"])
    meta = art.json(SOURCES[2])
    nfeat = df.groupby("variant").n_features.first()
    full_n = int(nfeat["full"])
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    facts = {
        "learner / source matrix": (
            f"{meta['model']} / {meta['dataset']} "
            f"(sha256 {meta['dataset_sha256'][:12]})"
        ),
        "design": (
            f"{len(meta['seeds'])} seeds x {meta['folds']}-fold CV, paired "
            f"within (split, seed, fold); grouped folds shuffled per seed"
        ),
    }
    offsets = {v: o for (v, _, _), o in zip(VARIANTS, (0.22, 0.0, -0.22))}
    for yi, split in enumerate(SPLITS):
        y = len(SPLITS) - 1 - yi
        d = df[df.split == split]
        full = d[d.variant == "full"].set_index("seed").macro_f1
        band = full.std(ddof=1)
        ax.add_patch(plt.Rectangle((-band, y - 0.38), 2 * band, 0.76,
                                   color=S.GREY_WASH, zorder=0, linewidth=0))
        facts[f"{S.SPLIT_LABEL[split]} full model"] = (
            f"{full.mean():.4f} ± {band:.4f} (seed SD)"
        )
        for v, lab, col in VARIANTS:
            drop = d[d.variant == v].set_index("seed").macro_f1
            per_seed = (full - drop.reindex(full.index))
            m = float(per_seed.mean())
            yy = y + offsets[v]
            # The summary stores the delta as (without block - full); the
            # contribution plotted here is the other sign, so the interval
            # flips with it.
            row = summary.loc[(split, v)]
            lo = -float(row.delta_ci95_hi)
            hi = -float(row.delta_ci95_lo)
            resolved = bool(row.delta_excludes_zero)
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.2, alpha=0.55,
                    zorder=2, solid_capstyle="round")
            ax.scatter(m, yy, s=36, color=col, edgecolor=S.SURFACE, linewidth=1.2,
                       zorder=3)
            ax.text(max(hi, m) + 0.0012, yy, f"{m:+.4f}", va="center",
                    fontsize=6.6, color=S.TEXT_PRIMARY, bbox=S.LABEL_BOX, zorder=5)
            facts[f"{S.SPLIT_LABEL[split]} / {lab}"] = (
                f"{m:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                f"({'resolved from zero' if resolved else 'within noise'}; "
                f"without block {drop.mean():.4f}; {int(nfeat[v])} features)"
            )
    ax.axvline(0, color=S.TEXT_PRIMARY, lw=0.8, zorder=1)
    ax.set_yticks(range(len(SPLITS)))
    ax.set_yticklabels(["Stratified\n(optimistic)", "Scaffold-grouped",
                        "Receptor-grouped\n(unseen receptor)"])
    ax.tick_params(axis="y", length=0)
    lim = 0.032
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.55, len(SPLITS) - 0.45)
    ax.set_xlabel("Contribution to macro-F1 (full − without block; positive = block helps)")
    S.xgrid(ax)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", ms=6,
                          markeredgecolor=S.SURFACE)
               for _, _, c in VARIANTS]
    labels = [f"{lab} ({full_n - int(nfeat[v])} cols)" for v, lab, _ in VARIANTS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=S.GREY_WASH))
    labels.append("±1 SD of full model across seeds")
    handles.append(plt.Line2D([], [], color=S.GREY, lw=1.2))
    labels.append("95% CI of the paired contribution")
    ax.legend(handles, labels, ncol=2, loc="upper center",
              bbox_to_anchor=(0.45, -0.17))
    facts["columns: full / receptor-dependent / ligand 3D shape / both"] = (
        f"{full_n} / {full_n - int(nfeat['drop_receptor_dependent'])} / "
        f"{full_n - int(nfeat['drop_ligand_3d_shape'])} / "
        f"{full_n - int(nfeat['drop_both'])}"
    )
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
