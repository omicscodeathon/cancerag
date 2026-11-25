"""Block-wise structural ablation, with the interval that judges each block."""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt

from . import _style as S
from ._data import Artefacts

NAME = "fig08_ablation"
TITLE = "The structural arm earns 0.02, and only on unseen chemistry"
DESCRIPTION = """
Does receptor-derived structure earn its place? (a) Contribution of each
structural block to four-class macro-F1, measured as full model − model
without the block (positive = the block helps), for the selected learner under
each split (`scripts/ablation_by_block.py`; the learner, seed list and source
matrix are recorded in the run's `ablation_by_block.meta.json` and reported in
the facts below). Blocks: receptor-dependent columns (docking energetics,
ProLIF contacts and contact counts; need a receptor to compute), ligand
docked-pose 3D shape (computable from the ligand alone) and both together.

The comparison is paired: within every (split, seed, fold) all four variants
are fitted on identical rows with an identical model seed, so each
contribution is a difference on the same data rather than a gap between two
independently noisy means. The coloured bar is its two-sided 95% confidence
interval over seeds; a bar crossing the zero line is a block this design
cannot resolve from zero. Grouped folds are reshuffled per seed
(`make_grouped_cv(shuffle=True)`); before that, `GroupKFold` returned one
fixed partition whatever seed it was given, every grouped seed was the same
measurement repeated, and the spread was exactly zero — a delta compared
against zero has not been compared at all (HAL-6168).

The independent bake-off's fold-to-fold spread for the same learner and split
(`cv_results_summary.csv`, 25 fits; 0.056, 0.112 and 0.054 for the stratified,
scaffold- and receptor-grouped splits) is reported in the facts rather than
drawn. It is the scatter of an absolute score, not of a paired difference, so
a contribution can lie well inside it and still be resolved from zero — as the
scaffold-grouped receptor-dependent block is — and at its width every interval
here would collapse onto the zero line. (b) The same question on the
well-posed
binary contrast (G protein vs β-arrestin, n = 316), where four feature views
are scored under each split. Dropping every structural column raises
receptor-held-out macro-F1 rather than lowering it, and dropping the 243
receptor-qualified contact columns together with the two fingerprint status
flags — 245 columns — changes nothing at all: that view is identical to the
full matrix to sixteen decimal places under every split.
"""
# Indexed positionally by both panels, so entries are appended, never inserted.
ABLATION, BAKEOFF, BINARY, DECISION, PAIRED, META = range(6)
SOURCES = [
    "processed/ml_models/extras/ablation_by_block.csv",
    "processed/ml_models/cv_results_summary.csv",
    "processed/ml_models/binary_erk_splits.json",
    "processed/ml_models/selection_decision.json",
    "processed/ml_models/extras/ablation_by_block_summary.csv",
    "processed/ml_models/extras/ablation_by_block.meta.json",
]

SPLITS = ["receptor_grouped", "scaffold_grouped", "stratified"]
# Ablation split name -> the bake-off's name for the same split.
BAKEOFF_SPLIT = {
    "stratified": "stratified_kfold",
    "scaffold_grouped": "scaffold_kfold",
    "receptor_grouped": "receptor_kfold",
}
VARIANTS = [
    ("drop_receptor_dependent", "Receptor-dependent", S.BLUE),
    ("drop_ligand_3d_shape", "Ligand pose 3D shape", S.ORANGE),
    ("drop_both", "Both blocks", S.AQUA),
]
VIEWS = [
    ("all", "All 5,511 features", S.GREY_DARK),
    ("bw_only", "− 245 receptor-qualified\nand status columns", S.BLUE),
    ("ligand_only", "Ligand only (no structure)", S.ORANGE),
    ("robust_11", "The 11 regime-robust features", S.AQUA),
]
BIN_SPLITS = ["stratified", "scaffold_grouped", "receptor_grouped"]


def _panel_a(ax, art, facts):
    df = art.csv(SOURCES[ABLATION])
    bakeoff = art.csv(SOURCES[BAKEOFF]).set_index(["model", "split"])
    # The paired intervals, from the run that produced the deltas. `summary`
    # stores each delta as (without block − full); the contribution plotted
    # here is the other sign, so its interval flips with it.
    paired = art.csv(SOURCES[PAIRED]).set_index(["split", "variant"])
    meta = art.json(SOURCES[META])
    winner = art.selected_model
    nfeat = df.groupby("variant").n_features.first()
    full_n = int(nfeat["full"])
    offsets = {v: o for (v, _, _), o in zip(VARIANTS, (0.24, 0.0, -0.24))}

    bounds = [abs(float(b)) for sp in SPLITS for v, _, _ in VARIANTS
              for b in (paired.loc[(sp, v), "delta_ci95_lo"],
                        paired.loc[(sp, v), "delta_ci95_hi"])]
    # The scale follows the intervals, and only the intervals. The bake-off's
    # fold-to-fold SD is about five times the widest of them, so drawing it as
    # a band would set a scale on which every interval collapses onto the zero
    # line — which is how a delta came to be reported against no comparison at
    # all. It stays in the caption and the facts, where it can be read.
    lim = max(bounds) * 1.5

    for yi, split in enumerate(SPLITS):
        y = len(SPLITS) - 1 - yi
        d = df[df.split == split]
        full = d[d.variant == "full"].set_index("seed").macro_f1
        band = float(bakeoff.loc[(winner, BAKEOFF_SPLIT[split]), "macro_f1_std"])
        facts[f"{S.SPLIT_LABEL[split]} full model"] = (
            f"{full.mean():.4f} ± {full.std(ddof=1):.4f} (ablation, "
            f"{len(meta['seeds'])} seeds × {meta['folds']} folds); fold-to-fold "
            f"SD {band:.4f} ({winner}, bake-off, 25 fits)"
        )
        for v, lab, col in VARIANTS:
            drop = d[d.variant == v].set_index("seed").macro_f1
            per_seed = full - drop.reindex(full.index)
            m = float(per_seed.mean())
            row = paired.loc[(split, v)]
            lo, hi = -float(row.delta_ci95_hi), -float(row.delta_ci95_lo)
            resolved = bool(row.delta_excludes_zero)
            yy = y + offsets[v]
            ax.plot([lo, hi], [yy, yy], color=col, lw=1.4, alpha=0.55,
                    zorder=2, solid_capstyle="round")
            ha, dx = ("left", 0.0022) if m >= 0 else ("right", -0.0022)
            ax.scatter(m, yy, s=34, color=col, edgecolor=S.SURFACE, linewidth=1.1,
                       zorder=3)
            ax.text((hi if m >= 0 else lo) + dx, yy, f"{m:+.4f}", va="center",
                    ha=ha, fontsize=6.4, color=S.TEXT_PRIMARY, bbox=S.LABEL_BOX,
                    zorder=5)
            facts[f"{S.SPLIT_LABEL[split]} / {lab}"] = (
                f"{m:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                f"({'resolved from zero' if resolved else 'within noise'}; "
                f"without block {drop.mean():.4f}; {int(nfeat[v])} features)"
            )
    ax.axvline(0, color=S.TEXT_PRIMARY, lw=0.8, zorder=1)
    ax.set_yticks(range(len(SPLITS)))
    ax.set_yticklabels(["Stratified\n(optimistic)", "Scaffold-grouped\n(unseen chemistry)",
                        "Receptor-grouped\n(unseen receptor)"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, len(SPLITS) - 0.4)
    ax.set_xlabel("Contribution to four-class macro-F1\n"
                  "(full − without block; positive = the block helps)")
    S.xgrid(ax)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", ms=5.5,
                          markeredgecolor=S.SURFACE) for _, _, c in VARIANTS]
    labels = [f"{lab} ({full_n - int(nfeat[v])} cols)" for v, lab, _ in VARIANTS]
    handles.append(plt.Line2D([], [], color=S.GREY, lw=1.4))
    labels.append("95% CI of the paired contribution")
    ax.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.46, -0.30),
              columnspacing=1.0)
    facts["design"] = (
        f"{len(meta['seeds'])} seeds × {meta['folds']}-fold CV on "
        f"{meta['dataset']} (sha256 {meta['dataset_sha256'][:12]}), paired "
        f"within (split, seed, fold); grouped folds reshuffled per seed"
    )
    facts["columns: full / receptor-dependent / ligand 3D shape / both"] = (
        f"{full_n} / {full_n - int(nfeat['drop_receptor_dependent'])} / "
        f"{full_n - int(nfeat['drop_ligand_3d_shape'])} / "
        f"{full_n - int(nfeat['drop_both'])}"
    )
    facts["selected learner (selection_decision.json)"] = winner


def _panel_b(ax, art, facts):
    b = json.loads(art.path(SOURCES[BINARY]).read_text())["binary"]
    rows = {(r["view"], r["regime"]): r for r in b["rows"]}
    floor = float(b["floor"])
    xs = range(len(BIN_SPLITS))
    styles = {
        "all": dict(marker="o", ms=6.5, lw=1.6, ls="-"),
        "bw_only": dict(marker="o", ms=11, lw=0, ls="none", mfc="none", mew=1.3),
        "ligand_only": dict(marker="o", ms=6.5, lw=1.6, ls="-"),
        "robust_11": dict(marker="o", ms=5.5, lw=1.3, ls=(0, (3, 2))),
    }
    for view, lab, col in VIEWS:
        ys = []
        for split in BIN_SPLITS:
            r = rows.get((view, split))
            ys.append(None if r is None else r["macro_f1"])
            if r is not None:
                facts[f"binary / {lab.replace(chr(10), ' ')} / {S.SPLIT_LABEL[split]}"] = (
                    f"macro-F1 {r['macro_f1']:.4f}, AUC {r['auc']:.4f}, "
                    f"MCC {r['mcc']:.4f}, {r['n']} columns"
                )
        ax.plot(list(xs), ys, color=col, label=lab, zorder=3,
                markeredgecolor=col if view == "bw_only" else S.SURFACE,
                **styles[view])
    # The two comparisons the panel exists to make.
    a_rec = rows[("all", "receptor_grouped")]["macro_f1"]
    l_rec = rows[("ligand_only", "receptor_grouped")]["macro_f1"]
    ax.annotate("", xy=(2.12, l_rec), xytext=(2.12, a_rec),
                arrowprops=dict(arrowstyle="-|>", color=S.TEXT_PRIMARY, lw=0.9,
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.text(2.18, (a_rec + l_rec) / 2,
            f"dropping all structure:\n{a_rec:.3f} → {l_rec:.3f}", fontsize=6.1,
            va="center", ha="left", color=S.TEXT_PRIMARY, zorder=5)
    ax.text(0.04, rows[("all", "stratified")]["macro_f1"] + 0.035,
            "the 245-column view sits exactly on the full matrix,\n"
            "under every split, to sixteen decimal places",
            fontsize=6.1, va="bottom", ha="left", color=S.TEXT_SECONDARY, zorder=5)
    ax.axhline(floor, color=S.TEXT_PRIMARY, lw=0.9, ls=(0, (4, 2)), zorder=2)
    ax.text(-0.08, floor - 0.018, f"majority-class floor {floor:.3f}", fontsize=6.1,
            ha="left", va="top", color=S.TEXT_SECONDARY, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Stratified", "Scaffold-\ngrouped", "Receptor-\ngrouped"])
    ax.set_xlim(-0.18, 3.35)
    ax.set_ylim(0.30, 0.90)
    ax.set_ylabel(f"Binary macro-F1 (G protein vs β-arrestin, n = {b['n']})")
    S.ygrid(ax)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              columnspacing=1.0, handlelength=1.8)
    facts["binary contrast"] = (
        f"n = {b['n']}, n_pos = {b.get('n_pos', '?')}, majority floor {floor:.4f}"
    )


def build(art: Artefacts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6),
                                   gridspec_kw={"width_ratios": [1.08, 1.0],
                                                "wspace": 0.44})
    facts: dict = {}
    _panel_a(ax1, art, facts)
    _panel_b(ax2, art, facts)
    S.panel_label(ax1, "a", x=-0.30)
    S.panel_label(ax2, "b", x=-0.14)
    fig.text(0.013, -0.255,
             "Grouped folds are reshuffled per seed, so the interval in (a) is "
             "resampling variance; before HAL-6168 GroupKFold returned one fixed "
             "partition and that spread was zero by construction.",
             fontsize=5.9, color=S.TEXT_MUTED, ha="left", va="top")
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
