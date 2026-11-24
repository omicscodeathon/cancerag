"""Block-wise structural ablation, against the noise floor that judges it."""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt

from . import _style as S
from ._data import Artefacts

NAME = "fig08_ablation"
TITLE = "Removing the structural arm costs nothing"
DESCRIPTION = """
Does receptor-derived structure earn its place? (a) Contribution of each
structural block to four-class macro-F1, measured as full model − model
without the block (positive = the block helps), for the selected learner over
3 seeds × 3 folds under each split (`scripts/ablation_by_block.py`). Blocks:
receptor-dependent columns (docking energetics, ProLIF contacts and contact
counts; need a receptor to compute), ligand docked-pose 3D shape (computable
from the ligand alone) and both together. The grey band is the fold-to-fold
spread of the same learner under the same split in the independent bake-off
(`cv_results_summary.csv`, 25 fits), the noise any contribution has to clear;
every contribution sits inside it. The coloured bar is the range across the
three ablation seeds, which is zero on both grouped splits because
``GroupKFold`` accepts no shuffle — seeds change the model, not the folds, and
on these data they change nothing. (b) The same question on the well-posed
binary contrast (G protein vs β-arrestin, n = 316), where four feature views
are scored under each split. Dropping every structural column raises
receptor-held-out macro-F1 rather than lowering it, and dropping the 243
receptor-qualified contact columns together with the two fingerprint status
flags — 245 columns — changes nothing at all: that view is identical to the
full matrix to sixteen decimal places under every split.
"""
SOURCES = [
    "processed/ml_models/extras/ablation_by_block.csv",
    "processed/ml_models/cv_results_summary.csv",
    "processed/ml_models/binary_erk_splits.json",
    "processed/ml_models/selection_decision.json",
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
    df = art.csv(SOURCES[0])
    summary = art.csv(SOURCES[1]).set_index(["model", "split"])
    winner = art.selected_model
    nfeat = df.groupby("variant").n_features.first()
    full_n = int(nfeat["full"])
    offsets = {v: o for (v, _, _), o in zip(VARIANTS, (0.24, 0.0, -0.24))}

    for yi, split in enumerate(SPLITS):
        y = len(SPLITS) - 1 - yi
        d = df[df.split == split]
        full = d[d.variant == "full"].set_index("seed").macro_f1
        band = float(summary.loc[(winner, BAKEOFF_SPLIT[split]), "macro_f1_std"])
        ax.add_patch(plt.Rectangle((-band, y - 0.42), 2 * band, 0.84,
                                   color=S.GREY_WASH, zorder=0, linewidth=0))
        facts[f"{S.SPLIT_LABEL[split]} full model"] = (
            f"{full.mean():.4f} (ablation, 3 seeds × 3 folds); "
            f"fold-to-fold SD {band:.4f} ({winner}, bake-off, 25 fits)"
        )
        for v, lab, col in VARIANTS:
            drop = d[d.variant == v].set_index("seed").macro_f1
            per_seed = full - drop.reindex(full.index)
            m = float(per_seed.mean())
            yy = y + offsets[v]
            ax.plot([per_seed.min(), per_seed.max()], [yy, yy], color=col,
                    lw=1.4, alpha=0.55, zorder=2, solid_capstyle="round")
            ha, dx = ("left", 0.0022) if m >= 0 else ("right", -0.0022)
            edge = per_seed.max() if m >= 0 else per_seed.min()
            ax.scatter(m, yy, s=34, color=col, edgecolor=S.SURFACE, linewidth=1.1,
                       zorder=3)
            ax.text(edge + dx, yy, f"{m:+.4f}", va="center", ha=ha, fontsize=6.4,
                    color=S.TEXT_PRIMARY, bbox=S.LABEL_BOX, zorder=5)
            facts[f"{S.SPLIT_LABEL[split]} / {lab}"] = (
                f"{m:+.4f} (without block {drop.mean():.4f}; "
                f"{int(nfeat[v])} features; per-seed "
                f"{', '.join(f'{x:+.4f}' for x in per_seed.values)})"
            )
    ax.axvline(0, color=S.TEXT_PRIMARY, lw=0.8, zorder=1)
    ax.set_yticks(range(len(SPLITS)))
    ax.set_yticklabels(["Stratified\n(optimistic)", "Scaffold-grouped\n(unseen chemistry)",
                        "Receptor-grouped\n(unseen receptor)"])
    ax.tick_params(axis="y", length=0)
    lim = 0.125
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.6, len(SPLITS) - 0.4)
    ax.set_xlabel("Contribution to four-class macro-F1\n"
                  "(full − without block; positive = the block helps)")
    S.xgrid(ax)
    handles = [plt.Line2D([], [], color=c, marker="o", ls="none", ms=5.5,
                          markeredgecolor=S.SURFACE) for _, _, c in VARIANTS]
    labels = [f"{lab} ({full_n - int(nfeat[v])} cols)" for v, lab, _ in VARIANTS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=S.GREY_WASH))
    labels.append("±1 SD fold-to-fold (bake-off)")
    ax.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.46, -0.30),
              columnspacing=1.0)
    facts["columns: full / receptor-dependent / ligand 3D shape / both"] = (
        f"{full_n} / {full_n - int(nfeat['drop_receptor_dependent'])} / "
        f"{full_n - int(nfeat['drop_ligand_3d_shape'])} / "
        f"{full_n - int(nfeat['drop_both'])}"
    )
    facts["selected learner (selection_decision.json)"] = winner


def _panel_b(ax, art, facts):
    b = json.loads(art.path(SOURCES[2]).read_text())["binary"]
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
             "Seed-to-seed spread is zero on both grouped splits by construction: "
             "GroupKFold accepts no shuffle, so every seed sees the same folds.",
             fontsize=5.9, color=S.TEXT_MUTED, ha="left", va="top")
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
