"""Composition of the 615-row modelled dataset."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts, receptor_label

NAME = "fig01_dataset_overview"
TITLE = "Dataset composition"
DESCRIPTION = """
The 615 modelled rows (443 train-eligible, year < 2018; 172 temporal holdout,
year ≥ 2018). (a) Bias-class counts. (b) The 15 receptors with most rows,
split by bias class — many receptors carry a single class. (c) Publication
year, split at the 2018 temporal-holdout cutoff. (d) Class counts in the
train-eligible and holdout sets, with the train-eligible class share.
"""
SOURCES = [
    "processed/ml_ready_dataset.parquet",
    "holdout/dataset_holdout.parquet",
    "processed/binding_sites.json",
]

TRAIN_COL = S.GREY_DARK
HOLD_COL = S.GREY_LIGHT


def build(art: Artefacts):
    df = art.modelled
    plot_order = ["G protein", "β Arrestin", "G protein selectivity", "ERK"]
    counts = df.bias_category.value_counts()

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.55, wspace=0.55)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (a) class counts
    ys = np.arange(len(plot_order))[::-1]
    vals = [int(counts.get(c, 0)) for c in plot_order]
    ax_a.barh(ys, vals, height=0.6, color=[S.CLASS_COLOR[c] for c in plot_order])
    for y, v in zip(ys, vals):
        ax_a.text(v + 6, y, f"{v} ({v / len(df):.1%})", va="center",
                  fontsize=S.FS_SMALL, color=S.TEXT_PRIMARY)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([S.CLASS_LABEL[c] for c in plot_order])
    ax_a.tick_params(axis="y", length=0)
    ax_a.set_xlim(0, max(vals) * 1.45)
    ax_a.set_xlabel("Rows (n = 615)")
    S.xgrid(ax_a)
    S.panel_label(ax_a, "a", x=-0.42)

    # (b) top-15 receptors stacked by class
    top = df.receptor_uniprot.value_counts().head(15)
    ct = (df[df.receptor_uniprot.isin(top.index)]
          .groupby(["receptor_uniprot", "bias_category"]).size().unstack(fill_value=0)
          .reindex(top.index))
    yb = np.arange(len(top))[::-1]
    left = np.zeros(len(top))
    for c in plot_order:
        v = ct.get(c, 0)
        v = v.values if hasattr(v, "values") else np.zeros(len(top))
        ax_b.barh(yb, v, left=left, height=0.7, color=S.CLASS_COLOR[c],
                  edgecolor=S.SURFACE, linewidth=0.8, label=S.CLASS_LABEL[c])
        left += v
    for y, tot in zip(yb, top.values):
        ax_b.text(tot + 1.5, y, str(int(tot)), va="center", fontsize=6.6,
                  color=S.TEXT_PRIMARY)
    ax_b.set_yticks(yb)
    ax_b.set_yticklabels([receptor_label(art, u) for u in top.index], fontsize=6.8)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlabel("Rows")
    ax_b.set_xlim(0, top.max() * 1.12)
    S.xgrid(ax_b)
    ax_b.legend(loc="lower right", fontsize=6.6)
    S.panel_label(ax_b, "b", x=-0.36)
    n_single = int((ct.gt(0).sum(axis=1) == 1).sum())

    # (c) year histogram
    yrs = df.year.astype(int)
    bins = np.arange(yrs.min(), yrs.max() + 2) - 0.5
    ax_c.hist(yrs[yrs < 2018], bins=bins, color=TRAIN_COL, edgecolor=S.SURFACE,
              linewidth=0.6, label="train-eligible (< 2018)")
    ax_c.hist(yrs[yrs >= 2018], bins=bins, color=HOLD_COL, edgecolor=S.SURFACE,
              linewidth=0.6, label="temporal holdout (≥ 2018)")
    ax_c.axvline(2017.5, color=S.TEXT_PRIMARY, lw=0.8, ls=(0, (3, 2)))
    ax_c.set_ylim(0, yrs.value_counts().max() * 1.35)
    ax_c.text(2017.0, ax_c.get_ylim()[1] * 0.62, "2018 cutoff", ha="right",
              va="center", rotation=90, fontsize=6.6, color=S.TEXT_SECONDARY)
    ax_c.set_xlabel("Publication year")
    ax_c.set_ylabel("Rows")
    ax_c.legend(loc="upper left", fontsize=6.6)
    S.ygrid(ax_c)
    S.panel_label(ax_c, "c", x=-0.2)

    # (d) train vs holdout per class
    tr = df[df.split == "train-eligible"].bias_category.value_counts()
    ho = df[df.split == "temporal holdout"].bias_category.value_counts()
    x = np.arange(len(plot_order))
    w = 0.38
    trv = [int(tr.get(c, 0)) for c in plot_order]
    hov = [int(ho.get(c, 0)) for c in plot_order]
    ax_d.bar(x - w / 2, trv, w, color=TRAIN_COL, edgecolor=S.SURFACE, linewidth=0.8,
             label=f"train-eligible (n = {sum(trv)})")
    ax_d.bar(x + w / 2, hov, w, color=HOLD_COL, edgecolor=S.SURFACE, linewidth=0.8,
             label=f"temporal holdout (n = {sum(hov)})")
    for xi, a, b in zip(x, trv, hov):
        ax_d.text(xi - w / 2, a + 4, f"{a}\n{a / sum(trv):.1%}", ha="center",
                  va="bottom", fontsize=6.3, color=S.TEXT_PRIMARY)
        ax_d.text(xi + w / 2, b + 4, str(b), ha="center", va="bottom",
                  fontsize=6.3, color=S.TEXT_PRIMARY)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([S.CLASS_LABEL[c].replace("G-protein selectivity",
                                                   "G-protein\nselectivity")
                          for c in plot_order])
    ax_d.tick_params(axis="x", length=0)
    ax_d.set_ylabel("Rows")
    ax_d.set_ylim(0, max(trv) * 1.35)
    ax_d.legend(loc="upper right", fontsize=6.6)
    S.ygrid(ax_d)
    S.panel_label(ax_d, "d", x=-0.16)

    facts = {
        "rows / train-eligible / holdout": f"{len(df)} / {sum(trv)} / {sum(hov)}",
        "receptors / unique ligands": f"{df.receptor_uniprot.nunique()} / {df.inchikey.nunique()}",
        "class counts (all 615)": {S.CLASS_LABEL[c]: int(counts.get(c, 0)) for c in plot_order},
        "train-eligible class share": {S.CLASS_LABEL[c]: f"{tr.get(c, 0) / sum(trv):.1%}"
                                       for c in plot_order},
        "holdout class counts": {S.CLASS_LABEL[c]: int(ho.get(c, 0)) for c in plot_order},
        "top-15 receptors": {receptor_label(art, u): int(n) for u, n in top.items()},
        "single-class receptors among top 15": n_single,
        "year range": f"{yrs.min()}–{yrs.max()}",
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
