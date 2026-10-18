"""Label-free structure of the feature matrix: the bias classes do not cluster."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from . import _style as S
from ._data import Artefacts

NAME = "fig14_unsupervised_clustering"
TITLE = "Bias classes do not form natural unsupervised clusters"
DESCRIPTION = """
K-means on the standardised feature matrix of all 615 modelled rows
(`cancerag.ml.clustering`; the number of non-constant features used is read
from the artefact). (a) Silhouette score for k = 2–10, with the best k and
k = 4 (the number of classes) marked. (b) K-means inertia; no elbow. (c)
Cluster × class contingency at k = 4, coloured by each cluster's class
composition, with the adjusted Rand index and adjusted mutual information
against the bias labels. (d) PCA and (e) UMAP projections coloured by true
bias class. (f) The same UMAP with each k = 4 cluster highlighted in turn
(small multiples rather than a fifth-to-eighth colour). UMAP panels are
zoomed to the central cloud (5th–95th percentile plus a 25 % margin); the number of
outlying points left outside the view is printed. All values come from
the current (post-leakage-fix) clustering artefact.
"""
SOURCES = [
    "processed/ml_models/clustering/clustering_results.json",
    "processed/ml_models/clustering/cluster_assignments.csv",
]

ORDER = ["G protein", "β Arrestin", "G protein selectivity", "ERK"]


def _scatter_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _core_limits(x, y, lo=5.0, hi=95.0, pad=0.25):
    """Axis limits covering the central point cloud; returns limits and #outside."""
    xl, xh = np.percentile(x, [lo, hi])
    yl, yh = np.percentile(y, [lo, hi])
    dx, dy = (xh - xl) * pad, (yh - yl) * pad
    xl, xh, yl, yh = xl - dx, xh + dx, yl - dy, yh + dy
    outside = int(((x < xl) | (x > xh) | (y < yl) | (y > yh)).sum())
    return (xl, xh), (yl, yh), outside


def build(art: Artefacts):
    res = art.json(SOURCES[0])
    ca = art.csv(SOURCES[1])
    sw = res["sweep"]
    ks, sil, inert = sw["k_values"], sw["silhouettes"], sw["inertias"]
    kbest = res["k_star"]
    ext = res["external_agreement_k_classes"]
    pca = res["pca_explained_variance"]
    n_feat = res["n_features_used"]

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.42)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_d, ax_e = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    gsf = gs[1, 2].subgridspec(2, 2, hspace=0.35, wspace=0.12)

    # (a) silhouette
    ax_a.axhline(0, color=S.GREY, lw=0.6)
    ax_a.plot(ks, sil, color=S.BLUE, lw=1.5, zorder=2)
    ax_a.scatter(ks, sil, s=18, color=S.BLUE, edgecolor=S.SURFACE, linewidth=0.8, zorder=3)
    for k, lab, dy in ((kbest, f"best k = {kbest}\n{sil[ks.index(kbest)]:.4f}", 8),
                       (4, f"k = 4\n{sil[ks.index(4)]:.4f}", 8)):
        ax_a.annotate(lab, (k, sil[ks.index(k)]), xytext=(6, dy),
                      textcoords="offset points", fontsize=6.3,
                      color=S.TEXT_PRIMARY, va="bottom")
    ax_a.set_xticks(ks[::2])
    ax_a.set_xlabel("Number of clusters k")
    ax_a.set_ylabel("Silhouette score")
    ax_a.set_ylim(min(sil) - 0.01, max(sil) + 0.025)
    S.ygrid(ax_a)
    S.panel_label(ax_a, "a", x=-0.32)

    # (b) inertia
    ax_b.plot(ks, np.array(inert) / 1e6, color=S.GREY_DARK, lw=1.5)
    ax_b.scatter(ks, np.array(inert) / 1e6, s=18, color=S.GREY_DARK,
                 edgecolor=S.SURFACE, linewidth=0.8, zorder=3)
    ax_b.set_xticks(ks[::2])
    ax_b.set_xlabel("Number of clusters k")
    ax_b.set_ylabel("Inertia (×10⁶)")
    S.ygrid(ax_b)
    S.panel_label(ax_b, "b", x=-0.32)

    # (c) contingency
    cbc = res["cluster_by_class_k_classes"]
    clusters = sorted(cbc, key=int)
    mat = np.array([[cbc[c][k] for k in ORDER] for c in clusters])
    share = mat / mat.sum(axis=1, keepdims=True)
    ax_c.imshow(share, cmap=S.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax_c.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=6.8,
                      color=S.ink_on(S.cmap_hex(S.SEQ_CMAP, share[i, j])))
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(["G prot.", "β-arr.", "GPS", "ERK"], fontsize=6.6)
    ax_c.set_yticks(range(len(clusters)))
    ax_c.set_yticklabels([f"C{c} (n={mat[i].sum()})" for i, c in enumerate(clusters)],
                         fontsize=6.6)
    ax_c.tick_params(length=0)
    for sp in ax_c.spines.values():
        sp.set_visible(False)
    ax_c.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax_c.set_yticks(np.arange(-0.5, len(clusters), 1), minor=True)
    ax_c.grid(which="minor", color=S.SURFACE, linewidth=1.2)
    ax_c.tick_params(which="minor", length=0)
    ax_c.set_title(f"k = 4: ARI {ext['adjusted_rand_index']:+.3f}, "
                   f"AMI {ext['adjusted_mutual_info']:.3f}",
                   fontsize=6.8, fontweight="normal", color=S.TEXT_SECONDARY)
    S.panel_label(ax_c, "c", x=-0.42)

    # (d, e) PCA and UMAP by class
    for ax, (xc, yc), title, letter in (
        (ax_d, ("pca1", "pca2"), f"PCA (PC1 + PC2 = {sum(pca):.2%} of variance)", "d"),
        (ax_e, ("umap1", "umap2"), "UMAP, coloured by class", "e"),
    ):
        for cls in ORDER:
            d = ca[ca.true_class == cls]
            ax.scatter(d[xc], d[yc], s=7, color=S.CLASS_COLOR[cls], alpha=0.75,
                       edgecolor=S.SURFACE, linewidth=0.3,
                       label=f"{S.CLASS_LABEL[cls]} ({len(d)})")
        _scatter_axes(ax)
        if xc == "umap1":
            xlim, ylim, n_out = _core_limits(ca.umap1.values, ca.umap2.values)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.text(0.98, 0.02, f"{n_out} outlying points outside view",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                    color=S.TEXT_MUTED)
        ax.set_xlabel(xc.upper().replace("1", " 1"))
        ax.set_ylabel(yc.upper().replace("2", " 2"))
        ax.set_title(title, fontsize=6.8, fontweight="normal", color=S.TEXT_SECONDARY)
        S.panel_label(ax, letter, x=-0.1)
    ax_d.legend(loc="upper center", bbox_to_anchor=(1.1, -0.15), ncol=4,
                fontsize=6.6, markerscale=2.2, handletextpad=0.2, columnspacing=1.0)

    # (f) UMAP small multiples per cluster
    first = None
    for i, c in enumerate(clusters):
        ax = fig.add_subplot(gsf[i // 2, i % 2])
        first = first or ax
        other = ca[ca.cluster_k_classes != int(c)]
        mine = ca[ca.cluster_k_classes == int(c)]
        ax.scatter(other.umap1, other.umap2, s=2, color=S.GREY_LIGHT, linewidth=0)
        ax.scatter(mine.umap1, mine.umap2, s=3, color=S.TEXT_PRIMARY, linewidth=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"C{c} (n = {len(mine)})", fontsize=6.3, fontweight="normal",
                     color=S.TEXT_SECONDARY, pad=2)
        _scatter_axes(ax)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(S.GRID)
    S.panel_label(first, "f", x=-0.1, y=1.18)

    facts = {
        "n rows / features used": f"{res['n_samples']} / {n_feat}",
        "silhouette by k": {k: round(s, 4) for k, s in zip(ks, sil)},
        "best k (silhouette)": f"{kbest} ({sil[ks.index(kbest)]:.4f})",
        "silhouette at k = 4": f"{sil[ks.index(4)]:.4f}",
        "ARI / AMI at k = 4": f"{ext['adjusted_rand_index']:+.4f} / {ext['adjusted_mutual_info']:.4f}",
        "ARI / AMI at best k": (
            f"{res['external_agreement_k_star']['adjusted_rand_index']:+.4f} / "
            f"{res['external_agreement_k_star']['adjusted_mutual_info']:.4f}"),
        "UMAP points outside zoomed view": n_out,
        "PCA variance PC1 + PC2": f"{pca[0]:.4f} + {pca[1]:.4f} = {sum(pca):.2%}",
        "cluster × class (k = 4; G/βarr/GPS/ERK)": {f"C{c}": mat[i].tolist()
                                                    for i, c in enumerate(clusters)},
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
