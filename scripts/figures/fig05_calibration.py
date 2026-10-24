"""Per-class reliability diagrams for the calibrated final model on the holdout."""

from __future__ import annotations

import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder

from . import _style as S
from ._data import Artefacts

NAME = "fig05_calibration"
TITLE = "Per-class reliability diagrams, calibrated random forest, temporal holdout"
DESCRIPTION = """
One-vs-rest reliability diagrams for the isotonic-calibrated random forest on
the temporal holdout. Holdout probabilities are not persisted, so they are
recomputed by loading the saved calibrated model and scoring the holdout
parquet; the build asserts that the recomputed class predictions match
`holdout_predictions.csv` row for row and that the mean Brier score matches
`model_meta.json`. Each class's predictions are split into eight
equal-count (quantile) bins, because the calibrated probabilities occupy a
narrow range and equal-width bins leave most of the axis empty; the shaded band is the 2.5–97.5 % range of the per-bin
observed frequency over 1,000 bootstrap resamples of the holdout. Each bin
holds about 21 of the 172 holdout rows. Bins with fewer than three
samples are not drawn. Axes are zoomed per class to the range the
predictions occupy (the diagonal is still y = x), so the rare-class panels
are legible.
"""
SOURCES = [
    "processed/ml_models/random_forest_final_calibrated.joblib",
    "processed/ml_ready_dataset.parquet",
    "holdout/dataset_holdout.parquet",
    "processed/ml_models/holdout_predictions.csv",
    "processed/ml_models/model_meta.json",
]

N_BINS = 8
N_BOOT = 1000
MIN_BIN = 3


def holdout_probabilities(art: Artefacts):
    """Recompute calibrated holdout probabilities exactly as model_training did."""
    from cancerag.ml.preprocessing import get_X_y_groups

    model = joblib.load(art.path(SOURCES[0]))
    tr, ho = art.train, art.holdout
    le = LabelEncoder().fit(tr["bias_category"].astype(str))
    assert list(le.classes_) == art.model_meta["label_classes"]
    X, _, _, _, _ = get_X_y_groups(tr, label_encoder=le)
    hX, hy, _, _, _ = get_X_y_groups(ho, label_encoder=le)
    for c in X.columns:
        if c not in hX.columns:
            hX[c] = 0.0
    hX = hX[X.columns]
    proba = model.predict_proba(hX)
    ref = art.csv(SOURCES[3])
    model_pred = le.classes_[model.predict(hX)]
    if not (model_pred == ref.y_pred.values).all():
        raise AssertionError("recomputed predictions differ from holdout_predictions.csv")
    return proba, hy, list(le.classes_)


def build(art: Artefacts):
    proba, y, classes = holdout_probabilities(art)
    brier = {c: brier_score_loss((y == i).astype(int), proba[:, i])
             for i, c in enumerate(classes)}
    mean_brier = float(np.mean(list(brier.values())))
    ref_brier = art.model_meta["holdout_metrics"]["mean_brier"]
    if abs(mean_brier - ref_brier) > 1e-9:
        raise AssertionError(f"Brier {mean_brier} != model_meta {ref_brier}")

    rng = np.random.default_rng(42)
    order = ["G protein", "β Arrestin", "ERK", "G protein selectivity"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35))
    facts = {"mean Brier": f"{mean_brier:.3f}"}
    for ax, cname, letter in zip(axes, order, "abcd"):
        i = classes.index(cname)
        p, t = proba[:, i], (y == i).astype(int)
        edges = np.unique(np.quantile(p, np.linspace(0, 1, N_BINS + 1)))
        nb = len(edges) - 1
        b = np.clip(np.digitize(p, edges[1:-1], right=True), 0, nb - 1)
        mp, fp, cnt, lo, hi = [], [], [], [], []
        boot = np.empty((N_BOOT, nb))
        idx = rng.integers(0, len(p), size=(N_BOOT, len(p)))
        for r in range(N_BOOT):
            bb, tt = b[idx[r]], t[idx[r]]
            for k in range(nb):
                m = bb == k
                boot[r, k] = tt[m].mean() if m.sum() >= MIN_BIN else np.nan
        for k in range(nb):
            m = b == k
            if m.sum() < MIN_BIN:
                continue
            mp.append(p[m].mean())
            fp.append(t[m].mean())
            cnt.append(int(m.sum()))
            lo.append(np.nanpercentile(boot[:, k], 2.5))
            hi.append(np.nanpercentile(boot[:, k], 97.5))
        col = S.CLASS_COLOR[cname]
        ax.plot([0, 1], [0, 1], color=S.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.fill_between(mp, lo, hi, color=col, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(mp, fp, color=col, lw=1.6, zorder=3)
        ax.scatter(mp, fp, s=22, color=col, edgecolor=S.SURFACE, linewidth=1.0,
                   zorder=4)
        lim = float(min(1.0, np.ceil(max(max(mp), np.nanmax(hi)) * 1.1 * 10) / 10))
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ticks = np.linspace(0, lim, 3)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_title(f"{S.CLASS_LABEL[cname]}\nn = {int(t.sum())} of 172, Brier {brier[cname]:.3f}",
                     fontsize=S.FS_SMALL, fontweight="normal", color=S.TEXT_PRIMARY)
        ax.set_xlabel("Predicted probability")
        S.panel_label(ax, letter, x=-0.12, y=1.12)
        facts[S.CLASS_LABEL[cname]] = (
            f"n_pos={int(t.sum())}, Brier={brier[cname]:.3f}, bins drawn={len(mp)} "
            f"(rows per bin {min(cnt)}–{max(cnt)})"
        )
    axes[0].set_ylabel("Observed frequency")
    fig.tight_layout(w_pad=1.2)
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
