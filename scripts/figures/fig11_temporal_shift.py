"""Train-eligible vs temporal-holdout distribution shift per feature."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import _style as S
from ._data import Artefacts

NAME = "fig11_temporal_shift"
TITLE = "Temporal shift: per-feature Wasserstein distance, train-eligible vs holdout"
DESCRIPTION = """
(a) The 20 features with the largest Wasserstein-1 distance between the
train-eligible and temporal-holdout marginals, as stored in
`temporal_shift.csv` (log scale). The raw distance is in each feature's own
units, so it ranks large-valued descriptors (Ipc, principal moments of
inertia) first. (b) The same distance divided by the feature's train-eligible
standard deviation (computed from `ml_ready_dataset.parquet`), a scale-free
ranking restricted to continuous features (more than two distinct values):
for sparse binary fingerprint bits the ratio is dominated by the tiny
standard deviation of rare bits and is not comparable. Features that need a docked
pose are tagged [pose].
"""
SOURCES = [
    "processed/ml_models/extras/temporal_shift.csv",
    "processed/ml_ready_dataset.parquet",
]
TOP = 20
POSE = {"Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
        "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex"}


def _lab(f: str) -> str:
    return f + ("  [pose]" if f in POSE else "")


def build(art: Artefacts):
    ts = art.csv(SOURCES[0]).sort_values("wasserstein_distance", ascending=False)
    tr = art.train
    num = {c: pd.to_numeric(tr[c], errors="coerce") for c in ts.feature if c in tr.columns}
    ts["train_sd"] = ts.feature.map({c: float(v.std()) for c, v in num.items()})
    ts["n_distinct"] = ts.feature.map({c: int(v.nunique()) for c, v in num.items()})
    ts["w_std"] = ts.wasserstein_distance / ts.train_sd
    continuous = np.isfinite(ts.w_std) & (ts.train_sd > 0) & (ts.n_distinct > 2)
    std_rank = ts[continuous].sort_values("w_std", ascending=False)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 4.2),
                                 gridspec_kw=dict(wspace=0.95))
    raw = ts.head(TOP)
    ys = np.arange(TOP)[::-1]
    ax.barh(ys, raw.wasserstein_distance, height=0.62, color=S.BLUE)
    ax.set_xscale("log")
    for y, v in zip(ys, raw.wasserstein_distance):
        ax.text(v * 1.25, y, f"{v:.3g}", va="center", fontsize=6.2,
                color=S.TEXT_PRIMARY)
    ax.set_xlim(raw.wasserstein_distance.min() / 2, raw.wasserstein_distance.max() * 40)
    ax.set_yticks(ys)
    ax.set_yticklabels([_lab(f) for f in raw.feature], fontsize=6.6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Wasserstein-1 distance\n(feature units, log scale)")
    S.xgrid(ax)
    S.panel_label(ax, "a", x=-0.55, y=1.01)

    st = std_rank.head(TOP)
    bx.barh(ys, st.w_std, height=0.62, color=S.AQUA)
    for y, v in zip(ys, st.w_std):
        bx.text(v + st.w_std.max() * 0.02, y, f"{v:.2f}", va="center",
                fontsize=6.2, color=S.TEXT_PRIMARY)
    bx.set_xlim(0, st.w_std.max() * 1.22)
    bx.set_yticks(ys)
    bx.set_yticklabels([_lab(f) for f in st.feature], fontsize=6.6)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("Wasserstein-1 / train SD\n(continuous features, scale-free)")
    S.xgrid(bx)
    S.panel_label(bx, "b", x=-0.55, y=1.01)

    facts = {
        "features scored": len(ts),
        "continuous features in panel b ranking": int(continuous.sum()),
        "top-5 raw": {r.feature: float(f"{r.wasserstein_distance:.4g}")
                      for _, r in raw.head(5).iterrows()},
        "top-5 standardised": {r.feature: round(float(r.w_std), 3)
                               for _, r in st.head(5).iterrows()},
        "overlap of raw and standardised top-20": len(set(raw.feature) & set(st.feature)),
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
