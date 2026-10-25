"""Resolve the ERK signal properly.

The 100-draw null gave p = 0.0495 -- so close to the threshold that one more
extreme permutation would have put it at 0.059. A result that flips on a
single draw is not a result. 1,000 draws resolve p to 0.001 and give the null
distribution enough shape to quote.

G-protein selectivity is not re-run: at p = 0.42, with the observed value
below the null mean, more draws cannot change the conclusion.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from cancerag.ml.model_training import MODEL_FACTORIES  # noqa: E402
from cancerag.ml.preprocessing import (  # noqa: E402
    build_full_pipeline, get_X_y_groups, make_grouped_cv,
)

N_PERM = 1000
SEEDS = [42, 7, 13]
OUT = Path("data/processed/ml_models/erk_null_1000.json")

tr = pd.read_parquet("data/processed/ml_ready_dataset.parquet")
X, _, _, _, _ = get_X_y_groups(tr)
yt = (tr["bias_category"] == "ERK").astype(int).to_numpy()
print(f"ERK: {yt.sum()} positives of {len(yt)}; X {X.shape}")


def one(ys, seed):
    splits = make_grouped_cv(tr, group_col="receptor_uniprot",
                             n_splits=5, seed=seed)
    if not splits:
        return None
    sc = []
    for tri, tei in splits:
        if len(np.unique(ys[tri])) < 2:
            continue
        p = build_full_pipeline(
            MODEL_FACTORIES["lightgbm"](n_classes=2, random_state=seed),
            variance_threshold=0.0, correlation_threshold=1.01)
        p.fit(X.iloc[tri], ys[tri])
        sc.append(f1_score(ys[tei], p.predict(X.iloc[tei]),
                           average="macro", zero_division=0))
    return float(np.mean(sc)) if sc else None


obs = float(np.mean([v for v in (one(yt, s) for s in SEEDS) if v is not None]))
print(f"observed (mean over {len(SEEDS)} seeds): {obs:.4f}", flush=True)

rng = np.random.default_rng(0)
null = []
for i in range(N_PERM):
    v = one(rng.permutation(yt), 42)
    if v is not None:
        null.append(v)
    if (i + 1) % 100 == 0:
        a = np.array(null)
        print(f"  {i+1}/{N_PERM}  running p = "
              f"{((a >= obs).sum() + 1) / (len(a) + 1):.4f}", flush=True)

null = np.array(null)
p = float(((null >= obs).sum() + 1) / (len(null) + 1))
print("\n" + "=" * 60)
print(f"observed            {obs:.4f}")
print(f"null mean +/- sd    {null.mean():.4f} +/- {null.std():.4f}")
print(f"null 95th / 99th    {np.percentile(null, 95):.4f} / "
      f"{np.percentile(null, 99):.4f}")
print(f"p ({len(null)} draws)   {p:.4f}")
print("verdict:", "above chance" if p < 0.05 else "NOT distinguishable from chance")
print(f"effect over the null mean: {obs - null.mean():+.4f} macro-F1")

OUT.write_text(json.dumps({
    "observed": obs, "null_mean": float(null.mean()),
    "null_sd": float(null.std()), "p_value": p, "n_perm": int(len(null)),
    "n_positive": int(yt.sum()),
    "null_pct": {"p95": float(np.percentile(null, 95)),
                 "p99": float(np.percentile(null, 99))},
}, indent=2))
print(f"written: {OUT}")
