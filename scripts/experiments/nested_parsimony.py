"""Parsimony without the selection peeking at the test fold.

The first ladder ranked features by SHAP selection frequency computed across
folds of the whole dataset, then scored subsets of that ranking by CV. The
ranking had therefore seen every row it was later tested on, so the numbers
are optimistic -- they show a parsimonious subset *exists*, not that you would
find it from a cold start.

Here selection happens strictly inside the training fold:

    for each (regime, seed, fold):
        fit on train only, with every feature   -> gain importances
        for each k: take the top-k of THOSE importances,
                    refit on train only, score the held-out fold

Nothing about the test fold informs which features are chosen. The gap between
this and the first ladder is the size of the optimism, and that gap is itself
the result worth reporting.

Two controls:
  random    k features drawn at random inside the same fold
  all       every feature, the incumbent
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from cancerag.ml.model_training import MODEL_FACTORIES  # noqa: E402
from cancerag.ml.preprocessing import (  # noqa: E402
    build_full_pipeline, get_X_y_groups, make_grouped_cv,
)

SEEDS = [42, 7, 13]
KS = [2, 5, 11, 20, 40, 80, 160, 320]
OUT = Path("data/processed/ml_models/nested_parsimony.json")

tr = pd.read_parquet("data/processed/ml_ready_dataset.parquet")
X, y, sw, _, _ = get_X_y_groups(tr)
NC = len(np.unique(y))
floor = f1_score(y, np.full_like(y, np.bincount(y).argmax()),
                 average="macro", zero_division=0)
print(f"X {X.shape}  floor {floor:.4f}")

REGIMES = {
    "stratified": lambda s: list(
        StratifiedKFold(5, shuffle=True, random_state=s).split(X, y)),
    "scaffold_grouped": lambda s: make_grouped_cv(
        tr, group_col="scaffold", n_splits=5, seed=s),
    "receptor_grouped": lambda s: make_grouped_cv(
        tr, group_col="receptor_uniprot", n_splits=5, seed=s),
}


def fit_score(cols, tri, tei, seed):
    p = build_full_pipeline(
        MODEL_FACTORIES["lightgbm"](n_classes=NC, random_state=seed),
        variance_threshold=0.0, correlation_threshold=1.01)
    last = p.steps[-1][0]
    try:
        p.fit(X[cols].iloc[tri], y[tri], **{f"{last}__sample_weight": sw[tri]})
    except TypeError:
        p.fit(X[cols].iloc[tri], y[tri])
    except ValueError:
        return None, None
    return p, f1_score(y[tei], p.predict(X[cols].iloc[tei]),
                       average="macro", zero_division=0)


rows = []
for regime, mk in REGIMES.items():
    for seed in SEEDS:
        splits = mk(seed)
        if not splits:
            continue
        rng = np.random.default_rng(seed)
        per_k: dict = {k: [] for k in KS}
        per_k_rand: dict = {k: [] for k in KS}
        full = []
        for tri, tei in splits:
            # one fit on the training fold only, to rank features
            pipe, s_all = fit_score(list(X.columns), tri, tei, seed)
            if pipe is None:
                continue
            full.append(s_all)
            model = pipe.steps[-1][1]
            imp = getattr(model, "feature_importances_", None)
            if imp is None:
                continue
            # the pipeline may drop columns; recover the surviving names
            try:
                names = pipe[:-1].get_feature_names_out()
            except Exception:
                names = np.array(X.columns)
            if len(names) != len(imp):
                names = np.array(X.columns)[:len(imp)]
            order = [n for n in pd.Series(imp, index=names)
                     .sort_values(ascending=False).index
                     if n in X.columns]
            for k in KS:
                sel = order[:k]
                if sel:
                    _, sc = fit_score(sel, tri, tei, seed)
                    if sc is not None:
                        per_k[k].append(sc)
                rsel = list(rng.choice(X.columns, size=k, replace=False))
                _, rsc = fit_score(rsel, tri, tei, seed)
                if rsc is not None:
                    per_k_rand[k].append(rsc)
        if full:
            rows.append({"regime": regime, "seed": seed, "arm": "all",
                         "k": int(X.shape[1]), "score": float(np.mean(full))})
        for k in KS:
            if per_k[k]:
                rows.append({"regime": regime, "seed": seed, "arm": "nested",
                             "k": k, "score": float(np.mean(per_k[k]))})
            if per_k_rand[k]:
                rows.append({"regime": regime, "seed": seed, "arm": "random",
                             "k": k, "score": float(np.mean(per_k_rand[k]))})
        print(f"  {regime} seed={seed} done", flush=True)

d = pd.DataFrame(rows)
print("\n" + "=" * 74)
print("NESTED PARSIMONY  (selection inside the training fold only)")
print("=" * 74)
for regime in REGIMES:
    sub = d[d.regime == regime]
    if sub.empty:
        continue
    piv = sub.pivot_table(index="k", columns="arm", values="score",
                          aggfunc=["mean", "std"])
    print(f"\n--- {regime} ---")
    print(piv.round(4).to_string())

OUT.write_text(json.dumps({"rows": rows, "floor": float(floor),
                           "n_features": int(X.shape[1])}, indent=2))
print(f"\nwritten: {OUT}")
