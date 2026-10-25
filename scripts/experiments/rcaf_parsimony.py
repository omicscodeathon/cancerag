"""RCAF stages 3, 5 and 6 — and the parsimony test.

The thesis under test (Anamnesis §2): *conditions, not complexity, are what hide
a parsimonious rule.* We have 5,511 features and four regimes of increasing
generalisation difficulty. If a small set of regime-surviving features matches
the full matrix, the complexity was never carrying the signal.

Three questions, one run:

  A. CONTRASTIVE PROFILES (RCAF 5-6)
     Attribution under the easiest regime vs the hardest. Features important
     under stratified and gone under receptor-grouped are dataset-specific
     shortcuts. The skill is explicit that the dropouts are reported too --
     concealing them hides how the filter was applied.

  B. PARSIMONY LADDER
     Macro-F1 against number of features, k = 1..all, features ordered by
     worst-case selection frequency across the four regimes (so the ordering
     is itself regime-filtered). Where is the knee?

  C. FEATURE VIEWS
     all / bw_only / ligand_only / structure_only, under every regime. Tests
     directly whether the receptor-comparable BW representation is what buys
     cross-receptor transfer, and whether structure adds over chemistry.

Baselines throughout: majority-class macro-F1, and k random features matched
to each rung of the ladder, so "11 features do as well" is measured against
"any 11 features do as well".
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

M = Path("data/processed/ml_models")
OUT = M / "rcaf_parsimony.json"
SEEDS = [42, 7, 13]
RUNG = [1, 2, 3, 5, 8, 11, 20, 40, 80, 160, 320, 800, 2000, None]  # None = all

tr = pd.read_parquet("data/processed/ml_ready_dataset.parquet")
X, y, sw, _, _ = get_X_y_groups(tr)
print(f"X {X.shape}  classes {np.bincount(y)}")

maj = f1_score(y, np.full_like(y, np.bincount(y).argmax()), average="macro",
               zero_division=0)
print(f"majority-class macro-F1 floor: {maj:.4f}\n")

# ---------------------------------------------------------------- the regimes
# Ordered easy -> hard. The ordering is the filter.
REGIMES = {
    "stratified": lambda seed: list(
        StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)),
    "scaffold_grouped": lambda seed: make_grouped_cv(
        tr, group_col="scaffold", n_splits=5, seed=seed),
    "receptor_grouped": lambda seed: make_grouped_cv(
        tr, group_col="receptor_uniprot", n_splits=5, seed=seed),
}


def cv_macro_f1(cols, regime, seed):
    splits = REGIMES[regime](seed)
    if not splits:
        return None
    Xc = X[cols]
    scores = []
    for tri, tei in splits:
        # The variance/correlation filters are for pruning the full matrix.
        # On a deliberately chosen handful they can drop every column and
        # leave StandardScaler with nothing, so stand them down here.
        p = build_full_pipeline(
            MODEL_FACTORIES["lightgbm"](n_classes=len(np.unique(y)),
                                        random_state=seed),
            variance_threshold=0.0, correlation_threshold=1.01)
        last = p.steps[-1][0]
        try:
            p.fit(Xc.iloc[tri], y[tri], **{f"{last}__sample_weight": sw[tri]})
        except TypeError:
            p.fit(Xc.iloc[tri], y[tri])
        except ValueError:
            # every chosen column was constant in this fold
            return None
        scores.append(f1_score(y[tei], p.predict(Xc.iloc[tei]),
                               average="macro", zero_division=0))
    return float(np.mean(scores)) if scores else None


# ============================================================ A. contrastive
print("=" * 74)
print("A. REGIME-CONTRASTIVE PROFILES")
print("=" * 74)
cr = pd.read_csv(M / "interpretability" / "shap_cross_regime.csv")
sf = ["sf_stratified", "sf_scaffold", "sf_receptor", "sf_temporal"]
cr["worst_sf"] = cr[sf].min(axis=1)

EASY, HARD = "sf_stratified", "sf_receptor"
top_easy = set(cr.nlargest(50, EASY)["feature"])
top_hard = set(cr.nlargest(50, HARD)["feature"])
shortcuts = cr[cr.feature.isin(top_easy - top_hard)].nlargest(15, EASY)
survivors = cr[cr.feature.isin(top_easy & top_hard)].nlargest(15, HARD)

print(f"top-50 under the EASIEST regime that are NOT top-50 under the HARDEST:"
      f" {len(top_easy - top_hard)} of 50")
print("\nSHORTCUTS (strong when generalisation is easy, gone when it is hard):")
print(shortcuts[["feature", EASY, HARD]].to_string(index=False))
print("\nSURVIVORS (strong under both):")
print(survivors[["feature", EASY, HARD]].to_string(index=False))

robust = cr[(cr[sf] >= 0.5).all(axis=1)].sort_values("worst_sf", ascending=False)
ranked = cr.sort_values("worst_sf", ascending=False)["feature"].tolist()
ranked = [f for f in ranked if f in X.columns]
print(f"\nrobust in all four regimes: {len(robust)}")
print(f"features rankable and present in X: {len(ranked)}")

# ============================================================ B. parsimony
print()
print("=" * 74)
print("B. PARSIMONY LADDER  (features ranked by worst-case regime survival)")
print("=" * 74)
rows = []
rng = np.random.default_rng(0)
for k in RUNG:
    cols = list(X.columns) if k is None else ranked[:k]
    if not cols:
        continue
    label = "all" if k is None else str(k)
    for regime in REGIMES:
        vals = [cv_macro_f1(cols, regime, s) for s in SEEDS]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        rows.append({"arm": "ranked", "k": len(cols), "k_label": label,
                     "regime": regime, "mean": float(np.mean(vals)),
                     "sd": float(np.std(vals))})
    # matched random control -- "do ANY k features do this well?"
    if k is not None and k <= 320:
        for regime in REGIMES:
            vals = []
            for s in SEEDS:
                rc = list(rng.choice(X.columns, size=k, replace=False))
                v = cv_macro_f1(rc, regime, s)
                if v is not None:
                    vals.append(v)
            if vals:
                rows.append({"arm": "random", "k": k, "k_label": label,
                             "regime": regime, "mean": float(np.mean(vals)),
                             "sd": float(np.std(vals))})
    print(f"  rung k={label} done", flush=True)

lad = pd.DataFrame(rows)
for regime in REGIMES:
    print(f"\n--- {regime} ---")
    sub = lad[lad.regime == regime].pivot_table(
        index="k_label", columns="arm", values=["mean", "sd"])
    order = [r for r in (["all"] + [str(r) for r in RUNG if r]) if r in sub.index]
    print(sub.reindex(order).round(4).to_string())

# ============================================================ C. views
print()
print("=" * 74)
print("C. FEATURE VIEWS")
print("=" * 74)
ifp = [c for c in X.columns if c.startswith("ifp_")]
bw = [c for c in ifp if c.startswith("ifp_BW")]
struct_pref = ("ifp_", "vina_", "pose_", "sasa", "rmsd")
structure = [c for c in X.columns if c.startswith(struct_pref)]
VIEWS = {
    "all": list(X.columns),
    "bw_only": [c for c in X.columns if c not in ifp or c in bw],
    "ligand_only": [c for c in X.columns if c not in structure],
    "structure_only": structure,
}
vrows = []
for v, cols in VIEWS.items():
    for regime in REGIMES:
        vals = [cv_macro_f1(cols, regime, s) for s in SEEDS]
        vals = [x for x in vals if x is not None]
        if vals:
            vrows.append({"view": v, "n": len(cols), "regime": regime,
                          "mean": float(np.mean(vals)),
                          "sd": float(np.std(vals))})
    print(f"  view {v} ({len(cols)}) done", flush=True)
vd = pd.DataFrame(vrows)
print()
print(vd.pivot_table(index=["view", "n"], columns="regime",
                     values="mean").round(4).to_string())

OUT.write_text(json.dumps({
    "floor_macro_f1": float(maj),
    "n_robust_all_regimes": int(len(robust)),
    "robust_features": robust["feature"].tolist(),
    "shortcuts_top_easy_not_hard": sorted(top_easy - top_hard),
    "ladder": rows, "views": vrows,
}, indent=2))
print(f"\nwritten: {OUT}")
