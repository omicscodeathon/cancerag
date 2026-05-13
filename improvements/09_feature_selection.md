# 09 — Feature Selection (Boruta)

Module covered: `src/cancerag/ml/feature_selection.py`

## What the code does today

576 lines. Boruta-based feature selection:
- Line 254-255: `X = X.fillna(X.median())` — global median imputation on the full matrix.
- Line 287, 363, 402: `LabelEncoder().fit_transform(y)` — re-encoded each call.
- Line 307, 324, 369, 412: `boruta.fit(X, y_encoded)` / `selector.fit(X, y_encoded)` / `rf.fit(X, y_encoded)` — all run on the **full matrix**, not on per-fold training data.

A separate `fit_transform` API exists (line 172-183) but the orchestration (in `ml_pipeline.py`) appears to call selection before CV.

## Problems

### Tier 1 — leakage

**P9.1 (E2 part 2) Boruta runs on the full feature matrix, including test rows.**
Lines 307 and 324: `boruta.fit(X.values, y_encoded)`. The selected feature set is informed by the test rows. This is leakage independent of and additive to the imputation leakage in [08_ml_preprocessing.md](08_ml_preprocessing.md).

**P9.2 Median imputation on full X before Boruta** (line 255). Same issue as P8.1 — propagates into selection.

**P9.3 No selection stability reporting.**
Boruta on n=504 with ~200 features gives a different shortlist for different bootstrap resamples. A single Boruta run is a single sample from a noisy distribution. Reporting "Boruta selected these features" is meaningless without selection frequency.

### Tier 2 — community-norm gaps

**P9.4 No alternative selection methods reported.**
For small datasets the literature recommends comparing:
- **Stability selection** (bootstrap + Lasso aggregation) — the gold standard for high-dim small-n.
- **Recursive Feature Elimination with CV** (RFECV).
- **Mutual information** + chi² for categorical descriptors.
- **L1-Logistic** as a cheap baseline.

A single method (Boruta) without ablation is a methodological weakness Reviewer 1 or 2 could pick up on a resubmission.

**P9.5 Boruta on Random Forest only.**
Boruta wraps RF; it inherits RF's biases (favors high-cardinality features, depresses correlated features). This is well known in the ML feature-selection literature.

**P9.6 No collinearity prefilter.**
Boruta is sensitive to collinearity (correlated features split importance). Should be preceded by the correlation/VIF filter in [06_featurization.md](06_featurization.md) F6.5.

**P9.7 No domain-knowledge constraints.**
For a pair-level model, *some* features should always be retained (the pair-level interaction-fingerprint columns). Currently Boruta can drop them based on noise.

### Tier 3 — engineering

**P9.8 `LabelEncoder` re-fit per call.**
Lines 287, 363, 402. Re-encoding `y` at each entrypoint risks inconsistent label mapping across calls.

**P9.9 No persistence of selected features per fold.**
For nested CV, each fold should write `selected_features_fold_<k>.json`. Reproducibility and stability analysis need this.

**P9.10 576 lines for what is conceptually 50 lines.**
Suggests duplicated entry points (`fit`, `fit_transform`, `select_features`, `select_features_with_random_forest`, etc.). Refactor into one selector class with method dispatch.

## Standard approach

1. **Selection inside CV folds, never on the full matrix.**
2. **Stability selection**: run Boruta on K bootstrap resamples; report selection frequency per feature; keep features selected in ≥ 80% of resamples.
3. **Compare ≥ 2 selection methods** as ablation.
4. **Force-keep pair-level features** (interaction fingerprint, docking ensemble features) regardless of selector verdict — they are the methodological contribution.

## Concrete fixes

### F9.1 Make selector a Pipeline step (not a separate phase)

```python
# ml/feature_selection.py
from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin

class BorutaSelector(BaseEstimator, TransformerMixin):
    def __init__(self, force_keep_prefixes=("ifp_", "vina_pose_"),
                 estimator=None, random_state=42):
        self.force_keep_prefixes = force_keep_prefixes
        self.estimator = estimator or RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=random_state)
        self.random_state = random_state

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        boruta = BorutaPy(self.estimator, n_estimators="auto",
                          random_state=self.random_state, max_iter=100)
        boruta.fit(X.values, y)
        keep = set(np.array(self.feature_names_)[boruta.support_])
        keep |= {c for c in self.feature_names_
                 if any(c.startswith(p) for p in self.force_keep_prefixes)}
        self.selected_ = sorted(keep)
        return self

    def transform(self, X):
        return X[self.selected_]
```

This selector is then a step in the pipeline from [08_ml_preprocessing.md](08_ml_preprocessing.md) F8.1, automatically running per fold.

### F9.2 Stability selection

```python
# ml/stability_selection.py
def stability_selection(X, y, selector_factory, n_boot=50, sample_frac=0.8,
                        seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    counts = pd.Series(0, index=X.columns)
    n = len(X)
    for _ in range(n_boot):
        idx = rng.choice(n, int(sample_frac * n), replace=False)
        sel = selector_factory()
        sel.fit(X.iloc[idx], y[idx])
        for f in sel.selected_:
            counts[f] += 1
    return (counts / n_boot).sort_values(ascending=False)
```

Report `selection_frequency` per feature. Use a threshold (e.g., 0.8) to define the stable feature set.

### F9.3 Compare selectors as ablation

```python
# ml/selection_ablation.py
SELECTORS = {
    "boruta": lambda: BorutaSelector(),
    "rfecv": lambda: RFECVSelector(),
    "stability_lasso": lambda: StabilityLassoSelector(),
    "mutual_info_top_k": lambda: MITopKSelector(k=50),
    "no_selection": lambda: PassThroughSelector(),
}

def ablation_run(X, y, groups, splits) -> pd.DataFrame:
    results = []
    for name, factory in SELECTORS.items():
        for fold, (tr, te) in enumerate(splits):
            sel = factory().fit(X.iloc[tr], y[tr])
            X_tr = sel.transform(X.iloc[tr])
            X_te = sel.transform(X.iloc[te])
            model = RandomForestClassifier().fit(X_tr, y[tr])
            f1 = f1_score(y[te], model.predict(X_te), average="macro")
            results.append({"selector": name, "fold": fold, "macro_f1": f1,
                            "n_features": len(sel.selected_)})
    return pd.DataFrame(results)
```

Report this ablation table in the manuscript supplementary.

### F9.4 Persist per-fold selections

```python
# In nested-CV loop:
fold_selections = {}
for fold, (tr, te) in enumerate(outer_splits):
    sel = pipeline.named_steps["select"].fit(X.iloc[tr], y[tr])
    fold_selections[fold] = sel.selected_
Path(f"results/feature_selection/{run_id}.json").write_text(
    json.dumps(fold_selections, indent=2))
```

Then a downstream analysis can compute selection frequency across the 5 outer folds.

### F9.5 Drop the duplicate entry points

Collapse the selector module from 576 lines to a single `BorutaSelector` class plus the stability-selection helper. Move ablation logic to `ml/selection_ablation.py`.

### F9.6 Force-keep critical features

The interaction-fingerprint and pose-ensemble columns are the methodological contribution. Use the `force_keep_prefixes` argument in F9.1 to ensure they survive selection regardless of whether a single Boruta run drops them.

## Acceptance criteria

- [ ] `BorutaSelector` is a Pipeline-compatible transformer; it is fitted only on training folds.
- [ ] No `fillna(X.median())` or `boruta.fit` calls operate on the full matrix.
- [ ] Stability-selection frequencies are reported per feature in `results/feature_selection/stability.csv`.
- [ ] An ablation table compares ≥ 3 selectors (Boruta, RFECV, stability-Lasso, MI top-k, no-selection) on macro-F1.
- [ ] Per-fold selected features are persisted to JSON.
- [ ] Pair-level features (`ifp_*`, `vina_pose_*`) are force-kept regardless of selector output.
- [ ] Module size drops below 200 lines after dedup.
- [ ] `LabelEncoder` is fit once at pipeline entry, not per call.
