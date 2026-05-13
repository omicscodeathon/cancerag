# 08 — ML Preprocessing (Splits, Imputation, Scaling)

Module covered: `src/cancerag/ml/preprocessing.py`

## What the code does today

`DataPreprocessor.prepare_dataset(df)`:

1. Drops metadata columns; computes `X` and `y`.
2. **Imputation on full X** (line 330): `X = self.handle_missing_values(X, strategy="median")`.
3. **Constant-column drop on full X** (lines 333-339).
4. Encode target.
5. **Then split** (line 348): `train_test_split(X, y_encoded, test_size=0.05, stratify=y, random_state=42)`.
6. Fit scaler on train only (line 351).
7. `handle_class_imbalance(X_train_scaled, y_train)` (line 355) — implementation not inspected; if SMOTE-like, this needs to be inside CV folds.

`config.yaml`: `test_size: 0.05`, `random_state: 42`.

## Problems

### Tier 1 — leakage and split-design failures

**P8.1 (E2) Imputation runs on the full X before the split.**
`SimpleImputer(strategy="median")` fits on all 504 rows. The median used to fill train-set NaNs has seen the test rows. Textbook leakage.

**P8.2 (E2) Constant-column pruning runs on full X before the split.**
A column constant in train but variable in test (or vice versa) gets dropped or kept based on the *full* matrix. The decision boundary has seen the test set.

**P8.3 (E3) Random stratified split with `test_size=0.05`.**
26 test samples on a 5-class problem with 15:1 class imbalance. The 95% CI on accuracy is roughly ±16 points. Reviewer 2 demanded an 80/20 split or nested CV.

**P8.4 (E4) No scaffold split.**
`train_test_split` shuffles ligand-receptor pairs. Congeneric series (close analogs of the same Murcko scaffold) end up on both sides of the split. Reviewer 2's "data leakage" concern is essentially proven by the code.

**P8.5 No receptor-grouped split.**
With 7 GPCR families and 5 bias classes, stratifying only on the bias label can put all examples of a given receptor on one side, or all on the other. The model can either:
- Memorize per-receptor pocket descriptors (leakage by receptor identity), or
- Fail entirely on unseen receptors at test time (but this case is hidden because the test set always contains the same receptors as training).

A **GroupShuffleSplit by receptor** answers a different generalization question and should be reported alongside scaffold-split metrics.

**P8.6 Class imbalance handled outside CV.**
Line 355 calls `handle_class_imbalance` on the training set as a whole, then this resampled set is used downstream. If resampling generates synthetic samples (SMOTE), they bleed into CV folds. Resampling must occur inside each fold via `imblearn.pipeline.Pipeline`.

**P8.7 Single seed (`random_state=42`).**
No reporting of seed-to-seed variance. With n_test ≈ 26, one seed change can move accuracy by 10+ points.

### Tier 2 — community-norm violations

**P8.8 No `sklearn.Pipeline`.**
Imputation, scaling, feature selection, and model fit are imperative steps with manual state management. The community norm is to compose a `Pipeline` and `cross_val_score(pipeline, X, y, cv=outer_cv)` so all preprocessing automatically lives inside each fold.

**P8.9 No nested CV.**
Reviewer 2 asked for it. Required for unbiased estimation of test performance after hyperparameter selection.

**P8.10 No reporting of bootstrap CIs on the macro-F1.**
With small test sets, point estimates are useless. Bootstrap (1000 resamples of test predictions) gives proper uncertainty bands.

**P8.11 No applicability-domain (AD) check.**
For each test prediction, the standard cheminformatics report includes Tanimoto similarity to the nearest training ligand and leverage in descriptor space. Predictions on out-of-distribution ligands should be flagged.

**P8.12 Stratification is on label only, not on `(label × receptor)`.**
See P8.5.

### Tier 3 — engineering

**P8.13 `logging.basicConfig` at module top** (line 18).

**P8.14 Manual `feature_columns` tracking.**
Lines 324, 337-339 manually track which columns survive each step. A `Pipeline` would do this implicitly.

**P8.15 No persistence of split assignments.**
After splitting, the `pair_key` of each row in train/val/test is not saved. Reproducing or auditing the split requires re-running with the same seed.

## Standard approach

1. **Move all preprocessing inside `sklearn.Pipeline`** so it runs inside CV folds.
2. **Scaffold split + receptor-grouped split** as two separate evaluation regimes; report both.
3. **5×5 nested CV**: outer loop estimates generalization, inner loop tunes hyperparameters.
4. **Bootstrap CIs** on the test-fold metrics.
5. **Persist split assignments** as a CSV with `pair_key, fold, split` for reproducibility.
6. **Applicability domain** check on every prediction.

## Concrete fixes

### F8.1 sklearn-Pipeline-ize everything

```python
# ml/pipelines.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def make_pipeline(estimator, *, impute_strategy="median",
                  per_family_impute: bool = True) -> Pipeline:
    steps = [
        ("variance", VarianceThreshold(threshold=0.0)),
        ("impute", _per_family_imputer() if per_family_impute else SimpleImputer(strategy=impute_strategy)),
        ("decorrelate", CorrelationFilter(threshold=0.95)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ]
    return Pipeline(steps)
```

All preprocessing then automatically respects fold boundaries.

### F8.2 Scaffold split

```python
# ml/splits.py
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "INVALID"
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)

def scaffold_groups(df: pd.DataFrame) -> np.ndarray:
    return df["canonical_smiles"].map(murcko_scaffold).factorize()[0]

def scaffold_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    groups = scaffold_groups(df)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return train_idx, test_idx
```

### F8.3 Receptor-grouped split

```python
def receptor_grouped_split(df, test_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, groups=df["receptor_uniprot"]))
```

Report both scaffold-split and receptor-split test metrics in the manuscript.

### F8.4 Nested CV

```python
# ml/nested_cv.py
def nested_cv(X, y, groups, pipeline, param_grid, *,
              outer_n=5, inner_n=5, scoring="f1_macro", seed=42):
    outer = StratifiedGroupKFold(n_splits=outer_n, shuffle=True, random_state=seed)
    out = []
    for fold_id, (tr, te) in enumerate(outer.split(X, y, groups)):
        inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=seed)
        gs = GridSearchCV(pipeline, param_grid, scoring=scoring, cv=inner,
                          refit=True, n_jobs=-1)
        gs.fit(X.iloc[tr], y[tr], groups=groups[tr])
        y_pred = gs.predict(X.iloc[te])
        out.append({
            "fold": fold_id,
            "best_params": gs.best_params_,
            "test_macro_f1": f1_score(y[te], y_pred, average="macro"),
            "test_balanced_acc": balanced_accuracy_score(y[te], y_pred),
        })
    return pd.DataFrame(out)
```

### F8.5 Bootstrap CIs

```python
def bootstrap_ci(y_true, y_pred, metric, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric(y_true[idx], y_pred[idx]))
    return np.percentile(scores, [2.5, 50, 97.5])
```

### F8.6 Per-family imputation

```python
# ml/imputers.py
class PerReceptorFamilyImputer(BaseEstimator, TransformerMixin):
    def __init__(self, family_col="receptor_family"):
        self.family_col = family_col
    def fit(self, X, y=None):
        self.medians_ = X.groupby(self.family_col).median()
        return self
    def transform(self, X):
        X = X.copy()
        for fam, meds in self.medians_.iterrows():
            mask = X[self.family_col] == fam
            X.loc[mask] = X.loc[mask].fillna(meds)
        return X.drop(columns=[self.family_col])
```

Replaces global median imputation; addresses Reviewer 2's specific concern.

### F8.7 Resampling inside the pipeline

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

pipeline = ImbPipeline([
    ("variance", VarianceThreshold()),
    ("impute", PerReceptorFamilyImputer()),
    ("decorrelate", CorrelationFilter(0.95)),
    ("scale", StandardScaler()),
    ("resample", SMOTENC(categorical_features=[...], random_state=42)),
    ("model", estimator),
])
```

`imblearn.pipeline.Pipeline` correctly applies the resampler only during `fit`, not `predict` — so CV folds remain clean.

### F8.8 Applicability-domain check

```python
# ml/applicability_domain.py
def tanimoto_to_nearest(query_fp, train_fps) -> float:
    return max(DataStructs.TanimotoSimilarity(query_fp, f) for f in train_fps)

def ad_flag(query_fp, train_fps, threshold=0.4) -> bool:
    return tanimoto_to_nearest(query_fp, train_fps) >= threshold
```

Predictions where the nearest-neighbor Tanimoto is below threshold are flagged "outside applicability domain" and excluded from the headline metric.

### F8.9 Persist split assignments

```python
# After splitting:
pd.DataFrame({
    "pair_key": df["pair_key"],
    "fold": fold_assignments,
    "split": split_assignments,  # train/val/test
    "scaffold": df["scaffold"],
}).to_csv(f"results/splits/{run_id}_split.csv", index=False)
```

### F8.10 Update `config.yaml`

```yaml
ml_model:
  outer_cv_folds: 5
  inner_cv_folds: 5
  test_size: 0.20                         # raise from 0.05
  split_strategy: "scaffold"              # primary regime
  also_report_split: "receptor_grouped"   # additional regime
  random_seeds: [42, 7, 1234, 99, 2024]   # report mean ± std across seeds
  bootstrap_ci_n: 1000
  applicability_domain_tanimoto: 0.4
```

## Acceptance criteria

- [ ] All preprocessing (impute, scale, decorrelate, resample, feature-select) lives inside an `sklearn.Pipeline` / `imblearn.pipeline.Pipeline`.
- [ ] Scaffold split is the primary evaluation regime; receptor-grouped split is reported alongside.
- [ ] 5×5 nested CV is run; per-fold metrics are persisted to CSV.
- [ ] Bootstrap 95% CIs are reported for macro-F1, balanced accuracy, AUC.
- [ ] Per-receptor-family median imputation replaces the global median imputer.
- [ ] Resampling is inside the pipeline, not before CV.
- [ ] Five seeds are run; mean ± std reported.
- [ ] Each prediction has an applicability-domain flag.
- [ ] Split assignments are persisted as `results/splits/<run_id>_split.csv`.
- [ ] `test_size = 0.05` is removed from config.
