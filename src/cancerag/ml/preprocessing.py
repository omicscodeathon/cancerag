"""
Stage 08 — ML preprocessing.

Defines the sklearn-Pipeline-based preprocessing layer that fits on train
folds only (no leakage from test/holdout). Provides:

- :func:`load_dataset` — loads ``ml_ready_dataset.parquet``; **refuses**
  to load anything under ``data/holdout/``.
- :func:`identify_columns` — splits the wide matrix into
  ``(metadata_cols, feature_cols, target_col, weight_col)``.
- :func:`build_preprocessing_pipeline` — returns an unfitted sklearn
  ``Pipeline`` with imputer → scaler → optional correlation filter.
- :class:`CorrelationFilter`, :class:`PerReceptorFamilyImputer` — sklearn-
  compatible transformers for use inside CV folds.
- :func:`run_preprocessing` — pre-fits the preprocessing pipeline on the
  full train-eligible matrix and persists it to disk so the trainer can
  inspect / reuse, **without** baking train/test split decisions in.
- Group-aware split helpers (kept here for tests + downstream stages):
  ``murcko_scaffold``, ``scaffold_split``, ``receptor_grouped_split``,
  ``scaffold_groups``, ``assignment_dataframe``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# Columns that are bookkeeping / labels — never go into X.
META_COLS = frozenset({
    "pair_key", "inchikey", "inchikey14", "receptor_uniprot",
    "canonical_smiles", "canonical_smiles_std", "ligand_name",
    "receptor_subtype", "bias_category", "bias_pathway",
    "reference_ligand", "assay_1", "assay_2", "pmid", "year", "doi",
    "source", "std_status", "label_status", "scaffold",
    "docking_success", "sample_weight",
})

DATASET_PATH = Path("data/processed/ml_ready_dataset.parquet")
HOLDOUT_DIR = Path("data/holdout")


# ----------------------------------------------------------------- loading


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Load the ML-ready dataset. **Refuses** the holdout file path."""
    path = Path(path).resolve()
    holdout_resolved = HOLDOUT_DIR.resolve()
    if holdout_resolved in path.parents or path.parent == holdout_resolved:
        raise RuntimeError(
            f"Refusing to load from holdout path: {path}. "
            "The holdout set must only be touched by the final evaluation step."
        )
    logger.info("Loading dataset from %s", path)
    return pd.read_parquet(path)


def identify_columns(df: pd.DataFrame) -> dict:
    """Split the wide matrix into role-tagged column lists.

    The categorical receptor-confidence flag (``docking_confidence``) is
    one-hot encoded into the feature set; the original string column is
    excluded from features.
    """
    cols = list(df.columns)
    feature_cols = [
        c for c in cols
        if c not in META_COLS and not c.endswith("_lig")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_features = ["docking_confidence"]
    return {
        "target_col": "bias_category",
        "weight_col": "sample_weight",
        "meta_cols": [c for c in cols if c in META_COLS],
        "feature_cols": feature_cols,
        "categorical_features": [c for c in categorical_features if c in cols],
        "n_features": len(feature_cols),
    }


# ----------------------------------------------- sklearn-compatible transformers


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop columns whose absolute Pearson correlation with another retained
    column exceeds ``threshold``.

    Pure-pandas; no fancy hierarchical clustering. Stable across seeds — walks
    columns in their input order and keeps the first member of each correlated
    cluster.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "CorrelationFilter":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CorrelationFilter requires a pandas DataFrame")
        numeric = X.select_dtypes(include=[np.number])
        corr = numeric.corr().abs().to_numpy()
        n = corr.shape[0]
        # Vectorised greedy walk: O(p²) instead of the previous O(p³).
        to_drop_set: set[int] = set()
        cols = list(numeric.columns)
        for i in range(n):
            if i in to_drop_set:
                continue
            # Indices j > i with corr > threshold.
            j_idx = np.flatnonzero(corr[i, i + 1:] > self.threshold) + (i + 1)
            for j in j_idx:
                to_drop_set.add(int(j))
        to_drop = [cols[j] for j in sorted(to_drop_set)]
        self.dropped_columns_ = to_drop
        drop_set = set(to_drop)
        self.kept_columns_ = [c for c in X.columns if c not in drop_set]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CorrelationFilter.transform requires a DataFrame")
        return X[self.kept_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


class PerReceptorFamilyImputer(BaseEstimator, TransformerMixin):
    """Per-receptor-family median imputer.

    Reviewer-2 explicitly flagged that filling missing structural features
    with the median across all 7 GPCR families is "physically unsound". This
    imputer learns a per-family median during ``fit`` and falls back to the
    global median if a family is unseen at predict time.
    """

    def __init__(self, family_col: str = "receptor_family"):
        self.family_col = family_col

    def fit(self, X: pd.DataFrame, y=None) -> "PerReceptorFamilyImputer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PerReceptorFamilyImputer requires a DataFrame")
        if self.family_col not in X.columns:
            raise KeyError(
                f"PerReceptorFamilyImputer: family column "
                f"{self.family_col!r} missing"
            )
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols_ = numeric_cols
        self.family_medians_ = (
            X.groupby(self.family_col)[numeric_cols].median(numeric_only=True)
        )
        self.global_medians_ = X[numeric_cols].median(numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "PerReceptorFamilyImputer.transform requires a DataFrame"
            )
        out = X.copy()
        for fam, group_idx in out.groupby(self.family_col).groups.items():
            meds = (
                self.family_medians_.loc[fam]
                if fam in self.family_medians_.index
                else self.global_medians_
            )
            out.loc[group_idx, self.numeric_cols_] = (
                out.loc[group_idx, self.numeric_cols_].fillna(meds)
            )
        out[self.numeric_cols_] = out[self.numeric_cols_].fillna(self.global_medians_)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None


# ----------------------------------------------- group-aware splits


def murcko_scaffold(smiles: str) -> str:
    """Canonical Murcko scaffold SMILES, or ``"INVALID"`` on parse failure."""
    if not isinstance(smiles, str) or not smiles:
        return "INVALID"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return "INVALID"


def scaffold_groups(
    df: pd.DataFrame, smiles_col: str = "canonical_smiles"
) -> np.ndarray:
    """Integer-encoded Murcko-scaffold group ids."""
    if smiles_col not in df.columns:
        raise KeyError(f"scaffold_groups: column {smiles_col!r} missing")
    scaffolds = df[smiles_col].map(murcko_scaffold)
    codes, _ = pd.factorize(scaffolds)
    return np.asarray(codes)


def scaffold_split(
    df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42,
    smiles_col: str = "canonical_smiles",
) -> tuple[np.ndarray, np.ndarray]:
    """``(train_idx, test_idx)`` such that no Murcko scaffold straddles the split."""
    groups = scaffold_groups(df, smiles_col=smiles_col)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=groups))
    return train_idx, test_idx


def receptor_grouped_split(
    df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42,
    receptor_col: str = "receptor_uniprot",
) -> tuple[np.ndarray, np.ndarray]:
    """``(train_idx, test_idx)`` such that no receptor straddles the split."""
    if receptor_col not in df.columns:
        raise KeyError(f"receptor_grouped_split: column {receptor_col!r} missing")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df[receptor_col].astype(str)))
    return train_idx, test_idx


def make_grouped_cv(
    df: pd.DataFrame,
    *,
    group_col: str,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build (train_idx, test_idx) splits for K-fold by group.

    Used for scaffold- and receptor-grouped cross-validation. Falls back to
    fewer folds if the group count is < n_splits (rare but possible for
    receptor-grouped on small subsets).
    """
    from sklearn.model_selection import GroupKFold
    groups = df[group_col].astype(str).values
    n_groups = len(set(groups))
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        return []
    gkf = GroupKFold(n_splits=n_splits)
    return [
        (np.asarray(tr), np.asarray(te))
        for tr, te in gkf.split(df, groups=groups)
    ]


def assignment_dataframe(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    pair_key_col: str = "pair_key",
    extra_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build a ``(pair_key, split[, extra_cols])`` table to persist the split."""
    cols = [pair_key_col] + list(extra_cols)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise KeyError("assignment_dataframe: no available identifying columns")
    rows = []
    for idx in train_idx:
        row = {"split": "train"}
        for c in cols:
            row[c] = df.iloc[int(idx)][c]
        rows.append(row)
    for idx in test_idx:
        row = {"split": "test"}
        for c in cols:
            row[c] = df.iloc[int(idx)][c]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------- pipeline assembly


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Median imputer that returns a DataFrame (preserves column names so
    downstream transformers like CorrelationFilter still work)."""

    def __init__(self, strategy: str = "median"):
        self.strategy = strategy

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameImputer requires a DataFrame")
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(X)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameImputer.transform requires a DataFrame")
        out = self.imputer_.transform(X)
        return pd.DataFrame(out, columns=self.columns_, index=X.index)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that returns a DataFrame."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameScaler requires a DataFrame")
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameScaler.transform requires a DataFrame")
        out = self.scaler_.transform(X)
        return pd.DataFrame(out, columns=self.columns_, index=X.index)


class DataFrameVarianceFilter(BaseEstimator, TransformerMixin):
    """Drop columns with variance below ``threshold``. Returns a DataFrame.

    Cheap pre-screen — at our dataset size (~3000 cols, mostly sparse IFP
    bits + Morgan bits) this typically halves the column count, making the
    downstream CorrelationFilter (which is O(p²)) tractable.
    """

    def __init__(self, threshold: float = 1e-4):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameVarianceFilter requires a DataFrame")
        var = X.var(numeric_only=True)
        self.kept_columns_ = var.index[var > self.threshold].tolist()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameVarianceFilter.transform requires a DataFrame")
        return X[self.kept_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


def build_full_pipeline(
    model,
    *,
    impute: bool = True,
    variance_threshold: float = 1e-4,
    correlation_threshold: float = 0.97,
    scale: bool = True,
    selector=None,
) -> Pipeline:
    """Return a sklearn ``Pipeline`` for per-fold use:
        imputer -> variance_filter -> correlation_filter -> scaler ->
        [selector] -> model

    All transformers preserve DataFrame structure so downstream selectors
    (which need column names) work. The trainer fits this on each CV fold
    so no test-fold information leaks into the fitted preprocessing.

    The variance filter is a cheap O(p) pre-screen that halves the column
    count before the O(p²) correlation filter — necessary for tractability
    at our ~3000-column dataset.
    """
    steps: list[tuple[str, BaseEstimator]] = []
    if impute:
        steps.append(("imputer", DataFrameImputer(strategy="median")))
    if variance_threshold > 0:
        steps.append(("variance_filter",
                      DataFrameVarianceFilter(threshold=variance_threshold)))
    if correlation_threshold and correlation_threshold < 1.0:
        steps.append(("correlation_filter",
                      CorrelationFilter(threshold=correlation_threshold)))
    if scale:
        steps.append(("scaler", DataFrameScaler()))
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", model))
    return Pipeline(steps)


def get_X_y_groups(
    df: pd.DataFrame, *, label_encoder=None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Standard extraction: returns (X, y_encoded, sample_weight, scaffolds, cols)."""
    cols = identify_columns(df)
    X = df[cols["feature_cols"]].copy()
    if cols["categorical_features"]:
        cat = pd.get_dummies(
            df[cols["categorical_features"]].fillna("unknown"),
            prefix=cols["categorical_features"], drop_first=False,
        ).astype(np.float32)
        X = pd.concat([X.reset_index(drop=True), cat.reset_index(drop=True)],
                      axis=1)
    y_str = df[cols["target_col"]].astype(str).values
    if label_encoder is None:
        label_encoder = LabelEncoder().fit(y_str)
    y = label_encoder.transform(y_str)
    sample_weight = df[cols["weight_col"]].astype(float).values
    return X, y, sample_weight, label_encoder, cols


# Legacy-compatible alias (kept for any old callers)
def build_preprocessing_pipeline(
    *,
    impute_strategy: str = "median",
    correlation_threshold: float = 0.97,
    scale: bool = True,
    drop_correlated: bool = True,
) -> Pipeline:
    """Compatibility wrapper around :func:`build_full_pipeline` without a model."""
    from sklearn.dummy import DummyClassifier
    pipe = build_full_pipeline(
        DummyClassifier(),
        impute=True,
        correlation_threshold=correlation_threshold if drop_correlated else 1.0,
        scale=scale,
    )
    # Drop the dummy model step so callers get a transformer-only pipeline
    return Pipeline(pipe.steps[:-1])


# ----------------------------------------------- orchestrator


def run_preprocessing(
    config: dict | None = None,
    dataset_path: Path | str | None = None,
    *,
    output_dir: Path | str = "data/processed/ml_preprocessed",
    correlation_threshold: float = 0.97,
) -> dict:
    """Pre-fit a preprocessing layer on the full train-eligible dataset and
    persist the fitted artifacts to ``output_dir``. **Does not** split the
    dataset — that's the trainer's job inside CV folds. We pre-fit here so a
    reviewer can inspect the global statistics (column drop list, medians,
    label encoder) before training.
    """
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(dataset_path or DATASET_PATH)
    cols = identify_columns(df)
    logger.info(
        "Dataset loaded: %d rows × %d feature cols (%d total cols)",
        len(df), cols["n_features"], len(df.columns),
    )

    X = df[cols["feature_cols"]].copy()
    # One-hot encode categorical features (only `docking_confidence`).
    if cols["categorical_features"]:
        cat = pd.get_dummies(
            df[cols["categorical_features"]].fillna("unknown"),
            prefix=cols["categorical_features"], drop_first=False,
        ).astype(np.float32)
        X = pd.concat([X.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    y = df[cols["target_col"]].astype(str).values
    sample_weight = df[cols["weight_col"]].astype(float).values

    # 1. Median imputer (fit on all train-eligible).
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X),
                         columns=X.columns, index=X.index)

    # 2. Correlation filter (fit on imputed train-eligible).
    cf = CorrelationFilter(threshold=correlation_threshold)
    cf.fit(X_imp)
    X_filt = cf.transform(X_imp)
    logger.info(
        "Correlation filter dropped %d features (threshold=%.2f), kept %d",
        len(cf.dropped_columns_), correlation_threshold, len(cf.kept_columns_),
    )

    # 3. Standard scaler (fit on filtered train-eligible).
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_filt), columns=X_filt.columns, index=X_filt.index
    )

    # 4. Label encoder (fit on train-eligible target).
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.info("Label classes: %s", list(le.classes_))

    # 5. Persist artefacts. Trainer is responsible for re-fitting per CV fold.
    joblib.dump(imputer, output_dir / "global_imputer.joblib")
    joblib.dump(cf, output_dir / "global_correlation_filter.joblib")
    joblib.dump(scaler, output_dir / "global_scaler.joblib")
    joblib.dump(le, output_dir / "label_encoder.joblib")

    X_scaled.to_parquet(output_dir / "X_train_eligible.parquet", index=False)
    np.save(output_dir / "y_train_eligible.npy", y_enc)
    np.save(output_dir / "sample_weight_train_eligible.npy", sample_weight)

    meta = {
        "schema_version": 1,
        "preprocessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path or DATASET_PATH),
        "n_rows": int(len(df)),
        "n_features_input": int(len(cols["feature_cols"])),
        "n_features_after_correlation_filter": int(len(cf.kept_columns_)),
        "correlation_threshold": correlation_threshold,
        "label_classes": [str(c) for c in le.classes_],
        "label_distribution": {
            str(c): int((y_enc == i).sum()) for i, c in enumerate(le.classes_)
        },
        "sample_weight_summary": {
            "min": float(sample_weight.min()),
            "max": float(sample_weight.max()),
            "median": float(np.median(sample_weight)),
        },
        "dropped_correlated_features_first_20": cf.dropped_columns_[:20],
        "n_dropped_correlated": int(len(cf.dropped_columns_)),
    }
    (output_dir / "preprocessing_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )
    logger.info("Preprocessing artifacts written to %s", output_dir)

    return {
        "metadata": {
            "n_samples_train": int(len(df)),
            "n_samples_test": 0,  # split happens inside CV at training time
            "n_features": int(len(cf.kept_columns_)),
            "n_classes": int(len(le.classes_)),
            "class_names": list(le.classes_),
        },
        "output_dir": str(output_dir),
        "kept_features": cf.kept_columns_,
        "dropped_correlated": cf.dropped_columns_,
    }


# ----------------------------------------------- module entry


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    result = run_preprocessing()
    print("\nPreprocessing summary:")
    for k, v in result["metadata"].items():
        print(f"  {k}: {v}")
    print(f"  output_dir: {result['output_dir']}")
    print(f"  features dropped by correlation filter: {len(result['dropped_correlated'])}")


if __name__ == "__main__":
    main()
