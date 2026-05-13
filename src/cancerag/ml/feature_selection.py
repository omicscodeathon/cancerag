"""
Stage 09 — Feature selection.

Whittles the wide preprocessed matrix (~2300 columns at n=443) down to a
defensible subset for downstream model training. At this n/p ratio Boruta on
the full matrix is impractical, so we layer cheap pre-screens before Boruta:

  1. **Variance threshold** — drop near-constant columns (mostly all-zero
     ProLIF bits that aren't activated by any pair).
  2. **Univariate ANOVA F-test** — keep the top-K columns by F-score against
     ``bias_category``. Cheap, leakage-free if fit on train fold only.
  3. **Boruta with stability selection** — bootstrap-resample (X, y); fit
     ``BorutaSelector`` on each subsample; keep features selected in ≥ 80%
     of resamples. The sklearn-compatible :class:`BorutaSelector` lives in
     this module and is intended for use **inside** CV folds — but we also
     offer a "global" pre-screen here that the trainer can use as a sanity
     baseline.
  4. **Force-keep prefixes** — methodology-essential columns (Vina pose-
     ensemble, ProLIF IFP bits, 3D pose descriptors) survive selection
     regardless of statistical verdict. They ARE the manuscript's contribution.

The legacy 250-line ``FeatureSelector`` class with its in-module fit+save
side-effects is removed. The sklearn-compatible :class:`BorutaSelector` and
:func:`stability_selection` are preserved.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

logger = logging.getLogger(__name__)


FORCE_KEEP_PREFIXES: tuple[str, ...] = (
    "vina_affinity", "vina_pose_", "vina_n_distinct",
    "ifp_", "n_residues_contacted", "n_total_contacts",
    "Asphericity", "Eccentricity", "NPR", "PMI", "RadiusOfGyration",
    "SpherocityIndex", "InertialShapeFactor",
    "redock_rmsd", "gnina_cnn",
)


# ----------------------------------------------- sklearn-compatible Boruta


class BorutaSelector(BaseEstimator, TransformerMixin):
    """Pipeline-compatible Boruta wrapper. ``force_keep_prefixes`` lets the
    methodologically important pair-level columns survive selection regardless
    of Boruta's verdict — they ARE the contribution and shouldn't be dropped
    due to one noisy fit."""

    def __init__(
        self,
        force_keep_prefixes: Sequence[str] = FORCE_KEEP_PREFIXES,
        n_estimators: int | str = "auto",
        max_iter: int = 100,
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.force_keep_prefixes = tuple(force_keep_prefixes)
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y) -> "BorutaSelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BorutaSelector requires a pandas DataFrame")
        if X.isna().any().any():
            raise ValueError(
                "BorutaSelector received NaN values; impute upstream in the Pipeline"
            )
        self.feature_names_ = list(X.columns)
        rf = RandomForestClassifier(
            n_jobs=-1, class_weight="balanced",
            random_state=self.random_state,
        )
        try:
            from boruta import BorutaPy
            boruta = BorutaPy(
                rf, n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_iter=self.max_iter, verbose=self.verbose,
            )
            boruta.fit(X.values, np.asarray(y))
            selected = list(np.array(self.feature_names_)[boruta.support_])
        except Exception as exc:
            logger.warning(
                "Boruta failed (%s); falling back to no-selection", exc,
            )
            selected = list(self.feature_names_)
        forced = [
            f for f in self.feature_names_
            if any(f.startswith(p) for p in self.force_keep_prefixes)
        ]
        self.selected_ = sorted(set(selected) | set(forced))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("BorutaSelector.transform requires a DataFrame")
        cols = [c for c in self.selected_ if c in X.columns]
        return X[cols].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_)


class L1LogRegSelector(BaseEstimator, TransformerMixin):
    """L1-penalized multinomial Logistic Regression as a feature selector.

    Standard high-dim small-n approach. Coefficients with non-zero magnitude
    after L1 are kept. ``force_keep_prefixes`` preserves methodology-essential
    columns regardless of the L1 verdict.
    """

    def __init__(
        self,
        force_keep_prefixes: Sequence[str] = FORCE_KEEP_PREFIXES,
        C: float = 0.1,
        max_iter: int = 2000,
        random_state: int = 42,
    ):
        self.force_keep_prefixes = tuple(force_keep_prefixes)
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y) -> "L1LogRegSelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("L1LogRegSelector requires a DataFrame")
        self.feature_names_ = list(X.columns)
        from sklearn.linear_model import LogisticRegression
        try:
            lr = LogisticRegression(
                penalty="l1", solver="saga", multi_class="multinomial",
                C=self.C, max_iter=self.max_iter, n_jobs=-1,
                random_state=self.random_state, class_weight="balanced",
            )
            lr.fit(X.values, np.asarray(y))
            mask = np.any(np.abs(lr.coef_) > 1e-8, axis=0)
            selected = [c for c, m in zip(self.feature_names_, mask) if m]
        except Exception as exc:
            logger.warning(
                "L1LogRegSelector failed (%s); keeping all features", exc,
            )
            selected = list(self.feature_names_)
        forced = [
            f for f in self.feature_names_
            if any(f.startswith(p) for p in self.force_keep_prefixes)
        ]
        self.selected_ = sorted(set(selected) | set(forced))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("L1LogRegSelector.transform requires a DataFrame")
        cols = [c for c in self.selected_ if c in X.columns]
        return X[cols].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_)


class RFECVSelector(BaseEstimator, TransformerMixin):
    """Recursive Feature Elimination with CV (RandomForest base estimator)."""

    def __init__(
        self,
        force_keep_prefixes: Sequence[str] = FORCE_KEEP_PREFIXES,
        step: float | int = 0.2,
        cv: int = 3,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.force_keep_prefixes = tuple(force_keep_prefixes)
        self.step = step
        self.cv = cv
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y) -> "RFECVSelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RFECVSelector requires a DataFrame")
        self.feature_names_ = list(X.columns)
        from sklearn.feature_selection import RFECV
        try:
            base = RandomForestClassifier(
                n_estimators=self.n_estimators, n_jobs=-1,
                class_weight="balanced", random_state=self.random_state,
            )
            rfe = RFECV(
                estimator=base, step=self.step, cv=self.cv,
                scoring="f1_macro", n_jobs=-1, min_features_to_select=10,
            )
            rfe.fit(X.values, np.asarray(y))
            selected = [c for c, m in zip(self.feature_names_, rfe.support_) if m]
        except Exception as exc:
            logger.warning(
                "RFECVSelector failed (%s); keeping all features", exc,
            )
            selected = list(self.feature_names_)
        forced = [
            f for f in self.feature_names_
            if any(f.startswith(p) for p in self.force_keep_prefixes)
        ]
        self.selected_ = sorted(set(selected) | set(forced))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RFECVSelector.transform requires a DataFrame")
        cols = [c for c in self.selected_ if c in X.columns]
        return X[cols].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_)


class MISelector(BaseEstimator, TransformerMixin):
    """Mutual-information top-K selector."""

    def __init__(
        self,
        force_keep_prefixes: Sequence[str] = FORCE_KEEP_PREFIXES,
        k: int = 200,
        random_state: int = 42,
    ):
        self.force_keep_prefixes = tuple(force_keep_prefixes)
        self.k = k
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y) -> "MISelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MISelector requires a DataFrame")
        self.feature_names_ = list(X.columns)
        from sklearn.feature_selection import mutual_info_classif
        try:
            mi = mutual_info_classif(
                X.values, np.asarray(y), random_state=self.random_state,
            )
            order = np.argsort(mi)[::-1]
            k = min(self.k, len(self.feature_names_))
            keep_idx = set(order[:k].tolist())
            selected = [c for i, c in enumerate(self.feature_names_) if i in keep_idx]
        except Exception as exc:
            logger.warning("MISelector failed (%s); keeping all features", exc)
            selected = list(self.feature_names_)
        forced = [
            f for f in self.feature_names_
            if any(f.startswith(p) for p in self.force_keep_prefixes)
        ]
        self.selected_ = sorted(set(selected) | set(forced))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("MISelector.transform requires a DataFrame")
        cols = [c for c in self.selected_ if c in X.columns]
        return X[cols].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_)


SELECTORS = {
    "boruta": BorutaSelector,
    "l1_logreg": L1LogRegSelector,
    "rfecv": RFECVSelector,
    "mutual_info": MISelector,
}


def run_selector_ablation(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/selector_ablation",
    selectors: list[str] = None,
    models: list[str] = None,
    seed: int = 42,
    cv_n_folds: int = 3,
) -> pd.DataFrame:
    """Run a selector × model ablation matrix on scaffold-grouped CV.

    For each (selector, model) cell, fit a per-fold pipeline with that
    selector + model on scaffold-grouped 3-fold CV at one seed and report
    macro-F1 mean + std. Disable force_keep_prefixes for the ablation so
    selectors compete on equal terms.

    Output: ``selector_ablation.csv`` with columns
    [selector, model, fold, macro_f1, n_features_selected].
    """
    import joblib
    from sklearn.model_selection import StratifiedKFold
    from cancerag.ml.preprocessing import (
        build_full_pipeline, get_X_y_groups, make_grouped_cv,
    )
    from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    selectors = selectors or list(SELECTORS.keys())
    models = models or ["xgboost", "lightgbm", "elastic_lr", "random_forest"]

    df = pd.read_parquet(dataset_path)
    le = joblib.load(label_encoder_path)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    splits = make_grouped_cv(df, group_col="scaffold",
                              n_splits=cv_n_folds, seed=seed)
    if not splits:
        splits = list(StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                                       random_state=seed).split(X, y))

    rows: list[dict] = []
    for sname in selectors:
        SelClass = SELECTORS[sname]
        for mname in models:
            mfactory = MODEL_FACTORIES[mname]
            for fi, (tr, te) in enumerate(splits):
                model = mfactory(n_classes=n_classes, random_state=seed)
                # Disable force_keep for the ablation (fair comparison)
                selector = SelClass(force_keep_prefixes=())
                pipe = build_full_pipeline(
                    model, impute=True, variance_threshold=1e-4,
                    correlation_threshold=0.97, scale=True,
                    selector=selector,
                )
                cw = _combined_weight(y[tr], sw[tr])
                last = pipe.steps[-1][0]
                try:
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        pipe.fit(X.iloc[tr], y[tr],
                                 **{f"{last}__sample_weight": cw})
                    y_pred = pipe.predict(X.iloc[te])
                    from sklearn.metrics import f1_score
                    macro_f1 = float(f1_score(y[te], y_pred,
                                               average="macro", zero_division=0))
                    n_sel = len(pipe.named_steps["selector"].selected_)
                except Exception as exc:
                    logger.warning(
                        "[%s|%s|fold=%d] failed: %s", sname, mname, fi, exc,
                    )
                    macro_f1, n_sel = float("nan"), 0
                rows.append({
                    "selector": sname, "model": mname, "fold": fi,
                    "macro_f1": macro_f1, "n_features_selected": n_sel,
                })
                logger.info(
                    "[%s|%s|fold=%d] macro_f1=%.3f, n_features=%d",
                    sname, mname, fi, macro_f1, n_sel,
                )
    abl = pd.DataFrame(rows)
    abl.to_csv(output_dir / "selector_ablation_long.csv", index=False)

    # Aggregate to a 4×4 matrix
    summary = (
        abl.groupby(["selector", "model"])["macro_f1"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary.to_csv(output_dir / "selector_ablation_summary.csv", index=False)
    pivot = summary.pivot(index="selector", columns="model", values="mean")
    pivot.to_csv(output_dir / "selector_ablation_matrix.csv")
    logger.info("Selector × model matrix:\n%s", pivot.to_string())
    return abl


def stability_selection(
    X: pd.DataFrame, y, selector_factory, *,
    n_boot: int = 30, sample_frac: float = 0.8, seed: int = 42,
) -> pd.Series:
    """Bootstrap-resample (X, y); fit ``selector_factory()`` on each
    subsample; return per-feature selection frequency as a ``Series``."""
    rng = np.random.default_rng(seed)
    counts = pd.Series(0, index=X.columns, dtype=int)
    n = len(X)
    if n == 0:
        return counts.astype(float)
    for i in range(n_boot):
        idx = rng.choice(n, size=max(1, int(round(sample_frac * n))), replace=False)
        sel = selector_factory()
        sel.fit(X.iloc[idx], np.asarray(y)[idx])
        for f in getattr(sel, "selected_", []):
            if f in counts.index:
                counts[f] += 1
        if (i + 1) % 5 == 0:
            logger.info("stability_selection: %d/%d bootstraps done", i + 1, n_boot)
    return (counts / n_boot).sort_values(ascending=False)


# ----------------------------------------------- pre-screens


def variance_prescreen(X: pd.DataFrame, *, threshold: float = 0.001) -> list[str]:
    """Drop near-constant columns. Returns the surviving column names."""
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X)
    kept = X.columns[vt.get_support()].tolist()
    logger.info(
        "variance_prescreen: %d -> %d columns (threshold=%g)",
        len(X.columns), len(kept), threshold,
    )
    return kept


def univariate_topk(
    X: pd.DataFrame, y, *, k: int = 500
) -> list[str]:
    """Top-K columns by ANOVA F-statistic against ``y``."""
    k = min(k, X.shape[1])
    skb = SelectKBest(f_classif, k=k)
    skb.fit(X, y)
    kept = X.columns[skb.get_support()].tolist()
    logger.info(
        "univariate_topk: %d -> %d columns (k=%d)",
        len(X.columns), len(kept), k,
    )
    return kept


# ----------------------------------------------- orchestrator


def run_feature_selection(
    config: dict | None = None,
    *,
    preprocessed_dir: Path | str = "data/processed/ml_preprocessed",
    output_dir: Path | str = "data/processed/ml_selected_features",
    variance_threshold: float = 0.001,
    univariate_k: int = 500,
    use_boruta: bool = True,
    boruta_n_boot: int = 10,
    boruta_stability_threshold: float = 0.6,
) -> tuple[pd.DataFrame, dict]:
    """Apply the layered feature-selection pipeline once on the full
    train-eligible matrix and persist the selected feature list. The actual
    fit-inside-CV happens in the trainer (Stage 10).
    """
    preprocessed_dir = Path(preprocessed_dir); output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(preprocessed_dir / "X_train_eligible.parquet")
    y = np.load(preprocessed_dir / "y_train_eligible.npy")
    logger.info("Loaded preprocessed data: %d rows × %d cols", *X.shape)

    # 1. Variance pre-screen
    survivors = variance_prescreen(X, threshold=variance_threshold)
    X1 = X[survivors]

    # 2. Univariate top-K
    topk = univariate_topk(X1, y, k=univariate_k)
    X2 = X1[topk]

    # 3. Force-keep methodologically important columns even if pre-screens
    # dropped them
    forced = [
        c for c in X.columns
        if any(c.startswith(p) for p in FORCE_KEEP_PREFIXES)
    ]
    survivors_set = set(topk) | set(forced)
    X3 = X[sorted(survivors_set)]
    logger.info("After force-keep: %d columns", X3.shape[1])

    # 4. Boruta with stability selection
    selected = sorted(survivors_set)
    stability_freq: pd.Series = pd.Series(dtype=float)
    if use_boruta:
        logger.info(
            "Running Boruta + stability selection (n_boot=%d, threshold=%.2f)",
            boruta_n_boot, boruta_stability_threshold,
        )
        try:
            stability_freq = stability_selection(
                X3, y,
                selector_factory=lambda: BorutaSelector(
                    force_keep_prefixes=FORCE_KEEP_PREFIXES,
                    max_iter=50,
                ),
                n_boot=boruta_n_boot,
            )
            stable = stability_freq[
                stability_freq >= boruta_stability_threshold
            ].index.tolist()
            forced_set = set(forced)
            selected = sorted(set(stable) | forced_set)
            logger.info(
                "Boruta-stable selection: %d columns (forced kept: %d)",
                len(selected), len(forced_set),
            )
        except Exception as exc:
            logger.warning(
                "Boruta stage failed (%s); using pre-screen survivors", exc,
            )

    X_sel = X3[selected]

    # 5. Persist
    X_sel.to_parquet(output_dir / "X_train_eligible_selected.parquet", index=False)
    np.save(output_dir / "y_train_eligible.npy", y)
    if (preprocessed_dir / "sample_weight_train_eligible.npy").exists():
        np.save(
            output_dir / "sample_weight_train_eligible.npy",
            np.load(preprocessed_dir / "sample_weight_train_eligible.npy"),
        )
    (output_dir / "selected_features.json").write_text(
        json.dumps(selected, indent=2)
    )
    if not stability_freq.empty:
        stability_freq.to_csv(output_dir / "stability_selection_freq.csv",
                              header=["frequency"])

    summary = {
        "method": (
            "variance + univariate F + Boruta stability selection"
            if use_boruta else "variance + univariate F (Boruta disabled)"
        ),
        "original_features": int(X.shape[1]),
        "after_variance_filter": len(survivors),
        "after_univariate_topk": len(topk),
        "force_kept": len(forced),
        "selected_features": len(selected),
        "reduction_percentage": float(100 * (1 - len(selected) / max(1, X.shape[1]))),
        "boruta_stability_threshold": boruta_stability_threshold,
        "boruta_n_boot": boruta_n_boot if use_boruta else 0,
    }
    (output_dir / "feature_selection_meta.json").write_text(
        json.dumps({
            "schema_version": 1,
            "selected_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
        }, indent=2, sort_keys=True)
    )
    logger.info("Feature selection done: %s", summary)
    return X_sel, summary


# ----------------------------------------------- entry point


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    X_sel, summary = run_feature_selection()
    print("\nFeature selection summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  output shape: {X_sel.shape}")


if __name__ == "__main__":
    main()
