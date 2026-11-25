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


# Pair-level structural columns protected from selector pruning. These all vary
# with the (ligand, receptor) pair, so force-keeping them preserves the
# structural contribution rather than a shortcut.
#
# ``redock_rmsd`` and ``gnina_cnn`` were previously force-kept here too. They are
# per-receptor constants (see preprocessing.RECEPTOR_QC_COLS) and no longer enter
# X at all, so protecting them would only shield a receptor-identity proxy from
# the one mechanism that could have dropped it.
FORCE_KEEP_PREFIXES: tuple[str, ...] = (
    "vina_affinity", "vina_pose_", "vina_n_distinct",
    "ifp_", "n_residues_contacted", "n_total_contacts",
    "Asphericity", "Eccentricity", "NPR", "PMI", "RadiusOfGyration",
    "SpherocityIndex", "InertialShapeFactor",
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
        # A missing `boruta` is an environment fault, not a data fault, and it
        # must not be absorbed. Absorbing it was worse than a crash: the
        # fallback keeps every column, the fitted pipeline still scores, and
        # the ablation then files that score in a row labelled `boruta`. The
        # artefact says Boruta was measured when no selection happened at all.
        # Seen 2026-11-25: twelve `boruta` cells recorded in a sandbox that
        # never had the package installed.
        from boruta import BorutaPy
        self.fallback_reason_ = None
        try:
            boruta = BorutaPy(
                rf, n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_iter=self.max_iter, verbose=self.verbose,
            )
            boruta.fit(X.values, np.asarray(y))
            selected = list(np.array(self.feature_names_)[boruta.support_])
        except Exception as exc:
            # A genuine failure to converge on this fold is recoverable, but
            # it is recorded on the estimator so a caller can refuse to
            # publish the number as Boruta's.
            self.fallback_reason_ = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Boruta failed (%s); falling back to no-selection. This cell "
                "is NOT a Boruta measurement.", exc,
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
        # Constructed outside the try: a bad keyword is a fault in this repo,
        # not a property of the fold. `multi_class="multinomial"` was removed
        # in scikit-learn 1.7 and used to be raised and absorbed here, so
        # every `l1_logreg` cell kept all columns and the ablation filed a
        # no-selection score under L1's name (measured 2026-11-25: L1 and
        # RFECV returned byte-identical macro-F1 on the same fold).
        # Multinomial is now the default for multiclass on `saga`.
        lr = LogisticRegression(
            penalty="l1", solver="saga",
            C=self.C, max_iter=self.max_iter, n_jobs=-1,
            random_state=self.random_state, class_weight="balanced",
        )
        self.fallback_reason_ = None
        try:
            lr.fit(X.values, np.asarray(y))
            mask = np.any(np.abs(lr.coef_) > 1e-8, axis=0)
            selected = [c for c, m in zip(self.feature_names_, mask) if m]
        except TypeError:
            raise
        except Exception as exc:
            self.fallback_reason_ = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "L1LogRegSelector failed (%s); falling back to no-selection. "
                "This cell is NOT an L1 measurement.", exc,
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
        # Constructed outside the try, for the reason given in L1LogRegSelector.
        base = RandomForestClassifier(
            n_estimators=self.n_estimators, n_jobs=-1,
            class_weight="balanced", random_state=self.random_state,
        )
        rfe = RFECV(
            estimator=base, step=self.step, cv=self.cv,
            scoring="f1_macro", n_jobs=-1, min_features_to_select=10,
        )
        self.fallback_reason_ = None
        try:
            rfe.fit(X.values, np.asarray(y))
            selected = [c for c, m in zip(self.feature_names_, rfe.support_) if m]
        except TypeError:
            raise
        except Exception as exc:
            self.fallback_reason_ = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "RFECVSelector failed (%s); falling back to no-selection. "
                "This cell is NOT an RFECV measurement.", exc,
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
        self.fallback_reason_ = None
        try:
            mi = mutual_info_classif(
                X.values, np.asarray(y), random_state=self.random_state,
            )
            order = np.argsort(mi)[::-1]
            k = min(self.k, len(self.feature_names_))
            keep_idx = set(order[:k].tolist())
            selected = [c for i, c in enumerate(self.feature_names_) if i in keep_idx]
        except TypeError:
            # A signature fault, not a fold property; see L1LogRegSelector.
            raise
        except Exception as exc:
            self.fallback_reason_ = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "MISelector failed (%s); falling back to no-selection. This "
                "cell is NOT a mutual-information measurement.", exc,
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
    seeds: Sequence[int] | None = None,
    shuffle_folds: bool = False,
    cv_n_folds: int = 3,
) -> pd.DataFrame:
    """Run a selector × model ablation matrix on scaffold-grouped CV.

    For each (selector, model) cell, fit a per-fold pipeline with that
    selector + model on scaffold-grouped CV and report macro-F1 mean + std.
    Disable force_keep_prefixes for the ablation so selectors compete on
    equal terms.

    ``seeds`` and ``shuffle_folds`` exist together, and one is useless
    without the other. Plain ``GroupKFold`` is deterministic, so sweeping
    seeds over unshuffled folds re-fits on a single partition and varies only
    model initialisation: the resulting spread is not resampling variance and
    cannot say whether one selector really beats another. Passing several
    ``seeds`` with ``shuffle_folds=True`` reshuffles the scaffold-to-fold
    assignment per seed, which is what makes a ranking between selectors
    testable rather than an artefact of one split (HAL-6168).

    Defaults reproduce the earlier single-seed, unshuffled behaviour.

    Output: ``selector_ablation_long.csv`` with columns
    [selector, model, seed, fold, macro_f1, n_features_selected,
    fallback_reason].
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

    seed_list = list(seeds) if seeds else [seed]

    def splits_for(s_: int):
        sp = make_grouped_cv(df, group_col="scaffold", n_splits=cv_n_folds,
                             seed=s_, shuffle=shuffle_folds)
        if not sp:
            sp = list(StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                                      random_state=s_).split(X, y))
        return sp

    rows: list[dict] = []
    for sname in selectors:
        SelClass = SELECTORS[sname]
        for mname in models:
            mfactory = MODEL_FACTORIES[mname]
            for cv_seed in seed_list:
              for fi, (tr, te) in enumerate(splits_for(cv_seed)):
                  model = mfactory(n_classes=n_classes, random_state=cv_seed)
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
                      # A selector that fell back to keeping every column did
                      # not measure what this row is named after. Carry the
                      # reason into the artefact so the number can never be
                      # quoted as that selector's.
                      fallback = getattr(
                          pipe.named_steps["selector"], "fallback_reason_", None)
                  except Exception as exc:
                      logger.warning(
                          "[%s|%s|fold=%d] failed: %s", sname, mname, fi, exc,
                      )
                      macro_f1, n_sel = float("nan"), 0
                      fallback = f"{type(exc).__name__}: {exc}"
                  rows.append({
                      "selector": sname, "model": mname, "seed": cv_seed,
                      "fold": fi,
                      "macro_f1": macro_f1, "n_features_selected": n_sel,
                      "fallback_reason": fallback,
                  })
                  logger.info(
                      "[%s|%s|seed=%d|fold=%d] macro_f1=%.3f, n_features=%d%s",
                      sname, mname, cv_seed, fi, macro_f1, n_sel,
                      f"  FALLBACK({fallback})" if fallback else "",
                  )
    abl = flag_no_selection_collisions(pd.DataFrame(rows))
    abl.to_csv(output_dir / "selector_ablation_long.csv", index=False)

    collisions = abl[abl["collides_with"].astype(bool)]
    if len(collisions):
        logger.error(
            "%d of %d fits are identical to another selector's fit on the same "
            "(model, seed, fold). Two selectors cannot be told apart here, and "
            "the usual cause is both falling back to no selection:\n%s",
            len(collisions), len(abl),
            collisions[["selector", "model", "seed", "fold", "macro_f1",
                        "n_features_selected", "collides_with"]].to_string(index=False),
        )

    n_fallback = int(abl["fallback_reason"].notna().sum())
    if n_fallback:
        logger.error(
            "%d of %d cells fell back and are NOT measurements of the "
            "selector they are filed under: %s", n_fallback, len(abl),
            sorted(set(abl.loc[abl["fallback_reason"].notna(), "selector"])),
        )

    # Aggregate to a 4×4 matrix
    summary = (
        abl.groupby(["selector", "model"])["macro_f1"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    # How many of each cell's folds are real measurements of that selector.
    ok = (abl[abl["fallback_reason"].isna()]
          .groupby(["selector", "model"]).size().rename("n_not_fallback"))
    summary = summary.merge(ok, on=["selector", "model"], how="left")
    summary["n_not_fallback"] = summary["n_not_fallback"].fillna(0).astype(int)
    n_coll = (abl[abl["collides_with"].astype(bool)]
              .groupby(["selector", "model"]).size().rename("n_collisions"))
    summary = summary.merge(n_coll, on=["selector", "model"], how="left")
    summary["n_collisions"] = summary["n_collisions"].fillna(0).astype(int)
    summary["n_seeds"] = len(seed_list)
    summary["shuffle_folds"] = bool(shuffle_folds)
    summary.to_csv(output_dir / "selector_ablation_summary.csv", index=False)

    # One complete CV estimate per (selector, model, seed). The folds inside a
    # seed are not independent, so this -- not the fold list -- is the unit a
    # ranking between selectors should be tested on.
    per_seed = (abl.groupby(["selector", "model", "seed"], as_index=False)
                   .macro_f1.mean())
    per_seed.to_csv(output_dir / "selector_ablation_seeds.csv", index=False)
    pivot = summary.pivot(index="selector", columns="model", values="mean")
    pivot.to_csv(output_dir / "selector_ablation_matrix.csv")
    logger.info("Selector × model matrix:\n%s", pivot.to_string())
    return abl


SELECTOR_ABLATION_ARTEFACTS: tuple[str, ...] = (
    "selector_ablation_long.csv",
    "selector_ablation_matrix.csv",
    "selector_ablation_summary.csv",
    "selector_ablation_seeds.csv",
)


def flag_no_selection_collisions(abl: pd.DataFrame) -> pd.DataFrame:
    """Mark fits where two different selectors returned the identical result.

    A selector that falls back keeps every column it was handed, so two
    fallen-back selectors fit the same model on the same rows with the same
    features and score the same to the last digit. That collision is
    detectable from the artefact alone, without any selector reporting its own
    failure -- which is exactly the case this exists for. On 2026-11-25 only
    `BorutaSelector` set ``fallback_reason_``; L1 and RFECV each absorbed their
    failure silently and the run's own fallback count read zero, while
    ``l1_logreg`` and ``rfecv`` carried byte-identical macro-F1 (0.4306958549)
    on fold 2 of every learner.

    Adds a ``collides_with`` column naming the other selectors that produced an
    identical ``(macro_f1, n_features_selected)`` for the same
    ``(model, seed, fold)``, empty where there is no collision. Two selectors
    that genuinely agree on a subset collide too, and that is reported rather
    than excused: the ablation cannot tell them apart either.
    """
    abl = abl.copy()
    keys = [c for c in ("model", "seed", "fold") if c in abl.columns]
    collides: dict[int, str] = {}
    for _, g in abl.groupby(keys, dropna=False):
        scored = g[g.macro_f1.notna()]
        for _, pair in scored.groupby(["macro_f1", "n_features_selected"]):
            if pair.selector.nunique() < 2:
                continue
            for idx, row in pair.iterrows():
                others = sorted(set(pair.selector) - {row.selector})
                collides[idx] = ", ".join(others)
    abl["collides_with"] = pd.Series(collides, dtype="object").reindex(abl.index)
    abl["collides_with"] = abl["collides_with"].fillna("")
    return abl


def stale_selector_ablation_artefacts(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    output_dir: Path | str = "data/processed/ml_models/selector_ablation",
) -> list[str]:
    """Names of selector-ablation artefacts older than the matrix they describe.

    The ablation's numbers are only about the feature matrix that produced
    them. When ``ml_ready_dataset.parquet`` is rebuilt — as it was by the
    2026-08 QC-leakage fix — any artefact left behind from before that
    rebuild silently describes a matrix that no longer exists, while still
    reading as a current result to anyone citing it.

    Returns the basenames of stale artefacts, newest-wins; empty when every
    artefact postdates the dataset. A missing dataset or a missing artefact is
    not reported here: this answers "is what is on disk consistent", not "is
    everything present".
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    if not dataset_path.exists():
        return []
    dataset_mtime = dataset_path.stat().st_mtime
    stale: list[str] = []
    for name in SELECTOR_ABLATION_ARTEFACTS:
        artefact = output_dir / name
        if artefact.exists() and artefact.stat().st_mtime < dataset_mtime:
            stale.append(name)
    return stale


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
