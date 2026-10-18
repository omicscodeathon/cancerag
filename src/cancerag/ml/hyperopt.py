"""
Phase 6 — Hyperparameter tuning + per-class threshold optimization.

Two complementary "free lift" techniques on top of the bake-off winner:

  1. **Optuna tuning** of the winning model (LightGBM by default) over its
     standard hyperparameter space. Optimizes scaffold-grouped 5-fold CV
     macro-F1 (the chemistry-realistic objective) with TPE sampler +
     median-pruner. Persists the best params + retrained model.

  2. **Per-class threshold optimization** on the calibrated model's
     `predict_proba` output. The default `argmax` decision rule favours the
     majority class (G protein, ~57% of train-eligible rows); per-class
     multipliers learned on out-of-fold probabilities trade some G-protein
     recall for minority-class recall. The multipliers are only meaningful
     for the model whose probabilities they were fit on, so the out-of-fold
     model is the same estimator, with the same hyperparameters and the same
     calibration, as the saved model they are applied to at holdout (the
     tuned LightGBM when it exists, otherwise the bake-off winner).

Run:
    python -m cancerag.ml.hyperopt tune
    python -m cancerag.ml.hyperopt threshold
    python -m cancerag.ml.hyperopt all
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from cancerag.ml.preprocessing import (
    build_full_pipeline, get_X_y_groups, make_grouped_cv,
)

logger = logging.getLogger(__name__)


OUTPUT_DIR = Path("data/processed/ml_models/advanced")
SELECTION_PATH = Path("data/processed/ml_models/selection_decision.json")
DATASET_PATH = Path("data/processed/ml_ready_dataset.parquet")
HOLDOUT_PATH = Path("data/holdout/dataset_holdout.parquet")
LE_PATH = Path("data/processed/ml_preprocessed/label_encoder.joblib")
MODELS_DIR = Path("data/processed/ml_models")

# The Optuna search space (``_build_lightgbm_objective``) is LightGBM's, so the
# tuned model is always LightGBM whatever the bake-off chose.
TUNED_MODEL = "lightgbm"
TUNED_MODEL_FILE = "lightgbm_tuned_calibrated.joblib"
TUNING_META_FILE = "optuna_tuning_meta.json"


# --------------------------------------------- Optuna tuning


def _build_lightgbm_objective(X, y, sw, n_classes, splits):
    """Closure that returns an Optuna objective for LightGBM."""
    import lightgbm as lgb
    from cancerag.ml.model_training import _combined_weight

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        scores = []
        for fi, (tr, te) in enumerate(splits):
            model = lgb.LGBMClassifier(
                objective="multiclass" if n_classes > 2 else "binary",
                num_class=n_classes if n_classes > 2 else None,
                n_jobs=4, verbose=-1, class_weight="balanced",
                random_state=42, **params,
            )
            pipe = build_full_pipeline(model)
            cw = _combined_weight(y[tr], sw[tr])
            last = pipe.steps[-1][0]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X.iloc[tr], y[tr],
                             **{f"{last}__sample_weight": cw})
                y_pred = pipe.predict(X.iloc[te])
                scores.append(float(f1_score(y[te], y_pred,
                                              average="macro", zero_division=0)))
            except Exception as exc:
                logger.warning("Optuna fold %d failed: %s", fi, exc)
                scores.append(0.0)
            # Pruning: report partial score after each fold
            trial.report(float(np.mean(scores)), step=fi)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(scores))

    return objective


def tuning_meta(
    *, winner: str, n_trials: int, best_score: float, best_params: dict,
    holdout_metrics: dict,
) -> dict:
    """The record written to ``optuna_tuning_meta.json``.

    ``model`` is the estimator that was tuned and saved (always LightGBM: the
    search space is LightGBM's). ``bake_off_winner`` is kept separately so the
    two are never conflated; earlier runs wrote the winner under ``model``,
    which labelled a tuned LightGBM as a random forest.
    """
    return {
        "schema_version": 2,
        "tuned_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": TUNED_MODEL,
        "bake_off_winner": winner,
        "n_trials": n_trials,
        "best_cv_macro_f1": best_score,
        "best_params": best_params,
        "holdout_metrics": holdout_metrics,
    }


def make_tuned_lightgbm(*, n_classes: int, random_state: int, best_params: dict):
    """The tuned LightGBM exactly as ``run_optuna`` builds it for the final fit."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else None,
        n_jobs=4, verbose=-1, class_weight="balanced",
        random_state=random_state, **best_params,
    )


def run_optuna(
    *, n_trials: int = 50, seed: int = 42, output_dir: Path | str = OUTPUT_DIR,
) -> dict:
    """Tune the bake-off winner with Optuna; persist best params + retrained
    final + calibrated model."""
    global optuna
    import optuna
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    decision = json.loads(SELECTION_PATH.read_text())
    winner = decision["chosen"]
    logger.info("Tuning %s with Optuna (n_trials=%d; bake-off winner %s)",
                TUNED_MODEL, n_trials, winner)
    if winner != "lightgbm":
        logger.warning(
            "Optuna search space currently defined for lightgbm only; "
            "winner is %s — running on lightgbm anyway", winner,
        )

    df = pd.read_parquet(DATASET_PATH)
    le = joblib.load(LE_PATH)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)
    splits = make_grouped_cv(df, group_col="scaffold", n_splits=5, seed=seed)
    if not splits:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=seed).split(X, y))

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=f"cancerag_{TUNED_MODEL}_scaffold_cv",
    )
    objective = _build_lightgbm_objective(X, y, sw, n_classes, splits)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = float(study.best_value)
    logger.info("Best CV macro-F1: %.4f", best_score)
    logger.info("Best params: %s", best_params)

    # Retrain on all train-eligible with best params
    from cancerag.ml.model_training import _combined_weight
    from sklearn.calibration import CalibratedClassifierCV

    model = make_tuned_lightgbm(
        n_classes=n_classes, random_state=seed, best_params=best_params,
    )
    pipe = build_full_pipeline(model)
    cw_all = _combined_weight(y, sw)
    last = pipe.steps[-1][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y, **{f"{last}__sample_weight": cw_all})
    joblib.dump(pipe, output_dir / "lightgbm_tuned_final.joblib")

    # Calibrate
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3).fit(
                X, y, **{f"{last}__sample_weight": cw_all},
            )
        joblib.dump(cal, output_dir / TUNED_MODEL_FILE)
    except Exception as exc:
        logger.warning("Calibration failed: %s", exc)
        cal = pipe

    # Holdout eval
    holdout_metrics = {}
    if HOLDOUT_PATH.exists():
        from cancerag.ml.model_evaluation import report_metrics
        ho = pd.read_parquet(HOLDOUT_PATH)
        ho_X, ho_y, _, _, _ = get_X_y_groups(ho, label_encoder=le)
        for c in X.columns:
            if c not in ho_X.columns:
                ho_X[c] = 0.0
        ho_X = ho_X[X.columns]
        ho_pred = cal.predict(ho_X)
        rep = report_metrics(ho_y, ho_pred, bootstrap_n=1000, seed=seed)
        holdout_metrics = {
            "n": rep["n_test"], "macro_f1": rep["macro_f1"],
            "balanced_accuracy": rep["balanced_accuracy"],
            "per_class_f1": rep["per_class_f1"],
        }

    out = tuning_meta(
        winner=winner, n_trials=n_trials, best_score=best_score,
        best_params=best_params, holdout_metrics=holdout_metrics,
    )
    (output_dir / TUNING_META_FILE).write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str)
    )

    # Also persist the trial history for transparency
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    logger.info("Optuna tuning complete: trials -> %s",
                output_dir / "optuna_trials.csv")
    return out


# --------------------------------------------- per-class threshold optimization


def final_estimator(model):
    """Unwrap calibration and pipeline wrappers down to the classifier."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    seen = 0
    while seen < 10:
        seen += 1
        if isinstance(model, CalibratedClassifierCV):
            fitted = getattr(model, "calibrated_classifiers_", None)
            model = fitted[0].estimator if fitted else model.estimator
        elif isinstance(model, Pipeline):
            model = model.steps[-1][1]
        else:
            return model
    raise ValueError("could not unwrap model to a final estimator")


def _fit_kwargs(model, sample_weight) -> dict:
    """Route sample weights to the classifier step, as the trainer does."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline

    inner = model.estimator if isinstance(model, CalibratedClassifierCV) else model
    if isinstance(inner, Pipeline):
        return {f"{inner.steps[-1][0]}__sample_weight": sample_weight}
    return {"sample_weight": sample_weight}


@dataclass
class ThresholdModel:
    """The model whose probabilities the thresholds are learned on and
    applied to.

    ``make_oof_model`` builds an unfitted copy for the out-of-fold loop;
    ``load_holdout_model`` loads the saved model the thresholds are applied
    to. Calibration settings are read from the saved model itself, so the
    out-of-fold probabilities pass through the same calibration as the
    holdout ones and the two are on the same scale.
    """

    name: str
    factory_name: str
    params: dict
    holdout_model_path: Path
    _holdout: object = field(default=None, repr=False, compare=False)

    def load_holdout_model(self):
        if self._holdout is None:
            self._holdout = joblib.load(self.holdout_model_path)
        return self._holdout

    def calibration(self) -> dict | None:
        from sklearn.calibration import CalibratedClassifierCV
        ho = self.load_holdout_model()
        if isinstance(ho, CalibratedClassifierCV):
            return {"method": ho.method, "cv": ho.cv}
        return None

    def make_estimator(self, *, n_classes: int, seed: int):
        if self.name == f"{TUNED_MODEL}_tuned":
            return make_tuned_lightgbm(
                n_classes=n_classes, random_state=seed, best_params=self.params,
            )
        from cancerag.ml.model_training import resolve_model_factory
        factory = resolve_model_factory(self.factory_name)
        return factory(n_classes=n_classes, random_state=seed, **self.params)

    def make_oof_model(self, *, n_classes: int, seed: int):
        from sklearn.calibration import CalibratedClassifierCV
        pipe = build_full_pipeline(
            self.make_estimator(n_classes=n_classes, seed=seed)
        )
        cal = self.calibration()
        if cal is None:
            return pipe
        return CalibratedClassifierCV(pipe, method=cal["method"], cv=cal["cv"])

    def check_matches_holdout(self, *, n_classes: int) -> None:
        """Refuse to learn thresholds on a different model than the one they
        will be applied to."""
        oof = final_estimator(self.make_oof_model(n_classes=n_classes, seed=0))
        ho = final_estimator(self.load_holdout_model())
        if type(oof) is not type(ho):
            raise ValueError(
                "threshold OOF model and holdout model are not the same model: "
                f"OOF {type(oof).__name__} vs holdout {type(ho).__name__} "
                f"({self.holdout_model_path})"
            )
        ho_params = ho.get_params()
        diff = {k: (v, ho_params.get(k)) for k, v in self.params.items()
                if ho_params.get(k) != v}
        if diff:
            raise ValueError(
                "threshold OOF model and holdout model are not the same model: "
                f"hyperparameters differ {diff}"
            )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "estimator": self.factory_name,
            "params": self.params,
            "calibration": self.calibration(),
            "holdout_model_path": str(self.holdout_model_path),
        }


def resolve_threshold_model(
    *, winner: str, output_dir: Path | str = OUTPUT_DIR,
    models_dir: Path | str = MODELS_DIR,
) -> ThresholdModel:
    """Pick the model thresholds are learned on and applied to.

    The tuned LightGBM when both its saved model and its ``best_params`` are
    present; otherwise the bake-off winner's saved (calibrated) final model.
    """
    output_dir, models_dir = Path(output_dir), Path(models_dir)
    tuned_path = output_dir / TUNED_MODEL_FILE
    meta_path = output_dir / TUNING_META_FILE
    if tuned_path.exists() and meta_path.exists():
        best_params = json.loads(meta_path.read_text()).get("best_params")
        if best_params:
            return ThresholdModel(
                name=f"{TUNED_MODEL}_tuned", factory_name=TUNED_MODEL,
                params=dict(best_params), holdout_model_path=tuned_path,
            )
    path = models_dir / f"{winner}_final_calibrated.joblib"
    if not path.exists():
        path = models_dir / f"{winner}_final.joblib"
    return ThresholdModel(
        name=winner, factory_name=winner, params={}, holdout_model_path=path,
    )


def predict_with_thresholds(p, thr):
    """Scale each class column by 1/threshold, then argmax."""
    return (np.asarray(p) / np.asarray(thr)[None, :]).argmax(axis=1)


THRESHOLD_GRID = np.linspace(0.5, 2.0, 16)


def search_thresholds(y, proba, *, grid=None, n_passes: int = 3):
    """Coordinate descent over per-class multipliers maximising macro-F1."""
    n_classes = proba.shape[1]
    grid = THRESHOLD_GRID if grid is None else grid
    best_thr = np.ones(n_classes, dtype=float)
    best_f1 = float(f1_score(y, proba.argmax(axis=1),
                             average="macro", zero_division=0))
    for _ in range(n_passes):
        for ci in range(n_classes):
            for v in grid:
                cand = best_thr.copy(); cand[ci] = v
                f1 = float(f1_score(y, predict_with_thresholds(proba, cand),
                                    average="macro", zero_division=0))
                if f1 > best_f1:
                    best_f1, best_thr = f1, cand
    return best_thr, best_f1


def paired_bootstrap_delta(y, pred_a, pred_b, *, n_boot=1000, seed=42) -> dict:
    """Macro-F1(b) - macro-F1(a) on the same resampled rows."""
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (
            f1_score(y[idx], pred_b[idx], average="macro", zero_division=0)
            - f1_score(y[idx], pred_a[idx], average="macro", zero_division=0)
        )
    lo, mid, hi = np.percentile(deltas, [2.5, 50.0, 97.5])
    point = (f1_score(y, pred_b, average="macro", zero_division=0)
             - f1_score(y, pred_a, average="macro", zero_division=0))
    return {"point_estimate": float(point), "ci_lo": float(lo),
            "ci_median": float(mid), "ci_hi": float(hi),
            "frac_boot_positive": float((deltas > 0).mean())}


def optimize_thresholds(
    *, output_dir: Path | str = OUTPUT_DIR, n_seeds: int = 3,
    models_dir: Path | str = MODELS_DIR,
) -> dict:
    """Find per-class probability multipliers that maximise macro-F1 on
    out-of-fold predictions of the model they will be applied to.

    Uses scaffold-grouped CV to avoid leakage; thresholds are then frozen and
    applied to the saved model's holdout probabilities.
    """
    from cancerag.ml.model_training import _combined_weight
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    decision = json.loads(SELECTION_PATH.read_text())
    winner = decision["chosen"]
    df = pd.read_parquet(DATASET_PATH)
    le = joblib.load(LE_PATH)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    spec = resolve_threshold_model(
        winner=winner, output_dir=output_dir, models_dir=models_dir,
    )
    spec.check_matches_holdout(n_classes=n_classes)
    logger.info("Threshold-opt model: %s (holdout model %s, calibration %s)",
                spec.name, spec.holdout_model_path, spec.calibration())

    # Out-of-fold probabilities, averaged over the seeds that scored each row
    oof_sum = np.zeros((len(X), n_classes), dtype=np.float64)
    n_scored = np.zeros(len(X), dtype=int)
    n_failed = 0
    for seed in range(42, 42 + n_seeds):
        splits = make_grouped_cv(df, group_col="scaffold", n_splits=5, seed=seed)
        if not splits:
            continue
        for tr, te in splits:
            model = spec.make_oof_model(n_classes=n_classes, seed=seed)
            cw = _combined_weight(y[tr], sw[tr])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X.iloc[tr], y[tr], **_fit_kwargs(model, cw))
                oof_sum[te] += model.predict_proba(X.iloc[te])
                n_scored[te] += 1
            except Exception as exc:
                n_failed += 1
                logger.warning("threshold-opt fold failed: %s", exc)

    seen = n_scored > 0
    oof_proba = np.zeros_like(oof_sum)
    oof_proba[seen] = oof_sum[seen] / n_scored[seen, None]
    eligible = seen & (y >= 0)
    y_sub = y[eligible]; p_sub = oof_proba[eligible]

    oof_argmax_f1 = float(f1_score(y_sub, p_sub.argmax(axis=1),
                                   average="macro", zero_division=0))
    best_thr, best_f1 = search_thresholds(y_sub, p_sub)
    logger.info(
        "Threshold-opt: OOF argmax macro-F1=%.4f -> tuned=%.4f (thresholds=%s)",
        oof_argmax_f1, best_f1, best_thr.tolist(),
    )
    oof_per_class = f1_score(
        y_sub, predict_with_thresholds(p_sub, best_thr), average=None,
        zero_division=0, labels=list(range(n_classes)),
    )
    oof_per_class_argmax = f1_score(
        y_sub, p_sub.argmax(axis=1), average=None, zero_division=0,
        labels=list(range(n_classes)),
    )

    holdout_metrics = {}
    if HOLDOUT_PATH.exists():
        from cancerag.ml.model_evaluation import report_metrics
        cal = spec.load_holdout_model()
        ho = pd.read_parquet(HOLDOUT_PATH)
        ho_X, ho_y, _, _, _ = get_X_y_groups(ho, label_encoder=le)
        for c in X.columns:
            if c not in ho_X.columns:
                ho_X[c] = 0.0
        ho_X = ho_X[X.columns]
        ho_proba = cal.predict_proba(ho_X)
        ho_pred_argmax = ho_proba.argmax(axis=1)
        ho_pred_tuned = predict_with_thresholds(ho_proba, best_thr)
        rep = report_metrics(ho_y, ho_pred_tuned, bootstrap_n=1000, seed=42)
        rep0 = report_metrics(ho_y, ho_pred_argmax, bootstrap_n=1000, seed=42)
        holdout_metrics = {
            "n": rep["n_test"],
            "macro_f1_tuned_thresholds": rep["macro_f1"],
            "per_class_f1_tuned": rep["per_class_f1"],
            "macro_f1_argmax": rep0["macro_f1"],
            "per_class_f1_argmax": rep0["per_class_f1"],
            "delta_macro_f1_paired_bootstrap": paired_bootstrap_delta(
                ho_y, ho_pred_argmax, ho_pred_tuned, n_boot=1000, seed=42,
            ),
        }

    classes = [str(c) for c in le.classes_]
    out = {
        "schema_version": 2,
        "optimized_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_classes": classes,
        "threshold_model": spec.describe(),
        "bake_off_winner": winner,
        "n_seeds": n_seeds,
        "n_oof_rows": int(eligible.sum()),
        "n_failed_folds": n_failed,
        "thresholds": best_thr.tolist(),
        "oof_macro_f1_argmax": oof_argmax_f1,
        "best_oof_macro_f1": best_f1,
        "notes": {
            "best_oof_macro_f1": (
                "evaluated on the same out-of-fold probabilities the "
                "thresholds were searched on, so it is an optimistic, "
                "in-sample figure; the holdout delta is the unbiased check"
            ),
            "thresholds_at_grid_edge": [
                classes[i] for i, t in enumerate(best_thr)
                if np.isclose(t, THRESHOLD_GRID.min())
                or np.isclose(t, THRESHOLD_GRID.max())
            ],
        },
        "oof_per_class_f1_argmax": dict(zip(classes, map(float, oof_per_class_argmax))),
        "oof_per_class_f1_tuned": dict(zip(classes, map(float, oof_per_class))),
        "holdout_metrics_with_tuned_thresholds": holdout_metrics,
    }
    (output_dir / "thresholds_meta.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str)
    )
    return out


# --------------------------------------------- entry point


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("tune", "all"):
        logger.info("=== Optuna tuning ===")
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 and cmd == "tune" else 50
        out = run_optuna(n_trials=n_trials)
        print(f"Optuna best CV macro-F1: {out['best_cv_macro_f1']:.4f}")
        if out["holdout_metrics"]:
            m = out["holdout_metrics"]["macro_f1"]
            print(f"Optuna-tuned holdout macro-F1: {m['point_estimate']:.3f} "
                  f"[CI {m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")
    if cmd in ("threshold", "all"):
        logger.info("=== Per-class threshold optimization ===")
        out = optimize_thresholds()
        print(f"Threshold-tuned OOF macro-F1: {out['best_oof_macro_f1']:.4f}")
        if out["holdout_metrics_with_tuned_thresholds"]:
            m = out["holdout_metrics_with_tuned_thresholds"]["macro_f1_tuned_thresholds"]
            print(f"Threshold-tuned holdout macro-F1: {m['point_estimate']:.3f} "
                  f"[CI {m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
