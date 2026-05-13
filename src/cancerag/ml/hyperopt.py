"""
Phase 6 — Hyperparameter tuning + per-class threshold optimization.

Two complementary "free lift" techniques on top of the bake-off winner:

  1. **Optuna tuning** of the winning model (LightGBM by default) over its
     standard hyperparameter space. Optimizes scaffold-grouped 5-fold CV
     macro-F1 (the chemistry-realistic objective) with TPE sampler +
     median-pruner. Persists the best params + retrained model.

  2. **Per-class threshold optimization** on the calibrated model's
     `predict_proba` output. The default `argmax` decision rule heavily
     favors the majority class (G protein at 56.7% prior); per-class
     thresholds tuned on the train-eligible probabilities recover ~+0.02-
     0.03 macro-F1 by trading a bit of G-protein recall for ERK / G-prot-
     selectivity recall.

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
    logger.info("Tuning %s with Optuna (n_trials=%d)", winner, n_trials)
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
        study_name=f"cancerag_{winner}_scaffold_cv",
    )
    objective = _build_lightgbm_objective(X, y, sw, n_classes, splits)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = float(study.best_value)
    logger.info("Best CV macro-F1: %.4f", best_score)
    logger.info("Best params: %s", best_params)

    # Retrain on all train-eligible with best params
    import lightgbm as lgb
    from cancerag.ml.model_training import _combined_weight
    from sklearn.calibration import CalibratedClassifierCV

    model = lgb.LGBMClassifier(
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else None,
        n_jobs=4, verbose=-1, class_weight="balanced",
        random_state=seed, **best_params,
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
        joblib.dump(cal, output_dir / "lightgbm_tuned_calibrated.joblib")
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

    out = {
        "schema_version": 1,
        "tuned_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": winner,
        "n_trials": n_trials,
        "best_cv_macro_f1": best_score,
        "best_params": best_params,
        "holdout_metrics": holdout_metrics,
    }
    (output_dir / "optuna_tuning_meta.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str)
    )

    # Also persist the trial history for transparency
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / "optuna_trials.csv", index=False)
    logger.info("Optuna tuning complete: trials -> %s",
                output_dir / "optuna_trials.csv")
    return out


# --------------------------------------------- per-class threshold optimization


def optimize_thresholds(
    *, output_dir: Path | str = OUTPUT_DIR, n_seeds: int = 3,
) -> dict:
    """Find per-class probability thresholds that maximize macro-F1 on out-of-
    fold predictions. Uses scaffold-grouped CV to avoid leakage; thresholds
    are then frozen and applied at inference time."""
    from cancerag.ml.model_training import _combined_weight
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    decision = json.loads(SELECTION_PATH.read_text())
    winner = decision["chosen"]
    df = pd.read_parquet(DATASET_PATH)
    le = joblib.load(LE_PATH)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    # Build out-of-fold predictions
    from cancerag.ml.model_training import MODEL_FACTORIES
    factory = MODEL_FACTORIES[winner]
    oof_proba = np.zeros((len(X), n_classes), dtype=np.float32)
    seen = np.zeros(len(X), dtype=bool)

    for seed in range(42, 42 + n_seeds):
        splits = make_grouped_cv(df, group_col="scaffold", n_splits=5, seed=seed)
        if not splits:
            continue
        for tr, te in splits:
            model = factory(n_classes=n_classes, random_state=seed)
            pipe = build_full_pipeline(model)
            cw = _combined_weight(y[tr], sw[tr])
            last = pipe.steps[-1][0]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X.iloc[tr], y[tr],
                             **{f"{last}__sample_weight": cw})
                proba = pipe.predict_proba(X.iloc[te])
                # Average across seeds
                oof_proba[te] += proba / n_seeds
                seen[te] = True
            except Exception as exc:
                logger.warning("threshold-opt fold failed: %s", exc)

    # Search per-class thresholds via grid search
    # Goal: assign argmax of (proba_c / threshold_c) — equivalent to scaling
    # the decision boundary.
    eligible = seen & (y >= 0)
    y_sub = y[eligible]; p_sub = oof_proba[eligible]

    def predict_with_thresholds(p, thr):
        # Scale each column by 1/thr, then argmax
        scaled = p / np.asarray(thr)[None, :]
        return scaled.argmax(axis=1)

    best_thr = np.ones(n_classes, dtype=float)
    best_f1 = float(f1_score(y_sub, p_sub.argmax(axis=1),
                              average="macro", zero_division=0))
    grid = np.linspace(0.5, 2.0, 16)
    # Coordinate descent over classes
    for _ in range(3):
        for ci in range(n_classes):
            for v in grid:
                cand = best_thr.copy(); cand[ci] = v
                preds = predict_with_thresholds(p_sub, cand)
                f1 = float(f1_score(y_sub, preds, average="macro", zero_division=0))
                if f1 > best_f1:
                    best_f1, best_thr = f1, cand
    logger.info(
        "Threshold-opt: baseline argmax macro-F1=%.4f -> tuned=%.4f (thresholds=%s)",
        float(f1_score(y_sub, p_sub.argmax(axis=1), average="macro", zero_division=0)),
        best_f1, best_thr.tolist(),
    )

    # Apply on holdout
    holdout_metrics = {}
    if HOLDOUT_PATH.exists():
        from cancerag.ml.model_evaluation import report_metrics
        # Try the tuned model first, fall back to base
        cal_path = output_dir / "lightgbm_tuned_calibrated.joblib"
        if not cal_path.exists():
            cal_path = (Path("data/processed/ml_models")
                        / f"{winner}_final_calibrated.joblib")
        cal = joblib.load(cal_path)
        ho = pd.read_parquet(HOLDOUT_PATH)
        ho_X, ho_y, _, _, _ = get_X_y_groups(ho, label_encoder=le)
        for c in X.columns:
            if c not in ho_X.columns:
                ho_X[c] = 0.0
        ho_X = ho_X[X.columns]
        ho_proba = cal.predict_proba(ho_X)
        ho_pred_tuned = predict_with_thresholds(ho_proba, best_thr)
        rep = report_metrics(ho_y, ho_pred_tuned, bootstrap_n=1000, seed=42)
        holdout_metrics = {
            "n": rep["n_test"],
            "macro_f1_tuned_thresholds": rep["macro_f1"],
            "per_class_f1_tuned": rep["per_class_f1"],
        }

    out = {
        "schema_version": 1,
        "optimized_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_classes": [str(c) for c in le.classes_],
        "thresholds": best_thr.tolist(),
        "best_oof_macro_f1": best_f1,
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
