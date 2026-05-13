"""
Phase 6 — MLflow run tracking.

Walks the persisted CV results, baselines, and advanced-training outputs and
re-emits each as an MLflow run for reviewer auditability. Uses file-based
backend (no server required) so the entire run history is a single
``mlruns/`` directory the reviewer can inspect.

Run:
    python -m cancerag.ml.mlflow_logging
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


MLRUNS_DIR = Path("data/processed/ml_models/mlruns")
ML_MODELS = Path("data/processed/ml_models")


def _safe_log_metric(mlflow, key, value):
    try:
        mlflow.log_metric(key, float(value))
    except Exception as exc:
        logger.debug("log_metric(%s, %s) skipped: %s", key, value, exc)


def _safe_log_param(mlflow, key, value):
    try:
        mlflow.log_param(key, str(value))
    except Exception as exc:
        logger.debug("log_param(%s, %s) skipped: %s", key, value, exc)


def log_all_runs(*, mlruns_dir: Path | str = MLRUNS_DIR) -> dict:
    """Walk all known result files and emit MLflow runs."""
    import mlflow
    mlruns_dir = Path(mlruns_dir).resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment("cancerag")
    n_runs = 0

    # 1. Bake-off CV results — one MLflow run per (model, split, fold, seed)
    cv_path = ML_MODELS / "cv_results_long.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
        for _, row in cv.iterrows():
            with mlflow.start_run(run_name=f"{row['model']}|{row['split']}|fold{int(row['fold'])}|s{int(row['seed'])}"):
                _safe_log_param(mlflow, "model", row["model"])
                _safe_log_param(mlflow, "split", row["split"])
                _safe_log_param(mlflow, "fold", int(row["fold"]))
                _safe_log_param(mlflow, "seed", int(row["seed"]))
                _safe_log_metric(mlflow, "macro_f1", row["macro_f1"])
                _safe_log_metric(mlflow, "macro_f1_ci_lo", row.get("macro_f1_ci_lo"))
                _safe_log_metric(mlflow, "macro_f1_ci_hi", row.get("macro_f1_ci_hi"))
                _safe_log_metric(mlflow, "balanced_accuracy", row.get("balanced_accuracy"))
                _safe_log_metric(mlflow, "accuracy", row.get("accuracy"))
                _safe_log_metric(mlflow, "n_train", row.get("n_train"))
                _safe_log_metric(mlflow, "n_test", row.get("n_test"))
                n_runs += 1
        logger.info("Logged %d bake-off CV runs", len(cv))

    # 2. ChemBERTa baseline
    cb_path = ML_MODELS / "baselines/chemberta/chemberta_summary.json"
    if cb_path.exists():
        cb = json.loads(cb_path.read_text())
        with mlflow.start_run(run_name="chemberta_baseline_summary"):
            _safe_log_param(mlflow, "model", "chemberta_logreg")
            _safe_log_param(mlflow, "checkpoint", cb.get("checkpoint"))
            _safe_log_param(mlflow, "embedding_dim", cb.get("embedding_dim"))
            for s in cb.get("cv_summary", []):
                _safe_log_metric(mlflow, f"{s['split']}_macro_f1_mean", s["mean"])
                _safe_log_metric(mlflow, f"{s['split']}_macro_f1_std", s["std"])
            ho = cb.get("holdout", {})
            if ho.get("macro_f1"):
                _safe_log_metric(mlflow, "holdout_macro_f1",
                                  ho["macro_f1"]["point_estimate"])
                _safe_log_metric(mlflow, "holdout_macro_f1_ci_lo",
                                  ho["macro_f1"]["ci_lo"])
                _safe_log_metric(mlflow, "holdout_macro_f1_ci_hi",
                                  ho["macro_f1"]["ci_hi"])
            n_runs += 1
        logger.info("Logged ChemBERTa baseline run")

    # 3. Selector ablation cells
    sa_path = ML_MODELS / "selector_ablation/selector_ablation_long.csv"
    if sa_path.exists():
        sa = pd.read_csv(sa_path)
        for _, row in sa.iterrows():
            with mlflow.start_run(
                run_name=f"selector_ablation|{row['selector']}|{row['model']}|f{int(row['fold'])}"
            ):
                _safe_log_param(mlflow, "selector", row["selector"])
                _safe_log_param(mlflow, "model", row["model"])
                _safe_log_param(mlflow, "experiment", "selector_ablation")
                _safe_log_metric(mlflow, "macro_f1", row["macro_f1"])
                _safe_log_metric(mlflow, "n_features_selected",
                                  row["n_features_selected"])
                n_runs += 1
        logger.info("Logged %d selector-ablation runs", len(sa))

    # 4. Advanced training results
    for name, path in (
        ("optuna_tuning", ML_MODELS / "advanced/optuna_tuning_meta.json"),
        ("threshold_optimization", ML_MODELS / "advanced/thresholds_meta.json"),
        ("stacking_ensemble", ML_MODELS / "advanced/stacking_meta.json"),
        ("focal_loss_xgb", ML_MODELS / "advanced/focal_loss_meta.json"),
    ):
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        with mlflow.start_run(run_name=name):
            _safe_log_param(mlflow, "experiment", name)
            for k, v in meta.items():
                if isinstance(v, (int, float)):
                    _safe_log_metric(mlflow, k, v)
                elif isinstance(v, dict):
                    # Flatten one level for nested metrics like holdout_metrics
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            _safe_log_metric(mlflow, f"{k}_{k2}", v2)
                        elif isinstance(v2, dict) and "point_estimate" in v2:
                            _safe_log_metric(mlflow, f"{k}_{k2}", v2["point_estimate"])
                else:
                    _safe_log_param(mlflow, k, str(v)[:200])
            n_runs += 1
        logger.info("Logged %s run", name)

    # 5. Reviewer extras
    for name, path in (
        ("learning_curve", ML_MODELS / "extras/learning_curve.csv"),
        ("ablation_no_structural", ML_MODELS / "extras/ablation_no_structural.csv"),
        ("loro", ML_MODELS / "extras/loro_results.csv"),
    ):
        if not path.exists():
            continue
        d = pd.read_csv(path)
        for _, row in d.iterrows():
            with mlflow.start_run(run_name=f"{name}|row{int(row.name)}"):
                _safe_log_param(mlflow, "experiment", name)
                for col in d.columns:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        _safe_log_metric(mlflow, col, val)
                    else:
                        _safe_log_param(mlflow, col, str(val))
                n_runs += 1
        logger.info("Logged %d %s runs", len(d), name)

    return {"n_runs_logged": n_runs, "mlruns_dir": str(mlruns_dir)}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    out = log_all_runs()
    print(f"Logged {out['n_runs_logged']} runs to {out['mlruns_dir']}")
    print("Inspect with: mlflow ui --backend-store-uri file://"
          f"{out['mlruns_dir']}")


if __name__ == "__main__":
    main()
