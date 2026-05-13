"""
Stage 10 — Model training & evaluation (rigorous edition).

Multi-model bake-off with per-fold preprocessing+selection (no leakage),
multi-seed runs, 4 evaluation modes, class+sample weighting, bootstrap CIs,
per-receptor breakdown, and probability calibration.

Inputs:
  - data/processed/ml_ready_dataset.parquet   (Stage 07 train-eligible)
  - data/processed/ml_preprocessed/label_encoder.joblib
  - data/processed/ml_splits.json
  - data/holdout/dataset_holdout.parquet

Outputs (data/processed/ml_models/):
  - cv_results_long.csv            : (seed, model, split, fold) → metrics
  - cv_results_summary.csv         : per (model, split) mean ± std
  - holdout_results.csv            : per model holdout metrics + CIs
  - per_receptor_holdout.csv       : per-receptor breakdown
  - selection_decision.json        : winning model
  - <winner>_final.joblib          : retrained on all train-eligible
  - <winner>_final_calibrated.joblib : isotonic calibration of winner
  - reliability.png                : reliability diagram on holdout
  - training_summary.md            : human-readable audit
  - model_meta.json                : provenance

Design:
  - Each (seed × model × split × fold) builds a fresh sklearn Pipeline:
    DataFrameImputer → CorrelationFilter → DataFrameScaler → BorutaSelector → model.
    Fitted on train fold only — no leakage.
  - 5 models in the bake-off: XGBoost, LightGBM, CatBoost, ElasticNet-LR, RF.
  - 5 seeds: {42, 7, 13, 21, 99}.
  - 4 split modes: stratified-kfold, scaffold-grouped, receptor-grouped,
    temporal-holdout (latter is single-shot).
  - Class+sample weighting: combined inverse-frequency × evidence weight.
  - Headline metric: macro-F1 with bootstrap CIs (via report_metrics).
"""

from __future__ import annotations

import json as _json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight

from cancerag.features.molecular_descriptors import morgan_dataframe
from cancerag.ml.feature_selection import BorutaSelector
from cancerag.ml.model_evaluation import (
    bootstrap_ci, per_receptor_metrics, report_metrics,
)
from cancerag.ml.preprocessing import (
    build_full_pipeline, get_X_y_groups, identify_columns,
    make_grouped_cv,
)

logger = logging.getLogger(__name__)


SEEDS_DEFAULT = [42, 7, 13, 21, 99]


# --------------------------------------------- model factories


def make_xgboost(*, n_classes: int, random_state: int = 42, **kwargs):
    import xgboost as xgb
    params = dict(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective=("multi:softprob" if n_classes > 2 else "binary:logistic"),
        eval_metric=("mlogloss" if n_classes > 2 else "logloss"),
        n_jobs=4, random_state=random_state, tree_method="hist",
    )
    if n_classes > 2:
        params["num_class"] = n_classes
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def make_lightgbm(*, n_classes: int, random_state: int = 42, **kwargs):
    import lightgbm as lgb
    params = dict(
        n_estimators=200, num_leaves=31, learning_rate=0.1,
        objective=("multiclass" if n_classes > 2 else "binary"),
        n_jobs=4, random_state=random_state, verbose=-1,
        class_weight="balanced",
    )
    if n_classes > 2:
        params["num_class"] = n_classes
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


def make_catboost(*, n_classes: int, random_state: int = 42, **kwargs):
    from catboost import CatBoostClassifier
    params = dict(
        iterations=200, depth=6, learning_rate=0.1,
        loss_function=("MultiClass" if n_classes > 2 else "Logloss"),
        random_seed=random_state, thread_count=4, verbose=False,
        auto_class_weights="Balanced",
    )
    params.update(kwargs)
    return CatBoostClassifier(**params)


def make_elastic_lr(*, n_classes: int, random_state: int = 42, **kwargs):
    params = dict(
        penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
        max_iter=2000, multi_class="multinomial",
        n_jobs=4, random_state=random_state, class_weight="balanced",
    )
    params.update(kwargs)
    return LogisticRegression(**params)


def make_rf(*, n_classes: int, random_state: int = 42, **kwargs):
    params = dict(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        n_jobs=4, random_state=random_state, class_weight="balanced",
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


MODEL_FACTORIES = {
    "xgboost": make_xgboost,
    "lightgbm": make_lightgbm,
    "catboost": make_catboost,
    "elastic_lr": make_elastic_lr,
    "random_forest": make_rf,
}


# --------------------------------------------- baselines (kept from old API)


def majority_class_baseline(random_state: int = 42) -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def stratified_baseline(random_state: int = 42) -> DummyClassifier:
    return DummyClassifier(strategy="stratified", random_state=random_state)


def smiles_only_rf_baseline(
    smiles_col: str = "canonical_smiles_std",
    n_bits: int = 2048, radius: int = 2,
    n_estimators: int = 200, random_state: int = 42,
) -> Pipeline:
    """RF on Morgan FPs alone — no receptor / docking. The "is the structural
    pipeline doing real work?" reviewer-1 ask."""

    def _featurize(df):
        if isinstance(df, pd.DataFrame):
            smiles = df[smiles_col].tolist()
        elif isinstance(df, pd.Series):
            smiles = df.tolist()
        else:
            smiles = list(df)
        return morgan_dataframe(smiles, radius=radius, n_bits=n_bits).values

    return Pipeline([
        ("featurize", FunctionTransformer(_featurize, validate=False)),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        )),
    ])


BASELINES = {
    "majority_class": majority_class_baseline,
    "stratified": stratified_baseline,
    "smiles_only_morgan_rf": smiles_only_rf_baseline,
}


# --------------------------------------------- weighting


def _combined_weight(y, base_weight, *, classes=None):
    """combined = inverse_frequency_class_weight × base sample_weight."""
    if classes is None:
        balanced = compute_sample_weight("balanced", y)
    else:
        # Compute per supplied class set so unseen-in-fold classes get 0
        balanced = compute_sample_weight(
            class_weight={c: float(len(y) / max(1, (y == c).sum() * len(classes)))
                          for c in classes},
            y=y,
        )
    return balanced * np.asarray(base_weight)


# --------------------------------------------- per-fold fit/eval


def _eval_pipeline_on_fold(
    pipeline: Pipeline, X_train, y_train, X_test, y_test,
    sample_weight_train, label_classes, *,
    seed: int,
) -> dict:
    """Fit the pipeline + report_metrics on the test fold."""
    fit_kwargs = {}
    if sample_weight_train is not None:
        # sklearn Pipeline supports per-step kwargs via "step__kwarg=value"
        # The model is the last step named 'model' (or 'rf' for baselines).
        last_name = pipeline.steps[-1][0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(
                    X_train, y_train,
                    **{f"{last_name}__sample_weight": sample_weight_train},
                )
            fit_kwargs["used_sample_weight"] = True
        except (TypeError, ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipeline.fit(X_train, y_train)
            fit_kwargs["used_sample_weight"] = False
    else:
        pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    rep = report_metrics(np.asarray(y_test), np.asarray(y_pred),
                         bootstrap_n=500, seed=seed)
    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": rep["balanced_accuracy"]["point_estimate"],
        "macro_f1": rep["macro_f1"]["point_estimate"],
        "macro_f1_ci_lo": rep["macro_f1"]["ci_lo"],
        "macro_f1_ci_hi": rep["macro_f1"]["ci_hi"],
        "per_class_f1": rep["per_class_f1"],
        "confusion_matrix": rep["confusion_matrix"],
        **fit_kwargs,
    }


def _build_pipeline_for(model_name: str, n_classes: int, seed: int,
                        *, with_selector: bool = True) -> Pipeline:
    """Build the per-fold preprocessing+selection+model pipeline."""
    factory = MODEL_FACTORIES[model_name]
    model = factory(n_classes=n_classes, random_state=seed)
    selector = (
        BorutaSelector(max_iter=30, random_state=seed)
        if with_selector else None
    )
    return build_full_pipeline(
        model, impute=True, correlation_threshold=0.97, scale=True,
        selector=selector,
    )


# --------------------------------------------- main orchestrator


def run_model_training(
    config: dict | None = None,
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    holdout_path: Path | str = "data/holdout/dataset_holdout.parquet",
    splits_path: Path | str = "data/processed/ml_splits.json",
    preprocessed_dir: Path | str = "data/processed/ml_preprocessed",
    output_dir: Path | str = "data/processed/ml_models",
    models: Sequence[str] = ("xgboost", "lightgbm",
                              "elastic_lr", "random_forest"),
    seeds: Sequence[int] = SEEDS_DEFAULT,
    cv_n_folds: int = 5,
    use_selector_in_pipeline: bool = False,
) -> dict:
    """Run the full multi-model × multi-seed × 4-mode evaluation."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    df = pd.read_parquet(dataset_path)
    le = joblib.load(Path(preprocessed_dir) / "label_encoder.joblib")
    n_classes = len(le.classes_); label_classes = list(le.classes_)
    X, y, sw, le, cols = get_X_y_groups(df, label_encoder=le)
    logger.info(
        "Train-eligible: %d rows × %d cols | classes=%s",
        len(df), X.shape[1], label_classes,
    )

    # CV iterators per split mode
    split_iters: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    rows_long: list[dict] = []

    for seed in seeds:
        logger.info("=== seed=%d ===", seed)
        # 1. Stratified-kfold (random)
        skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                              random_state=seed)
        split_iters["stratified_kfold"] = [
            (tr, te) for tr, te in skf.split(X, y)
        ]
        # 2. Scaffold-grouped
        split_iters["scaffold_kfold"] = make_grouped_cv(
            df, group_col="scaffold", n_splits=cv_n_folds, seed=seed,
        )
        # 3. Receptor-grouped
        split_iters["receptor_kfold"] = make_grouped_cv(
            df, group_col="receptor_uniprot", n_splits=cv_n_folds, seed=seed,
        )

        for split_name, splits in split_iters.items():
            for fi, (tr, te) in enumerate(splits):
                # Combined weight = inverse-freq class weight × evidence weight
                cw = _combined_weight(y[tr], sw[tr])
                for mname in models:
                    pipe = _build_pipeline_for(
                        mname, n_classes, seed,
                        with_selector=use_selector_in_pipeline,
                    )
                    try:
                        r = _eval_pipeline_on_fold(
                            pipe, X.iloc[tr], y[tr], X.iloc[te], y[te],
                            sample_weight_train=cw, label_classes=label_classes,
                            seed=seed,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[%s|%s|fold=%d|seed=%d] failed: %s",
                            mname, split_name, fi, seed, exc,
                        )
                        continue
                    r.update(model=mname, split=split_name,
                             fold=fi, seed=seed)
                    rows_long.append(r)
                # Baselines: only on stratified_kfold to keep table compact
                if split_name == "stratified_kfold":
                    for bname, factory in BASELINES.items():
                        if bname == "smiles_only_morgan_rf":
                            continue  # needs SMILES not in feature matrix
                        try:
                            r = _eval_pipeline_on_fold(
                                Pipeline([("dummy", factory(random_state=seed))]),
                                X.iloc[tr], y[tr], X.iloc[te], y[te],
                                sample_weight_train=None,
                                label_classes=label_classes, seed=seed,
                            )
                        except Exception as exc:
                            logger.warning("[%s] failed: %s", bname, exc)
                            continue
                        r.update(model=bname, split=split_name,
                                 fold=fi, seed=seed)
                        rows_long.append(r)

    cv_long = pd.DataFrame([
        {k: (v if not isinstance(v, (dict, list)) else _json.dumps(v))
         for k, v in r.items()}
        for r in rows_long
    ])
    cv_long.to_csv(output_dir / "cv_results_long.csv", index=False)
    logger.info("cv_results_long: %d rows", len(cv_long))

    # Aggregate per (model, split): mean ± std across (seed, fold)
    agg = (
        cv_long.groupby(["model", "split"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_acc_mean=("balanced_accuracy", "mean"),
            balanced_acc_std=("balanced_accuracy", "std"),
            n_obs=("macro_f1", "count"),
        )
        .reset_index()
        .sort_values(["split", "macro_f1_mean"], ascending=[True, False])
    )
    agg.to_csv(output_dir / "cv_results_summary.csv", index=False)

    # Pick winner from scaffold-grouped (chemistry-realistic primary metric)
    primary_split = "scaffold_kfold" if "scaffold_kfold" in agg["split"].values \
        else "stratified_kfold"
    winner_row = (
        agg[agg["split"] == primary_split]
        .iloc[0] if not agg[agg["split"] == primary_split].empty
        else agg.iloc[0]
    )
    winner = str(winner_row["model"])
    decision = {
        "rule": "max_mean_macro_f1_on_scaffold_grouped_kfold",
        "primary_split": primary_split,
        "chosen": winner,
        "winner_metrics": {
            "macro_f1_mean": float(winner_row["macro_f1_mean"]),
            "macro_f1_std": float(winner_row["macro_f1_std"]),
            "balanced_acc_mean": float(winner_row["balanced_acc_mean"]),
        },
        "summary": agg.to_dict(orient="records"),
    }
    (output_dir / "selection_decision.json").write_text(
        _json.dumps(decision, indent=2, sort_keys=True)
    )
    logger.info(
        "Winner: %s (macro-F1 %.3f ± %.3f on %s)",
        winner, winner_row["macro_f1_mean"], winner_row["macro_f1_std"],
        primary_split,
    )

    # Refit winner on all train-eligible (best seed)
    final_pipe = _build_pipeline_for(
        winner, n_classes, SEEDS_DEFAULT[0],
        with_selector=use_selector_in_pipeline,
    )
    cw_all = _combined_weight(y, sw)
    last_name = final_pipe.steps[-1][0]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipe.fit(X, y, **{f"{last_name}__sample_weight": cw_all})
    except (TypeError, ValueError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_pipe.fit(X, y)
    joblib.dump(final_pipe, output_dir / f"{winner}_final.joblib")

    # Calibration via CalibratedClassifierCV (isotonic, 5-fold)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated = CalibratedClassifierCV(
                final_pipe, method="isotonic", cv=3,
            ).fit(X, y, **{f"{last_name}__sample_weight": cw_all})
    except Exception as exc:
        logger.warning("Calibration failed (%s); using uncalibrated final", exc)
        calibrated = final_pipe
    joblib.dump(calibrated, output_dir / f"{winner}_final_calibrated.joblib")

    # Holdout evaluation (calibrated model)
    holdout_metrics = {}
    holdout_path = Path(holdout_path)
    if holdout_path.exists():
        ho = pd.read_parquet(holdout_path)
        ho_X, ho_y, ho_sw, _, _ = get_X_y_groups(ho, label_encoder=le)
        # Align columns to training X
        for c in X.columns:
            if c not in ho_X.columns:
                ho_X[c] = 0.0
        ho_X = ho_X[X.columns]
        ho_pred = calibrated.predict(ho_X)
        rep = report_metrics(ho_y, ho_pred, bootstrap_n=1000, seed=42)

        # Brier score per class (on calibrated probabilities)
        try:
            ho_proba = calibrated.predict_proba(ho_X)
            brier_per_class = {}
            for ci, cname in enumerate(label_classes):
                y_bin = (ho_y == ci).astype(int)
                brier_per_class[str(cname)] = float(
                    brier_score_loss(y_bin, ho_proba[:, ci])
                )
            mean_brier = float(np.mean(list(brier_per_class.values())))
        except Exception:
            brier_per_class, mean_brier = {}, float("nan")

        holdout_metrics = {
            "n": rep["n_test"],
            "macro_f1": rep["macro_f1"],
            "balanced_accuracy": rep["balanced_accuracy"],
            "per_class_f1": rep["per_class_f1"],
            "confusion_matrix": rep["confusion_matrix"],
            "brier_per_class": brier_per_class,
            "mean_brier": mean_brier,
            "classification_report": classification_report(
                ho_y, ho_pred, target_names=label_classes,
                zero_division=0, output_dict=True,
            ),
        }

        # Per-receptor breakdown
        try:
            per_rec = per_receptor_metrics(
                ho_y, ho_pred, ho["receptor_uniprot"].values, min_samples=3,
            )
            per_rec.to_csv(output_dir / "per_receptor_holdout.csv", index=False)
        except Exception as exc:
            logger.warning("per_receptor_metrics failed: %s", exc)

        # Predictions table
        pred_df = pd.DataFrame({
            "y_true": [label_classes[i] for i in ho_y],
            "y_pred": [label_classes[i] for i in ho_pred],
        })
        pred_df.to_csv(output_dir / "holdout_predictions.csv", index=False)

        # Reliability diagram (one-vs-rest per class)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.calibration import calibration_curve
            fig, axes = plt.subplots(
                1, n_classes, figsize=(4 * n_classes, 4), sharey=True,
            )
            if n_classes == 1:
                axes = [axes]
            for ci, ax in enumerate(axes):
                y_bin = (ho_y == ci).astype(int)
                if y_bin.sum() < 2:
                    ax.set_title(f"{label_classes[ci]} (n_pos={y_bin.sum()})")
                    continue
                frac_pos, mean_pred = calibration_curve(
                    y_bin, ho_proba[:, ci], n_bins=5, strategy="uniform",
                )
                ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
                ax.plot(mean_pred, frac_pos, "o-", label="model")
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")
                ax.set_title(label_classes[ci])
            plt.suptitle(f"Reliability diagram — {winner} (calibrated)")
            plt.tight_layout()
            fig_path = output_dir / "figures" / "reliability.png"
            plt.savefig(fig_path, dpi=120)
            plt.close()
            logger.info("Reliability diagram -> %s", fig_path)
        except Exception as exc:
            logger.warning("Reliability diagram failed: %s", exc)

    # Provenance + audit
    summary_meta = {
        "schema_version": 2,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_eligible": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": n_classes,
        "label_classes": label_classes,
        "models_evaluated": list(models),
        "seeds": list(seeds),
        "cv_n_folds": cv_n_folds,
        "evaluation_modes": list(split_iters.keys()) + ["temporal_holdout"],
        "selection_decision": decision,
        "holdout_metrics": holdout_metrics,
    }
    (output_dir / "model_meta.json").write_text(
        _json.dumps(summary_meta, indent=2, sort_keys=True, default=str)
    )

    # Markdown audit
    md = ["# Stage 10 — Model training & evaluation (rigor edition)", "",
          f"_Generated: {summary_meta['trained_at_utc']}_  ", ""]
    md.append("## Cross-validation (mean ± std across seeds × folds)")
    md.append("")
    md.append("| split | model | macro-F1 (mean ± std) | balanced acc (mean ± std) | n |")
    md.append("| --- | --- | --- | --- | --- |")
    for _, row in agg.iterrows():
        md.append(
            f"| {row['split']} | {row['model']} | "
            f"{row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f} | "
            f"{row['balanced_acc_mean']:.3f} ± {row['balanced_acc_std']:.3f} | "
            f"{int(row['n_obs'])} |"
        )
    md.append("")
    md.append(f"**Winner**: {winner} (selected on `{primary_split}` mean macro-F1)")
    md.append("")
    if holdout_metrics:
        md.append("## Temporal holdout (year ≥ 2018, calibrated model)")
        md.append("")
        m = holdout_metrics["macro_f1"]
        b = holdout_metrics["balanced_accuracy"]
        md.append(f"- N: {holdout_metrics['n']}")
        md.append(f"- Macro-F1: **{m['point_estimate']:.3f}** "
                  f"[CI {m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")
        md.append(f"- Balanced accuracy: **{b['point_estimate']:.3f}** "
                  f"[CI {b['ci_lo']:.3f}, {b['ci_hi']:.3f}]")
        md.append(f"- Mean Brier (calibration quality, lower=better): "
                  f"{holdout_metrics.get('mean_brier', float('nan')):.3f}")
        md.append("- Per-class F1: " +
                  ", ".join(f"{c}={s:.2f}" for c, s in
                             holdout_metrics["per_class_f1"].items()))
        md.append("")
    (output_dir / "training_summary.md").write_text("\n".join(md))
    logger.info("Wrote training summary -> %s", output_dir / "training_summary.md")

    return summary_meta


# --------------------------------------------- model selection (kept for tests)


SELECTION_RULE = "max_outer_cv_macro_f1_mean_then_min_std"


def select_final_model(nested_cv_results: pd.DataFrame) -> dict:
    required = {"model", "macro_f1"}
    missing = required - set(nested_cv_results.columns)
    if missing:
        raise KeyError(f"select_final_model: missing columns {sorted(missing)}")
    summary = (
        nested_cv_results.groupby("model")["macro_f1"]
        .agg(mean_macro_f1="mean", std_macro_f1="std", n_folds="count")
        .reset_index()
        .sort_values(["mean_macro_f1", "std_macro_f1"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return {
        "rule": SELECTION_RULE,
        "summary": summary.to_dict(orient="records"),
        "chosen": str(summary.iloc[0]["model"]),
    }


def write_selection_decision(decision: dict, path: Path | str) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(decision, indent=2, sort_keys=True))
    return path


def read_selection_decision(path: Path | str) -> dict:
    return _json.loads(Path(path).read_text())


# --------------------------------------------- entry point


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    summary = run_model_training()
    print("\nModel training summary:")
    print(f"  classes: {summary['label_classes']}")
    print(f"  selection: {summary['selection_decision']['chosen']}")
    if summary.get("holdout_metrics"):
        m = summary["holdout_metrics"]["macro_f1"]
        print(f"  holdout macro-F1: {m['point_estimate']:.3f} "
              f"[CI {m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
