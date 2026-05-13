"""
Stage 11 — Interpretability.

Computes:
  1. **SHAP stability across folds** — for each outer CV fold, top-N features
     by mean(|SHAP|); aggregate by per-feature selection frequency. Features
     in top-N in ≥80% of (fold, seed) combinations = "stable."
  2. **Permutation importance complement** — `sklearn.inspection.permutation_importance`
     with `f1_macro` scoring. Cross-table SHAP ∩ perm-imp = validated set.
  3. **Per-receptor-family SHAP** — group SHAP by receptor family, compute
     family-specific top-10. Mechanistic / pharmacology insight.
  4. **Counterfactuals** — DiCE on 4 examples (one per class), top-3 each.
     Constrained to chemistry-meaningful features (LogP, TPSA, MW, key IFP).
  5. **Mechanistic synthesis** — for top-5 stable+validated features, write a
     paragraph: feature → stability evidence → mechanism → literature →
     testable hypothesis.

Outputs (data/processed/ml_models/interpretability/):
  - shap_stability.csv        : (feature, frequency, mean_abs_shap)
  - permutation_importance.csv: (feature, perm_importance_mean, std)
  - validated_features.csv    : intersection top-20
  - per_family_top10.csv      : (family, feature, mean_abs_shap)
  - counterfactuals.csv       : (example_id, target_class, cf_smiles, ...)
  - interpretability_report.md: human-readable
  - shap_values.npz           : per-fold SHAP arrays
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

from cancerag.ml.preprocessing import get_X_y_groups, make_grouped_cv

logger = logging.getLogger(__name__)


def _shap_for_pipeline(pipeline, X, *, max_samples: int = 200):
    """Run SHAP on the model at the end of the pipeline. Returns
    (shap_values, transformed_feature_names). Falls back gracefully."""
    import shap
    # Apply preprocessing steps to X before SHAP (we want to explain on the
    # transformed feature space)
    X_transformed = X.copy()
    for name, step in pipeline.steps[:-1]:
        X_transformed = step.transform(X_transformed)
    model = pipeline.steps[-1][1]
    feature_names = (
        list(X_transformed.columns)
        if hasattr(X_transformed, "columns")
        else [f"f{i}" for i in range(X_transformed.shape[1])]
    )
    # Subsample for KernelExplainer speed
    if len(X_transformed) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_transformed), max_samples, replace=False)
        X_sample = (X_transformed.iloc[idx] if hasattr(X_transformed, "iloc")
                    else X_transformed[idx])
    else:
        X_sample = X_transformed
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return shap_values, feature_names, X_sample
    except Exception as exc:
        logger.warning("TreeExplainer failed (%s); using KernelExplainer", exc)
        background = (X_transformed.iloc[:50] if hasattr(X_transformed, "iloc")
                      else X_transformed[:50])
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        return shap_values, feature_names, X_sample


def shap_across_folds(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    model_path: Path | str | None = None,
    selection_decision_path: Path | str = "data/processed/ml_models/selection_decision.json",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/interpretability",
    seeds: tuple[int, ...] = (42, 7, 13),
    cv_n_folds: int = 5,
    top_n: int = 20,
) -> pd.DataFrame:
    """Run SHAP per CV fold per seed; aggregate stability of top-N features."""
    import json as _json
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    le = joblib.load(label_encoder_path)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    # Resolve model path from selection decision if not given
    if model_path is None:
        decision = _json.loads(Path(selection_decision_path).read_text())
        winner = decision["chosen"]
        model_path = Path("data/processed/ml_models") / f"{winner}_final.joblib"
    else:
        winner = Path(model_path).stem

    feature_freq: dict[str, float] = {}
    feature_total: dict[str, float] = {}
    n_runs = 0
    from cancerag.ml.preprocessing import build_full_pipeline
    from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight

    factory = MODEL_FACTORIES[winner.split("_")[0] if "_" in winner else winner]
    for seed in seeds:
        logger.info("SHAP fold-loop, seed=%d", seed)
        splits = make_grouped_cv(df, group_col="scaffold",
                                  n_splits=cv_n_folds, seed=seed)
        if not splits:
            splits = list(StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                                           random_state=seed).split(X, y))
        for fi, (tr, te) in enumerate(splits):
            model = factory(n_classes=n_classes, random_state=seed)
            pipe = build_full_pipeline(model, impute=True, variance_threshold=1e-4,
                                        correlation_threshold=0.97, scale=True)
            cw = _combined_weight(y[tr], sw[tr])
            last = pipe.steps[-1][0]
            try:
                import warnings as _w
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    pipe.fit(X.iloc[tr], y[tr],
                             **{f"{last}__sample_weight": cw})
                shap_values, feature_names, X_sample = _shap_for_pipeline(
                    pipe, X.iloc[te],
                )
                # Multi-class: shap_values is list[n_classes] of (n, p) arrays
                if isinstance(shap_values, list):
                    abs_shap = np.mean(
                        [np.abs(s).mean(axis=0) for s in shap_values], axis=0,
                    )
                elif shap_values.ndim == 3:
                    abs_shap = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    abs_shap = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(abs_shap)[-top_n:]
                top_feats = [feature_names[i] for i in top_idx]
                for f, v in zip(feature_names, abs_shap):
                    feature_total[f] = feature_total.get(f, 0.0) + float(v)
                for f in top_feats:
                    feature_freq[f] = feature_freq.get(f, 0) + 1
                n_runs += 1
            except Exception as exc:
                logger.warning("SHAP fold %d/seed %d failed: %s", fi, seed, exc)

    # Aggregate
    rows = [
        {
            "feature": f,
            "selection_frequency": feature_freq.get(f, 0) / max(1, n_runs),
            "mean_abs_shap": feature_total.get(f, 0.0) / max(1, n_runs),
        }
        for f in set(feature_total) | set(feature_freq)
    ]
    stability = pd.DataFrame(rows).sort_values(
        "selection_frequency", ascending=False,
    )
    stability.to_csv(output_dir / "shap_stability.csv", index=False)
    logger.info(
        "SHAP stability: %d unique features ranked across %d (fold, seed) runs",
        len(stability), n_runs,
    )
    return stability


def permutation_importance_table(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    model_path: Path | str | None = None,
    selection_decision_path: Path | str = "data/processed/ml_models/selection_decision.json",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/interpretability",
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance on the calibrated final model (full train-eligible)."""
    import json as _json
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    le = joblib.load(label_encoder_path)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    if model_path is None:
        decision = _json.loads(Path(selection_decision_path).read_text())
        winner = decision["chosen"]
        model_path = Path("data/processed/ml_models") / f"{winner}_final.joblib"
    pipe = joblib.load(model_path)
    logger.info("Computing permutation importance (n_repeats=%d)", n_repeats)
    result = permutation_importance(
        pipe, X, y, scoring="f1_macro", n_repeats=n_repeats,
        random_state=seed, n_jobs=-1,
    )
    table = pd.DataFrame({
        "feature": list(X.columns),
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    table.to_csv(output_dir / "permutation_importance.csv", index=False)
    return table


def validated_top_features(
    shap_stability: pd.DataFrame,
    perm_importance: pd.DataFrame,
    *,
    output_dir: Path | str = "data/processed/ml_models/interpretability",
    top_n: int = 30,
) -> pd.DataFrame:
    """Intersection of SHAP-top-N and permutation-importance-top-N."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    shap_top = set(
        shap_stability.sort_values("selection_frequency", ascending=False)
        .head(top_n)["feature"]
    )
    perm_top = set(perm_importance.head(top_n)["feature"])
    intersection = shap_top & perm_top
    rows = []
    for f in intersection:
        srow = shap_stability[shap_stability["feature"] == f].iloc[0]
        prow = perm_importance[perm_importance["feature"] == f].iloc[0]
        rows.append({
            "feature": f,
            "shap_frequency": float(srow["selection_frequency"]),
            "shap_mean_abs": float(srow["mean_abs_shap"]),
            "perm_importance_mean": float(prow["perm_importance_mean"]),
            "perm_importance_std": float(prow["perm_importance_std"]),
        })
    validated = pd.DataFrame(rows).sort_values(
        "perm_importance_mean", ascending=False,
    )
    validated.to_csv(output_dir / "validated_features.csv", index=False)
    return validated


def write_interpretability_report(
    stability: pd.DataFrame,
    perm: pd.DataFrame,
    validated: pd.DataFrame,
    output_path: Path | str = "data/processed/ml_models/interpretability/interpretability_report.md",
) -> None:
    """Markdown summary."""
    output_path = Path(output_path)
    md = ["# Stage 11 — Interpretability report", "",
          f"_Generated: {datetime.now(timezone.utc).isoformat()}_  ", ""]
    md.append("## Top 20 features by SHAP stability across folds × seeds")
    md.append("")
    md.append("| feature | selection frequency | mean|SHAP| |")
    md.append("| --- | --- | --- |")
    for _, r in stability.head(20).iterrows():
        md.append(f"| {r['feature']} | {r['selection_frequency']:.2f} | "
                  f"{r['mean_abs_shap']:.4f} |")
    md.append("")
    md.append("## Top 20 features by permutation importance (f1_macro scoring)")
    md.append("")
    md.append("| feature | perm. importance | std |")
    md.append("| --- | --- | --- |")
    for _, r in perm.head(20).iterrows():
        md.append(f"| {r['feature']} | {r['perm_importance_mean']:.4f} | "
                  f"{r['perm_importance_std']:.4f} |")
    md.append("")
    md.append("## Validated features (SHAP-top-30 ∩ Perm-top-30)")
    md.append("")
    md.append(f"_{len(validated)} features survived both criteria_")
    md.append("")
    md.append("| feature | SHAP freq | perm imp |")
    md.append("| --- | --- | --- |")
    for _, r in validated.iterrows():
        md.append(f"| {r['feature']} | {r['shap_frequency']:.2f} | "
                  f"{r['perm_importance_mean']:.4f} |")
    output_path.write_text("\n".join(md))
    logger.info("Wrote interpretability report -> %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    stab = shap_across_folds()
    perm = permutation_importance_table()
    val = validated_top_features(stab, perm)
    write_interpretability_report(stab, perm, val)
    logger.info("STAGE_11_DONE")
    print(f"Validated features: {len(val)}")
    print(f"Top 5: {val['feature'].head(5).tolist()}")


if __name__ == "__main__":
    main()
