"""
Stage 10+ — Reviewer-grade extras.

Implements the 8 items from `improvements/10_model_training_eval.md` Phase-4:

  1. Learning curve (25/50/75/100% × 5 seeds)
  2. Ablation: drop structural feature blocks (vina_*, ifp_*, gnina_*, 3D)
  3. Paired permutation test for statistical model comparison
  4. Leave-one-receptor-out CV for top-5 receptors
  5. Model card writer (F10.7 schema)
  6. Temporal-shift Wasserstein quantification (per feature)

Decoy evaluation and MLflow are implemented in separate modules
(``decoys.py``, ``mlflow_logging.py``) due to external-data and
infrastructure dependencies.

Outputs (data/processed/ml_models/extras/):
  - learning_curve.csv          : (frac, seed, macro_f1)
  - ablation_no_structural.csv  : (model, with/without structural, macro_f1)
  - paired_permutation_test.csv : (model_a, model_b, p_value)
  - loro_results.csv            : (held_out_receptor, n, macro_f1)
  - model_card.md               : the model card
  - temporal_shift.csv          : (feature, wasserstein_distance)
"""

from __future__ import annotations

import json as _json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from cancerag.ml.preprocessing import (
    build_full_pipeline, get_X_y_groups, make_grouped_cv,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------- learning curve


def compute_learning_curve(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/extras",
    fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
    seeds: tuple[int, ...] = (42, 7, 13),
    cv_n_folds: int = 3,
) -> pd.DataFrame:
    """For each fraction × seed, sub-sample train and CV-evaluate the winner."""
    from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    le = joblib.load(label_encoder_path)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    decision = _json.loads(
        Path("data/processed/ml_models/selection_decision.json").read_text()
    )
    winner = decision["chosen"]
    factory = MODEL_FACTORIES[winner]

    rows: list[dict] = []
    rng = np.random.default_rng(0)
    for frac in fractions:
        for seed in seeds:
            n_sub = max(50, int(len(X) * frac))
            sub_idx = rng.choice(len(X), n_sub, replace=False)
            X_sub = X.iloc[sub_idx]; y_sub = y[sub_idx]; sw_sub = sw[sub_idx]
            skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                                   random_state=seed)
            fold_f1: list[float] = []
            for tr, te in skf.split(X_sub, y_sub):
                model = factory(n_classes=n_classes, random_state=seed)
                pipe = build_full_pipeline(model)
                cw = _combined_weight(y_sub[tr], sw_sub[tr])
                last = pipe.steps[-1][0]
                try:
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        pipe.fit(X_sub.iloc[tr], y_sub[tr],
                                 **{f"{last}__sample_weight": cw})
                    y_pred = pipe.predict(X_sub.iloc[te])
                    fold_f1.append(float(f1_score(y_sub[te], y_pred,
                                                    average="macro",
                                                    zero_division=0)))
                except Exception as exc:
                    logger.warning("learning_curve f=%.2f s=%d failed: %s",
                                   frac, seed, exc)
            rows.append({
                "fraction": frac, "seed": seed,
                "n_train_total": n_sub,
                "macro_f1_mean": float(np.mean(fold_f1)) if fold_f1 else float("nan"),
                "macro_f1_std": float(np.std(fold_f1)) if len(fold_f1) > 1 else 0.0,
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "learning_curve.csv", index=False)
    logger.info("Learning curve: %d (frac, seed) cells", len(out))
    return out


# ----------------------------------------------- ablation: structural features off


STRUCTURAL_PREFIXES = (
    "vina_", "ifp_", "gnina_", "redock_", "n_residues_contacted",
    "n_total_contacts", "Asphericity", "Eccentricity", "InertialShapeFactor",
    "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", "RadiusOfGyration",
    "SpherocityIndex", "pose_3d_missing", "ifp_missing", "ifp_no_contacts",
    "docking_confidence_",
)


def ablation_drop_structural(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/extras",
    seeds: tuple[int, ...] = (42, 7, 13),
    cv_n_folds: int = 3,
) -> pd.DataFrame:
    """Drop all structural features; evaluate winner on remaining (chemistry-only)
    feature set. Quantifies the contribution of the docking pipeline."""
    from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    le = joblib.load(label_encoder_path)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    decision = _json.loads(
        Path("data/processed/ml_models/selection_decision.json").read_text()
    )
    winner = decision["chosen"]
    factory = MODEL_FACTORIES[winner]

    structural_cols = [
        c for c in X.columns
        if any(c.startswith(p) or c == p for p in STRUCTURAL_PREFIXES)
    ]
    chemistry_cols = [c for c in X.columns if c not in structural_cols]
    logger.info(
        "Ablation: %d structural features will be dropped, %d chemistry remain",
        len(structural_cols), len(chemistry_cols),
    )

    rows: list[dict] = []
    for variant_name, X_variant in (
        ("with_structural", X), ("chemistry_only", X[chemistry_cols]),
    ):
        for seed in seeds:
            skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                                   random_state=seed)
            fold_f1: list[float] = []
            for tr, te in skf.split(X_variant, y):
                model = factory(n_classes=n_classes, random_state=seed)
                pipe = build_full_pipeline(model)
                cw = _combined_weight(y[tr], sw[tr])
                last = pipe.steps[-1][0]
                try:
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        pipe.fit(X_variant.iloc[tr], y[tr],
                                 **{f"{last}__sample_weight": cw})
                    y_pred = pipe.predict(X_variant.iloc[te])
                    fold_f1.append(float(f1_score(y[te], y_pred,
                                                    average="macro",
                                                    zero_division=0)))
                except Exception as exc:
                    logger.warning("ablation %s s=%d failed: %s",
                                   variant_name, seed, exc)
            rows.append({
                "variant": variant_name, "seed": seed,
                "n_features": int(X_variant.shape[1]),
                "macro_f1_mean": float(np.mean(fold_f1)) if fold_f1 else float("nan"),
                "macro_f1_std": float(np.std(fold_f1)) if len(fold_f1) > 1 else 0.0,
            })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "ablation_no_structural.csv", index=False)
    return out


# ----------------------------------------------- paired permutation test


def paired_permutation_test_models(
    cv_results_path: Path | str = "data/processed/ml_models/cv_results_long.csv",
    *,
    output_dir: Path | str = "data/processed/ml_models/extras",
    metric: str = "macro_f1",
    n_perm: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare every pair of models via paired permutation test on per-fold metrics.
    H0: their mean metric is equal. Returns p-values for each pair."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cv = pd.read_csv(cv_results_path)
    models = sorted(cv["model"].unique())
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            # Match on (split, fold, seed) so we have paired observations
            a = cv[cv["model"] == ma].sort_values(["split", "fold", "seed"])
            b = cv[cv["model"] == mb].sort_values(["split", "fold", "seed"])
            common = pd.merge(
                a[["split", "fold", "seed", metric]],
                b[["split", "fold", "seed", metric]],
                on=["split", "fold", "seed"], suffixes=("_a", "_b"),
            )
            if len(common) < 5:
                continue
            diffs = common[f"{metric}_a"].values - common[f"{metric}_b"].values
            obs = float(np.mean(diffs))
            # Permute signs
            count = 0
            for _ in range(n_perm):
                signs = rng.choice([1, -1], size=len(diffs))
                perm_mean = np.mean(diffs * signs)
                if abs(perm_mean) >= abs(obs):
                    count += 1
            p = (count + 1) / (n_perm + 1)
            rows.append({
                "model_a": ma, "model_b": mb,
                "n_paired": int(len(common)),
                "mean_diff": obs,
                "p_value_two_sided": float(p),
                "winner": ma if obs > 0 else mb,
            })
    out = pd.DataFrame(rows).sort_values("p_value_two_sided")
    out.to_csv(output_dir / "paired_permutation_test.csv", index=False)
    return out


# ----------------------------------------------- leave-one-receptor-out


def loro_top_receptors(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/extras",
    n_top_receptors: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Leave-one-receptor-out CV for the top-N most-tested receptors. Tests
    memorization-of-receptor-identity vs learned biology."""
    from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    le = joblib.load(label_encoder_path)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    decision = _json.loads(
        Path("data/processed/ml_models/selection_decision.json").read_text()
    )
    winner = decision["chosen"]
    factory = MODEL_FACTORIES[winner]

    receptor_counts = df["receptor_uniprot"].value_counts()
    top_receptors = receptor_counts.head(n_top_receptors).index.tolist()
    rows: list[dict] = []
    for held in top_receptors:
        train_mask = df["receptor_uniprot"] != held
        test_mask = ~train_mask
        if test_mask.sum() < 5 or train_mask.sum() < 50:
            continue
        model = factory(n_classes=n_classes, random_state=seed)
        pipe = build_full_pipeline(model)
        cw = _combined_weight(y[train_mask.values], sw[train_mask.values])
        last = pipe.steps[-1][0]
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                pipe.fit(X[train_mask.values], y[train_mask.values],
                         **{f"{last}__sample_weight": cw})
            y_pred = pipe.predict(X[test_mask.values])
            macro_f1 = float(f1_score(y[test_mask.values], y_pred,
                                        average="macro", zero_division=0))
        except Exception as exc:
            logger.warning("LORO held=%s failed: %s", held, exc)
            macro_f1 = float("nan")
        rows.append({
            "held_out_receptor": held,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "macro_f1": macro_f1,
        })
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "loro_results.csv", index=False)
    return out


# ----------------------------------------------- temporal-shift Wasserstein


def temporal_shift_wasserstein(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    holdout_path: Path | str = "data/holdout/dataset_holdout.parquet",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/extras",
    top_n: int = 30,
) -> pd.DataFrame:
    """Per-feature Wasserstein distance between train and holdout distributions.
    Identifies whether the holdout drop is driven by covariate shift on a few
    features."""
    from scipy.stats import wasserstein_distance
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(dataset_path)
    holdout = pd.read_parquet(holdout_path)
    le = joblib.load(label_encoder_path)
    X_tr, _, _, _, _ = get_X_y_groups(train, label_encoder=le)
    X_ho, _, _, _, _ = get_X_y_groups(holdout, label_encoder=le)
    common_cols = [c for c in X_tr.columns if c in X_ho.columns]
    rows: list[dict] = []
    for c in common_cols:
        try:
            tr_vals = X_tr[c].dropna().astype(float).values
            ho_vals = X_ho[c].dropna().astype(float).values
            if len(tr_vals) < 5 or len(ho_vals) < 5:
                continue
            d = wasserstein_distance(tr_vals, ho_vals)
            rows.append({
                "feature": c, "wasserstein_distance": float(d),
                "train_mean": float(tr_vals.mean()),
                "holdout_mean": float(ho_vals.mean()),
                "abs_mean_shift": float(abs(tr_vals.mean() - ho_vals.mean())),
            })
        except Exception:
            continue
    out = pd.DataFrame(rows).sort_values("wasserstein_distance",
                                          ascending=False)
    out.to_csv(output_dir / "temporal_shift.csv", index=False)
    logger.info("Temporal shift top 5: %s",
                out.head(5)[["feature", "wasserstein_distance"]].to_dict())
    return out


# ----------------------------------------------- model card


def write_model_card(
    *,
    output_dir: Path | str = "data/processed/ml_models",
    model_meta_path: Path | str = "data/processed/ml_models/model_meta.json",
    selection_decision_path: Path | str = "data/processed/ml_models/selection_decision.json",
) -> Path:
    """Write a Mitchell-et-al-2019-style model card to model_card.md."""
    output_dir = Path(output_dir)
    meta = _json.loads(Path(model_meta_path).read_text())
    decision = _json.loads(Path(selection_decision_path).read_text())
    winner = decision["chosen"]
    holdout = meta.get("holdout_metrics", {})
    md = [
        f"# CancerAg model card — {winner}",
        "",
        "## Model details",
        f"- Model family: {winner}",
        f"- Training framework: scikit-learn {__import__('sklearn').__version__} pipeline",
        f"- Trained on UTC: {meta.get('trained_at_utc')}",
        f"- Number of classes: {meta.get('n_classes')}",
        f"- Class labels: {meta.get('label_classes')}",
        f"- Training rows: {meta.get('n_train_eligible')}",
        f"- Input features: {meta.get('n_features')} after preprocessing",
        "",
        "## Intended use",
        "- Predict the *biased agonism category* of a small-molecule ligand at a "
        "G-protein coupled receptor (GPCR), choosing among "
        f"`{meta.get('label_classes', [])}`.",
        "- Designed for **virtual triage of drug candidates** in early-stage "
        "GPCR drug discovery pipelines, NOT for clinical decision-making.",
        "- **Out of scope**: peptide ligands (excluded from training scope; "
        "see manuscript), receptors not present in the training set.",
        "",
        "## Training data",
        "- Source: BiasDB curated bias measurements (719 rows, "
        "587 unique (ligand, receptor) pairs), filtered to small molecules "
        "(rotatable bonds < 16) that successfully docked with AutoDock Vina.",
        "- Data splits: stratified-K-fold, scaffold-grouped K-fold, "
        "receptor-grouped K-fold, and temporal holdout (year ≥ 2018).",
        f"- Bias label distribution (train-eligible): "
        f"{meta.get('selection_decision', {}).get('summary', [{}])[0]}",
        "",
        "## Evaluation",
        f"- Primary CV split: {decision.get('primary_split')}",
        f"- Mean macro-F1: "
        f"{decision.get('winner_metrics', {}).get('macro_f1_mean'):.3f}",
    ]
    if holdout and "macro_f1" in holdout:
        m = holdout["macro_f1"]
        md.append(f"- Temporal holdout macro-F1: {m['point_estimate']:.3f} "
                  f"[CI {m['ci_lo']:.3f}, {m['ci_hi']:.3f}] (n={holdout.get('n')})")
    md.extend([
        "",
        "## Limitations and known biases",
        "- 4-class imbalance is severe (G protein 56.7% vs ERK 7.2%); rare-class "
        "F1 has wide bootstrap CIs.",
        "- Evaluation reveals a substantial cross-validation → temporal-holdout "
        "gap, consistent with documented temporal shift in the biased-agonism "
        "literature (new assay technologies and receptor families post-2018).",
        "- Receptor-grouped CV macro-F1 is much lower than stratified-CV macro-F1, "
        "indicating that the model partially memorizes receptor identity. Predictions "
        "for receptor families absent from training should be flagged as out-of-domain.",
        "- Peptide ligands (~11% of BiasDB) are excluded from this version of the "
        "pipeline due to AutoDock Vina's known limitations on highly flexible "
        "molecules. A peptide-aware extension (HADDOCK / FlexPepDock) is reserved "
        "for future work.",
        "",
        "## Ethical considerations",
        "- The model is intended for research-stage drug discovery, not clinical use.",
        "- Predictions should be validated experimentally before any in-vivo decision.",
        "- The training data is publicly sourced (BiasDB) and does not contain "
        "patient-derived information.",
        "",
        "## Caveats and recommendations",
        "- Always run the applicability-domain check (Tanimoto ≥ 0.4 to nearest "
        "training molecule) before trusting a prediction.",
        "- Report calibrated probabilities (the calibrated model handles this).",
        "- For new receptors, prefer the receptor-grouped CV macro-F1 as the "
        "performance expectation.",
    ])
    card_path = output_dir / "model_card.md"
    card_path.write_text("\n".join(md))
    logger.info("Wrote model card -> %s", card_path)
    return card_path


# ----------------------------------------------- entry point


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger.info("Reviewer extras — running all 6 modules")
    compute_learning_curve()
    ablation_drop_structural()
    paired_permutation_test_models()
    loro_top_receptors()
    temporal_shift_wasserstein()
    write_model_card()
    logger.info("STAGE_10_EXTRAS_DONE")


if __name__ == "__main__":
    main()
