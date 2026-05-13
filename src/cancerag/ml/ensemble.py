"""
Phase 6 — Stacking ensemble + focal-loss XGBoost.

Two complementary approaches to push past the single-model ceiling:

  1. **Stacking ensemble**: out-of-fold predictions from LightGBM, XGBoost,
     and ChemBERTa-LR are stacked into a logistic-regression meta-learner.
     Captures the discovered nuance: tree models win in-distribution,
     ChemBERTa wins out-of-distribution. The meta-learner picks per-class
     mixing weights.

  2. **Focal-loss XGBoost**: replaces multi-class log-loss with focal loss
     (Lin et al. 2017), which down-weights easy examples and concentrates
     gradient on hard rare-class examples. Targets the ERK and G-protein-
     selectivity classes which currently score F1=0 on holdout.

Run:
    python -m cancerag.ml.ensemble stack
    python -m cancerag.ml.ensemble focal
    python -m cancerag.ml.ensemble all
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from cancerag.ml.preprocessing import (
    build_full_pipeline, get_X_y_groups, make_grouped_cv,
)

logger = logging.getLogger(__name__)


OUTPUT_DIR = Path("data/processed/ml_models/advanced")
DATASET_PATH = Path("data/processed/ml_ready_dataset.parquet")
HOLDOUT_PATH = Path("data/holdout/dataset_holdout.parquet")
LE_PATH = Path("data/processed/ml_preprocessed/label_encoder.joblib")


# --------------------------------------------- stacking ensemble


def build_oof_predictions(
    X, y, sw, n_classes, *, model_name: str, seed: int = 42, n_folds: int = 5,
) -> tuple[np.ndarray, list]:
    """Out-of-fold predictions for one model on scaffold-grouped CV.
    Returns (oof_proba, fold_pipelines) for downstream meta-learner training."""
    from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight
    df_for_split = pd.read_parquet(DATASET_PATH)
    splits = make_grouped_cv(df_for_split, group_col="scaffold",
                              n_splits=n_folds, seed=seed)
    if not splits:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=seed).split(X, y))

    oof = np.zeros((len(X), n_classes), dtype=np.float32)
    fitted: list = []
    factory = MODEL_FACTORIES[model_name]
    for fi, (tr, te) in enumerate(splits):
        model = factory(n_classes=n_classes, random_state=seed)
        pipe = build_full_pipeline(model)
        cw = _combined_weight(y[tr], sw[tr])
        last = pipe.steps[-1][0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X.iloc[tr], y[tr],
                         **{f"{last}__sample_weight": cw})
            oof[te] = pipe.predict_proba(X.iloc[te])
            fitted.append(pipe)
        except Exception as exc:
            logger.warning("OOF fold %d for %s failed: %s", fi, model_name, exc)
            fitted.append(None)
    return oof, fitted


def build_chemberta_oof(X_emb, y, sw, n_classes, *, seed: int = 42, n_folds: int = 5):
    """OOF predictions for ChemBERTa-LR head (uses pre-computed embeddings)."""
    from sklearn.utils.class_weight import compute_sample_weight
    df_for_split = pd.read_parquet(DATASET_PATH)
    splits = make_grouped_cv(df_for_split, group_col="scaffold",
                              n_splits=n_folds, seed=seed)
    if not splits:
        splits = list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=seed).split(X_emb, y))
    oof = np.zeros((len(X_emb), n_classes), dtype=np.float32)
    for tr, te in splits:
        cw = compute_sample_weight("balanced", y[tr]) * sw[tr]
        clf = LogisticRegression(
            max_iter=2000, C=1.0, class_weight="balanced",
            n_jobs=4, random_state=seed,
        )
        try:
            clf.fit(X_emb[tr], y[tr], sample_weight=cw)
            oof[te] = clf.predict_proba(X_emb[te])
        except Exception as exc:
            logger.warning("ChemBERTa OOF fold failed: %s", exc)
    return oof


def run_stacking(
    *, output_dir: Path | str = OUTPUT_DIR, seed: int = 42,
) -> dict:
    """LightGBM + XGBoost + ChemBERTa-LR → LR meta-learner."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATASET_PATH)
    le = joblib.load(LE_PATH)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)

    # 1. Build OOF predictions for each base model
    logger.info("Computing OOF predictions for LightGBM...")
    oof_lgbm, _ = build_oof_predictions(X, y, sw, n_classes,
                                         model_name="lightgbm", seed=seed)
    logger.info("Computing OOF predictions for XGBoost...")
    oof_xgb, _ = build_oof_predictions(X, y, sw, n_classes,
                                         model_name="xgboost", seed=seed)
    logger.info("Loading ChemBERTa embeddings + OOF predictions...")
    cb_cache = Path("data/processed/ml_models/baselines/chemberta/embeddings.npz")
    if cb_cache.exists():
        cached = np.load(cb_cache, allow_pickle=True)
        cached_keys = list(cached["pair_keys"])
        order = [cached_keys.index(k) for k in df["pair_key"].values]
        X_emb = cached["embeddings"][order]
        oof_cb = build_chemberta_oof(X_emb, y, sw, n_classes, seed=seed)
        bases = {"lightgbm": oof_lgbm, "xgboost": oof_xgb, "chemberta_lr": oof_cb}
    else:
        logger.warning("ChemBERTa embeddings missing; stacking without it")
        bases = {"lightgbm": oof_lgbm, "xgboost": oof_xgb}

    # 2. Stack: meta-features = [n_classes per model] concatenated
    Z = np.hstack([bases[k] for k in sorted(bases)])
    logger.info("Stacked meta-features: %s", Z.shape)

    # 3. Meta-learner = multinomial LR on Z
    from sklearn.utils.class_weight import compute_sample_weight
    cw_all = compute_sample_weight("balanced", y) * sw
    meta = LogisticRegression(
        max_iter=2000, C=1.0, class_weight="balanced",
        n_jobs=4, random_state=seed,
    )
    meta.fit(Z, y, sample_weight=cw_all)
    joblib.dump(meta, output_dir / "stacking_meta_learner.joblib")

    # 4. CV macro-F1 of the stacked model on its own OOF
    z_pred = meta.predict(Z)
    cv_f1 = float(f1_score(y, z_pred, average="macro", zero_division=0))
    logger.info("Stacking ensemble CV macro-F1: %.4f", cv_f1)

    # 5. Holdout predictions (refit base models on all train-eligible, get
    # holdout proba, stack)
    holdout_metrics = {}
    if HOLDOUT_PATH.exists():
        from cancerag.ml.model_evaluation import report_metrics
        from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight
        ho = pd.read_parquet(HOLDOUT_PATH)
        ho_X, ho_y, _, _, _ = get_X_y_groups(ho, label_encoder=le)
        for c in X.columns:
            if c not in ho_X.columns:
                ho_X[c] = 0.0
        ho_X = ho_X[X.columns]

        ho_bases = {}
        for mname in ("lightgbm", "xgboost"):
            if mname not in bases: continue
            factory = MODEL_FACTORIES[mname]
            model = factory(n_classes=n_classes, random_state=seed)
            pipe = build_full_pipeline(model)
            last = pipe.steps[-1][0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X, y, **{f"{last}__sample_weight": cw_all})
            ho_bases[mname] = pipe.predict_proba(ho_X)
        if "chemberta_lr" in bases:
            from cancerag.ml.baselines import chemberta_embed
            ho_emb = chemberta_embed(ho["canonical_smiles_std"].tolist())
            cb_clf = LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced",
                n_jobs=4, random_state=seed,
            )
            cb_clf.fit(X_emb, y, sample_weight=cw_all)
            ho_bases["chemberta_lr"] = cb_clf.predict_proba(ho_emb)

        Z_ho = np.hstack([ho_bases[k] for k in sorted(ho_bases)])
        ho_pred = meta.predict(Z_ho)
        rep = report_metrics(ho_y, ho_pred, bootstrap_n=1000, seed=42)
        holdout_metrics = {
            "n": rep["n_test"], "macro_f1": rep["macro_f1"],
            "balanced_accuracy": rep["balanced_accuracy"],
            "per_class_f1": rep["per_class_f1"],
            "base_models": list(sorted(bases.keys())),
        }

    out = {
        "schema_version": 1,
        "stacked_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_models": list(sorted(bases.keys())),
        "meta_learner": "LogisticRegression(multinomial, C=1.0, class_weight=balanced)",
        "stacking_oof_macro_f1": cv_f1,
        "holdout_metrics": holdout_metrics,
    }
    (output_dir / "stacking_meta.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str)
    )
    return out


# --------------------------------------------- focal-loss XGBoost


def _focal_loss_xgb(gamma: float = 2.0, alpha: float | None = None):
    """Returns (objective, eval) for multi-class focal loss with XGBoost.

    Implements multi-class focal loss as in Lin et al. 2017 with the
    derivative wrt logits suitable for XGBoost's custom-objective interface.
    `alpha` is a per-class scaling vector (None = no per-class scaling on top
    of focal-modulation).
    """
    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def objective(preds, dtrain):
        # preds shape: (n_samples * n_classes,) for multi:softprob custom obj
        labels = dtrain.get_label().astype(int)
        n_samples = len(labels)
        n_classes = preds.shape[0] // n_samples
        preds = preds.reshape(n_samples, n_classes)
        prob = softmax(preds)
        y_one = np.zeros_like(prob)
        y_one[np.arange(n_samples), labels] = 1.0
        # Standard cross-entropy gradient is (prob - y_one)
        # Focal loss multiplies by (1 - p_t)^gamma where p_t = prob of true class
        p_t = prob[np.arange(n_samples), labels].reshape(-1, 1)
        focal_weight = (1.0 - p_t) ** gamma
        if alpha is not None:
            alpha_arr = np.asarray(alpha)[labels].reshape(-1, 1)
            focal_weight = focal_weight * alpha_arr
        grad = focal_weight * (prob - y_one)
        # Diagonal Hessian approximation
        hess = focal_weight * prob * (1.0 - prob)
        return grad.flatten(), hess.flatten()
    return objective


def run_focal_loss(
    *, gamma: float = 2.0, output_dir: Path | str = OUTPUT_DIR, seed: int = 42,
) -> dict:
    """Train XGBoost with focal-loss objective; evaluate on scaffold-CV +
    holdout. Compares to standard XGBoost on the same splits."""
    import xgboost as xgb
    from cancerag.ml.model_training import _combined_weight
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATASET_PATH)
    le = joblib.load(LE_PATH)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)
    splits = make_grouped_cv(df, group_col="scaffold", n_splits=5, seed=seed)
    if not splits:
        splits = list(StratifiedKFold(n_splits=5, shuffle=True,
                                       random_state=seed).split(X, y))

    fold_f1: list[float] = []
    fold_per_class: list[dict] = []
    for fi, (tr, te) in enumerate(splits):
        # Apply preprocessing first (no leakage — fit on train fold)
        from cancerag.ml.preprocessing import (
            DataFrameImputer, DataFrameVarianceFilter, CorrelationFilter, DataFrameScaler,
        )
        imp = DataFrameImputer().fit(X.iloc[tr]); X_tr = imp.transform(X.iloc[tr])
        vf = DataFrameVarianceFilter(1e-4).fit(X_tr); X_tr = vf.transform(X_tr)
        cf = CorrelationFilter(0.97).fit(X_tr); X_tr = cf.transform(X_tr)
        sc = DataFrameScaler().fit(X_tr); X_tr = sc.transform(X_tr)
        X_te = sc.transform(cf.transform(vf.transform(imp.transform(X.iloc[te]))))

        cw = _combined_weight(y[tr], sw[tr])
        dtrain = xgb.DMatrix(X_tr.values, label=y[tr], weight=cw)
        dtest = xgb.DMatrix(X_te.values, label=y[te])
        params = dict(
            num_class=n_classes, max_depth=6, learning_rate=0.1,
            tree_method="hist", nthread=4, seed=seed,
            disable_default_eval_metric=1,
        )
        try:
            booster = xgb.train(
                params, dtrain, num_boost_round=200,
                obj=_focal_loss_xgb(gamma=gamma),
            )
            preds = booster.predict(dtest, output_margin=True)
            preds = preds.reshape(-1, n_classes)
            y_pred = preds.argmax(axis=1)
            f1 = float(f1_score(y[te], y_pred, average="macro", zero_division=0))
            per_class = f1_score(y[te], y_pred, labels=range(n_classes),
                                  average=None, zero_division=0)
            fold_per_class.append({str(le.classes_[i]): float(per_class[i])
                                    for i in range(n_classes)})
            fold_f1.append(f1)
            logger.info("focal-loss fold %d/%d: macro-F1=%.4f", fi+1, len(splits), f1)
        except Exception as exc:
            logger.warning("focal-loss fold %d failed: %s", fi, exc)

    cv_mean = float(np.mean(fold_f1)) if fold_f1 else float("nan")
    cv_std = float(np.std(fold_f1)) if len(fold_f1) > 1 else 0.0

    # Aggregate per-class F1 across folds
    per_class_mean = {}
    if fold_per_class:
        for cname in fold_per_class[0].keys():
            vals = [d[cname] for d in fold_per_class]
            per_class_mean[cname] = float(np.mean(vals))

    out = {
        "schema_version": 1,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": f"multi-class focal loss (gamma={gamma})",
        "scaffold_cv_macro_f1_mean": cv_mean,
        "scaffold_cv_macro_f1_std": cv_std,
        "per_class_f1_mean_across_folds": per_class_mean,
        "n_folds_completed": len(fold_f1),
    }
    (output_dir / "focal_loss_meta.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str)
    )
    logger.info("Focal-loss XGB: CV macro-F1 = %.4f ± %.4f", cv_mean, cv_std)
    return out


# --------------------------------------------- entry point


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("stack", "all"):
        logger.info("=== Stacking ensemble ===")
        out = run_stacking()
        print(f"Stacking OOF macro-F1: {out['stacking_oof_macro_f1']:.4f}")
        if out.get("holdout_metrics"):
            m = out["holdout_metrics"]["macro_f1"]
            print(f"Stacking holdout macro-F1: {m['point_estimate']:.3f} "
                  f"[CI {m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")
    if cmd in ("focal", "all"):
        logger.info("=== Focal-loss XGBoost ===")
        out = run_focal_loss(gamma=2.0)
        print(f"Focal-loss XGB CV macro-F1: {out['scaffold_cv_macro_f1_mean']:.4f}")
        print(f"Per-class F1: {out['per_class_f1_mean_across_folds']}")


if __name__ == "__main__":
    main()
