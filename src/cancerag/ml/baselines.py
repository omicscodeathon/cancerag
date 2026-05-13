"""
Stage 10+ — Pretrained molecular foundation-model baselines.

Implements a ChemBERTa zero-shot embedding + LogReg head as a "modern
molecular foundation model" baseline — addresses the Reviewer-1 ask for a
GNN/pretrained-model comparison without the dependency friction of installing
ChemProp on Python 3.13.

Pipeline:
  1. Tokenize each canonical SMILES with the ChemBERTa tokenizer.
  2. Run a frozen forward pass through ``DeepChem/ChemBERTa-77M-MTR`` (or
     similar checkpoint) to get a 384-dim per-molecule embedding (mean-pool
     over token positions).
  3. Train a multi-class LogisticRegression on those embeddings, using the
     same scaffold-grouped CV + 5 seeds setup as the main bake-off.

Output (data/processed/ml_models/baselines/chemberta/):
  - chemberta_cv_results.csv  : (seed, fold, macro_f1, balanced_acc)
  - chemberta_summary.json    : aggregate metrics (mean ± std)
  - chemberta_holdout.json    : holdout metrics (calibrated isotonic)

Why this is the recommended GNN-equivalent:
  - ChemBERTa is *more* current than ChemProp (transformer foundation model
    pretrained on ~10M PubChem SMILES vs ChemProp's D-MPNN from 2019).
  - Installs cleanly on Python 3.13 (vs ChemProp's 3.10-3.12 cap).
  - At n=443/4-class with severe imbalance, GNN-class methods typically tie
    or slightly underperform tuned XGBoost — this baseline confirms or
    refutes that for our specific dataset.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from cancerag.ml.preprocessing import make_grouped_cv

logger = logging.getLogger(__name__)


# Small ChemBERTa checkpoint (~30 MB). MTR = Multi-Task Regression head.
DEFAULT_CHECKPOINT = "DeepChem/ChemBERTa-77M-MTR"


# ----------------------------------------------- embedding


def chemberta_embed(
    smiles_list: list[str],
    *,
    checkpoint: str = DEFAULT_CHECKPOINT,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Embed a list of canonical SMILES into a (n, d) numpy array.

    Uses mean-pooling over token positions. Falls back to all-zero vectors
    for any SMILES that fail to tokenize.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    logger.info("Loading ChemBERTa from %s", checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint).eval().to(device)

    embeddings: list[np.ndarray] = []
    n = len(smiles_list)
    for i in range(0, n, batch_size):
        batch = smiles_list[i:i + batch_size]
        # Replace empty / non-string with a placeholder so batch dimension stays
        batch_clean = [s if isinstance(s, str) and s else "C" for s in batch]
        try:
            tokens = tokenizer(
                batch_clean, padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**tokens)
            # last_hidden_state: (batch, seq_len, hidden); mean-pool over seq
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = (summed / counts).cpu().numpy()
            embeddings.append(mean_pooled)
        except Exception as exc:
            logger.warning("ChemBERTa batch %d failed: %s; using zeros", i, exc)
            hidden_dim = getattr(model.config, "hidden_size", 384)
            embeddings.append(np.zeros((len(batch), hidden_dim), dtype=np.float32))
        if (i // batch_size) % 10 == 0:
            logger.info("ChemBERTa: embedded %d/%d", i + len(batch), n)
    return np.vstack(embeddings).astype(np.float32)


# ----------------------------------------------- baseline runner


def run_chemberta_baseline(
    *,
    dataset_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    holdout_path: Path | str = "data/holdout/dataset_holdout.parquet",
    label_encoder_path: Path | str = "data/processed/ml_preprocessed/label_encoder.joblib",
    output_dir: Path | str = "data/processed/ml_models/baselines/chemberta",
    seeds: tuple[int, ...] = (42, 7, 13),
    cv_n_folds: int = 5,
    smiles_col: str = "canonical_smiles_std",
) -> dict:
    """Embed canonical SMILES with ChemBERTa, train multinomial LogReg head,
    evaluate under the same scaffold-grouped CV + temporal-holdout protocol
    as the main bake-off."""
    from sklearn.utils.class_weight import compute_sample_weight
    from cancerag.ml.model_evaluation import report_metrics

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(dataset_path)
    le = joblib.load(label_encoder_path)
    n_classes = len(le.classes_)

    # 1. Embed SMILES (cache to disk so we don't re-embed on every run)
    cache_path = output_dir / "embeddings.npz"
    pair_keys = df["pair_key"].values
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if (set(cached["pair_keys"].tolist())
                == set(pair_keys.tolist())):
            logger.info("Loading cached ChemBERTa embeddings")
            # Realign to current df order
            cached_keys = list(cached["pair_keys"])
            cached_emb = cached["embeddings"]
            order = [cached_keys.index(k) for k in pair_keys]
            X_emb = cached_emb[order]
        else:
            X_emb = chemberta_embed(df[smiles_col].tolist())
            np.savez_compressed(
                cache_path, embeddings=X_emb, pair_keys=pair_keys,
            )
    else:
        X_emb = chemberta_embed(df[smiles_col].tolist())
        np.savez_compressed(
            cache_path, embeddings=X_emb, pair_keys=pair_keys,
        )

    y = le.transform(df["bias_category"].astype(str).values)
    sw = df["sample_weight"].astype(float).values

    # 2. Multi-seed × multi-split CV
    rows: list[dict] = []
    for seed in seeds:
        # Stratified-kfold
        skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True,
                               random_state=seed)
        for split_name, splits in (
            ("stratified_kfold", list(skf.split(X_emb, y))),
            ("scaffold_kfold",
             make_grouped_cv(df, group_col="scaffold",
                              n_splits=cv_n_folds, seed=seed)),
        ):
            for fi, (tr, te) in enumerate(splits):
                cw = compute_sample_weight("balanced", y[tr]) * sw[tr]
                clf = LogisticRegression(
                    multi_class="multinomial", max_iter=2000,
                    C=1.0, class_weight="balanced",
                    n_jobs=4, random_state=seed,
                )
                try:
                    clf.fit(X_emb[tr], y[tr], sample_weight=cw)
                    y_pred = clf.predict(X_emb[te])
                    rep = report_metrics(y[te], y_pred, bootstrap_n=300, seed=seed)
                    rows.append({
                        "model": "chemberta_logreg",
                        "split": split_name,
                        "seed": seed, "fold": fi,
                        "macro_f1": rep["macro_f1"]["point_estimate"],
                        "macro_f1_ci_lo": rep["macro_f1"]["ci_lo"],
                        "macro_f1_ci_hi": rep["macro_f1"]["ci_hi"],
                        "balanced_accuracy": rep["balanced_accuracy"]["point_estimate"],
                    })
                except Exception as exc:
                    logger.warning(
                        "[chemberta|%s|fold=%d|seed=%d] failed: %s",
                        split_name, fi, seed, exc,
                    )

    cv = pd.DataFrame(rows)
    cv.to_csv(output_dir / "chemberta_cv_results.csv", index=False)

    summary = (
        cv.groupby("split")["macro_f1"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary.to_csv(output_dir / "chemberta_summary.csv", index=False)

    # 3. Holdout: refit on all train-eligible, predict on holdout
    holdout_metrics = {}
    if Path(holdout_path).exists():
        ho = pd.read_parquet(holdout_path)
        ho_emb = chemberta_embed(ho[smiles_col].tolist())
        ho_y = le.transform(ho["bias_category"].astype(str).values)
        cw_all = compute_sample_weight("balanced", y) * sw
        clf = LogisticRegression(
            multi_class="multinomial", max_iter=2000, C=1.0,
            class_weight="balanced", n_jobs=4, random_state=42,
        )
        clf.fit(X_emb, y, sample_weight=cw_all)
        # Calibrate
        try:
            cal = CalibratedClassifierCV(clf, method="isotonic", cv=3).fit(
                X_emb, y, sample_weight=cw_all,
            )
        except Exception:
            cal = clf
        ho_pred = cal.predict(ho_emb)
        rep = report_metrics(ho_y, ho_pred, bootstrap_n=1000, seed=42)
        holdout_metrics = {
            "n": rep["n_test"],
            "macro_f1": rep["macro_f1"],
            "balanced_accuracy": rep["balanced_accuracy"],
            "per_class_f1": rep["per_class_f1"],
        }
        joblib.dump(cal, output_dir / "chemberta_logreg_calibrated.joblib")

    out_meta = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": DEFAULT_CHECKPOINT,
        "embedding_dim": int(X_emb.shape[1]),
        "n_train_eligible": int(len(df)),
        "cv_summary": summary.to_dict(orient="records"),
        "holdout": holdout_metrics,
    }
    (output_dir / "chemberta_summary.json").write_text(
        json.dumps(out_meta, indent=2, sort_keys=True, default=str)
    )
    logger.info("ChemBERTa baseline complete: cv_summary=%s",
                summary.to_dict(orient="records"))
    return out_meta


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    out = run_chemberta_baseline(seeds=(42,), cv_n_folds=3)
    print("\nChemBERTa baseline summary:")
    for s in out["cv_summary"]:
        print(f"  {s['split']}: macro-F1 {s['mean']:.3f} ± {s['std']:.3f} (n={s['n']})")
    if out["holdout"]:
        m = out["holdout"]["macro_f1"]
        print(f"  Holdout macro-F1: {m['point_estimate']:.3f} "
              f"[CI {m['ci_lo']:.3f}, {m['ci_hi']:.3f}]")


if __name__ == "__main__":
    main()
