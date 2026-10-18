"""Unsupervised clustering of the full labelled set (train + holdout, n=615).

Asks whether the feature space has natural groups that line up with the four
bias classes, without using the labels to form them. Writes
``clustering_results.json`` (k sweep, internal validity, external agreement
with the labels, PCA variance) and ``cluster_assignments.csv`` (per-row
clusters plus 2-D PCA and UMAP coordinates for plotting).

The procedure, which reproduces the shipped artefact to floating-point
precision:

1. Concatenate ``ml_ready_dataset.parquet`` and ``dataset_holdout.parquet``
   (row order: train then holdout).
2. Take every numeric column except ``year`` and ``sample_weight`` (booleans
   such as ``has_pains`` are not numeric dtypes and drop out here). This keeps
   the docking-QC columns ``redock_rmsd_angstrom``, ``gnina_cnn_score``,
   ``TPSA_lig`` and the two ``*_missing`` flags alongside the feature columns.
3. Fill missing values with the column median over all 615 rows.
4. Drop columns with population variance <= 1e-4 (constant columns and
   ``InertialShapeFactor``), leaving 2,242 columns.
5. Standardise (``StandardScaler``).
6. KMeans for k = 2..10 (``n_init=10, random_state=42``); k* is the k with
   the highest silhouette. Validity indices and ARI/AMI against the labels
   are reported at k* and at k = number of classes.
7. PCA (2 components, ``random_state=42``) and UMAP (library defaults,
   ``random_state=42``) on the standardised matrix.

Run:
    python -m cancerag.ml.unsupervised_clustering \
        [--output-dir data/processed/ml_models/clustering]
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DATASET_PATH = Path("data/processed/ml_ready_dataset.parquet")
HOLDOUT_PATH = Path("data/holdout/dataset_holdout.parquet")
OUTPUT_DIR = Path("data/processed/ml_models/clustering")
LABEL_COL = "bias_category"
EXCLUDE_NUMERIC = ("year", "sample_weight")
VARIANCE_THRESHOLD = 1e-4
K_VALUES = tuple(range(2, 11))
SEED = 42


def clustering_matrix(
    df: pd.DataFrame, *, exclude: tuple[str, ...] = EXCLUDE_NUMERIC,
    variance_threshold: float = VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Numeric columns, median-filled, low-variance columns dropped (unscaled)."""
    num = df.select_dtypes("number").drop(
        columns=[c for c in exclude if c in df.columns]
    )
    num = num.fillna(num.median())
    return num.loc[:, num.var(ddof=0) > variance_threshold]


def _validity(Z: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "silhouette": float(silhouette_score(Z, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(Z, labels)),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
    }


def _agreement(y: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "adjusted_rand_index": float(adjusted_rand_score(y, labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(y, labels)),
    }


def run_clustering(
    df: pd.DataFrame, *, label_col: str = LABEL_COL,
    k_values: tuple[int, ...] = K_VALUES, seed: int = SEED,
    with_umap: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Return ``(results, assignments)`` for a labelled frame."""
    X = clustering_matrix(df)
    Z = StandardScaler().fit_transform(X)
    y = df[label_col].astype(str).to_numpy()

    fits: dict[int, KMeans] = {}
    inertias, silhouettes = [], []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        fits[k] = km
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(Z, km.labels_)))
        logger.info("k=%d inertia=%.1f silhouette=%.5f", k, km.inertia_,
                    silhouettes[-1])
    k_star = int(k_values[int(np.argmax(silhouettes))])
    k_classes = int(pd.Series(y).nunique())
    if k_classes not in fits:
        fits[k_classes] = KMeans(n_clusters=k_classes, n_init=10,
                                 random_state=seed).fit(Z)
    lab_star, lab_cls = fits[k_star].labels_, fits[k_classes].labels_

    pca = PCA(n_components=2, random_state=seed).fit(Z)
    P = pca.transform(Z)

    crosstab = pd.crosstab(pd.Series(lab_cls, name="cluster"),
                           pd.Series(y, name="class"))
    results = {
        "n_samples": len(df),
        "n_features_used": int(X.shape[1]),
        "label_column": label_col,
        "class_counts": {str(k): int(v)
                         for k, v in pd.Series(y).value_counts().items()},
        "sweep": {
            "k_values": list(k_values),
            "inertias": inertias,
            "silhouettes": silhouettes,
            "k_best_silhouette": k_star,
        },
        "k_star": k_star,
        "validity_k_star": _validity(Z, lab_star),
        "validity_k_classes": _validity(Z, lab_cls),
        "external_agreement_k_classes": _agreement(y, lab_cls),
        "external_agreement_k_star": _agreement(y, lab_star),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "cluster_by_class_k_classes": {
            str(c): {str(k): int(v) for k, v in row.items()}
            for c, row in crosstab.iterrows()
        },
    }

    assignments = pd.DataFrame({
        "true_class": y,
        "cluster_k_classes": lab_cls,
        "cluster_k_star": lab_star,
        "pca1": P[:, 0],
        "pca2": P[:, 1],
    })
    if with_umap:
        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E = umap.UMAP(n_components=2, random_state=seed).fit_transform(Z)
        assignments["umap1"] = E[:, 0]
        assignments["umap2"] = E[:, 1]
    return results, assignments


def load_labelled_set(
    dataset_path: Path | str = DATASET_PATH,
    holdout_path: Path | str = HOLDOUT_PATH,
) -> pd.DataFrame:
    frames = [pd.read_parquet(dataset_path)]
    if Path(holdout_path).exists():
        frames.append(pd.read_parquet(holdout_path))
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Unsupervised clustering (n=615)")
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--holdout", default=str(HOLDOUT_PATH))
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--no-umap", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    df = load_labelled_set(args.dataset, args.holdout)
    results, assignments = run_clustering(df, with_umap=not args.no_umap)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clustering_results.json").write_text(json.dumps(results, indent=2))
    assignments.to_csv(out / "cluster_assignments.csv", index=False)
    print(f"n={results['n_samples']} features={results['n_features_used']} "
          f"k*={results['k_star']} -> {out}")


if __name__ == "__main__":
    main()
