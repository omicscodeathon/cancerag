"""Tests for cancerag.ml.unsupervised_clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cancerag.ml.unsupervised_clustering import clustering_matrix, run_clustering


def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["A", "B", "C"])[np.arange(n) % 3]
    df = pd.DataFrame({
        "bias_category": y,
        "canonical_smiles": ["C"] * n,
        "year": rng.integers(2000, 2020, n),
        "sample_weight": rng.uniform(0.2, 1.0, n),
        "has_pains": rng.integers(0, 2, n).astype(bool),
        "constant": np.ones(n),
        "tiny": 1.0 + rng.normal(scale=1e-4, size=n),
        "with_nan": np.where(np.arange(n) < 5, np.nan, rng.normal(size=n)),
    })
    for i in range(6):
        df[f"f{i}"] = rng.normal(size=n) + (y == "A") * 3.0 * (i < 3)
    return df


@pytest.mark.unit
class TestClusteringMatrix:
    def test_column_selection(self):
        X = clustering_matrix(_frame())
        assert "year" not in X and "sample_weight" not in X
        assert "has_pains" not in X        # bool is not a numeric dtype
        assert "canonical_smiles" not in X
        assert "constant" not in X and "tiny" not in X   # variance <= 1e-4
        assert {"with_nan", *(f"f{i}" for i in range(6))} == set(X.columns)

    def test_median_fill(self):
        df = _frame()
        X = clustering_matrix(df)
        assert not X.isna().any().any()
        assert (X["with_nan"].iloc[:5] == df["with_nan"].median()).all()


@pytest.mark.unit
class TestRunClustering:
    def test_schema_and_consistency(self):
        df = _frame()
        res, assign = run_clustering(df, k_values=(2, 3, 4), with_umap=False)
        assert res["n_samples"] == len(df) and res["n_features_used"] == 7
        sw = res["sweep"]
        assert sw["k_values"] == [2, 3, 4] and len(sw["silhouettes"]) == 3
        assert res["k_star"] == sw["k_values"][int(np.argmax(sw["silhouettes"]))]
        assert res["validity_k_classes"]["silhouette"] == pytest.approx(
            sw["silhouettes"][1])  # k_classes = 3
        assert sum(sum(r.values()) for r in
                   res["cluster_by_class_k_classes"].values()) == len(df)
        assert len(res["pca_explained_variance"]) == 2
        assert list(assign.columns) == [
            "true_class", "cluster_k_classes", "cluster_k_star", "pca1", "pca2"]
        assert assign["cluster_k_star"].nunique() == res["k_star"]

    def test_deterministic(self):
        """Same seed, same clusters. Inertia can differ in the last ulp
        between runs (threaded BLAS summation order), so it is compared
        approximately; everything else exactly."""
        df = _frame()
        a, la = run_clustering(df, k_values=(2, 3), with_umap=False)
        b, lb = run_clustering(df, k_values=(2, 3), with_umap=False)
        assert a["sweep"].pop("inertias") == pytest.approx(
            b["sweep"].pop("inertias"), rel=1e-12)
        assert a == b
        pd.testing.assert_frame_equal(la, lb, check_exact=False, rtol=1e-12)

    def test_recovers_planted_group(self):
        res, _ = run_clustering(_frame(n=90), k_values=(2, 3), with_umap=False)
        assert res["external_agreement_k_star"]["adjusted_rand_index"] > 0.3
