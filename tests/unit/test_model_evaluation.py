"""Tests for cancerag.ml.model_evaluation — bootstrap CI and reporting suite."""

from __future__ import annotations

import numpy as np
import pytest

from cancerag.ml.model_evaluation import (
    bootstrap_ci,
    per_receptor_metrics,
    report_metrics,
)


@pytest.mark.unit
class TestBootstrapCI:
    def test_three_ordered_values(self):
        from sklearn.metrics import f1_score
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, size=100)
        yhat = y.copy()
        yhat[::5] = (y[::5] + 1) % 3
        lo, mid, hi = bootstrap_ci(
            y, yhat, f1_score, n_boot=200, average="macro", seed=0
        )
        assert lo <= mid <= hi

    def test_empty_input_returns_nan(self):
        from sklearn.metrics import f1_score
        out = bootstrap_ci(
            np.array([]), np.array([]), f1_score, n_boot=10, average="macro"
        )
        assert all(x != x for x in out)


@pytest.mark.unit
class TestReportMetrics:
    def test_macro_f1_leads(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 5, size=200)
        yhat = y.copy()
        yhat[::3] = (y[::3] + 1) % 5
        out = report_metrics(y, yhat, bootstrap_n=100, seed=0)
        assert "macro_f1" in out
        assert "accuracy" not in out
        assert "balanced_accuracy" in out
        assert "per_class_f1" in out
        assert "confusion_matrix" in out
        assert out["n_test"] == 200


@pytest.mark.unit
class TestPerReceptorMetrics:
    def test_filters_small_groups(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        yhat = y.copy()
        recs = np.array(["A"] * 8 + ["B"] * 2)
        out = per_receptor_metrics(y, yhat, recs, min_samples=5)
        assert "A" in out["receptor"].values
        assert "B" not in out["receptor"].values

    def test_perfect_predictions_score_one(self):
        y = np.array([0, 1, 0, 1, 0])
        out = per_receptor_metrics(y, y, np.array(["A"] * 5), min_samples=3)
        assert out.iloc[0]["macro_f1"] == 1.0
