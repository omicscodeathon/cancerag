"""Tests for cancerag.ml.generate_final_report — model card writer plus
SHAP-stability and permutation-importance helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from cancerag.ml.generate_final_report import (
    cross_validate_top_features,
    permutation_importance_df,
    persist_shap_values,
    render_model_card,
    selection_frequency,
    write_model_card,
)


@pytest.mark.unit
class TestModelCard:
    def test_renders_with_required_sections(self):
        card = render_model_card(
            model_name="random_forest",
            git_sha="abc1234",
            trained_at_utc="2026-04-28T20:00:00Z",
            lib_versions={"sklearn": "1.7.2", "rdkit": "2025.3.6"},
            dataset_sha256="deadbeef",
            n_train=403,
            n_test=101,
            split_strategy="scaffold",
            hyperparameters={"n_estimators": 200},
            val_macro_f1="0.62 [0.58, 0.66]",
            test_macro_f1="0.60 [0.55, 0.65]",
            test_balanced_acc="0.61 [0.56, 0.66]",
            limitations=["small n", "single conformer per ligand"],
        )
        for section in (
            "# Model Card",
            "## Provenance",
            "## Training Data",
            "## Hyperparameters",
            "## Performance",
            "## Intended Use",
            "## Limitations",
        ):
            assert section in card
        assert "applicability domain" in card

    def test_write_to_disk(self, tmp_path: Path):
        path = write_model_card(
            tmp_path / "card.md",
            model_name="x",
            git_sha="x",
            trained_at_utc="2026-04-28T00:00:00Z",
            lib_versions={"a": "1"},
            dataset_sha256="x",
            n_train=1,
            n_test=1,
            split_strategy="x",
            hyperparameters={},
            val_macro_f1="x",
            test_macro_f1="x",
            test_balanced_acc="x",
            limitations=["x"],
        )
        assert path.exists()
        assert "Model Card" in path.read_text()


@pytest.mark.unit
class TestSelectionFrequency:
    def test_basic_count(self):
        long = pd.DataFrame(
            {
                "fold": [0, 0, 1, 1, 2],
                "feature": ["a", "b", "a", "c", "a"],
            }
        )
        freq = selection_frequency(long, n_folds=3)
        assert freq["a"] == pytest.approx(1.0)
        assert freq["b"] == pytest.approx(1 / 3)
        assert freq["c"] == pytest.approx(1 / 3)
        assert freq.index[0] == "a"

    def test_empty(self):
        assert selection_frequency(pd.DataFrame(), n_folds=3).empty


@pytest.mark.unit
class TestCrossValidateTopFeatures:
    def test_intersection_of_shap_stable_and_perm_top(self):
        shap_freq = pd.Series({"a": 1.0, "b": 0.9, "c": 0.5, "d": 0.85})
        perm = pd.DataFrame(
            {
                "feature": ["a", "x", "y", "d", "b"],
                "perm_importance_mean": [0.5, 0.4, 0.3, 0.2, 0.1],
                "perm_importance_std": [0.0] * 5,
            }
        )
        out = cross_validate_top_features(shap_freq, perm, top_k=5, min_freq=0.8)
        assert set(out) == {"a", "b", "d"}


@pytest.mark.unit
class TestPermutationImportance:
    def test_returns_sorted_dataframe(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "d"])
        y = ((X["a"] + X["b"]) > 0).astype(int).values
        model = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        out = permutation_importance_df(model, X, y, n_repeats=10, seed=0)
        assert list(out.columns) == [
            "feature", "perm_importance_mean", "perm_importance_std"
        ]
        assert (
            out["perm_importance_mean"].values
            == np.sort(out["perm_importance_mean"].values)[::-1]
        ).all()
        top2 = set(out.head(2)["feature"].tolist())
        assert top2 == {"a", "b"}


@pytest.mark.unit
class TestPersistShapValues:
    def test_single_array_round_trip(self, tmp_path: Path):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.1, 0.2, 0.3]})
        sv = np.array([[0.1, -0.2], [0.3, 0.0], [0.0, -0.4]])
        y = np.array([0, 1, 0])
        out = persist_shap_values(sv, X, y, tmp_path / "shap.npz")
        loaded = np.load(out, allow_pickle=True)
        assert "shap" in loaded
        assert loaded["shap"].shape == (3, 2)
        assert list(loaded["feature_names"]) == ["a", "b"]
        assert (loaded["y_true"] == y).all()

    def test_multiclass_list_round_trip(self, tmp_path: Path):
        X = pd.DataFrame({"a": [1.0, 2.0]})
        sv = [
            np.array([[0.1], [-0.2]]),
            np.array([[0.0], [0.3]]),
            np.array([[-0.1], [0.4]]),
        ]
        out = persist_shap_values(sv, X, np.array([0, 2]), tmp_path / "shap.npz")
        loaded = np.load(out, allow_pickle=True)
        for i in range(3):
            assert f"shap_class_{i}" in loaded
        assert loaded["shap_class_0"].shape == (2, 1)
