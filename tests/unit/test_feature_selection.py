"""Tests for cancerag.ml.feature_selection.

Covers:
- BorutaSelector force-keeps prefixed features (interaction-fingerprint /
  pose-ensemble columns must survive selection).
- BorutaSelector tolerates Boruta failures via no-selection fallback.
- BorutaSelector composes inside a sklearn Pipeline.
- BorutaSelector.transform tolerates columns missing at predict time.
- BorutaSelector raises a helpful error if its input contains NaNs.
- stability_selection returns frequencies in [0, 1].
- selector-ablation artefacts are not older than the matrix they describe.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from cancerag.ml.feature_selection import (
    SELECTOR_ABLATION_ARTEFACTS,
    BorutaSelector,
    stability_selection,
    stale_selector_ablation_artefacts,
)
from cancerag.ml.preprocessing import CorrelationFilter


def _toy_data(n_samples=80, n_features=20, n_informative=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    # First n_informative features drive y
    coef = np.zeros(n_features)
    coef[:n_informative] = rng.normal(size=n_informative)
    logits = X @ coef
    y = (logits > 0).astype(int)
    cols = [f"f{i}" for i in range(n_features - 2)] + [
        "ifp_DRD2_3.32_HBDonor",
        "vina_pose_diversity_rmsd",
    ]
    df = pd.DataFrame(X, columns=cols)
    return df, y


@pytest.mark.unit
class TestBorutaSelectorBasics:
    def test_force_keep_survives(self):
        df, y = _toy_data()
        sel = BorutaSelector(force_keep_prefixes=("ifp_", "vina_pose_"),
                             max_iter=10).fit(df, y)
        assert "ifp_DRD2_3.32_HBDonor" in sel.selected_
        assert "vina_pose_diversity_rmsd" in sel.selected_

    def test_transform_returns_only_selected_columns(self):
        df, y = _toy_data()
        sel = BorutaSelector(max_iter=10).fit(df, y)
        out = sel.transform(df)
        assert set(out.columns) == set(sel.selected_)
        assert len(out) == len(df)

    def test_transform_tolerates_missing_columns_at_predict(self):
        df, y = _toy_data()
        sel = BorutaSelector(force_keep_prefixes=("ifp_",), max_iter=10).fit(df, y)
        # Drop one of the selected columns at predict time
        if sel.selected_:
            df2 = df.drop(columns=[sel.selected_[0]])
            out = sel.transform(df2)
            assert sel.selected_[0] not in out.columns

    def test_rejects_nan_inputs(self):
        df, y = _toy_data()
        df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            BorutaSelector(max_iter=10).fit(df, y)

    def test_get_feature_names_out(self):
        df, y = _toy_data()
        sel = BorutaSelector(max_iter=10).fit(df, y)
        names = sel.get_feature_names_out()
        assert isinstance(names, np.ndarray)
        assert set(names) == set(sel.selected_)


@pytest.mark.unit
class TestBorutaInPipeline:
    def test_pipeline_fit_predict(self):
        df, y = _toy_data(n_samples=80)
        pipe = Pipeline([
            ("decorr", CorrelationFilter(threshold=0.99)),
            ("select", BorutaSelector(max_iter=10)),
            ("model", RandomForestClassifier(
                n_estimators=10, random_state=0, n_jobs=1)),
        ])
        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (80,)


@pytest.mark.unit
class TestStabilitySelection:
    def test_frequencies_in_unit_interval(self):
        df, y = _toy_data(n_samples=60)

        def factory():
            return BorutaSelector(max_iter=5,
                                   force_keep_prefixes=("ifp_",),
                                   random_state=0)

        freq = stability_selection(df, y, factory, n_boot=3, sample_frac=0.7, seed=0)
        assert isinstance(freq, pd.Series)
        assert (freq >= 0).all() and (freq <= 1).all()
        # Force-kept feature is selected in every bootstrap -> frequency == 1
        assert freq["ifp_DRD2_3.32_HBDonor"] == pytest.approx(1.0)

    def test_handles_empty_input(self):
        df = pd.DataFrame({"a": [], "b": []})
        freq = stability_selection(df, [], lambda: BorutaSelector(max_iter=1),
                                   n_boot=2)
        assert (freq == 0.0).all()


@pytest.mark.unit
class TestSelectorAblationFreshness:
    """The ablation artefact must not outlive the matrix it describes.

    HAL-3613: the committed artefact was produced on 2026-05-03, against a
    feature matrix that still carried the leaked per-receptor QC columns, and
    was still being cited for Figure 2 after the matrix was rebuilt in August.
    Nothing on disk said so.
    """

    @staticmethod
    def _write(path, mtime):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("selector,model,mean\n")
        os.utime(path, (mtime, mtime))

    def test_artefact_older_than_dataset_is_stale(self, tmp_path):
        dataset = tmp_path / "ml_ready_dataset.parquet"
        out = tmp_path / "selector_ablation"
        self._write(dataset, 2_000.0)
        for name in SELECTOR_ABLATION_ARTEFACTS:
            self._write(out / name, 1_000.0)

        stale = stale_selector_ablation_artefacts(
            dataset_path=dataset, output_dir=out,
        )
        assert sorted(stale) == sorted(SELECTOR_ABLATION_ARTEFACTS)

    def test_artefact_newer_than_dataset_is_fresh(self, tmp_path):
        dataset = tmp_path / "ml_ready_dataset.parquet"
        out = tmp_path / "selector_ablation"
        self._write(dataset, 1_000.0)
        for name in SELECTOR_ABLATION_ARTEFACTS:
            self._write(out / name, 2_000.0)

        assert stale_selector_ablation_artefacts(
            dataset_path=dataset, output_dir=out,
        ) == []

    def test_reports_only_the_stale_ones(self, tmp_path):
        dataset = tmp_path / "ml_ready_dataset.parquet"
        out = tmp_path / "selector_ablation"
        self._write(dataset, 2_000.0)
        self._write(out / "selector_ablation_long.csv", 1_000.0)
        self._write(out / "selector_ablation_matrix.csv", 3_000.0)
        self._write(out / "selector_ablation_summary.csv", 3_000.0)

        assert stale_selector_ablation_artefacts(
            dataset_path=dataset, output_dir=out,
        ) == ["selector_ablation_long.csv"]

    def test_missing_inputs_are_not_reported_as_stale(self, tmp_path):
        # No dataset: nothing to compare against.
        assert stale_selector_ablation_artefacts(
            dataset_path=tmp_path / "absent.parquet",
            output_dir=tmp_path / "selector_ablation",
        ) == []
        # Dataset but no artefacts: absence is not staleness.
        dataset = tmp_path / "ml_ready_dataset.parquet"
        self._write(dataset, 2_000.0)
        assert stale_selector_ablation_artefacts(
            dataset_path=dataset, output_dir=tmp_path / "selector_ablation",
        ) == []

    def test_repository_artefacts_are_not_stale(self):
        """Guards the real artefact when the gitignored data is present."""
        dataset = Path("data/processed/ml_ready_dataset.parquet")
        out = Path("data/processed/ml_models/selector_ablation")
        if not dataset.exists() or not any(
            (out / n).exists() for n in SELECTOR_ABLATION_ARTEFACTS
        ):
            pytest.skip("gitignored ML data not present in this checkout")
        stale = stale_selector_ablation_artefacts(
            dataset_path=dataset, output_dir=out,
        )
        assert stale == [], (
            f"selector-ablation artefacts predate {dataset}: {stale}. "
            "Re-run cancerag.ml.feature_selection.run_selector_ablation."
        )
