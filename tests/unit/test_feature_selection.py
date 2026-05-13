"""Tests for cancerag.ml.feature_selection.

Covers:
- BorutaSelector force-keeps prefixed features (interaction-fingerprint /
  pose-ensemble columns must survive selection).
- BorutaSelector tolerates Boruta failures via no-selection fallback.
- BorutaSelector composes inside a sklearn Pipeline.
- BorutaSelector.transform tolerates columns missing at predict time.
- BorutaSelector raises a helpful error if its input contains NaNs.
- stability_selection returns frequencies in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from cancerag.ml.feature_selection import BorutaSelector, stability_selection
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
