"""Tests for cancerag.ml.model_training — baselines and locked model selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cancerag.ml.model_training import (
    BASELINES,
    SELECTION_RULE,
    majority_class_baseline,
    read_selection_decision,
    select_final_model,
    smiles_only_rf_baseline,
    stratified_baseline,
    write_selection_decision,
)


@pytest.mark.unit
class TestBaselines:
    def test_majority_class(self):
        m = majority_class_baseline()
        m.fit(np.zeros((10, 1)), [0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        preds = m.predict(np.zeros((3, 1)))
        assert all(p == 0 for p in preds)

    def test_stratified(self):
        m = stratified_baseline(random_state=0)
        m.fit(np.zeros((20, 1)), [0] * 10 + [1] * 10)
        preds = m.predict(np.zeros((50, 1)))
        assert set(preds.tolist()).issubset({0, 1})

    def test_smiles_only_rf_runs(self):
        df = pd.DataFrame(
            {"canonical_smiles": ["CCO", "c1ccccc1", "CCN", "c1ccncc1"]}
        )
        y = np.array([0, 1, 0, 1])
        pipe = smiles_only_rf_baseline(n_bits=128, n_estimators=10)
        pipe.fit(df, y)
        preds = pipe.predict(df)
        assert preds.shape == (4,)

    def test_registry_intact(self):
        for name in ("majority_class", "stratified", "smiles_only_morgan_rf"):
            assert name in BASELINES


@pytest.mark.unit
class TestModelSelection:
    def test_picks_highest_mean(self):
        results = pd.DataFrame(
            {
                "model": ["rf"] * 5 + ["xgb"] * 5,
                "macro_f1": [0.55, 0.6, 0.62, 0.58, 0.6,
                              0.7, 0.71, 0.69, 0.72, 0.7],
            }
        )
        decision = select_final_model(results)
        assert decision["chosen"] == "xgb"
        assert decision["rule"] == SELECTION_RULE
        assert decision["summary"][0]["model"] == "xgb"

    def test_std_tiebreak(self):
        results = pd.DataFrame(
            {
                "model": ["rf"] * 4 + ["xgb"] * 4,
                "macro_f1": [0.7, 0.7, 0.7, 0.7,
                              0.5, 0.9, 0.5, 0.9],
            }
        )
        decision = select_final_model(results)
        assert decision["chosen"] == "rf"

    def test_persist_round_trip(self, tmp_path: Path):
        results = pd.DataFrame(
            {"model": ["rf", "rf"], "macro_f1": [0.5, 0.6]}
        )
        decision = select_final_model(results)
        out_path = write_selection_decision(decision, tmp_path / "sel.json")
        loaded = read_selection_decision(out_path)
        assert loaded["chosen"] == "rf"
        assert loaded["rule"] == SELECTION_RULE

    def test_missing_columns_raises(self):
        with pytest.raises(KeyError):
            select_final_model(pd.DataFrame({"x": [1]}))
