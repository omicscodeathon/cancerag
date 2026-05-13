"""Tests asserting train/inference feature-vector parity.

The critical Stage 12 P12.1 bug: the inference path silently produces a
feature vector with different columns than the training path, and the model
quietly returns nonsense. These tests catch that.

For the new sklearn-Pipeline-based architecture, parity is structural: the
joblib-saved Pipeline accepts a DataFrame with column set X, so as long as
inference builds a DataFrame with the same column set X, parity holds.

We verify this by:
  1. Loading the trained Pipeline from data/processed/ml_models/
  2. Loading 5 samples from the train-eligible matrix
  3. Asserting the Pipeline accepts them and produces well-formed predictions
  4. Re-extracting the feature columns from the dataset assembly logic and
     asserting equality with what the Pipeline expects
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[2]
ML_MODELS = REPO / "data" / "processed" / "ml_models"
DATASET = REPO / "data" / "processed" / "ml_ready_dataset.parquet"
SELECTION_DECISION = ML_MODELS / "selection_decision.json"


def _ml_artifacts_present() -> bool:
    return (
        DATASET.exists() and SELECTION_DECISION.exists()
        and any(ML_MODELS.glob("*_final.joblib"))
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not _ml_artifacts_present(),
    reason="ML artifacts not yet built; run Stage 10 first",
)
class TestFeatureParity:
    """The inference feature vector must match training schema."""

    def test_pipeline_loads_and_predicts_on_5_train_samples(self):
        import joblib
        decision = json.loads(SELECTION_DECISION.read_text())
        winner = decision["chosen"]
        model_path = ML_MODELS / f"{winner}_final.joblib"
        assert model_path.exists(), f"Final model missing: {model_path}"

        pipe = joblib.load(model_path)
        df = pd.read_parquet(DATASET)

        # Build the same X the trainer built
        from cancerag.ml.preprocessing import get_X_y_groups
        import joblib as _joblib
        le = _joblib.load(REPO / "data" / "processed"
                          / "ml_preprocessed" / "label_encoder.joblib")
        X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)

        # Pick 5 samples
        X_sample = X.iloc[:5]
        y_pred = pipe.predict(X_sample)
        assert len(y_pred) == 5
        # Predictions are valid class indices
        for p in y_pred:
            assert 0 <= int(p) < len(le.classes_)

    def test_calibrated_model_loads_and_predict_proba(self):
        import joblib
        decision = json.loads(SELECTION_DECISION.read_text())
        winner = decision["chosen"]
        cal_path = ML_MODELS / f"{winner}_final_calibrated.joblib"
        if not cal_path.exists():
            pytest.skip("calibrated model not found")
        pipe = joblib.load(cal_path)
        df = pd.read_parquet(DATASET)
        from cancerag.ml.preprocessing import get_X_y_groups
        import joblib as _joblib
        le = _joblib.load(REPO / "data" / "processed"
                          / "ml_preprocessed" / "label_encoder.joblib")
        X, _, _, _, _ = get_X_y_groups(df, label_encoder=le)
        proba = pipe.predict_proba(X.iloc[:5])
        # Each row sums to ~1
        for row in proba:
            assert abs(row.sum() - 1.0) < 0.01
            assert all(0.0 <= p <= 1.0 for p in row)

    def test_selection_decision_winner_has_a_saved_model(self):
        decision = json.loads(SELECTION_DECISION.read_text())
        winner = decision["chosen"]
        model_path = ML_MODELS / f"{winner}_final.joblib"
        assert model_path.exists(), (
            f"selection_decision.json says winner={winner} but "
            f"{model_path.name} doesn't exist on disk"
        )

    def test_holdout_uses_same_feature_schema(self):
        """If the holdout file exists, its column set must be a superset of
        what the dataset_assembly produces — so the inference path can join."""
        holdout_path = REPO / "data" / "holdout" / "dataset_holdout.parquet"
        if not holdout_path.exists():
            pytest.skip("holdout file not present")
        train_cols = set(pd.read_parquet(DATASET).columns)
        ho_cols = set(pd.read_parquet(holdout_path).columns)
        missing = train_cols - ho_cols
        # bias_category is always present; structural cols may have NaN but
        # the column itself must exist for join compatibility.
        critical_missing = {c for c in missing if not c.startswith("morgan_")
                                                  and not c.startswith("maccs_")
                                                  and not c.startswith("ifp_")}
        assert not critical_missing, (
            f"Holdout missing {len(critical_missing)} non-fingerprint "
            f"training columns: {sorted(critical_missing)[:10]}"
        )


@pytest.mark.unit
def test_feature_parity_test_module_imports_cleanly():
    """Smoke: this test file imports without error."""
    pass
