"""Tests for cancerag.ml.hyperopt — threshold optimisation and tuning metadata.

The per-class thresholds are multipliers on predicted probabilities, so they
are only meaningful for the model whose probabilities they were learned on.
These tests pin that the out-of-fold model used to learn them and the holdout
model they are applied to are the same estimator, configured the same way.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cancerag.ml.hyperopt import (
    ThresholdModel,
    final_estimator,
    resolve_threshold_model,
    tuning_meta,
)

lgb = pytest.importorskip("lightgbm")

BEST_PARAMS = {
    "colsample_bytree": 0.7, "learning_rate": 0.1, "max_depth": 4,
    "min_child_samples": 5, "n_estimators": 20, "num_leaves": 15,
    "reg_alpha": 0.01, "reg_lambda": 0.01, "subsample": 0.8,
}


def _toy(n=90, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = np.arange(n) % n_classes
    X[:, 0] += y  # make it learnable
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(6)]), y


def _write_tuned_artefacts(output_dir, *, n_classes=3):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "optuna_tuning_meta.json").write_text(json.dumps({
        "model": "lightgbm", "bake_off_winner": "random_forest",
        "best_params": BEST_PARAMS,
    }))
    X, y = _toy(n_classes=n_classes)
    pipe = Pipeline([("scale", StandardScaler()), ("model", lgb.LGBMClassifier(
        objective="multiclass", num_class=n_classes, verbose=-1,
        **BEST_PARAMS))])
    cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3).fit(X, y)
    joblib.dump(cal, output_dir / "lightgbm_tuned_calibrated.joblib")


def _write_winner_artefact(models_dir, winner="random_forest"):
    models_dir.mkdir(parents=True, exist_ok=True)
    X, y = _toy()
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", RandomForestClassifier(n_estimators=5))])
    cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3).fit(X, y)
    joblib.dump(cal, models_dir / f"{winner}_final_calibrated.joblib")


@pytest.mark.unit
class TestThresholdModelResolution:
    def test_tuned_lightgbm_is_used_when_present(self, tmp_path):
        adv = tmp_path / "advanced"
        _write_tuned_artefacts(adv)
        spec = resolve_threshold_model(
            winner="random_forest", output_dir=adv, models_dir=tmp_path,
        )
        assert spec.name == "lightgbm_tuned"
        assert spec.holdout_model_path == adv / "lightgbm_tuned_calibrated.joblib"

    def test_oof_and_holdout_models_are_the_same_estimator(self, tmp_path):
        """The regression this module had: OOF probabilities came from the
        bake-off winner (random forest) while the thresholds were applied to
        the tuned LightGBM at holdout."""
        adv = tmp_path / "advanced"
        _write_tuned_artefacts(adv)
        spec = resolve_threshold_model(
            winner="random_forest", output_dir=adv, models_dir=tmp_path,
        )
        oof_model = spec.make_oof_model(n_classes=3, seed=42)
        holdout_model = spec.load_holdout_model()

        oof_est, ho_est = final_estimator(oof_model), final_estimator(holdout_model)
        assert type(oof_est) is type(ho_est) is lgb.LGBMClassifier
        oof_params, ho_params = oof_est.get_params(), ho_est.get_params()
        for k, v in BEST_PARAMS.items():
            assert oof_params[k] == v == ho_params[k], k

    def test_oof_model_is_calibrated_like_the_holdout_model(self, tmp_path):
        adv = tmp_path / "advanced"
        _write_tuned_artefacts(adv)
        spec = resolve_threshold_model(
            winner="random_forest", output_dir=adv, models_dir=tmp_path,
        )
        oof_model = spec.make_oof_model(n_classes=3, seed=42)
        assert isinstance(oof_model, CalibratedClassifierCV)
        assert oof_model.method == "isotonic"
        assert oof_model.cv == 3

    def test_oof_model_fits_and_predicts(self, tmp_path):
        adv = tmp_path / "advanced"
        _write_tuned_artefacts(adv)
        spec = resolve_threshold_model(
            winner="random_forest", output_dir=adv, models_dir=tmp_path,
        )
        X, y = _toy()
        proba = spec.make_oof_model(n_classes=3, seed=0).fit(X, y).predict_proba(X)
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_falls_back_to_winner_without_tuned_model(self, tmp_path):
        adv = tmp_path / "advanced"
        adv.mkdir()
        _write_winner_artefact(tmp_path)
        spec = resolve_threshold_model(
            winner="random_forest", output_dir=adv, models_dir=tmp_path,
        )
        assert spec.name == "random_forest"
        oof_est = final_estimator(spec.make_oof_model(n_classes=3, seed=42))
        ho_est = final_estimator(spec.load_holdout_model())
        assert type(oof_est) is type(ho_est) is RandomForestClassifier

    def test_tuned_meta_without_tuned_model_falls_back(self, tmp_path):
        """best_params alone are not enough: thresholds are applied to a saved
        model, so without the saved tuned model the winner is used."""
        adv = tmp_path / "advanced"
        adv.mkdir()
        (adv / "optuna_tuning_meta.json").write_text(
            json.dumps({"model": "lightgbm", "best_params": BEST_PARAMS}))
        _write_winner_artefact(tmp_path)
        spec = resolve_threshold_model(
            winner="random_forest", output_dir=adv, models_dir=tmp_path,
        )
        assert spec.name == "random_forest"

    def test_mismatch_is_refused(self, tmp_path):
        adv = tmp_path / "advanced"
        _write_tuned_artefacts(adv)
        spec = resolve_threshold_model(
            winner="random_forest", output_dir=adv, models_dir=tmp_path,
        )
        wrong = ThresholdModel(
            name="random_forest", factory_name="random_forest", params={},
            holdout_model_path=spec.holdout_model_path,
        )
        with pytest.raises(ValueError, match="same model"):
            wrong.check_matches_holdout(n_classes=3)


@pytest.mark.unit
class TestTuningMeta:
    def test_records_the_model_actually_tuned(self):
        meta = tuning_meta(
            winner="random_forest", n_trials=50, best_score=0.5,
            best_params=BEST_PARAMS, holdout_metrics={},
        )
        assert meta["model"] == "lightgbm"
        assert meta["bake_off_winner"] == "random_forest"
        assert meta["best_params"] == BEST_PARAMS
