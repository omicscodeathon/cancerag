"""Tests for inference_app.src.predictor — model-name resolution,
applicability-domain check, confidence label, and health payload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _import_predictor():
    """Lazy-import the predictor module so we don't pull Gradio into
    every test session at collection time."""
    import importlib
    import sys

    inference_src = Path("inference_app/src")
    if str(inference_src) not in sys.path:
        sys.path.insert(0, str(inference_src))
    return importlib.import_module("predictor")


PRED = _import_predictor()
ADResult = PRED.ADResult
ApplicabilityChecker = PRED.ApplicabilityChecker
ad_to_dict = PRED.ad_to_dict
confidence_label = PRED.confidence_label
nearest_neighbor_tanimoto = PRED.nearest_neighbor_tanimoto
health_payload = PRED.health_payload
write_health_snapshot = PRED.write_health_snapshot


def _import_resolve_model_name():
    import importlib
    import sys

    inference_src = Path("inference_app/src")
    if str(inference_src) not in sys.path:
        sys.path.insert(0, str(inference_src))
    mod = importlib.import_module("predictor")
    return mod.resolve_model_name


@pytest.mark.unit
class TestResolveModelName:
    def test_falls_back_when_no_decision_file(self, tmp_path: Path):
        resolve = _import_resolve_model_name()
        # No results/model_selection_decision.json under tmp_path
        assert resolve(base_path=tmp_path) == "random_forest"

    def test_uses_decision_file_when_present(self, tmp_path: Path):
        resolve = _import_resolve_model_name()
        decision_path = tmp_path / "results" / "model_selection_decision.json"
        decision_path.parent.mkdir(parents=True)
        decision_path.write_text(
            json.dumps({"chosen": "xgboost", "rule": "max_outer_cv_macro_f1_mean"})
        )
        assert resolve(base_path=tmp_path) == "xgboost"

    def test_corrupt_file_falls_back(self, tmp_path: Path):
        resolve = _import_resolve_model_name()
        decision_path = tmp_path / "results" / "model_selection_decision.json"
        decision_path.parent.mkdir(parents=True)
        decision_path.write_text("not_json{")
        assert resolve(base_path=tmp_path, fallback="lightgbm") == "lightgbm"


@pytest.mark.unit
class TestApplicabilityChecker:
    def test_in_domain_for_training_molecule(self):
        train = ["CCO", "CCN", "c1ccccc1"]
        checker = ApplicabilityChecker(train, threshold=0.4)
        result = checker.check("CCO")
        assert result.in_domain is True
        assert result.nearest_neighbor_tanimoto == pytest.approx(1.0)

    def test_out_of_domain_for_distant_molecule(self):
        train = ["CCO"]  # ethanol — tiny aliphatic
        checker = ApplicabilityChecker(train, threshold=0.4)
        # A large polycyclic structure should be far from ethanol on Morgan
        result = checker.check("c1ccc2c(c1)ccc1c2cccc1")
        assert result.in_domain is False

    def test_invalid_smiles_is_out_of_domain(self):
        checker = ApplicabilityChecker(["CCO"], threshold=0.4)
        result = checker.check("garbage$$$")
        assert result.in_domain is False
        assert result.nearest_neighbor_tanimoto == 0.0

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            ApplicabilityChecker(["CCO"], threshold=2.0)

    def test_empty_training_raises(self):
        with pytest.raises(ValueError):
            ApplicabilityChecker([], threshold=0.4)

    def test_batch_check(self):
        checker = ApplicabilityChecker(["CCO", "CCN"], threshold=0.3)
        results = checker.batch_check(["CCO", "garbage", "CCO"])
        assert len(results) == 3
        assert results[0].in_domain
        assert not results[1].in_domain

    def test_ad_to_dict_round_trip(self):
        ad = ADResult(in_domain=True, nearest_neighbor_tanimoto=0.85, threshold=0.4)
        d = ad_to_dict(ad)
        assert d == {"in_domain": True, "nearest_neighbor_tanimoto": 0.85, "threshold": 0.4}

    def test_stateless_helper(self):
        sim = nearest_neighbor_tanimoto("CCO", ["CCO"])
        assert sim == pytest.approx(1.0)


@pytest.mark.unit
class TestConfidenceLabel:
    def test_out_of_domain_overrides_high_proba(self):
        ad = ADResult(in_domain=False, nearest_neighbor_tanimoto=0.1, threshold=0.4)
        assert confidence_label(ad, max_class_proba=0.95) == "out_of_domain"

    def test_buckets(self):
        ad = ADResult(in_domain=True, nearest_neighbor_tanimoto=0.9, threshold=0.4)
        assert confidence_label(ad, 0.8) == "high"
        assert confidence_label(ad, 0.55) == "medium"
        assert confidence_label(ad, 0.4) == "low"


@pytest.mark.unit
class TestHealthPayload:
    def test_includes_required_fields(self, tmp_path: Path):
        model_path = tmp_path / "rf.pkl"
        model_path.write_bytes(b"\x80\x04N.")  # tiny pickle bytes
        payload = health_payload(
            model_path=model_path,
            model_name="random_forest",
            dataset_version="v1.2.3",
        )
        assert payload["status"] == "ok"
        assert payload["model_name"] == "random_forest"
        assert payload["dataset_version"] == "v1.2.3"
        assert payload["model_sha256"] is not None
        assert "started_at_utc" in payload
        assert "python_version" in payload

    def test_missing_model_marked(self, tmp_path: Path):
        payload = health_payload(
            model_path=tmp_path / "ghost.pkl",
            model_name="ghost",
        )
        assert payload["status"] == "model_missing"
        assert payload["model_sha256"] is None

    def test_write_snapshot(self, tmp_path: Path):
        model_path = tmp_path / "rf.pkl"
        model_path.write_bytes(b"x")
        payload = health_payload(model_path=model_path, model_name="rf")
        snap = write_health_snapshot(payload, tmp_path / "snap.json")
        assert snap.exists()
        loaded = json.loads(snap.read_text())
        assert loaded["model_name"] == "rf"


# ----------------------------- ModernBiasPredictor tests --------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_MODELS = REPO_ROOT / "data" / "processed" / "ml_models"
DATASET = REPO_ROOT / "data" / "processed" / "ml_ready_dataset.parquet"
SELECTION_DECISION = ML_MODELS / "selection_decision.json"
LABEL_ENCODER = (
    REPO_ROOT / "data" / "processed" / "ml_preprocessed" / "label_encoder.joblib"
)


def _modern_artifacts_present() -> bool:
    return (
        DATASET.exists()
        and SELECTION_DECISION.exists()
        and LABEL_ENCODER.exists()
        and any(ML_MODELS.glob("*_final.joblib"))
    )


@pytest.mark.unit
@pytest.mark.skipif(
    not _modern_artifacts_present(),
    reason="Modern ML artifacts not yet built; run Stage 10 first",
)
class TestModernBiasPredictor:
    """Tests for the new sklearn-Pipeline-based predictor."""

    def _make(self):
        return PRED.ModernBiasPredictor(repo_root=REPO_ROOT)

    def test_construction_loads_cleanly(self):
        p = self._make()
        assert p.model is not None
        assert p.model_path is not None and p.model_path.exists()
        assert p.model_sha256 and len(p.model_sha256) == 64
        assert p.feature_columns, "feature columns should be populated"
        assert p.label_encoder is not None

    def test_predict_on_known_training_pair_is_deterministic(self):
        import pandas as pd
        df = pd.read_parquet(DATASET)
        sample = df.iloc[0]
        smiles = str(sample["canonical_smiles"])
        receptor = str(sample["receptor_uniprot"])
        p = self._make()
        r1 = p.predict(smiles, receptor, log_audit=False)
        r2 = p.predict(smiles, receptor, log_audit=False)
        assert r1["predicted_class"] == r2["predicted_class"]
        # Probabilities deterministic
        for k, v in r1["probabilities"].items():
            assert abs(v - r2["probabilities"][k]) < 1e-9
        assert 0.999 <= sum(r1["probabilities"].values()) <= 1.001
        assert r1["model_name"] is not None

    def test_audit_log_appended(self, tmp_path: Path, monkeypatch):
        import pandas as pd
        p = self._make()
        # Redirect audit log to a tmp dir by pointing repo_root at tmp_path
        # but keep model resources loaded. We monkeypatch the path attribute
        # by overriding repo_root used inside _append_audit.
        monkeypatch.setattr(p, "repo_root", tmp_path)
        df = pd.read_parquet(DATASET)
        smiles = str(df.iloc[0]["canonical_smiles"])
        receptor = str(df.iloc[0]["receptor_uniprot"])
        audit_path = tmp_path / "data" / "processed" / "inference_audit.jsonl"
        assert not audit_path.exists()
        p.predict(smiles, receptor, log_audit=True)
        assert audit_path.exists()
        lines = audit_path.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["receptor_uniprot"] == receptor
        assert "smiles_sha256" in entry and len(entry["smiles_sha256"]) == 64
        assert "predicted_class" in entry
        assert "probabilities" in entry
        assert "top_shap" in entry
        assert "model_sha256" in entry

    def test_top_shap_returns_at_most_5_tuples(self):
        import pandas as pd
        p = self._make()
        if p._explainer is None:
            pytest.skip("SHAP explainer not available in this environment")
        df = pd.read_parquet(DATASET)
        smiles = str(df.iloc[0]["canonical_smiles"])
        receptor = str(df.iloc[0]["receptor_uniprot"])
        result = p.predict(smiles, receptor, log_audit=False)
        top = result["top_shap"]
        assert isinstance(top, list)
        assert len(top) <= 5
        # 5 expected when explainer is available and feature count >= 5
        assert len(top) == 5
        for item in top:
            assert isinstance(item, tuple) and len(item) == 2
            name, val = item
            assert isinstance(name, str)
            assert isinstance(val, float)
