"""Smoke test for the production inference path (ModernBiasPredictor).

SABER predicts biased agonism for a (ligand, receptor) pair, so the predictor
requires a receptor UniProt accession. This test confirms the current 4-class
artifacts load and a single-pair prediction returns a well-formed result over
exactly the four bias classes (no legacy 5-class "Agonist" label).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.predictor import ModernBiasPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_CLASSES = {"ERK", "G protein", "G protein selectivity", "β Arrestin"}


def test_modern_predictor_four_class():
    predictor = ModernBiasPredictor(repo_root=REPO_ROOT)

    # The label encoder must carry exactly the four bias classes.
    assert predictor.label_encoder is not None
    classes = set(map(str, predictor.label_encoder.classes_))
    assert classes == EXPECTED_CLASSES, f"unexpected classes: {classes}"
    assert "Agonist" not in classes  # the defunct v1 label must not resurface

    # Single-pair prediction: aspirin against the κ-opioid receptor (P41145).
    result = predictor.predict("CC(=O)Oc1ccccc1C(=O)O", "P41145", log_audit=False)

    assert result["predicted_class"] in EXPECTED_CLASSES
    probs = result["probabilities"]
    assert set(probs) == EXPECTED_CLASSES
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert "applicability" in result and "confidence" in result


if __name__ == "__main__":
    test_modern_predictor_four_class()
    print("Inference smoke test passed: 4-class ModernBiasPredictor OK.")
