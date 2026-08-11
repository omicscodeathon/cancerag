"""Tests for config-driven feature selection in cancerag.ml.model_training.

Wrapper selection is the slowest step in the pipeline and the only one that
looks at the labels, so it is off by default and switched on through
``ml_model.feature_selection`` in configs/config.yaml rather than by editing
code. These tests pin the contract: the default reproduces published results,
a missing or malformed config degrades to OFF rather than crashing, and when
enabled the selector lands inside the Pipeline so it is refitted per fold.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cancerag.ml.model_training import (
    SELECTOR_DEFAULTS,
    _build_pipeline_for,
    build_selector,
    load_selector_config,
)


@pytest.mark.unit
class TestSelectorConfig:
    def test_shipped_config_keeps_selection_off(self):
        """The default must reproduce every result published to date."""
        cfg = load_selector_config("configs/config.yaml")
        assert cfg["enabled"] is False
        assert build_selector(cfg, seed=42) is None

    def test_missing_config_degrades_to_off(self, tmp_path: Path):
        cfg = load_selector_config(tmp_path / "does_not_exist.yaml")
        assert cfg == SELECTOR_DEFAULTS

    def test_malformed_config_degrades_to_off(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("ml_model: [this is not a mapping\n")
        assert load_selector_config(bad)["enabled"] is False

    def test_config_values_are_read(self, tmp_path: Path):
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "ml_model:\n"
            "  feature_selection:\n"
            "    enabled: true\n"
            "    method: rfecv\n"
            "    max_iter: 55\n"
            "    force_keep_structural: true\n"
        )
        cfg = load_selector_config(cfg_file)
        assert cfg["enabled"] is True
        assert cfg["method"] == "rfecv"
        assert cfg["max_iter"] == 55
        assert cfg["force_keep_structural"] is True

    def test_unknown_keys_are_ignored_not_fatal(self, tmp_path: Path):
        cfg_file = tmp_path / "c.yaml"
        cfg_file.write_text(
            "ml_model:\n  feature_selection:\n    enabled: true\n    typo_key: 3\n"
        )
        cfg = load_selector_config(cfg_file)
        assert cfg["enabled"] is True
        assert "typo_key" not in cfg

    @pytest.mark.parametrize(
        "method,expected",
        [("boruta", "BorutaSelector"), ("rfecv", "RFECVSelector"),
         ("mutual_info", "MISelector"), ("l1_logreg", "L1LogRegSelector")],
    )
    def test_each_method_builds(self, method, expected):
        sel = build_selector({"enabled": True, "method": method}, seed=1)
        assert type(sel).__name__ == expected

    def test_unknown_method_raises_with_valid_options(self):
        with pytest.raises(KeyError, match="boruta"):
            build_selector({"enabled": True, "method": "nonsense"}, seed=1)

    def test_force_keep_default_is_off(self):
        """Force-keeping exempts columns from the selector's verdict, so their
        survival stops being evidence. It must be opt-in."""
        sel = build_selector({"enabled": True, "method": "boruta"}, seed=1)
        assert sel.force_keep_prefixes == ()
        kept = build_selector(
            {"enabled": True, "method": "boruta", "force_keep_structural": True},
            seed=1,
        )
        assert len(kept.force_keep_prefixes) > 0


@pytest.mark.unit
class TestSelectorInPipeline:
    def test_selector_absent_when_disabled(self):
        pipe = _build_pipeline_for("lightgbm", 4, 42, with_selector=False)
        assert "selector" not in pipe.named_steps

    def test_selector_present_and_before_model(self):
        """Inside the Pipeline means refitted on each fold's training rows —
        the whole point, since selection looks at the labels."""
        pipe = _build_pipeline_for(
            "lightgbm", 4, 42, with_selector=True,
            selector_config={"enabled": True, "method": "mutual_info"},
        )
        names = list(pipe.named_steps)
        assert "selector" in names
        assert names.index("selector") < names.index("model")
