"""Tests for cancerag.ml.dataset_assembly.

Covers:
- The legacy ``fillna(-5.0)`` and ``fillna("unknown")`` hardcodes are gone
  from the assembly module's source.
- Helpers: pair_key composition, label-conflict resolution by evidence,
  temporal holdout split, evidence weight, missing indicators, ``-5.0``
  absence check.
"""

from __future__ import annotations

import inspect
import re

import pandas as pd
import pytest

from cancerag.ml import dataset_assembly
from cancerag.ml.dataset_assembly import (
    add_missing_indicators,
    decisions_to_dataframe,
    evidence_weight_column,
    has_no_default_docking_score,
    make_pair_key,
    resolve_label_conflicts,
    split_temporal_holdout,
)


@pytest.mark.unit
class TestLegacyHardcodesRemoved:
    def test_no_minus_5_default_score(self):
        src = inspect.getsource(dataset_assembly)
        # Allow the value to appear in comments; reject any actual call site.
        # Match `.fillna(-5.0)` (with optional whitespace) anywhere in code.
        assert not re.search(r"\.fillna\(\s*-5\.0\s*\)", src), (
            "Legacy -5.0 docking-score sentinel still present in dataset_assembly"
        )

    def test_no_unknown_label_default(self):
        src = inspect.getsource(dataset_assembly)
        assert not re.search(r'\.fillna\(\s*[\'"]unknown[\'"]\s*\)', src), (
            "Legacy 'unknown' bias-label fillna still present"
        )


@pytest.mark.unit
class TestPairKey:
    def test_combines_required_columns(self):
        df = pd.DataFrame(
            {
                "inchikey14": ["AAAA", "BBBB"],
                "receptor_uniprot": ["P14416", "P08908"],
                "assay_1": ["BRET", None],
                "assay_2": ["cAMP", "Tango"],
            }
        )
        keys = make_pair_key(df)
        assert keys.iloc[0] == "AAAA::P14416::BRET::cAMP"
        assert keys.iloc[1] == "BBBB::P08908::?::Tango"

    def test_missing_column_raises(self):
        df = pd.DataFrame({"inchikey14": ["X"]})
        with pytest.raises(KeyError):
            make_pair_key(df)


@pytest.mark.unit
class TestResolveLabelConflicts:
    def test_unanimous_kept(self):
        df = pd.DataFrame(
            {
                "pair_key": ["k1", "k1"],
                "primary_bias_label": ["G", "G"],
                "year": [2020, 2021],
            }
        )
        out, decisions = resolve_label_conflicts(df)
        assert len(out) == 1
        assert decisions[0].decision == "unanimous"
        assert decisions[0].chosen == "G"

    def test_evidence_wins(self):
        # Two different labels for the same key — the one with more recent
        # year (higher evidence weight) should win.
        df = pd.DataFrame(
            {
                "pair_key": ["k1", "k1"],
                "primary_bias_label": ["Arr", "G"],
                "year": [2010, 2024],
                "assay_type": ["F", "F"],
            }
        )
        out, decisions = resolve_label_conflicts(df)
        assert len(out) == 1
        assert decisions[0].chosen == "G"
        assert "evidence_weight" in decisions[0].decision

    def test_decisions_dataframe(self):
        df = pd.DataFrame(
            {"pair_key": ["k"], "primary_bias_label": ["G"], "year": [2020]}
        )
        _, decisions = resolve_label_conflicts(df)
        out_df = decisions_to_dataframe(decisions)
        assert {"pair_key", "chosen", "decision", "candidates"}.issubset(
            out_df.columns
        )

    def test_drops_rows_with_no_label(self):
        df = pd.DataFrame(
            {"pair_key": ["k"], "primary_bias_label": [None], "year": [2020]}
        )
        out, decisions = resolve_label_conflicts(df)
        assert out.empty
        assert decisions == []


@pytest.mark.unit
class TestSplitTemporalHoldout:
    def test_partitions_by_cutoff(self):
        df = pd.DataFrame(
            {"pair_key": ["a", "b", "c"], "year": [2018, 2024, 2025]}
        )
        train, holdout = split_temporal_holdout(df, cutoff_year=2024)
        assert list(train["pair_key"]) == ["a"]
        assert sorted(holdout["pair_key"].tolist()) == ["b", "c"]

    def test_missing_year_stays_in_train(self):
        df = pd.DataFrame(
            {"pair_key": ["a", "b"], "year": [None, 2024]}
        )
        train, holdout = split_temporal_holdout(df, cutoff_year=2024)
        assert "a" in train["pair_key"].tolist()
        assert "b" in holdout["pair_key"].tolist()

    def test_missing_year_column_raises(self):
        df = pd.DataFrame({"pair_key": ["a"]})
        with pytest.raises(KeyError):
            split_temporal_holdout(df, cutoff_year=2024)


@pytest.mark.unit
class TestEvidenceWeight:
    def test_functional_assay_baseline(self):
        df = pd.DataFrame({"assay_type": ["F"], "year": [2018]})
        w = evidence_weight_column(df)
        assert w.iloc[0] == pytest.approx(1.0)

    def test_binding_only_downweighted(self):
        df = pd.DataFrame({"assay_type": ["B"], "year": [2018]})
        w = evidence_weight_column(df)
        assert w.iloc[0] == pytest.approx(0.5)

    def test_recent_high_confidence_boosts(self):
        df = pd.DataFrame(
            {"assay_type": ["F"], "year": [2024], "confidence_score": [9]}
        )
        w = evidence_weight_column(df)
        assert w.iloc[0] > 1.0


@pytest.mark.unit
class TestMissingIndicators:
    def test_appends_indicator_columns(self):
        df = pd.DataFrame({"docking_score_HTR1A": [-7.5, None, -6.1]})
        out = add_missing_indicators(df, ["docking_score_HTR1A"])
        assert "docking_score_HTR1A_missing" in out.columns
        assert list(out["docking_score_HTR1A_missing"]) == [0, 1, 0]

    def test_skip_unknown_columns(self):
        df = pd.DataFrame({"a": [1]})
        out = add_missing_indicators(df, ["a", "ghost"])
        assert "a_missing" in out.columns
        assert "ghost_missing" not in out.columns


@pytest.mark.unit
class TestSentinelAbsence:
    def test_passes_when_no_minus_5(self):
        df = pd.DataFrame({"docking_score_HTR1A": [-7.5, -6.0]})
        assert has_no_default_docking_score(df) is True

    def test_fails_when_minus_5_present(self):
        df = pd.DataFrame({"docking_vina_HTR1A": [-7.5, -5.0, -6.0]})
        assert has_no_default_docking_score(df) is False
