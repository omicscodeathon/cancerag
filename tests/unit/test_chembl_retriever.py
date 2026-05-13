"""Tests for cancerag.data_collection.chembl_retriever.

The retriever now uses a ReceptorRegistry for UniProt-anchored target
resolution (no more "search ChEMBL by name and pick first hit") and applies
activity-table filters wired to config keys (formerly dead code).
ChEMBL output is tagged ``label_status="unlabeled"``; bias_category is
never auto-assigned (E1).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from cancerag.data_collection.chembl_retriever import (
    ChEMBLRetriever,
    UnknownReceptorError,
)
from cancerag.data_collection.registry import ReceptorRegistry


def _registry() -> ReceptorRegistry:
    return ReceptorRegistry.load("data/registry/receptors.tsv")


def _stub_retriever(tmp_path: Path, **kwargs) -> ChEMBLRetriever:
    """Build a retriever with a non-empty registry but no real network."""
    return ChEMBLRetriever(
        output_dir=tmp_path / "chembl",
        registry=_registry(),
        network_config={"max_retries": 0},
        **kwargs,
    )


@pytest.mark.unit
class TestRegistryStrictness:
    def test_unknown_receptor_raises(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        with pytest.raises(UnknownReceptorError, match="not in"):
            r._resolve_target_id("never-heard-of-this-receptor")

    def test_strict_run_aborts_on_unknown(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        with pytest.raises(UnknownReceptorError):
            r.run(["totally-fake-receptor"])

    def test_known_receptor_resolved(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        # 5-HT1A receptor is in the shipped registry
        uniprot, target_id = r._resolve_target_id("5HT1A receptor")
        assert uniprot == "P08908"
        assert target_id == "CHEMBL214"

    def test_non_strict_returns_none(self, tmp_path: Path):
        r = _stub_retriever(tmp_path, strict=False)
        assert r._resolve_target_id("nonexistent receptor") is None


@pytest.mark.unit
class TestActivityFilters:
    def test_confidence_filter_applied_client_side(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        with patch.object(
            ChEMBLRetriever, "_run_with_retry",
            return_value=[
                {"molecule_chembl_id": "CHEMBL_A", "confidence_score": 9},
                {"molecule_chembl_id": "CHEMBL_B", "confidence_score": 4},
                {"molecule_chembl_id": "CHEMBL_C", "confidence_score": 8},
                {"molecule_chembl_id": "CHEMBL_D", "confidence_score": None},
            ],
        ):
            rows = r._fetch_activities("CHEMBL214")
        # Only CHEMBL_A (score 9) and CHEMBL_C (score 8) pass the >=8 default
        ids = [row["molecule_chembl_id"] for row in rows]
        assert ids == ["CHEMBL_A", "CHEMBL_C"]


@pytest.mark.unit
class TestBuildDataFrame:
    def test_label_status_unlabeled_and_no_bias_category(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        activities = [
            {"molecule_chembl_id": "CHEMBL_A", "standard_type": "EC50",
             "standard_value": 100, "standard_units": "nM",
             "assay_type": "F", "confidence_score": 9},
        ]
        with patch.object(
            ChEMBLRetriever, "_fetch_molecule_smiles", return_value="CCO"
        ):
            df = r._build_dataframe(
                activities, "5HT1A receptor", "P08908", "CHEMBL214"
            )
        assert len(df) == 1
        # E1 fix verified at the row level: ChEMBL rows are unlabeled,
        # never auto-assigned to "Agonist".
        assert df.iloc[0]["label_status"] == "unlabeled"
        assert df.iloc[0]["bias_category"] is None
        assert df.iloc[0]["receptor_uniprot"] == "P08908"
        assert df.iloc[0]["chembl_target_id"] == "CHEMBL214"

    def test_max_per_receptor_caps_output(self, tmp_path: Path):
        r = _stub_retriever(tmp_path, max_per_receptor=2)
        activities = [
            {"molecule_chembl_id": f"CHEMBL_{i}", "standard_type": "EC50",
             "standard_value": 100, "standard_units": "nM",
             "assay_type": "F", "confidence_score": 9}
            for i in range(5)
        ]
        with patch.object(
            ChEMBLRetriever, "_fetch_molecule_smiles", return_value="CCO"
        ):
            df = r._build_dataframe(
                activities, "5HT1A receptor", "P08908", "CHEMBL214"
            )
        assert len(df) == 2

    def test_unique_per_molecule(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        activities = [
            {"molecule_chembl_id": "CHEMBL_A", "standard_type": "EC50",
             "standard_value": 50, "standard_units": "nM",
             "assay_type": "F", "confidence_score": 9},
            {"molecule_chembl_id": "CHEMBL_A", "standard_type": "EC50",
             "standard_value": 60, "standard_units": "nM",
             "assay_type": "F", "confidence_score": 9},
        ]
        with patch.object(
            ChEMBLRetriever, "_fetch_molecule_smiles", return_value="CCO"
        ):
            df = r._build_dataframe(activities, "X", "U", "T")
        assert len(df) == 1


@pytest.mark.unit
class TestRunWritesArtifacts:
    def test_writes_csv_and_meta_sidecar(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        activities = [
            {"molecule_chembl_id": "CHEMBL_A", "standard_type": "EC50",
             "standard_value": 50, "standard_units": "nM",
             "assay_type": "F", "confidence_score": 9},
        ]
        with patch.object(
            ChEMBLRetriever, "_fetch_activities", return_value=activities
        ), patch.object(
            ChEMBLRetriever, "_fetch_molecule_smiles", return_value="CCO"
        ):
            summary = r.run(["5HT1A receptor"])

        assert summary["total_records"] == 1
        out_csv = (
            tmp_path / "chembl"
            / "5ht1a_receptor__P08908__unlabeled.csv"
        )
        assert out_csv.exists()
        df = pd.read_csv(out_csv)
        assert (df["label_status"] == "unlabeled").all()
        assert "Agonist" not in str(df.get("bias_category", pd.Series()).values)

        meta_path = out_csv.with_suffix(out_csv.suffix + ".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["row_count"] == 1
        assert meta["query_params"]["target_chembl_id"] == "CHEMBL214"
        assert meta["query_params"]["uniprot"] == "P08908"

    def test_summary_json_emitted(self, tmp_path: Path):
        r = _stub_retriever(tmp_path)
        with patch.object(
            ChEMBLRetriever, "_fetch_activities", return_value=[]
        ):
            r.run(["5HT1A receptor"])
        assert (tmp_path / "chembl" / "chembl_summary.json").exists()


@pytest.mark.unit
class TestFromConfig:
    def test_wires_config_keys(self, tmp_path: Path):
        cfg = {
            "data_collection": {
                "chembl_activity_types": ["EC50", "Ki"],
                "chembl_min_confidence_score": 9,
                "chembl_activity_threshold_nm": 500,
                "chembl_max_agonists_per_receptor": 100,
            },
            "network": {"max_retries": 0},
        }
        r = ChEMBLRetriever.from_config(
            output_dir=tmp_path / "chembl",
            config=cfg,
            registry=_registry(),
        )
        assert r.activity_types == ("EC50", "Ki")
        assert r.min_confidence_score == 9
        assert r.activity_threshold_nm == 500
        assert r.max_per_receptor == 100
