"""Tests for cancerag.data_collection.registry."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cancerag.data_collection.registry import (
    REQUIRED_COLUMNS,
    ReceptorRegistry,
    RegistryError,
)


def _good_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "uniprot": "P08908",
                "gene_symbol": "HTR1A",
                "biasdb_name": "5-HT1A receptor",
                "chembl_target_id": "CHEMBL214",
                "gpcrdb_id": "5ht1a_human",
                "gpcrdb_class": "A",
                "gpcrdb_family": "Aminergic",
                "preferred_pdb_active": "7E2X",
                "preferred_pdb_inactive": "7E2Y",
                "alphafold_id": "AF-P08908-F1",
                "notes": "serotonin",
            },
            {
                "uniprot": "P14416",
                "gene_symbol": "DRD2",
                "biasdb_name": "D2 receptor",
                "chembl_target_id": "CHEMBL217",
                "gpcrdb_id": "drd2_human",
                "gpcrdb_class": "A",
                "gpcrdb_family": "Aminergic",
                "preferred_pdb_active": "6CM4",
                "preferred_pdb_inactive": "6LUQ",
                "alphafold_id": "AF-P14416-F1",
                "notes": "dopamine",
            },
        ]
    )


@pytest.mark.unit
class TestReceptorRegistry:
    def test_load_from_disk(self):
        reg = ReceptorRegistry.load("data/registry/receptors.tsv")
        assert len(reg) >= 5
        assert "P08908" in reg.all_uniprots()
        for col in REQUIRED_COLUMNS:
            assert col in reg.dataframe.columns

    def test_lookup_by_uniprot(self):
        reg = ReceptorRegistry(_good_df())
        row = reg.by_uniprot("P14416")
        assert row["gene_symbol"] == "DRD2"
        assert row["chembl_target_id"] == "CHEMBL217"

    def test_lookup_by_biasdb_name_case_insensitive(self):
        reg = ReceptorRegistry(_good_df())
        row = reg.by_biasdb_name("5-ht1a RECEPTOR")
        assert row is not None
        assert row["uniprot"] == "P08908"

    def test_lookup_missing_returns_none(self):
        reg = ReceptorRegistry(_good_df())
        assert reg.by_biasdb_name("nonexistent") is None
        with pytest.raises(KeyError):
            reg.by_uniprot("P00000")

    def test_rejects_duplicate_uniprot(self):
        df = _good_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        with pytest.raises(RegistryError, match="Duplicate UniProt"):
            ReceptorRegistry(df)

    def test_rejects_missing_columns(self):
        df = _good_df().drop(columns=["chembl_target_id"])
        with pytest.raises(RegistryError, match="missing required columns"):
            ReceptorRegistry(df)

    def test_rejects_blank_uniprot(self):
        df = _good_df()
        df.loc[0, "uniprot"] = pd.NA
        with pytest.raises(RegistryError, match="empty uniprot"):
            ReceptorRegistry(df)

    def test_load_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ReceptorRegistry.load(tmp_path / "no.tsv")
