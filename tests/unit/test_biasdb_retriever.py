"""
Unit tests for BiasDB retriever module.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from cancerag.data_collection import biasdb_retriever


@pytest.mark.unit
class TestBiasDBRetriever:
    """Test suite for BiasDB retriever."""

    def test_download_biasdb_data_success(self, tmp_path):
        """Test successful BiasDB data download."""
        output_path = tmp_path / "biasdb_data.csv"

        # Mock data that would come from BiasDB
        mock_data = {
            "receptor_subtype": ["5HT2C receptor", "D2 receptor"],
            "ligand_name": ["Lorcaserin", "Aripiprazole"],
            "smiles": [
                "Clc1ccc2[nH]cc(c2c1)-c1ccc(Br)cc1",
                "Clc1ccc2c(c1)nccc2-c1ncccn1",
            ],
            "bias_category": ["G protein-biased", "Balanced"],
        }
        mock_df = pd.DataFrame(mock_data)

        with patch("pandas.read_csv", return_value=mock_df):
            result = biasdb_retriever.download_biasdb_data(str(output_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "receptor_subtype" in result.columns

    def test_download_biasdb_data_empty(self, tmp_path):
        """Test BiasDB retriever with empty data."""
        output_path = tmp_path / "empty_biasdb.csv"

        empty_df = pd.DataFrame()

        with patch("pandas.read_csv", return_value=empty_df):
            result = biasdb_retriever.download_biasdb_data(str(output_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_biasdb_data_columns(self, sample_biasdb_data):
        """Test expected BiasDB data structure."""
        df = pd.DataFrame(sample_biasdb_data)

        # Check essential columns exist
        assert "receptor_subtype" in df.columns
        assert "ligand_name" in df.columns
        assert "smiles" in df.columns
        assert "bias_category" in df.columns

    def test_unique_receptors(self, sample_biasdb_data):
        """Test unique receptor extraction."""
        df = pd.DataFrame(sample_biasdb_data)
        unique_receptors = df["receptor_subtype"].unique()

        assert len(unique_receptors) == 3
        assert "5HT2C receptor" in unique_receptors
