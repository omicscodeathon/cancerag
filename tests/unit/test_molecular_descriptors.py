"""
Unit tests for molecular descriptor calculation.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from cancerag.features.molecular_descriptors import MolecularDescriptorCalculator


@pytest.mark.unit
class TestMolecularDescriptors:
    """Test suite for molecular descriptor calculation."""

    def test_descriptor_calculator_init(self, test_config):
        """Test MolecularDescriptorCalculator initialization."""
        calculator = MolecularDescriptorCalculator(test_config)

        assert calculator is not None
        assert hasattr(calculator, "descriptor_list")
        assert len(calculator.descriptor_list) > 0

    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.read_csv")
    def test_run_with_mock_data(
        self, mock_read, mock_write, test_config, sample_smiles
    ):
        """Test descriptor calculation with mock data."""
        # Mock input data
        mock_df = pd.DataFrame(
            {"smiles": sample_smiles, "receptor_subtype": ["R1", "R2", "R3"]}
        )
        mock_read.return_value = mock_df

        calculator = MolecularDescriptorCalculator(test_config)
        calculator.run()

        # Verify CSV was written
        mock_write.assert_called_once()

    def test_smiles_to_mol(self):
        """Test SMILES to RDKit molecule conversion."""
        from rdkit import Chem

        smiles = "CC(=O)O"  # Acetic acid
        mol = Chem.MolFromSmiles(smiles)

        assert mol is not None
        assert mol.GetNumAtoms() > 0

    def test_descriptor_calculation_single_mol(self):
        """Test descriptor calculation for single molecule."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        smiles = "CC(=O)O"  # Acetic acid
        mol = Chem.MolFromSmiles(smiles)

        # Calculate a few descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        assert mw > 0
        assert isinstance(logp, float)
        assert tpsa >= 0
