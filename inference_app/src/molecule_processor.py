"""
Molecule Preprocessing for Inference

This module handles molecule standardization and validation for inference.
"""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


class MoleculeProcessor:
    """
    Processes and standardizes molecules for inference.
    """

    def __init__(self):
        """Initialize the molecule processor."""
        self.unchoarger = rdMolStandardize.Uncharger()

    def standardize_molecule(self, smiles: str) -> Optional[Chem.Mol]:
        """
        Standardize a molecule from SMILES string.

        Args:
            smiles: SMILES string of the molecule

        Returns:
            Standardized RDKit molecule object, or None if standardization fails
        """
        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Failed to parse SMILES: {smiles}")
                return None

            # Remove salts and fragments, keeping the largest fragment
            mol = rdMolStandardize.Cleanup(mol)
            parent = rdMolStandardize.FragmentParent(mol)

            # Neutralize charges
            neutral = self.unchoarger.uncharge(parent)

            # Sanitize and prepare molecule
            Chem.SanitizeMol(neutral)
            neutral.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(neutral)

            return neutral

        except Exception as e:
            logger.warning(f"Could not standardize molecule {smiles}: {e}")
            return None

    def validate_smiles(self, smiles: str) -> tuple[bool, str]:
        """
        Validate a SMILES string.

        Args:
            smiles: SMILES string to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not smiles or not smiles.strip():
            return False, "SMILES string is empty"

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False, "Invalid SMILES string - cannot parse molecule"

        return True, ""

    def get_canonical_smiles(self, mol: Chem.Mol) -> str:
        """
        Get canonical SMILES from a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            Canonical SMILES string
        """
        return Chem.MolToSmiles(mol, canonical=True)
