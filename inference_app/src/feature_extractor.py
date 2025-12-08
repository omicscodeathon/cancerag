"""
Feature Extraction for Inference

This module extracts molecular descriptors for inference.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts molecular descriptors for inference.
    """

    def __init__(self):
        """Initialize the feature extractor."""
        # Get the list of all available 2D descriptors from RDKit
        self.descriptor_list = [desc[0] for desc in Descriptors._descList]
        logger.info(
            f"Initialized with {len(self.descriptor_list)} available RDKit descriptors."
        )

    def calculate_custom_features(self, mol: Optional[Chem.Mol]) -> Dict[str, float]:
        """
        Calculate custom features that match the training pipeline.

        Args:
            mol: RDKit molecule object

        Returns:
            Dictionary of custom feature values
        """
        if mol is None:
            return {
                "MW": None,
                "LogP": None,
                "HBD": None,
                "HBA": None,
                "TPSA": None,
                "Rotatable_Bonds": None,
                "Lipinski_Violations": None,
            }

        try:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)

            # Calculate Lipinski violations
            violations = (
                (1 if mw > 500 else 0)
                + (1 if logp > 5 else 0)
                + (1 if hbd > 5 else 0)
                + (1 if hba > 10 else 0)
            )

            return {
                "MW": mw,
                "LogP": logp,
                "HBD": hbd,
                "HBA": hba,
                "TPSA": tpsa,
                "Rotatable_Bonds": rot_bonds,
                "Lipinski_Violations": violations,
            }
        except Exception as e:
            logger.warning(f"Could not calculate custom features: {e}")
            return {
                "MW": None,
                "LogP": None,
                "HBD": None,
                "HBA": None,
                "TPSA": None,
                "Rotatable_Bonds": None,
                "Lipinski_Violations": None,
            }

    def calculate_descriptors(self, mol: Optional[Chem.Mol]) -> List[float]:
        """
        Calculate all RDKit descriptors for a molecule.

        Args:
            mol: RDKit molecule object

        Returns:
            List of descriptor values (None for failed calculations)
        """
        if mol is None:
            return [None] * len(self.descriptor_list)

        try:
            # Calculate all descriptors in the list
            return [func(mol) for name, func in Descriptors._descList]
        except Exception as e:
            logger.warning(f"Could not calculate descriptors for molecule: {e}")
            return [None] * len(self.descriptor_list)

    def extract_features(self, mol: Optional[Chem.Mol]) -> pd.DataFrame:
        """
        Extract features for a molecule and return as DataFrame.

        This includes both custom features and RDKit descriptors.

        Args:
            mol: RDKit molecule object

        Returns:
            DataFrame with one row containing all descriptors
        """
        # Calculate RDKit descriptors
        descriptors = self.calculate_descriptors(mol)
        df = pd.DataFrame([descriptors], columns=self.descriptor_list)

        # Add custom features (these may override RDKit descriptors with same names)
        custom_features = self.calculate_custom_features(mol)
        for key, value in custom_features.items():
            df[key] = value

        return df

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names.

        Returns:
            List of feature names (RDKit descriptors + custom features)
        """
        custom_feature_names = [
            "MW",
            "LogP",
            "HBD",
            "HBA",
            "TPSA",
            "Rotatable_Bonds",
            "Lipinski_Violations",
        ]
        # Combine, removing duplicates (custom features take precedence)
        all_features = custom_feature_names + [
            f for f in self.descriptor_list if f not in custom_feature_names
        ]
        return all_features
