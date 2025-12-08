"""
Complete Inference Pipeline

This module orchestrates the complete inference pipeline from SMILES to prediction.
"""

import logging
from typing import Dict, Optional

import pandas as pd

from .docking_extractor import DockingFeatureExtractor
from .feature_extractor import FeatureExtractor
from .molecule_processor import MoleculeProcessor
from .predictor import BiasPredictor

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Complete inference pipeline for biased agonism prediction.
    """

    def __init__(
        self,
        predictor: BiasPredictor,
        base_path: Optional[str] = None,
        enable_docking: bool = True,
    ):
        """
        Initialize the inference pipeline.

        Args:
            predictor: Initialized BiasPredictor instance
            base_path: Base path to project root (for docking resources)
            enable_docking: Whether to perform docking to extract docking features
        """
        self.predictor = predictor
        self.molecule_processor = MoleculeProcessor()
        self.feature_extractor = FeatureExtractor()
        self.enable_docking = enable_docking

        # Initialize docking extractor if enabled
        if self.enable_docking:
            try:
                self.docking_extractor = DockingFeatureExtractor(base_path=base_path)
                logger.info("Docking feature extraction enabled")
            except Exception as e:
                logger.warning(
                    f"Failed to initialize docking extractor: {e}. "
                    "Continuing without docking features."
                )
                self.enable_docking = False
                self.docking_extractor = None
        else:
            self.docking_extractor = None
            logger.info("Docking feature extraction disabled")

    def predict_from_smiles(self, smiles: str, return_details: bool = True) -> Dict:
        """
        Predict bias category from SMILES string.

        Args:
            smiles: SMILES string of the molecule
            return_details: Whether to return detailed information

        Returns:
            Dictionary with prediction results and details
        """
        result = {
            "success": False,
            "error": None,
            "smiles_input": smiles,
            "smiles_canonical": None,
            "predicted_class": None,
            "probabilities": {},
            "details": {},
        }

        try:
            # Validate SMILES
            is_valid, error_msg = self.molecule_processor.validate_smiles(smiles)
            if not is_valid:
                result["error"] = error_msg
                return result

            # Standardize molecule
            mol = self.molecule_processor.standardize_molecule(smiles)
            if mol is None:
                result["error"] = "Failed to standardize molecule"
                return result

            # Get canonical SMILES
            canonical_smiles = self.molecule_processor.get_canonical_smiles(mol)
            result["smiles_canonical"] = canonical_smiles

            # Extract molecular descriptors
            features_df = self.feature_extractor.extract_features(mol)

            # Extract docking features if enabled
            if self.enable_docking and self.docking_extractor:
                try:
                    logger.info("Extracting docking features...")
                    docking_features_df = (
                        self.docking_extractor.extract_docking_features(mol)
                    )

                    if not docking_features_df.empty:
                        # Merge docking features with molecular descriptors
                        features_df = pd.concat(
                            [features_df, docking_features_df], axis=1
                        )
                        logger.info(
                            f"Added {len(docking_features_df.columns)} docking features"
                        )
                    else:
                        logger.warning(
                            "Docking features extraction returned empty DataFrame"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract docking features: {e}. "
                        "Continuing with molecular descriptors only."
                    )

            # Make prediction
            predicted_class, probabilities = self.predictor.predict(features_df)

            result["predicted_class"] = predicted_class
            result["probabilities"] = probabilities
            result["success"] = True

            if return_details:
                result["details"] = {
                    "num_features": len(features_df.columns),
                    "molecule_valid": True,
                    "standardization_success": True,
                }

        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def predict_batch(self, smiles_list: list[str]) -> list[Dict]:
        """
        Predict bias categories for multiple SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of prediction result dictionaries
        """
        results = []
        for smiles in smiles_list:
            result = self.predict_from_smiles(smiles, return_details=True)
            results.append(result)
        return results
