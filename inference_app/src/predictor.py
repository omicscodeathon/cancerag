"""
Model Prediction Module

This module handles model loading and prediction for inference.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BiasPredictor:
    """
    Predicts biased agonism for GPCR ligands.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        metadata_path: str,
        imputer_path: Optional[str] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model pickle file
            scaler_path: Path to scaler pickle file
            metadata_path: Path to preprocessing metadata JSON file
            imputer_path: Optional path to imputer pickle file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.imputer_path = imputer_path

        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.label_mapping = None
        self.reverse_label_mapping = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all required artifacts (model, scaler, metadata)."""
        import json

        logger.info("Loading prediction artifacts...")

        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from: {self.model_path}")

        # Load scaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"Loaded scaler from: {self.scaler_path}")

        # Load imputer if provided
        if self.imputer_path and os.path.exists(self.imputer_path):
            with open(self.imputer_path, "rb") as f:
                self.imputer = pickle.load(f)
            logger.info(f"Loaded imputer from: {self.imputer_path}")

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.feature_columns = metadata.get("feature_columns", [])
        self.label_mapping = metadata.get("label_mapping", {})

        # Create reverse mapping (numeric -> label)
        if self.label_mapping:
            self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        logger.info(f"Loaded metadata with {len(self.feature_columns)} features")
        logger.info(f"Label mapping: {self.label_mapping}")

    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features to match training data format.

        Args:
            features_df: DataFrame with molecular descriptors

        Returns:
            Preprocessed feature array ready for prediction
        """
        # Select only the features used during training
        if self.feature_columns:
            # Get available features
            [
                col for col in self.feature_columns if col in features_df.columns
            ]
            missing_features = [
                col for col in self.feature_columns if col not in features_df.columns
            ]

            if missing_features:
                logger.warning(
                    f"Missing {len(missing_features)} features. Filling with NaN."
                )
                # Add missing features as NaN
                for col in missing_features:
                    features_df[col] = np.nan

            # Select features in the correct order
            features_df = features_df[self.feature_columns]

        # Handle missing values with imputer
        if self.imputer is not None:
            features_array = self.imputer.transform(features_df.values)
        else:
            # Simple median imputation if no imputer
            features_array = features_df.fillna(features_df.median()).values

        # Scale features
        features_scaled = self.scaler.transform(features_array)

        return features_scaled

    def predict(
        self, features_df: pd.DataFrame, return_proba: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """
        Predict bias category for given features.

        Args:
            features_df: DataFrame with molecular descriptors
            return_proba: Whether to return class probabilities

        Returns:
            Tuple of (predicted_class, class_probabilities_dict)
        """
        # Preprocess features
        features_processed = self.preprocess_features(features_df)

        # Make prediction
        predicted_class_idx = self.model.predict(features_processed)[0]

        # Convert to class label
        if self.reverse_label_mapping:
            predicted_class = self.reverse_label_mapping.get(
                int(predicted_class_idx), f"Class_{predicted_class_idx}"
            )
        else:
            predicted_class = f"Class_{predicted_class_idx}"

        # Get probabilities if available
        probabilities = {}
        if return_proba and hasattr(self.model, "predict_proba"):
            proba_array = self.model.predict_proba(features_processed)[0]
            class_indices = self.model.classes_

            for idx, prob in zip(class_indices, proba_array):
                if self.reverse_label_mapping:
                    class_name = self.reverse_label_mapping.get(
                        int(idx), f"Class_{idx}"
                    )
                else:
                    class_name = f"Class_{idx}"
                probabilities[class_name] = float(prob)

        return predicted_class, probabilities

    def get_class_names(self) -> List[str]:
        """
        Get list of class names.

        Returns:
            List of class names
        """
        if self.label_mapping:
            return list(self.label_mapping.keys())
        return []


def load_predictor(
    model_name: str = "random_forest",
    base_path: str = None,
) -> BiasPredictor:
    """
    Convenience function to load a predictor with default paths.

    Args:
        model_name: Name of the model to load (without .pkl extension)
        base_path: Base path to project root (defaults to parent of inference_app)

    Returns:
        Initialized BiasPredictor instance
    """
    if base_path is None:
        # Default to parent directory of inference_app
        base_path = Path(__file__).parent.parent.parent

    base_path = Path(base_path)

    model_path = base_path / "results" / "models" / f"{model_name}.pkl"
    scaler_path = base_path / "data" / "processed" / "ml_preprocessed" / "scaler.pkl"
    metadata_path = (
        base_path
        / "data"
        / "processed"
        / "ml_preprocessed"
        / "preprocessing_metadata.json"
    )
    imputer_path = base_path / "data" / "processed" / "ml_preprocessed" / "imputer.pkl"

    # Check if imputer exists
    if not imputer_path.exists():
        imputer_path = None

    return BiasPredictor(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
        imputer_path=str(imputer_path) if imputer_path else None,
    )
