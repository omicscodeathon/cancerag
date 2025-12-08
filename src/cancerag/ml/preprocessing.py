"""
Data Preprocessing Module for Machine Learning Pipeline

This module handles data preprocessing including missing value imputation,
class balancing, feature scaling, and train-test splitting.
"""

import logging
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles all data preprocessing steps for machine learning.
    """

    def __init__(self, config: Dict):
        """
        Initialize data preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get("ml_model", {})
        self.test_size = self.ml_config.get("test_size", 0.2)
        self.random_state = self.ml_config.get("random_state", 42)

        self.scaler = None
        self.label_encoder = None
        self.imputer = None
        self.feature_columns = None
        self.target_column = "primary_bias_label"

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """
        Load the dataset from CSV.

        Args:
            dataset_path: Path to dataset file

        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading dataset from: {dataset_path}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")

        return df

    def handle_missing_values(
        self, X: pd.DataFrame, strategy: str = "median"
    ) -> pd.DataFrame:
        """
        Handle missing values in the feature matrix.

        Args:
            X: Feature matrix
            strategy: Imputation strategy ('mean', 'median', 'most_frequent')

        Returns:
            Imputed feature matrix
        """
        logger.info(f"Handling missing values with strategy: {strategy}")

        missing_count = X.isnull().sum().sum()
        missing_percentage = (missing_count / (X.shape[0] * X.shape[1])) * 100

        logger.info(f"Missing values: {missing_count} ({missing_percentage:.2f}%)")

        # Columns with all-NaN values will be dropped by SimpleImputer. Preserve them by
        # filling with zeros (or a neutral value) before imputation so downstream code
        # keeps the original feature count.
        all_nan_cols = [col for col in X.columns if X[col].isna().all()]
        if all_nan_cols:
            logger.warning(
                "Detected %d columns with all NaN values. Filling them with 0 to "
                "preserve feature shape: %s",
                len(all_nan_cols),
                all_nan_cols[:10],
            )
            X = X.copy()
            for col in all_nan_cols:
                X[col] = 0.0
            # Recompute missing count after filling all-NaN columns
            missing_count = X.isnull().sum().sum()

        if missing_count > 0:
            if self.imputer is None:
                try:
                    self.imputer = SimpleImputer(
                        strategy=strategy, keep_empty_features=True
                    )
                except TypeError:
                    # keep_empty_features added in newer sklearn versions; fall back gracefully
                    self.imputer = SimpleImputer(strategy=strategy)
                X_imputed = pd.DataFrame(
                    self.imputer.fit_transform(X), columns=X.columns, index=X.index
                )
            else:
                X_imputed = pd.DataFrame(
                    self.imputer.transform(X), columns=X.columns, index=X.index
                )

            logger.info("Missing values imputed successfully")
            return X_imputed
        else:
            logger.info("No missing values found")
            return X

    def encode_target(self, y: pd.Series) -> Tuple[np.ndarray, Dict]:
        """
        Encode target labels to numerical values.

        Args:
            y: Target labels

        Returns:
            Tuple of (encoded labels, label mapping)
        """
        logger.info("Encoding target labels...")

        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)

        # Create label mapping
        label_mapping = dict(
            zip(
                self.label_encoder.classes_,
                self.label_encoder.transform(self.label_encoder.classes_),
            )
        )

        logger.info(f"Label mapping: {label_mapping}")

        return y_encoded, label_mapping

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using standardization.

        Args:
            X: Feature matrix
            fit: Whether to fit the scaler (True for training, False for test)

        Returns:
            Scaled feature matrix
        """
        logger.info("Scaling features...")

        if fit:
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X), columns=X.columns, index=X.index
            )
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted yet")
            X_scaled = pd.DataFrame(
                self.scaler.transform(X), columns=X.columns, index=X.index
            )

        logger.info("Features scaled successfully")

        return X_scaled

    def handle_class_imbalance(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Handle class imbalance using undersampling or oversampling.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Tuple of (balanced features, balanced labels)
        """
        logger.info("Analyzing class imbalance...")

        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))

        logger.info(f"Class distribution: {class_distribution}")

        # Calculate imbalance ratio
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count

        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")

        if imbalance_ratio > 2.0:
            logger.info("Significant class imbalance detected")
            logger.info(
                "Note: Using class_weight='balanced' in models to handle imbalance"
            )
            # Instead of resampling, we'll use class weights in models
            # This preserves the original data distribution
        else:
            logger.info("Class distribution is relatively balanced")

        return X, y

    def split_data(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        test_size: float = None,
        stratify: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.

        Args:
            X: Feature matrix
            y: Target labels
            test_size: Test set size (fraction)
            stratify: Whether to stratify by target

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if test_size is None:
            test_size = self.test_size

        logger.info(f"Splitting data: train={1 - test_size:.0%}, test={test_size:.0%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if stratify else None,
        )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        # Log class distribution
        train_unique, train_counts = np.unique(y_train, return_counts=True)
        test_unique, test_counts = np.unique(y_test, return_counts=True)

        logger.info(
            f"Training class distribution: {dict(zip(train_unique, train_counts))}"
        )
        logger.info(f"Test class distribution: {dict(zip(test_unique, test_counts))}")

        return X_train, X_test, y_train, y_test

    def prepare_dataset(self, df: pd.DataFrame, test_size: float = None) -> Dict:
        """
        Complete preprocessing pipeline.

        Args:
            df: Input dataframe
            test_size: Test set size (fraction)

        Returns:
            Dictionary containing preprocessed data and metadata
        """
        logger.info("Starting data preprocessing pipeline...")

        # Identify metadata columns
        metadata_columns = [
            "ligand_name",
            "canonical_smiles",
            "smiles",
            "smiles_duplicate",
            "canonical_smiles_standardized",
            "source",
            "receptor_family",
            "receptor",
            "receptor_subtype",
            "bias_category",
            "bias_pathway",
            "reference_ligand",
            "assay_1",
            "assay_2",
            "publication_title",
            "author",
            "doi",
            "pmid",
            "year",
            "receptor_count",
        ]

        # Keep only metadata columns that exist
        metadata_columns = [col for col in metadata_columns if col in df.columns]

        # Check target column
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataset"
            )

        # Separate features and target
        X = df.drop(columns=metadata_columns + [self.target_column], errors="ignore")
        y = df[self.target_column]

        # Store original feature columns
        self.feature_columns = X.columns.tolist()

        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Target: {self.target_column}")

        # Handle missing values
        X = self.handle_missing_values(X, strategy="median")

        # Remove constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant columns")
            X = X.drop(columns=constant_cols)
            self.feature_columns = [
                col for col in self.feature_columns if col not in constant_cols
            ]

        # Remove columns with too many missing values (>50% before imputation was already handled)
        # This step is now redundant after imputation

        # Encode target
        y_encoded, label_mapping = self.encode_target(y)

        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y_encoded, test_size)

        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)

        # Handle class imbalance (analysis only, will use class weights)
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(
            X_train_scaled, y_train
        )

        # Prepare return dictionary
        result = {
            "X_train": X_train_balanced,
            "X_test": X_test_scaled,
            "y_train": y_train_balanced,
            "y_test": y_test,
            "feature_columns": self.feature_columns,
            "label_mapping": label_mapping,
            "label_encoder": self.label_encoder,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "metadata": {
                "n_features": len(self.feature_columns),
                "n_samples_train": X_train_balanced.shape[0],
                "n_samples_test": X_test_scaled.shape[0],
                "n_classes": len(label_mapping),
                "class_names": list(label_mapping.keys()),
            },
        }

        logger.info("Preprocessing pipeline completed successfully!")
        logger.info(
            f"Final dataset: {result['metadata']['n_samples_train']} train samples, "
            f"{result['metadata']['n_samples_test']} test samples, "
            f"{result['metadata']['n_features']} features, "
            f"{result['metadata']['n_classes']} classes"
        )

        return result

    def save_preprocessed_data(self, result: Dict, output_dir: str) -> None:
        """
        Save preprocessed data and metadata.

        Args:
            result: Preprocessing result dictionary
            output_dir: Output directory
        """
        import json
        import pickle

        logger.info("Saving preprocessed data...")

        os.makedirs(output_dir, exist_ok=True)

        # Save training data
        result["X_train"].to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        result["X_test"].to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

        np.save(os.path.join(output_dir, "y_train.npy"), result["y_train"])
        np.save(os.path.join(output_dir, "y_test.npy"), result["y_test"])

        # Save preprocessing objects
        with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(result["label_encoder"], f)

        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(result["scaler"], f)

        if result["imputer"]:
            with open(os.path.join(output_dir, "imputer.pkl"), "wb") as f:
                pickle.dump(result["imputer"], f)

        # Save metadata
        metadata = result["metadata"].copy()
        metadata["label_mapping"] = result["label_mapping"]
        metadata["feature_columns"] = result["feature_columns"]

        def make_json_serializable(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_serializable(v) for v in obj]
            return obj

        metadata = make_json_serializable(metadata)

        with open(os.path.join(output_dir, "preprocessing_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved preprocessed data to: {output_dir}")


def run_preprocessing(config: Dict, dataset_path: str = None) -> Dict:
    """
    Main function to run data preprocessing.

    Args:
        config: Configuration dictionary
        dataset_path: Path to input dataset (optional)

    Returns:
        Preprocessing result dictionary
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)

    # Load dataset
    if dataset_path is None:
        # Try selected features dataset first, fallback to full dataset
        selected_path = os.path.join(
            config["paths"]["processed_data"], "ml_dataset_selected_features.csv"
        )
        full_path = os.path.join(
            config["paths"]["processed_data"], "unified_ml_dataset.csv"
        )

        if os.path.exists(selected_path):
            dataset_path = selected_path
            logger.info("Using dataset with selected features")
        elif os.path.exists(full_path):
            dataset_path = full_path
            logger.info("Using full dataset (feature selection not run yet)")
        else:
            raise FileNotFoundError("No dataset found for preprocessing")

    df = preprocessor.load_dataset(dataset_path)

    # Run preprocessing
    result = preprocessor.prepare_dataset(df)

    # Save preprocessed data
    output_dir = os.path.join(config["paths"]["processed_data"], "ml_preprocessed")
    preprocessor.save_preprocessed_data(result, output_dir)

    return result


if __name__ == "__main__":
    import yaml

    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run preprocessing
    result = run_preprocessing(config)

    print("\nPreprocessing Summary:")
    print(f"Training samples: {result['metadata']['n_samples_train']}")
    print(f"Test samples: {result['metadata']['n_samples_test']}")
    print(f"Features: {result['metadata']['n_features']}")
    print(f"Classes: {result['metadata']['n_classes']}")
    print(f"Class names: {result['metadata']['class_names']}")
