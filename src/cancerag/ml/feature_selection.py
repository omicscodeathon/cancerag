"""
Feature Selection Module for Machine Learning Pipeline

This module implements feature selection using Boruta algorithm and other methods
to identify the most important features for bias prediction.
"""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

# Try to import the boruta library for better implementation
try:
    from boruta import BorutaPy

    HAS_BORUTA_LIB = True
except ImportError:
    HAS_BORUTA_LIB = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BorutaFeatureSelector:
    """
    Boruta algorithm implementation for feature selection.
    Based on the idea of comparing feature importance with random probes.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 100,
        random_state: int = 42,
        alpha: float = 0.05,
    ):
        """
        Initialize Boruta feature selector.

        Args:
            n_estimators: Number of trees in random forest
            max_iter: Maximum number of iterations
            random_state: Random state for reproducibility
            alpha: Significance level for statistical tests
        """
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.support_ = None
        self.feature_importances_ = None
        self.selected_features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BorutaFeatureSelector":
        """
        Fit the Boruta algorithm to select important features.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            self: Fitted selector
        """
        logger.info("Starting Boruta feature selection...")

        n_features = X.shape[1]
        feature_names = X.columns.tolist()

        # Initialize decision arrays
        hits = np.zeros(n_features)
        tentative = np.ones(n_features, dtype=bool)
        confirmed = np.zeros(n_features, dtype=bool)
        rejected = np.zeros(n_features, dtype=bool)

        for iteration in range(self.max_iter):
            # Create shadow features (random permutations)
            X_shadow = X.apply(np.random.permutation)
            X_shadow.columns = [f"shadow_{col}" for col in X_shadow.columns]

            # Combine original and shadow features
            X_boruta = pd.concat([X, X_shadow], axis=1)

            # Train random forest
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_boruta, y)

            # Get feature importances
            feat_imp = rf.feature_importances_[:n_features]
            shadow_imp = rf.feature_importances_[n_features:]

            # Calculate maximum shadow importance
            shadow_max = np.max(shadow_imp)

            # Compare each feature to shadow maximum
            hits[feat_imp > shadow_max] += 1

            # Update tentative features
            if iteration > 0 and iteration % 10 == 0:
                # Statistical test using binomial distribution
                hit_rate = hits / (iteration + 1)

                # Confirm features that consistently beat shadows
                confirmed[hit_rate > 0.5] = True
                tentative[confirmed] = False

                # Reject features that consistently lose to shadows
                rejected[hit_rate < 0.1] = True
                tentative[rejected] = False

                logger.info(
                    f"Iteration {iteration}: Confirmed={confirmed.sum()}, "
                    f"Rejected={rejected.sum()}, Tentative={tentative.sum()}"
                )

                # Early stopping if all features decided
                if not np.any(tentative):
                    break

        # Final decisions
        self.support_ = confirmed
        self.selected_features_ = [
            feature_names[i] for i in range(n_features) if confirmed[i]
        ]

        # Get final feature importances
        if len(self.selected_features_) > 0:
            rf_final = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf_final.fit(X[self.selected_features_], y)
            self.feature_importances_ = dict(
                zip(self.selected_features_, rf_final.feature_importances_)
            )

        logger.info(
            f"Boruta selected {len(self.selected_features_)} features out of {n_features}"
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset to include only selected features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.selected_features_ is None:
            raise ValueError("Selector has not been fitted yet")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y).transform(X)


class FeatureSelector:
    """
    Comprehensive feature selection class with multiple methods.
    """

    def __init__(self, config: Dict):
        """
        Initialize feature selector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get("ml_model", {})
        self.random_state = self.ml_config.get("random_state", 42)

        self.selected_features_ = None
        self.feature_scores_ = None
        self.selection_method_ = None

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for feature selection by separating features and target.

        Args:
            df: Input dataset

        Returns:
            Tuple of (features, target, metadata_columns)
        """
        logger.info("Preparing data for feature selection...")

        # Identify target column
        target_col = "primary_bias_label"
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Identify metadata columns to exclude from features
        metadata_columns = [
            "ligand_name",
            "smiles",
            "smiles_duplicate",
            "canonical_smiles_standardized",
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

        # Separate features and target
        X = df.drop(columns=metadata_columns + [target_col], errors="ignore")
        y = df[target_col]

        # Handle missing values - fill with median for numerical columns
        X = X.fillna(X.median())

        # Remove constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant columns")
            X = X.drop(columns=constant_cols)

        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y, metadata_columns

    def select_features_boruta(
        self, X: pd.DataFrame, y: pd.Series, max_features: int = None
    ) -> List[str]:
        """
        Select features using Boruta algorithm.

        Args:
            X: Feature matrix
            y: Target labels
            max_features: Maximum number of features to select (None for all)

        Returns:
            List of selected feature names
        """
        logger.info("Running Boruta feature selection...")

        # Encode target labels if they are strings
        if y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y

        # Use library implementation if available, otherwise use custom
        if HAS_BORUTA_LIB:
            logger.info("Using boruta library implementation")
            rf = RandomForestClassifier(
                n_estimators=self.ml_config.get("n_estimators", 100),
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
            boruta = BorutaPy(
                estimator=rf,
                n_estimators="auto",
                max_iter=50,
                random_state=self.random_state,
                verbose=0,
            )
            boruta.fit(X.values, y_encoded)

            # Get selected features
            selected_mask = boruta.support_
            selected_features = X.columns[selected_mask].tolist()

            # Get feature importances
            self.feature_scores_ = dict(zip(X.columns, boruta.ranking_))
        else:
            logger.info("Using custom Boruta implementation")
            # Run Boruta (custom implementation)
            boruta = BorutaFeatureSelector(
                n_estimators=self.ml_config.get("n_estimators", 100),
                max_iter=50,  # Reduced for faster execution
                random_state=self.random_state,
            )

            boruta.fit(X, y_encoded)
            selected_features = boruta.selected_features_

            # Limit features if requested
            if max_features and len(selected_features) > max_features:
                # Sort by importance and take top features
                feature_importance = sorted(
                    boruta.feature_importances_.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                selected_features = [f[0] for f in feature_importance[:max_features]]
                logger.info(f"Limited to top {max_features} features by importance")

            self.feature_scores_ = boruta.feature_importances_

        self.selection_method_ = "boruta"
        logger.info(f"Boruta selected {len(selected_features)} features")
        return selected_features

    def select_features_univariate(
        self, X: pd.DataFrame, y: pd.Series, k: int = 100
    ) -> List[str]:
        """
        Select top K features using univariate statistical tests.

        Args:
            X: Feature matrix
            y: Target labels
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        logger.info(f"Running univariate feature selection (k={k})...")

        # Encode target labels if they are strings
        if y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y

        # Use mutual information for feature selection
        selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        selector.fit(X, y_encoded)

        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        # Get feature scores
        self.feature_scores_ = dict(zip(X.columns, selector.scores_))
        self.selection_method_ = "univariate"

        logger.info(f"Selected {len(selected_features)} features")

        return selected_features

    def select_features_random_forest(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.01
    ) -> List[str]:
        """
        Select features using Random Forest importance.

        Args:
            X: Feature matrix
            y: Target labels
            threshold: Minimum importance threshold

        Returns:
            List of selected feature names
        """
        logger.info("Running Random Forest feature selection...")

        # Encode target labels if they are strings
        if y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y

        # Train random forest
        rf = RandomForestClassifier(
            n_estimators=self.ml_config.get("n_estimators", 100),
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y_encoded)

        # Get feature importances
        importances = rf.feature_importances_
        self.feature_scores_ = dict(zip(X.columns, importances))

        # Select features above threshold
        selected_features = [
            feature
            for feature, importance in self.feature_scores_.items()
            if importance >= threshold
        ]

        self.selection_method_ = "random_forest"

        logger.info(
            f"Selected {len(selected_features)} features above threshold {threshold}"
        )

        return selected_features

    def run_feature_selection(
        self, df: pd.DataFrame, method: str = "boruta", **kwargs
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run feature selection and return transformed dataset.

        Args:
            df: Input dataset
            method: Selection method ('boruta', 'univariate', 'random_forest')
            **kwargs: Additional arguments for selection method

        Returns:
            Tuple of (transformed dataset, selection summary)
        """
        logger.info(f"Starting feature selection with method: {method}")

        # Prepare data
        X, y, metadata_columns = self.prepare_data(df)

        # Select features based on method
        if method == "boruta":
            selected_features = self.select_features_boruta(X, y, **kwargs)
        elif method == "univariate":
            selected_features = self.select_features_univariate(X, y, **kwargs)
        elif method == "random_forest":
            selected_features = self.select_features_random_forest(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        self.selected_features_ = selected_features

        # Create transformed dataset
        metadata_df = df[metadata_columns + ["primary_bias_label"]]
        features_df = df[selected_features]
        transformed_df = pd.concat([metadata_df, features_df], axis=1)

        # Create summary
        summary = {
            "method": method,
            "original_features": X.shape[1],
            "selected_features": len(selected_features),
            "reduction_percentage": ((X.shape[1] - len(selected_features)) / X.shape[1])
            * 100,
            "selected_feature_names": selected_features,
            "feature_scores": self.feature_scores_,
        }

        logger.info(
            f"Feature selection complete: {X.shape[1]} -> {len(selected_features)} features "
            f"({summary['reduction_percentage']:.1f}% reduction)"
        )

        return transformed_df, summary

    def save_results(self, summary: Dict, output_path: str) -> None:
        """
        Save feature selection results.

        Args:
            summary: Selection summary dictionary
            output_path: Output file path
        """
        import json

        # Prepare summary for serialization
        summary_serializable = summary.copy()

        # Convert numpy types
        if (
            "feature_scores" in summary_serializable
            and summary_serializable["feature_scores"]
        ):
            summary_serializable["feature_scores"] = {
                k: float(v) if hasattr(v, "item") else v
                for k, v in summary_serializable["feature_scores"].items()
            }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(summary_serializable, f, indent=2)

        logger.info(f"Saved feature selection results to: {output_path}")


def run_feature_selection(
    config: Dict, method: str = "random_forest"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to run feature selection.

    Args:
        config: Configuration dictionary
        method: Selection method

    Returns:
        Tuple of (transformed dataset, selection summary)
    """
    # Load unified dataset
    dataset_path = os.path.join(
        config["paths"]["processed_data"], "unified_ml_dataset.csv"
    )
    df = pd.read_csv(dataset_path)

    # Initialize selector
    selector = FeatureSelector(config)

    # Run feature selection
    # Use random_forest instead of boruta for faster execution
    transformed_df, summary = selector.run_feature_selection(
        df, method=method, threshold=0.001
    )

    # Save transformed dataset
    output_path = os.path.join(
        config["paths"]["processed_data"], "ml_dataset_selected_features.csv"
    )
    transformed_df.to_csv(output_path, index=False)
    logger.info(f"Saved transformed dataset to: {output_path}")

    # Save feature selection summary
    summary_path = os.path.join(
        config["paths"]["processed_data"], "feature_selection_summary.json"
    )
    selector.save_results(summary, summary_path)

    return transformed_df, summary


if __name__ == "__main__":
    import yaml

    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run feature selection
    dataset, summary = run_feature_selection(config)

    print("\nFeature Selection Summary:")
    print(f"Method: {summary['method']}")
    print(f"Original features: {summary['original_features']}")
    print(f"Selected features: {summary['selected_features']}")
    print(f"Reduction: {summary['reduction_percentage']:.1f}%")
