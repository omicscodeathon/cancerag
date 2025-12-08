"""
Model Training Module for Machine Learning Pipeline

This module implements training for multiple classification algorithms
including Logistic Regression, Random Forest, XGBoost, and CatBoost.
"""

import logging
import os
import pickle
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Optional advanced models
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates multiple machine learning models.
    """

    def __init__(self, config: Dict):
        """
        Initialize model trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get("ml_model", {})
        self.random_state = self.ml_config.get("random_state", 42)
        self.n_estimators = self.ml_config.get("n_estimators", 100)
        self.max_depth = self.ml_config.get("max_depth", 10)

        self.models = {}
        self.results = {}

    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get configuration for all models.

        Returns:
            Dictionary of model configurations
        """
        configs = {
            "logistic_regression": {
                "model": LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                "name": "Logistic Regression",
            },
            "random_forest": {
                "model": RandomForestClassifier(
                    n_estimators=500,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    max_features="log2",
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                "name": "Random Forest",
            },
            "random_forest_deep": {
                "model": RandomForestClassifier(
                    n_estimators=500,
                    max_depth=25,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                "name": "Random Forest (Deep)",
            },
            "extra_trees": {
                "model": ExtraTreesClassifier(
                    n_estimators=500,
                    max_depth=22,
                    min_samples_split=6,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                "name": "Extra Trees",
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(
                    n_estimators=400,
                    max_depth=18,
                    learning_rate=0.03,
                    subsample=0.85,
                    min_samples_split=8,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                ),
                "name": "Gradient Boosting",
            },
        }

        # Add XGBoost if available
        if HAS_XGBOOST:
            configs["xgboost"] = {
                "model": xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=18,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=4,
                    gamma=0.1,
                    random_state=self.random_state,
                    n_jobs=-1,
                    tree_method="hist",
                    eval_metric="mlogloss",
                ),
                "name": "XGBoost",
            }

        # Add LightGBM if available
        if HAS_LIGHTGBM:
            configs["lightgbm"] = {
                "model": lgb.LGBMClassifier(
                    n_estimators=400,
                    max_depth=18,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_samples=4,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1,
                    class_weight="balanced",
                ),
                "name": "LightGBM",
            }

        # Add CatBoost if available
        if HAS_CATBOOST:
            configs["catboost"] = {
                "model": cb.CatBoostClassifier(
                    iterations=self.n_estimators,
                    depth=self.max_depth,
                    random_seed=self.random_state,
                    learning_rate=0.1,
                    verbose=False,
                    class_weights="Balanced",
                ),
                "name": "CatBoost",
            }

        return configs

    def train_model(
        self, model: Any, X_train: pd.DataFrame, y_train: np.ndarray, model_name: str
    ) -> Tuple[Any, Dict]:
        """
        Train a single model.

        Args:
            model: Model instance
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model

        Returns:
            Tuple of (trained model, training info)
        """
        logger.info(f"Training {model_name}...")

        start_time = time.time()

        # Train model
        model.fit(X_train, y_train)

        training_time = time.time() - start_time

        logger.info(f"{model_name} trained in {training_time:.2f} seconds")

        training_info = {
            "training_time": training_time,
            "n_samples": X_train.shape[0],
            "n_features": X_train.shape[1],
        }

        return model, training_info

    def evaluate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        label_mapping: Dict,
        dataset_name: str = "test",
    ) -> Dict:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            label_mapping: Mapping of label names to indices
            dataset_name: Name of dataset being evaluated

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {dataset_name} set...")

        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = (
            model.predict_proba(X) if hasattr(model, "predict_proba") else None
        )

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision_macro = precision_score(y, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y, y_pred, average="macro", zero_division=0)

        precision_weighted = precision_score(
            y, y_pred, average="weighted", zero_division=0
        )
        recall_weighted = recall_score(y, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y, y_pred, average="weighted", zero_division=0)

        # Calculate ROC-AUC for multiclass
        try:
            if y_pred_proba is not None and len(np.unique(y)) > 2:
                roc_auc = roc_auc_score(
                    y, y_pred_proba, multi_class="ovr", average="macro"
                )
            elif y_pred_proba is not None:
                roc_auc = roc_auc_score(y, y_pred_proba[:, 1])
            else:
                roc_auc = None
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Classification report
        class_names = sorted(label_mapping.keys(), key=lambda x: label_mapping[x])
        report = classification_report(
            y, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        logger.info(f"{dataset_name.capitalize()} Set Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")
        if roc_auc is not None:
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")

        return metrics

    def get_feature_importance(
        self, model: Any, feature_names: List[str], model_name: str
    ) -> Dict[str, float]:
        """
        Extract feature importance from model.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model

        Returns:
            Dictionary of feature importances
        """
        importance_dict = {}

        try:
            if hasattr(model, "feature_importances_"):
                # Tree-based models
                importances = model.feature_importances_
                importance_dict = dict(zip(feature_names, importances.tolist()))

                # Log top 10 features
                top_features = sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )[:10]
                logger.info(f"Top 10 features for {model_name}:")
                for feat, imp in top_features:
                    logger.info(f"  {feat}: {imp:.4f}")

            elif hasattr(model, "coef_"):
                # Linear models
                coef = (
                    np.abs(model.coef_).mean(axis=0)
                    if len(model.coef_.shape) > 1
                    else np.abs(model.coef_)
                )
                importance_dict = dict(zip(feature_names, coef.tolist()))

                # Log top 10 features
                top_features = sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )[:10]
                logger.info(
                    f"Top 10 features for {model_name} (by coefficient magnitude):"
                )
                for feat, imp in top_features:
                    logger.info(f"  {feat}: {imp:.4f}")

        except Exception as e:
            logger.warning(
                f"Could not extract feature importance for {model_name}: {e}"
            )

        return importance_dict

    def train_and_evaluate_all(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        label_mapping: Dict,
        feature_names: List[str],
    ) -> Dict:
        """
        Train and evaluate all models.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            label_mapping: Label name to index mapping
            feature_names: List of feature names

        Returns:
            Dictionary of all results
        """
        logger.info("Starting model training and evaluation...")

        model_configs = self.get_model_configs()
        all_results = {}

        for model_key, model_config in model_configs.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {model_config['name']}")
            logger.info(f"{'=' * 60}")

            try:
                # Train model
                model, training_info = self.train_model(
                    model_config["model"], X_train, y_train, model_config["name"]
                )

                # Evaluate on training set
                train_metrics = self.evaluate_model(
                    model, X_train, y_train, label_mapping, "train"
                )

                # Evaluate on test set
                test_metrics = self.evaluate_model(
                    model, X_test, y_test, label_mapping, "test"
                )

                # Get feature importance
                feature_importance = self.get_feature_importance(
                    model, feature_names, model_config["name"]
                )

                # Store model and results
                self.models[model_key] = model

                all_results[model_key] = {
                    "model_name": model_config["name"],
                    "training_info": training_info,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "feature_importance": feature_importance,
                }

                logger.info(f"✓ {model_config['name']} completed successfully")

            except Exception as e:
                logger.error(f"✗ {model_config['name']} failed: {e}")
                all_results[model_key] = {
                    "model_name": model_config["name"],
                    "error": str(e),
                }

        self.results = all_results

        # Create ensemble models from best base models
        logger.info("\n" + "=" * 60)
        logger.info("CREATING ENSEMBLE MODELS")
        logger.info("=" * 60)

        ensemble_results = self.create_ensembles(
            X_train, y_train, X_test, y_test, label_mapping, feature_names
        )

        # Add ensemble results to all_results
        all_results.update(ensemble_results)
        self.results = all_results

        logger.info("\n" + "=" * 60)
        logger.info("All models trained and evaluated")
        logger.info("=" * 60)

        return all_results

    def create_ensembles(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        label_mapping: Dict,
        feature_names: List[str],
    ) -> Dict:
        """
        Create ensemble models from trained base models.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            label_mapping: Label name to index mapping
            feature_names: List of feature names

        Returns:
            Dictionary of ensemble results
        """
        # Select best base models for ensemble (top 3 by accuracy)
        base_model_results = [
            (k, v)
            for k, v in self.results.items()
            if "error" not in v and "voting" not in k and "stacking" not in k
        ]
        base_model_results.sort(
            key=lambda x: x[1]["test_metrics"]["accuracy"], reverse=True
        )
        top_models = base_model_results[:3]

        if len(top_models) < 2:
            logger.warning("Not enough models for ensemble, skipping...")
            return {}

        ensemble_results = {}

        # Voting Classifier
        logger.info("\nCreating Voting Classifier...")
        try:
            estimators = [(k, self.models[k]) for k, _ in top_models]
            voting_clf = VotingClassifier(
                estimators=estimators, voting="soft", n_jobs=-1
            )
            voting_clf.fit(X_train, y_train)

            # Evaluate
            train_metrics = self.evaluate_model(
                voting_clf, X_train, y_train, label_mapping, "train"
            )
            test_metrics = self.evaluate_model(
                voting_clf, X_test, y_test, label_mapping, "test"
            )

            self.models["voting_ensemble"] = voting_clf
            ensemble_results["voting_ensemble"] = {
                "model_name": "Voting Ensemble",
                "training_info": {
                    "training_time": 0,
                    "n_samples": len(X_train),
                    "n_features": len(feature_names),
                },
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_importance": {},
            }
            logger.info("✓ Voting Ensemble created and evaluated")
        except Exception as e:
            logger.warning(f"Could not create voting ensemble: {e}")

        # Stacking Classifier with Logistic Regression meta-learner
        logger.info("\nCreating Stacking Classifier (LR meta-learner)...")
        try:
            estimators = [(k, self.models[k]) for k, _ in top_models]
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(
                    random_state=self.random_state,
                    max_iter=2000,
                    class_weight="balanced",
                    C=1.0,
                ),
                cv=5,
                n_jobs=-1,
            )
            stacking_clf.fit(X_train, y_train)

            # Evaluate
            train_metrics = self.evaluate_model(
                stacking_clf, X_train, y_train, label_mapping, "train"
            )
            test_metrics = self.evaluate_model(
                stacking_clf, X_test, y_test, label_mapping, "test"
            )

            self.models["stacking_ensemble"] = stacking_clf
            ensemble_results["stacking_ensemble"] = {
                "model_name": "Stacking Ensemble (LR)",
                "training_info": {
                    "training_time": 0,
                    "n_samples": len(X_train),
                    "n_features": len(feature_names),
                },
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_importance": {},
            }
            logger.info("✓ Stacking Ensemble created and evaluated")
        except Exception as e:
            logger.warning(f"Could not create stacking ensemble: {e}")

        return ensemble_results

    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to disk.

        Args:
            output_dir: Output directory for models
        """
        logger.info("Saving trained models...")

        os.makedirs(output_dir, exist_ok=True)

        for model_key, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_key}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            logger.info(f"Saved {model_key} to: {model_path}")

    def save_results(self, output_dir: str) -> None:
        """
        Save training results to disk.

        Args:
            output_dir: Output directory for results
        """
        import json

        logger.info("Saving training results...")

        os.makedirs(output_dir, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        results_serializable = convert_types(self.results)

        results_path = os.path.join(output_dir, "training_results.json")
        with open(results_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Saved results to: {results_path}")

        # Create summary table
        summary_data = []
        for model_key, result in self.results.items():
            if "error" not in result:
                summary_data.append(
                    {
                        "Model": result["model_name"],
                        "Train Accuracy": result["train_metrics"]["accuracy"],
                        "Test Accuracy": result["test_metrics"]["accuracy"],
                        "Test F1 (macro)": result["test_metrics"]["f1_macro"],
                        "Test F1 (weighted)": result["test_metrics"]["f1_weighted"],
                        "Training Time (s)": result["training_info"]["training_time"],
                    }
                )

        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, "model_comparison_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        logger.info(f"Saved summary to: {summary_path}")
        logger.info("\nModel Comparison Summary:")
        logger.info(summary_df.to_string(index=False))


def run_model_training(config: Dict, preprocessed_data: Dict = None) -> Dict:
    """
    Main function to run model training.

    Args:
        config: Configuration dictionary
        preprocessed_data: Preprocessed data dictionary (optional)

    Returns:
        Training results dictionary
    """
    # Load preprocessed data if not provided
    if preprocessed_data is None:
        logger.info("Loading preprocessed data...")

        preprocessed_dir = os.path.join(
            config["paths"]["processed_data"], "ml_preprocessed"
        )

        X_train = pd.read_csv(os.path.join(preprocessed_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(preprocessed_dir, "X_test.csv"))
        y_train = np.load(os.path.join(preprocessed_dir, "y_train.npy"))
        y_test = np.load(os.path.join(preprocessed_dir, "y_test.npy"))

        import json

        with open(
            os.path.join(preprocessed_dir, "preprocessing_metadata.json"), "r"
        ) as f:
            metadata = json.load(f)

        feature_names = metadata["feature_columns"]
        label_mapping = metadata["label_mapping"]

    else:
        X_train = preprocessed_data["X_train"]
        X_test = preprocessed_data["X_test"]
        y_train = preprocessed_data["y_train"]
        y_test = preprocessed_data["y_test"]
        feature_names = preprocessed_data["feature_columns"]
        label_mapping = preprocessed_data["label_mapping"]

    # Initialize trainer
    trainer = ModelTrainer(config)

    # Train and evaluate all models
    results = trainer.train_and_evaluate_all(
        X_train, X_test, y_train, y_test, label_mapping, feature_names
    )

    # Save models
    models_dir = config["paths"]["models"]
    trainer.save_models(models_dir)

    # Save results
    results_dir = config["paths"]["reports"]
    trainer.save_results(results_dir)

    return results


if __name__ == "__main__":
    import yaml

    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run model training
    results = run_model_training(config)

    print("\n" + "=" * 60)
    print("Model Training Complete")
    print("=" * 60)
