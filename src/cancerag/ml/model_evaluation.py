"""
Model Evaluation Module for Machine Learning Pipeline

This module implements comprehensive model evaluation including cross-validation,
SHAP analysis for interpretability, and detailed performance metrics.
"""

import logging
import os
import pickle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation including cross-validation and interpretability.
    """

    def __init__(self, config: Dict):
        """
        Initialize model evaluator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get("ml_model", {})
        self.random_state = self.ml_config.get("random_state", 42)
        self.figures_dir = config["paths"]["figures"]

        os.makedirs(self.figures_dir, exist_ok=True)

    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        cv: int = 5,
        model_name: str = "Model",
    ) -> Dict:
        """
        Perform cross-validation on a model.

        Args:
            model: Model instance
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            model_name: Name of the model

        Returns:
            Dictionary of cross-validation results
        """
        logger.info(f"Running {cv}-fold cross-validation for {model_name}...")

        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        # Calculate different metrics
        scoring_metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

        cv_results = {}

        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                "scores": scores.tolist(),
                "mean": scores.mean(),
                "std": scores.std(),
            }

            logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_results

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        model_name: str,
        output_path: str = None,
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: List of class names
            model_name: Name of the model
            output_path: Output file path
        """
        plt.figure(figsize=(10, 8))

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Create heatmap
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()

        if output_path is None:
            output_path = os.path.join(
                self.figures_dir,
                f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png",
            )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved confusion matrix to: {output_path}")

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        model_name: str,
        top_n: int = 20,
        output_path: str = None,
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_importance: Dictionary of feature importances
            model_name: Name of the model
            top_n: Number of top features to show
            output_path: Output file path
        """
        if not feature_importance:
            logger.warning(f"No feature importance available for {model_name}")
            return

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.tight_layout()

        if output_path is None:
            output_path = os.path.join(
                self.figures_dir,
                f"feature_importance_{model_name.replace(' ', '_').lower()}.png",
            )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved feature importance plot to: {output_path}")

    def plot_roc_curves(
        self,
        models_data: Dict,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        label_mapping: Dict,
        output_path: str = None,
    ) -> None:
        """
        Plot ROC curves for all models.

        Args:
            models_data: Dictionary of model data
            X_test: Test features
            y_test: Test labels
            label_mapping: Label mapping dictionary
            output_path: Output file path
        """
        logger.info("Plotting ROC curves...")

        n_classes = len(label_mapping)

        # For binary classification
        if n_classes == 2:
            plt.figure(figsize=(10, 8))

            for model_key, model in models_data.items():
                if hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)

                    plt.plot(fpr, tpr, label=f"{model_key} (AUC = {roc_auc:.2f})")

            plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves - Model Comparison")
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if output_path is None:
                output_path = os.path.join(
                    self.figures_dir, "roc_curves_comparison.png"
                )

            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved ROC curves to: {output_path}")

        else:
            logger.info("Multi-class ROC curves not plotted (use per-class analysis)")

    def analyze_model_performance(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        label_mapping: Dict,
        model_name: str,
    ) -> Dict:
        """
        Comprehensive performance analysis for a model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            label_mapping: Label mapping
            model_name: Name of the model

        Returns:
            Dictionary of analysis results
        """
        logger.info(f"Analyzing performance for {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        _ = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }

        # Per-class metrics
        class_names = sorted(label_mapping.keys(), key=lambda x: label_mapping[x])
        per_class_metrics = {}

        for idx, class_name in enumerate(class_names):
            mask = y_test == idx
            if mask.sum() > 0:
                _ = y_pred == idx
                per_class_metrics[class_name] = {
                    "precision": precision_score(
                        y_test == idx, y_pred == idx, zero_division=0
                    ),
                    "recall": recall_score(
                        y_test == idx, y_pred == idx, zero_division=0
                    ),
                    "f1": f1_score(y_test == idx, y_pred == idx, zero_division=0),
                    "support": mask.sum(),
                }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        analysis = {
            "metrics": metrics,
            "per_class_metrics": per_class_metrics,
            "confusion_matrix": cm.tolist(),
        }

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, class_names, model_name)

        return analysis

    def run_shap_analysis(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        feature_names: List[str],
        model_name: str,
        max_samples: int = 100,
    ) -> Dict:
        """
        Run SHAP analysis for model interpretability.

        Note: This is a simplified version. For full SHAP analysis,
        install shap package: pip install shap

        Args:
            model: Trained model
            X_sample: Sample of features for SHAP
            feature_names: List of feature names
            model_name: Name of the model
            max_samples: Maximum samples for SHAP analysis

        Returns:
            Dictionary of SHAP analysis results
        """
        logger.info(f"Running simplified interpretability analysis for {model_name}...")

        try:
            # Try importing SHAP
            import shap

            # Limit sample size for performance
            if X_sample.shape[0] > max_samples:
                sample_indices = np.random.choice(
                    X_sample.shape[0], max_samples, replace=False
                )
                X_shap = X_sample.iloc[sample_indices]
            else:
                X_shap = X_sample

            # Create explainer based on model type
            if hasattr(model, "tree_"):
                # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:
                # Use Kernel explainer for other models
                explainer = shap.KernelExplainer(model.predict_proba, X_shap)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_shap)

            # Get mean absolute SHAP values
            if isinstance(shap_values, list):
                # Multi-class
                mean_shap = np.mean(
                    [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
                )
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)

            shap_importance = dict(zip(feature_names, mean_shap))

            # Save SHAP summary plot
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                shap.summary_plot(
                    shap_values[0],
                    X_shap,
                    feature_names=feature_names,
                    show=False,
                    max_display=20,
                )
            else:
                shap.summary_plot(
                    shap_values,
                    X_shap,
                    feature_names=feature_names,
                    show=False,
                    max_display=20,
                )

            output_path = os.path.join(
                self.figures_dir,
                f"shap_summary_{model_name.replace(' ', '_').lower()}.png",
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved SHAP summary plot to: {output_path}")

            return {"shap_importance": shap_importance}

        except ImportError:
            logger.warning("SHAP package not installed. Skipping SHAP analysis.")
            logger.info("To enable SHAP analysis, install: pip install shap")
            return {}
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
            return {}

    def compare_models(self, results: Dict, output_path: str = None) -> None:
        """
        Create comprehensive model comparison visualizations.

        Args:
            results: Dictionary of model results
            output_path: Output directory path
        """
        logger.info("Creating model comparison visualizations...")

        if output_path is None:
            output_path = self.figures_dir

        # Extract metrics for comparison
        model_names = []
        test_accuracy = []
        test_f1 = []
        train_time = []

        for model_key, result in results.items():
            if "error" not in result:
                model_names.append(result["model_name"])
                test_accuracy.append(result["test_metrics"]["accuracy"])
                test_f1.append(result["test_metrics"]["f1_weighted"])
                train_time.append(result["training_info"]["training_time"])

        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Accuracy comparison
        axes[0].bar(model_names, test_accuracy, color="skyblue")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Test Accuracy Comparison")
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis="x", rotation=45)

        # F1 Score comparison
        axes[1].bar(model_names, test_f1, color="lightgreen")
        axes[1].set_ylabel("F1 Score (Weighted)")
        axes[1].set_title("Test F1 Score Comparison")
        axes[1].set_ylim([0, 1])
        axes[1].tick_params(axis="x", rotation=45)

        # Training time comparison
        axes[2].bar(model_names, train_time, color="lightcoral")
        axes[2].set_ylabel("Training Time (seconds)")
        axes[2].set_title("Training Time Comparison")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        comparison_path = os.path.join(output_path, "model_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved model comparison to: {comparison_path}")


def run_model_evaluation(
    config: Dict, models: Dict = None, preprocessed_data: Dict = None
) -> Dict:
    """
    Main function to run comprehensive model evaluation.

    Args:
        config: Configuration dictionary
        models: Dictionary of trained models (optional)
        preprocessed_data: Preprocessed data dictionary (optional)

    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(config)

    # Load models if not provided
    if models is None:
        logger.info("Loading trained models...")
        models = {}
        models_dir = config["paths"]["models"]

        for model_file in os.listdir(models_dir):
            if model_file.endswith(".pkl"):
                model_key = model_file.replace(".pkl", "")
                with open(os.path.join(models_dir, model_file), "rb") as f:
                    models[model_key] = pickle.load(f)

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

    # Run evaluation for each model
    evaluation_results = {}

    for model_key, model in models.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating {model_key}")
        logger.info(f"{'=' * 60}")

        # Cross-validation
        cv_results = evaluator.cross_validate_model(
            model, X_train, y_train, cv=5, model_name=model_key
        )

        # Performance analysis
        performance_analysis = evaluator.analyze_model_performance(
            model, X_test, y_test, label_mapping, model_key
        )

        # SHAP analysis
        shap_results = evaluator.run_shap_analysis(
            model, X_train, feature_names, model_key, max_samples=100
        )

        evaluation_results[model_key] = {
            "cross_validation": cv_results,
            "performance_analysis": performance_analysis,
            "shap_analysis": shap_results,
        }

    # Plot ROC curves
    evaluator.plot_roc_curves(models, X_test, y_test, label_mapping)

    # Load training results for comparison
    results_path = os.path.join(config["paths"]["reports"], "training_results.json")
    if os.path.exists(results_path):
        import json

        with open(results_path, "r") as f:
            training_results = json.load(f)
        evaluator.compare_models(training_results)

    # Save evaluation results
    output_path = os.path.join(config["paths"]["reports"], "evaluation_results.json")
    import json

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

    evaluation_results_serializable = convert_types(evaluation_results)

    with open(output_path, "w") as f:
        json.dump(evaluation_results_serializable, f, indent=2)

    logger.info(f"Saved evaluation results to: {output_path}")

    return evaluation_results


# =====================================================================
# Stage 10 — reporting suite (bootstrap CIs, per-receptor metrics).
# Macro-F1 leads (Reviewer 2 explicitly demanded this on the imbalanced
# 5-class problem). Accuracy is intentionally absent from `report_metrics`.
# =====================================================================


from typing import Callable  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Callable[..., float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    **metric_kwargs,
) -> tuple[float, float, float]:
    """Return ``(lower_2.5%, median, upper_97.5%)`` bootstrap CI for ``metric``."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            scores[i] = metric(y_true[idx], y_pred[idx], **metric_kwargs)
        except Exception:
            scores[i] = float("nan")
    lo, mid, hi = np.nanpercentile(scores, [2.5, 50.0, 97.5])
    return float(lo), float(mid), float(hi)


def report_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> dict:
    """Headline metrics for a held-out evaluation."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1_lo, macro_f1_med, macro_f1_hi = bootstrap_ci(
        y_true, y_pred, f1_score,
        n_boot=bootstrap_n, seed=seed, average="macro",
    )
    bal_acc_lo, bal_acc_med, bal_acc_hi = bootstrap_ci(
        y_true, y_pred, balanced_accuracy_score, n_boot=bootstrap_n, seed=seed
    )
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "macro_f1": {
            "point_estimate": float(
                f1_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "ci_lo": macro_f1_lo,
            "ci_median": macro_f1_med,
            "ci_hi": macro_f1_hi,
        },
        "balanced_accuracy": {
            "point_estimate": float(balanced_accuracy_score(y_true, y_pred)),
            "ci_lo": bal_acc_lo,
            "ci_median": bal_acc_med,
            "ci_hi": bal_acc_hi,
        },
        "per_class_f1": {
            str(c): float(s)
            for c, s in zip(
                classes,
                f1_score(
                    y_true, y_pred, average=None, zero_division=0, labels=classes
                ),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": int(len(y_true)),
    }


def per_receptor_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    receptor_ids: np.ndarray,
    *,
    min_samples: int = 5,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    receptor_ids = np.asarray(receptor_ids)
    rows = []
    for r in pd.unique(receptor_ids):
        mask = receptor_ids == r
        if int(mask.sum()) < min_samples:
            continue
        rows.append({
            "receptor": r,
            "n": int(mask.sum()),
            "macro_f1": float(
                f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_true[mask], y_pred[mask])
            ),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import yaml

    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run evaluation
    results = run_model_evaluation(config)

    print("\n" + "=" * 60)
    print("Model Evaluation Complete")
    print("=" * 60)
