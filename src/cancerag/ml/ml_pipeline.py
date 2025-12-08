"""
ML Pipeline Orchestrator

This module orchestrates the complete machine learning pipeline from
feature selection through model training, evaluation, and selection.
"""

import logging
import os
import time
from typing import Dict


from .feature_selection import run_feature_selection
from .model_evaluation import run_model_evaluation
from .model_training import run_model_training
from .preprocessing import run_preprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLPipeline:
    """
    Complete machine learning pipeline orchestrator.
    """

    def __init__(self, config: Dict):
        """
        Initialize ML pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}

    def run_complete_pipeline(self, skip_feature_selection: bool = False) -> Dict:
        """
        Run the complete ML pipeline.

        Args:
            skip_feature_selection: Whether to skip feature selection step

        Returns:
            Dictionary containing all pipeline results
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING MACHINE LEARNING PIPELINE")
        logger.info("=" * 80)

        pipeline_start_time = time.time()

        try:
            # Stage 1: Feature Selection (optional, can be slow)
            if not skip_feature_selection:
                logger.info("\n" + "-" * 80)
                logger.info("STAGE 1: FEATURE SELECTION")
                logger.info("-" * 80)

                feature_selection_start = time.time()

                # Use random_forest method for faster execution (Boruta is very slow)
                dataset, feature_summary = run_feature_selection(
                    self.config, method="random_forest"
                )

                feature_selection_time = time.time() - feature_selection_start

                self.results["feature_selection"] = {
                    "summary": feature_summary,
                    "execution_time": feature_selection_time,
                }

                logger.info(
                    f"✓ Feature selection completed in {feature_selection_time:.2f}s"
                )
                logger.info(
                    f"  Selected {feature_summary['selected_features']}/{feature_summary['original_features']} features"
                )

            else:
                logger.info("\n⊗ Skipping feature selection (using all features)")

            # Stage 2: Data Preprocessing
            logger.info("\n" + "-" * 80)
            logger.info("STAGE 2: DATA PREPROCESSING")
            logger.info("-" * 80)

            preprocessing_start = time.time()

            preprocessed_data = run_preprocessing(self.config)

            preprocessing_time = time.time() - preprocessing_start

            self.results["preprocessing"] = {
                "metadata": preprocessed_data["metadata"],
                "execution_time": preprocessing_time,
            }

            logger.info(f"✓ Preprocessing completed in {preprocessing_time:.2f}s")
            logger.info(
                f"  Training samples: {preprocessed_data['metadata']['n_samples_train']}"
            )
            logger.info(
                f"  Test samples: {preprocessed_data['metadata']['n_samples_test']}"
            )
            logger.info(f"  Features: {preprocessed_data['metadata']['n_features']}")
            logger.info(f"  Classes: {preprocessed_data['metadata']['n_classes']}")

            # Stage 3: Model Training
            logger.info("\n" + "-" * 80)
            logger.info("STAGE 3: MODEL TRAINING")
            logger.info("-" * 80)

            training_start = time.time()

            training_results = run_model_training(self.config, preprocessed_data)

            training_time = time.time() - training_start

            self.results["training"] = {
                "results": training_results,
                "execution_time": training_time,
            }

            logger.info(f"✓ Model training completed in {training_time:.2f}s")

            # Log training summary
            for model_key, result in training_results.items():
                if "error" not in result:
                    logger.info(f"  {result['model_name']}:")
                    logger.info(
                        f"    Test Accuracy: {result['test_metrics']['accuracy']:.4f}"
                    )
                    logger.info(
                        f"    Test F1 (weighted): {result['test_metrics']['f1_weighted']:.4f}"
                    )

            # Stage 4: Model Evaluation
            logger.info("\n" + "-" * 80)
            logger.info("STAGE 4: MODEL EVALUATION")
            logger.info("-" * 80)

            evaluation_start = time.time()

            # Load trained models for evaluation
            import pickle

            models = {}
            models_dir = self.config["paths"]["models"]

            for model_file in os.listdir(models_dir):
                if model_file.endswith(".pkl"):
                    model_key = model_file.replace(".pkl", "")
                    with open(os.path.join(models_dir, model_file), "rb") as f:
                        models[model_key] = pickle.load(f)

            evaluation_results = run_model_evaluation(
                self.config, models, preprocessed_data
            )

            evaluation_time = time.time() - evaluation_start

            self.results["evaluation"] = {
                "results": evaluation_results,
                "execution_time": evaluation_time,
            }

            logger.info(f"✓ Model evaluation completed in {evaluation_time:.2f}s")

            # Stage 5: Model Selection and Recommendation
            logger.info("\n" + "-" * 80)
            logger.info("STAGE 5: MODEL SELECTION")
            logger.info("-" * 80)

            best_model = self.select_best_model(training_results)

            self.results["best_model"] = best_model

            logger.info(f"✓ Best model selected: {best_model['model_name']}")
            logger.info(f"  Test Accuracy: {best_model['test_accuracy']:.4f}")
            logger.info(f"  Test F1 Score: {best_model['test_f1']:.4f}")
            logger.info(
                f"  Cross-Val F1: {best_model['cv_f1_mean']:.4f} (+/- {best_model['cv_f1_std']:.4f})"
            )

            # Calculate total pipeline time
            total_pipeline_time = time.time() - pipeline_start_time

            self.results["total_execution_time"] = total_pipeline_time

            logger.info("\n" + "=" * 80)
            logger.info("MACHINE LEARNING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(
                f"Total execution time: {total_pipeline_time:.2f}s ({total_pipeline_time / 60:.2f} minutes)"
            )

            # Save pipeline summary
            self.save_pipeline_summary()

            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            logger.exception("Full traceback:")
            raise

    def select_best_model(self, training_results: Dict) -> Dict:
        """
        Select the best performing model based on multiple criteria.

        Args:
            training_results: Dictionary of training results

        Returns:
            Dictionary with best model information
        """
        logger.info("Selecting best model based on test F1 score...")

        best_model = None
        best_f1 = -1

        for model_key, result in training_results.items():
            if "error" not in result:
                test_f1 = result["test_metrics"]["f1_weighted"]

                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_model = {
                        "model_key": model_key,
                        "model_name": result["model_name"],
                        "test_accuracy": result["test_metrics"]["accuracy"],
                        "test_f1": test_f1,
                        "train_accuracy": result["train_metrics"]["accuracy"],
                        "train_f1": result["train_metrics"]["f1_weighted"],
                        "training_time": result["training_info"]["training_time"],
                    }

        # Add cross-validation results if available
        if "evaluation" in self.results and best_model:
            model_key = best_model["model_key"]
            if model_key in self.results["evaluation"]["results"]:
                cv_results = self.results["evaluation"]["results"][model_key].get(
                    "cross_validation", {}
                )
                if "f1_macro" in cv_results:
                    best_model["cv_f1_mean"] = cv_results["f1_macro"]["mean"]
                    best_model["cv_f1_std"] = cv_results["f1_macro"]["std"]

        return best_model

    def save_pipeline_summary(self) -> None:
        """
        Save a comprehensive summary of the pipeline execution.
        """
        import json

        logger.info("Saving pipeline summary...")

        summary = {
            "pipeline_execution_time": self.results.get("total_execution_time", 0),
            "stages": {},
        }

        # Feature selection
        if "feature_selection" in self.results:
            summary["stages"]["feature_selection"] = {
                "execution_time": self.results["feature_selection"]["execution_time"],
                "original_features": self.results["feature_selection"]["summary"][
                    "original_features"
                ],
                "selected_features": self.results["feature_selection"]["summary"][
                    "selected_features"
                ],
            }

        # Preprocessing
        if "preprocessing" in self.results:
            summary["stages"]["preprocessing"] = {
                "execution_time": self.results["preprocessing"]["execution_time"],
                "train_samples": self.results["preprocessing"]["metadata"][
                    "n_samples_train"
                ],
                "test_samples": self.results["preprocessing"]["metadata"][
                    "n_samples_test"
                ],
                "features": self.results["preprocessing"]["metadata"]["n_features"],
                "classes": self.results["preprocessing"]["metadata"]["n_classes"],
            }

        # Training
        if "training" in self.results:
            summary["stages"]["training"] = {
                "execution_time": self.results["training"]["execution_time"],
                "models_trained": len(self.results["training"]["results"]),
            }

        # Evaluation
        if "evaluation" in self.results:
            summary["stages"]["evaluation"] = {
                "execution_time": self.results["evaluation"]["execution_time"]
            }

        # Best model
        if "best_model" in self.results:
            summary["best_model"] = self.results["best_model"]

        # Save to file
        output_path = os.path.join(
            self.config["paths"]["reports"], "ml_pipeline_summary.json"
        )

        def convert_types(obj):
            import numpy as np

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

        summary_serializable = convert_types(summary)

        with open(output_path, "w") as f:
            json.dump(summary_serializable, f, indent=2)

        logger.info(f"Saved pipeline summary to: {output_path}")


def run_ml_pipeline(config: Dict, skip_feature_selection: bool = True) -> Dict:
    """
    Main function to run the complete ML pipeline.

    Args:
        config: Configuration dictionary
        skip_feature_selection: Whether to skip feature selection (default True for speed)

    Returns:
        Pipeline results dictionary
    """
    pipeline = MLPipeline(config)
    return pipeline.run_complete_pipeline(skip_feature_selection=skip_feature_selection)


if __name__ == "__main__":
    import yaml

    # Load configuration
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Run ML pipeline
    # Skip feature selection by default for faster execution
    results = run_ml_pipeline(config, skip_feature_selection=True)

    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    if "best_model" in results:
        print(f"\nBest Model: {results['best_model']['model_name']}")
        print(f"  Test Accuracy: {results['best_model']['test_accuracy']:.4f}")
        print(f"  Test F1 Score: {results['best_model']['test_f1']:.4f}")

    print(f"\nTotal Execution Time: {results['total_execution_time']:.2f}s")
    print("=" * 80)
