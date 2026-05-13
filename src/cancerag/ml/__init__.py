"""
Machine Learning Module for CancerAg Pipeline

This module provides complete machine learning functionality including:
- Feature selection (Boruta, Random Forest, Univariate)
- Data preprocessing and scaling
- Model training (Logistic Regression, Random Forest, Gradient Boosting)
- Model evaluation with cross-validation
- SHAP analysis for interpretability
- Model comparison and selection
"""

from .dataset_assembly import run_dataset_assembly
from .feature_selection import run_feature_selection
from .ml_pipeline import MLPipeline, run_ml_pipeline
from .model_evaluation import ModelEvaluator, run_model_evaluation
from .model_training import run_model_training
from .preprocessing import run_preprocessing

__all__ = [
    "run_dataset_assembly",
    "run_feature_selection",
    "run_preprocessing",
    "run_model_training",
    "run_model_evaluation",
    "ModelEvaluator",
    "run_ml_pipeline",
    "MLPipeline",
]
