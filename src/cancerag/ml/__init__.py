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

from .dataset_assembly import DatasetAssembler, run_dataset_assembly
from .feature_selection import FeatureSelector, run_feature_selection
from .ml_pipeline import MLPipeline, run_ml_pipeline
from .model_evaluation import ModelEvaluator, run_model_evaluation
from .model_training import ModelTrainer, run_model_training
from .preprocessing import DataPreprocessor, run_preprocessing

__all__ = [
    "run_dataset_assembly",
    "DatasetAssembler",
    "run_feature_selection",
    "FeatureSelector",
    "run_preprocessing",
    "DataPreprocessor",
    "run_model_training",
    "ModelTrainer",
    "run_model_evaluation",
    "ModelEvaluator",
    "run_ml_pipeline",
    "MLPipeline",
]
