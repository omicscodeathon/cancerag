# Machine Learning Module

This module implements the complete machine learning pipeline for predicting GPCR signaling bias.

## Overview

The ML module consists of 6 main components:

1. **Dataset Assembly** - Combines molecular descriptors, docking results, and bias labels
2. **Feature Selection** - Identifies the most important features using multiple methods
3. **Data Preprocessing** - Handles missing values, scaling, and train-test splitting
4. **Model Training** - Trains multiple classification algorithms
5. **Model Evaluation** - Comprehensive evaluation with cross-validation and SHAP analysis
6. **ML Pipeline** - Orchestrates the entire workflow

## Components

### 1. Dataset Assembly (`dataset_assembly.py`)

Merges data from multiple sources into a unified ML dataset:
- Molecular descriptors (218 features)
- Docking affinities (36 receptor features)
- Bias labels and metadata

**Usage:**
```python
from cancerag.ml import run_dataset_assembly

dataset, summary = run_dataset_assembly(config)
```

### 2. Feature Selection (`feature_selection.py`)

Implements multiple feature selection methods:
- **Boruta Algorithm**: Iterative feature selection using shadow features
- **Random Forest Importance**: Fast tree-based selection (recommended)
- **Univariate Selection**: Statistical tests (f_classif, mutual_info)

**Usage:**
```python
from cancerag.ml import run_feature_selection

# Fast method (recommended)
dataset, summary = run_feature_selection(config, method='random_forest')

# Slow but thorough method
dataset, summary = run_feature_selection(config, method='boruta')
```

### 3. Data Preprocessing (`preprocessing.py`)

Handles all preprocessing steps:
- Missing value imputation (median strategy)
- Feature scaling (standardization)
- Train-test splitting (stratified)
- Class imbalance analysis
- Label encoding

**Usage:**
```python
from cancerag.ml import run_preprocessing

preprocessed_data = run_preprocessing(config)
# Returns: X_train, X_test, y_train, y_test, metadata
```

### 4. Model Training (`model_training.py`)

Trains multiple classification models:
- **Logistic Regression** (baseline, fast)
- **Random Forest** (robust, interpretable)
- **Gradient Boosting** (high performance)

All models use `class_weight='balanced'` to handle class imbalance.

**Usage:**
```python
from cancerag.ml import run_model_training

results = run_model_training(config, preprocessed_data)
```

### 5. Model Evaluation (`model_evaluation.py`)

Comprehensive evaluation including:
- **Cross-validation**: 5-fold stratified CV with multiple metrics
- **Performance analysis**: Accuracy, precision, recall, F1, ROC-AUC
- **Confusion matrices**: Visualization of classification errors
- **Feature importance**: Model-specific importance plots
- **SHAP analysis**: Interpretability analysis (optional, requires `shap` package)

**Usage:**
```python
from cancerag.ml import run_model_evaluation

evaluation_results = run_model_evaluation(config, models, preprocessed_data)
```

### 6. ML Pipeline (`ml_pipeline.py`)

Orchestrates the complete workflow:
1. Feature selection (optional)
2. Data preprocessing
3. Model training
4. Model evaluation
5. Model selection
6. Results summary

**Usage:**
```python
from cancerag.ml import run_ml_pipeline

# Run complete pipeline (skip feature selection for speed)
results = run_ml_pipeline(config, skip_feature_selection=True)

# Run with feature selection (slower)
results = run_ml_pipeline(config, skip_feature_selection=False)
```

## Running the ML Pipeline

### Option 1: From Main Pipeline

The ML pipeline is integrated into the main CancerAg pipeline:

```bash
python src/cancerag/main.py
```

### Option 2: Standalone Script

Run only the ML components (requires existing dataset):

```bash
# Fast mode (without feature selection)
python run_ml_pipeline.py

# With feature selection (slower)
python run_ml_pipeline.py --with-feature-selection
```

### Option 3: Python API

```python
import yaml
from cancerag.ml import run_ml_pipeline

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Run pipeline
results = run_ml_pipeline(config, skip_feature_selection=True)

# Access results
best_model = results['best_model']
print(f"Best model: {best_model['model_name']}")
print(f"Test F1 Score: {best_model['test_f1']:.4f}")
```

## Output Files

The ML pipeline generates the following outputs:

### Models (`results/models/`)
- `logistic_regression.pkl` - Trained logistic regression model
- `random_forest.pkl` - Trained random forest model
- `gradient_boosting.pkl` - Trained gradient boosting model

### Results (`results/reports/`)
- `training_results.json` - Detailed training metrics for all models
- `evaluation_results.json` - Cross-validation and evaluation results
- `model_comparison_summary.csv` - Side-by-side model comparison
- `ml_pipeline_summary.json` - Complete pipeline execution summary

### Figures (`results/figures/`)
- `model_comparison.png` - Bar charts comparing model performance
- `confusion_matrix_*.png` - Confusion matrices for each model
- `feature_importance_*.png` - Feature importance plots
- `shap_summary_*.png` - SHAP analysis plots (if available)
- `roc_curves_comparison.png` - ROC curves for all models

### Preprocessed Data (`data/processed/ml_preprocessed/`)
- `X_train.csv` - Training features
- `X_test.csv` - Test features
- `y_train.npy` - Training labels
- `y_test.npy` - Test labels
- `preprocessing_metadata.json` - Preprocessing configuration
- `label_encoder.pkl` - Label encoder for inverse transform
- `scaler.pkl` - Feature scaler for new predictions

## Configuration

Add ML-specific configuration to `configs/config.yaml`:

```yaml
ml_model:
  test_size: 0.2          # Test set size (20%)
  random_state: 42        # Random seed for reproducibility
  n_estimators: 100       # Number of trees for forest models
  max_depth: 10           # Maximum tree depth
```

## Performance Expectations

Based on the CancerAg dataset (727 samples, 908 features):

| Stage | Execution Time | Output |
|-------|---------------|--------|
| Feature Selection | 5-30 min | ~200 selected features |
| Preprocessing | < 1 min | Scaled, split data |
| Model Training | 1-5 min | 3 trained models |
| Evaluation | 2-10 min | Cross-validation, SHAP |
| **Total** | **3-46 min** | Complete ML pipeline |

*Note: Skipping feature selection reduces total time to 3-16 minutes.*

## Data Challenges

The CancerAg dataset has specific challenges:

1. **High Missing Data (69%)**: Sparse affinity matrix due to receptor specificity
   - Handled by median imputation

2. **Class Imbalance (6.5:1)**: G protein bias dominant
   - Handled by `class_weight='balanced'` in models

3. **Multi-class Classification**: 4+ bias categories
   - Using macro and weighted averaging for metrics

## Model Selection Criteria

The best model is selected based on:
1. Test F1 score (weighted) - Primary criterion
2. Cross-validation F1 score - Generalization check
3. Training time - Efficiency consideration

## Advanced Usage

### Custom Model Training

```python
from cancerag.ml import ModelTrainer

trainer = ModelTrainer(config)
results = trainer.train_and_evaluate_all(
    X_train, X_test, y_train, y_test,
    label_mapping, feature_names
)
```

### Custom Feature Selection

```python
from cancerag.ml import FeatureSelector

selector = FeatureSelector(config)
transformed_df, summary = selector.run_feature_selection(
    df, method='random_forest', threshold=0.001
)
```

### Loading Trained Models

```python
import pickle

# Load best model
with open('results/models/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessing objects
with open('data/processed/ml_preprocessed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

## Troubleshooting

### Dataset Not Found
```
ERROR: Dataset not found at data/processed/unified_ml_dataset.csv
```
**Solution**: Run the main pipeline first to generate the dataset:
```bash
python src/cancerag/main.py
```

### Memory Error During Feature Selection
```
MemoryError: Unable to allocate array
```
**Solution**: Skip feature selection or use random_forest method:
```python
results = run_ml_pipeline(config, skip_feature_selection=True)
```

### SHAP Import Error
```
ImportError: No module named 'shap'
```
**Solution**: SHAP is optional. Install it for interpretability analysis:
```bash
pip install shap
```

## Dependencies

Core dependencies (included in pyproject.toml):
- scikit-learn >= 1.7.2
- pandas >= 2.3.2
- numpy >= 2.3.3
- matplotlib >= 3.10.2
- seaborn >= 0.13.2

Optional dependencies:
- shap (for SHAP analysis)
- xgboost (for XGBoost models)
- catboost (for CatBoost models)

## Future Enhancements

Potential improvements:
1. Add XGBoost and CatBoost models
2. Implement hyperparameter optimization (GridSearchCV, RandomizedSearchCV)
3. Add ensemble methods (voting, stacking)
4. Implement SMOTE for class balancing
5. Add receptor-specific model training
6. Implement external validation on held-out receptors
7. Add model explainability dashboard
8. Implement online learning for new data

## Citation

If you use this ML module in your research, please cite:

```
CancerAg: Machine Learning Pipeline for GPCR Signaling Bias Prediction
https://github.com/yourusername/cancerag
```

## License

This module is part of the CancerAg project. See main LICENSE file for details.
