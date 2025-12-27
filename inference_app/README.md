# CancerAg Inference App

A Gradio-based web application for predicting biased agonism in GPCR ligands using the trained CancerAg machine learning models.

## Features

- **Single Molecule Prediction**: Predict bias category for a single molecule from SMILES string
- **Batch Prediction**: Predict bias categories for multiple molecules at once
- **Detailed Results**: View predicted class, class probabilities, and molecule details
- **User-Friendly Interface**: Clean, intuitive Gradio web interface

## Installation

### Option 1: Docker Deployment (Recommended)

The easiest way to deploy the inference app is using Docker:

```bash
cd inference_app
./docker-run.sh
```

Or manually:

```bash
docker-compose up --build
```

The app will be available at `http://localhost:7860`

See [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md) for detailed Docker deployment instructions.

### Option 2: Local Installation

#### Prerequisites

- Python 3.10 or higher
- AutoDock Vina (optional, for docking features)
- OpenBabel (optional, for docking features)
- Trained models and preprocessing artifacts from the main pipeline

#### Setup

1. **Install dependencies**:

```bash
cd inference_app
pip install -r requirements.txt
```

2. **Verify model files exist**:

The app expects the following files to exist in the project root:

- `results/models/random_forest.pkl` (or other model)
- `data/processed/ml_preprocessed/scaler.pkl`
- `data/processed/ml_preprocessed/preprocessing_metadata.json`
- `data/processed/ml_preprocessed/imputer.pkl` (optional)

## Usage

### Running the App

```bash
python app.py
```

The app will start on `http://localhost:7860` by default.

### Using the Interface

#### Single Prediction

1. Enter a SMILES string in the input box
2. Click "Predict"
3. View the predicted bias category, class probabilities, and molecule details

#### Batch Prediction

1. Go to the "Batch Prediction" tab
2. Enter multiple SMILES strings (one per line)
3. Click "Predict Batch"
4. View results for all molecules

### Example SMILES

- `CCO` - Ethanol
- `CC(=O)OC1=CC=CC=C1C(=O)O` - Aspirin
- `CN1CCC[C@H]1c2cccnc2` - Nicotine

## Architecture

The inference app consists of several modules:

- **`molecule_processor.py`**: Handles molecule standardization and validation
- **`feature_extractor.py`**: Extracts molecular descriptors using RDKit
- **`predictor.py`**: Loads models and makes predictions
- **`inference_pipeline.py`**: Orchestrates the complete inference pipeline
- **`app.py`**: Gradio web interface

## Model Selection

By default, the app uses the `random_forest` model. To use a different model, modify the `initialize_predictor()` call in `app.py`:

```python
initialize_predictor(model_name="xgboost")  # or "catboost", "lightgbm", etc.
```

Available models:

- `random_forest` (default, best performance)
- `xgboost`
- `catboost`
- `lightgbm`
- `logistic_regression`

## Output Format

### Single Prediction

- **Predicted Class**: The most likely bias category
- **Class Probabilities**: Probability distribution across all classes
- **Details**: Canonical SMILES, number of features, etc.

### Batch Prediction

- Results for each molecule including:
  - Input SMILES
  - Predicted class
  - Confidence score

## Troubleshooting

### Model Not Found

If you see "Model file not found" errors:

1. Ensure you've run the training pipeline first
2. Check that model files exist in `results/models/`
3. Verify the model name matches available files

### SMILES Parsing Errors

If a SMILES string fails to parse:

- Verify the SMILES string is valid
- Check for typos or formatting issues
- Try standardizing the SMILES using RDKit first

### Feature Mismatch

If you see feature-related errors:

- Ensure preprocessing artifacts are from the same pipeline run as the model
- Check that `preprocessing_metadata.json` exists and is valid

## Performance

- **Single prediction**: ~1-2 seconds
- **Batch prediction**: ~1-2 seconds per molecule

Performance depends on:

- Molecule complexity
- System resources
- Model complexity

## Limitations

1. **Model Accuracy**: The model achieves ~73% accuracy on test data. Predictions should be validated experimentally.

2. **Class Imbalance**: The model performs better on majority classes (G protein) than minority classes (G protein selectivity).

3. **Feature Requirements**: The model requires all molecular descriptors to be calculated. Some molecules may fail if they cannot be standardized.

4. **Receptor Information**: The current model does not use receptor-specific information. All predictions are based on ligand properties only.

## Future Improvements

- Add receptor selection for receptor-specific predictions
- Visualize molecular structures
- Export results to CSV/JSON
- Add confidence thresholds
- Support for SDF file uploads
- Integration with docking predictions

## License

Same as the main CancerAg project.

## Contact

For issues or questions, please refer to the main project repository.
