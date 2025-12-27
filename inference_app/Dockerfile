# Dockerfile for CancerAg Inference Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenBabel for molecular format conversion
    openbabel \
    # AutoDock Vina for docking (optional but included)
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install AutoDock Vina
RUN wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 \
    && chmod +x vina_1.2.5_linux_x86_64 \
    && mv vina_1.2.5_linux_x86_64 /usr/local/bin/vina \
    && vina --version

# Verify OpenBabel installation
RUN obabel -V || echo "OpenBabel check"

# Copy requirements first for better caching
COPY inference_app/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the main cancerag package and pyproject.toml for editable install
COPY pyproject.toml /app/pyproject.toml
COPY src/cancerag/ /app/src/cancerag/

# Install cancerag package in editable mode (only core dependencies needed for inference)
RUN pip install --no-cache-dir -e .

# Copy inference app code
COPY inference_app/ /app/inference_app/

# Copy essential model files for inference (bundled in image)
COPY results/models/random_forest.pkl /app/results/models/random_forest.pkl
COPY results/models/xgboost.pkl /app/results/models/xgboost.pkl
COPY results/models/lightgbm.pkl /app/results/models/lightgbm.pkl
COPY results/models/logistic_regression.pkl /app/results/models/logistic_regression.pkl

# Copy preprocessing artifacts (required for inference)
COPY data/processed/ml_preprocessed/scaler.pkl /app/data/processed/ml_preprocessed/scaler.pkl
COPY data/processed/ml_preprocessed/preprocessing_metadata.json /app/data/processed/ml_preprocessed/preprocessing_metadata.json
COPY data/processed/ml_preprocessed/label_encoder.pkl /app/data/processed/ml_preprocessed/label_encoder.pkl

# Copy receptor structures for docking (ONLY selected 46 high-yield receptors, ~30MB)
COPY data/processed/receptors_selected/ /app/data/processed/receptors/
COPY data/processed/binding_sites.json /app/data/processed/binding_sites.json

# Create additional necessary directories
RUN mkdir -p /app/data/processed/interim/docking_results/receptors \
    /app/logs \
    /app/temp_receptors

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV BASE_PATH=/app

# Expose Gradio port
EXPOSE 7860

# Health check (increased start-period for model/receptor loading)
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" || exit 1

# Run the application
WORKDIR /app/inference_app
CMD ["python", "app.py"]

