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
    # Boost libraries required by AutoDock Vina binary
    libboost-all-dev \
    # XML utilities for parsing
    libxml2-utils \
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

# Copy the main cancerag package and minimal pyproject.toml for editable install
COPY inference_app/pyproject.toml /app/pyproject.toml
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

# Copy binding sites configuration FIRST
COPY data/processed/binding_sites.json /app/data/processed/binding_sites.json

# Copy and filter receptor structures (only those with binding sites defined)
RUN mkdir -p /app/data/processed/receptors
COPY data/processed/receptors_selected/ /tmp/receptors_temp/
RUN python3 << 'EOF'
import json, shutil
from pathlib import Path

with open('/app/data/processed/binding_sites.json') as f:
    sites = json.load(f)

valid = {site['source_pdb'] + '.pdb' for site in sites.values()}
src = Path('/tmp/receptors_temp')
dst = Path('/app/data/processed/receptors')

copied = 0
for pdb in src.glob('*.pdb'):
    if pdb.name in valid:
        shutil.copy(pdb, dst / pdb.name)
        copied += 1

shutil.rmtree(src)
print(f'✓ Copied {copied} receptors with binding sites (expected: {len(sites)})')
EOF

# Pre-convert receptors to PDBQT for faster docking (eliminates 30-60s cold start delay)
RUN python3 << 'EOF'
import subprocess, sys
from pathlib import Path

receptors_dir = Path('/app/data/processed/receptors')
prepared_dir = Path('/app/data/processed/receptors_prepared')
prepared_dir.mkdir(parents=True, exist_ok=True)

print("Pre-converting receptors to PDBQT format...")
converted = failed = 0

for pdb_file in sorted(receptors_dir.glob('*.pdb')):
    pdbqt_path = prepared_dir / pdb_file.with_suffix('.pdbqt').name

    result = subprocess.run(
        ['obabel', '-ipdb', str(pdb_file), '-opdbqt', '-xr', '-O', str(pdbqt_path)],
        capture_output=True, text=True
    )

    if result.returncode == 0 and pdbqt_path.exists() and pdbqt_path.stat().st_size > 100:
        print(f'  ✓ {pdb_file.stem}')
        converted += 1
    else:
        print(f'  ✗ {pdb_file.stem} FAILED: {result.stderr[:100] if result.stderr else "unknown error"}')
        failed += 1

print(f'\nReceptor pre-conversion complete: {converted} success, {failed} failed')
if failed > 0:
    print('WARNING: Some receptors failed to convert - docking will use runtime conversion as fallback')
EOF

# Create additional necessary directories
RUN mkdir -p /app/data/interim/docking_results/receptors \
    /app/logs \
    /app/temp_receptors

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV BASE_PATH=/app
# PORT will be set by Cloud Run, defaults to 8080
ENV PORT=8080

# Expose port (Cloud Run uses 8080 by default)
EXPOSE 8080

# Health check (increased start-period for model/receptor loading)
# Note: Cloud Run has its own health checks, but this is useful for local testing
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8080)}')" || exit 1

# Run the application
WORKDIR /app/inference_app
CMD ["python", "app.py"]

