# Dockerfile for CancerAg Inference Application
# Post-rebuild deployment surface: packages the sklearn-Pipeline model
# artefacts from data/processed/ml_models/ + 61 prepared receptor PDBs
# and PDBQTs (one per UniProt) + the cached feature blocks the
# ModernBiasPredictor uses for known (ligand, receptor) pair lookups.

FROM python:3.11-slim

WORKDIR /app

# ---------------- system dependencies ----------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenBabel for molecular format conversion
    openbabel \
    # AutoDock Vina prerequisites
    wget \
    ca-certificates \
    libboost-all-dev \
    libxml2-utils \
    && rm -rf /var/lib/apt/lists/*

# Install AutoDock Vina (pinned to 1.2.5 per configs/config.yaml)
RUN wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 \
    && chmod +x vina_1.2.5_linux_x86_64 \
    && mv vina_1.2.5_linux_x86_64 /usr/local/bin/vina \
    && vina --version

RUN obabel -V || echo "OpenBabel check"

# ---------------- Python dependencies ----------------
COPY inference_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ---------------- cancerag package (editable) ----------------
COPY inference_app/pyproject.toml /app/pyproject.toml
COPY src/cancerag/ /app/src/cancerag/
RUN pip install --no-cache-dir -e .

# ---------------- inference app code ----------------
COPY inference_app/ /app/inference_app/

# ---------------- rebuilt ML artefacts ----------------
# Post-rebuild model surface (Stage 10 + Phase 6 advanced training).
# These are the joblib-pickled sklearn Pipelines the ModernBiasPredictor
# loads via data/processed/ml_models/selection_decision.json.

RUN mkdir -p \
    /app/data/processed/ml_models \
    /app/data/processed/ml_models/advanced \
    /app/data/processed/ml_preprocessed

COPY data/processed/ml_models/selection_decision.json \
     /app/data/processed/ml_models/selection_decision.json
COPY data/processed/ml_models/lightgbm_final.joblib \
     /app/data/processed/ml_models/lightgbm_final.joblib
COPY data/processed/ml_models/lightgbm_final_calibrated.joblib \
     /app/data/processed/ml_models/lightgbm_final_calibrated.joblib

# Optional Phase-6 advanced models (stacking ensemble + Optuna-tuned)
COPY data/processed/ml_models/advanced/stacking_meta_learner.joblib \
     /app/data/processed/ml_models/advanced/stacking_meta_learner.joblib
COPY data/processed/ml_models/advanced/lightgbm_tuned_calibrated.joblib \
     /app/data/processed/ml_models/advanced/lightgbm_tuned_calibrated.joblib

# Label encoder + dataset schema metadata
COPY data/processed/ml_preprocessed/label_encoder.joblib \
     /app/data/processed/ml_preprocessed/label_encoder.joblib

# Cached feature blocks for known (inchikey, receptor_uniprot) pair lookups
RUN mkdir -p /app/data/processed
COPY data/processed/ml_ready_dataset.parquet \
     /app/data/processed/ml_ready_dataset.parquet
COPY data/processed/docking_features.csv \
     /app/data/processed/docking_features.csv
COPY data/processed/pose_3d_features.csv \
     /app/data/processed/pose_3d_features.csv
COPY data/processed/interaction_fingerprints.parquet \
     /app/data/processed/interaction_fingerprints.parquet
COPY data/processed/ligand_features.parquet \
     /app/data/processed/ligand_features.parquet
COPY data/processed/unified_ligands.csv \
     /app/data/processed/unified_ligands.csv

# ---------------- binding sites + receptors ----------------
COPY data/processed/binding_sites.json \
     /app/data/processed/binding_sites.json

# All 61 prepared receptor PDBs (one per UniProt accession, chain-resolved
# with the Stage-03 DBREF UniProt mapping)
COPY data/processed/receptors/ /app/data/processed/receptors/

# All 61 prepared receptor PDBQTs (consolidated from .redock_work/<uniprot>/
# into a flat data/processed/receptors_pdbqt/<uniprot>.pdbqt layout for a
# clean COPY; obabel-retry-without-charges fix applied for the 4 chimeric
# receptors whose Gasteiger pass produced empty files)
COPY data/processed/receptors_pdbqt/ /app/data/processed/receptors_pdbqt/

# Verify packaging completeness at build time
RUN python3 <<'EOF'
import json, sys
from pathlib import Path

errors = []

# 1. binding_sites.json
bs_path = Path('/app/data/processed/binding_sites.json')
if not bs_path.exists():
    errors.append('binding_sites.json missing')
else:
    payload = json.loads(bs_path.read_text())
    sites = payload.get('binding_sites', [])
    if not isinstance(sites, list):
        errors.append('binding_sites.json wrong schema (expected list)')
    print(f'  binding_sites: {len(sites)} entries')

# 2. receptor PDBs
pdb_dir = Path('/app/data/processed/receptors')
pdbs = list(pdb_dir.glob('*.pdb'))
print(f'  receptor PDBs:   {len(pdbs)}')

# 3. receptor PDBQTs
pdbqt_dir = Path('/app/data/processed/receptors_pdbqt')
pdbqts = list(pdbqt_dir.glob('*.pdbqt'))
print(f'  receptor PDBQTs: {len(pdbqts)}')
empty = [p for p in pdbqts if p.stat().st_size < 100]
if empty:
    errors.append(f'{len(empty)} receptor PDBQTs are empty/tiny: {[p.name for p in empty[:5]]}')

# 4. ML artefacts
required = [
    '/app/data/processed/ml_models/selection_decision.json',
    '/app/data/processed/ml_models/lightgbm_final_calibrated.joblib',
    '/app/data/processed/ml_preprocessed/label_encoder.joblib',
    '/app/data/processed/ml_ready_dataset.parquet',
    '/app/data/processed/docking_features.csv',
    '/app/data/processed/pose_3d_features.csv',
    '/app/data/processed/interaction_fingerprints.parquet',
    '/app/data/processed/ligand_features.parquet',
]
for f in required:
    if not Path(f).exists():
        errors.append(f'MISSING: {f}')
    else:
        sz = Path(f).stat().st_size
        print(f'  OK ({sz:>10,} bytes) {f}')

if errors:
    print('\nPACKAGING ERRORS:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
print('\nDeployment packaging verified.')
EOF

# Create runtime scratch directories
RUN mkdir -p \
    /app/data/interim/docking_results/receptors \
    /app/logs \
    /app/temp_receptors

# ---------------- environment ----------------
ENV PYTHONPATH=/app:$PYTHONPATH
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV BASE_PATH=/app
ENV PORT=8080
EXPOSE 8080

# ---------------- healthcheck ----------------
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 8080)}')" || exit 1

WORKDIR /app/inference_app
CMD ["python", "app.py"]
