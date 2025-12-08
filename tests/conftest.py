"""
Pytest configuration and shared fixtures for CancerAg tests.
"""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_config():
    """Provide a minimal test configuration."""
    return {
        "paths": {
            "raw_data": "tests/data/raw",
            "processed_data": "tests/data/processed",
            "interim_data": "tests/data/interim",
            "models": "tests/results/models",
            "figures": "tests/results/figures",
            "reports": "tests/results/reports",
            "biasdb_input": "tests/data/raw/biasdb_data.csv",
            "pdb_summary": "tests/data/pdb",
        },
        "data_collection": {
            "max_pdb_files_per_receptor": 5,
            "enable_chembl": False,
        },
        "preprocessing": {
            "activity_threshold_nm": 1000,
            "lipinski_strict": False,
        },
        "docking": {
            "exhaustiveness": 1,
            "num_modes": 1,
            "num_cpu": 1,
        },
        "ml_model": {
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 10,
            "max_depth": 3,
        },
    }


@pytest.fixture
def sample_smiles():
    """Provide sample SMILES strings for testing."""
    return [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CC1=CC=C(C=C1)C(C)CC(=O)O",  # Naproxen
    ]


@pytest.fixture
def sample_biasdb_data():
    """Provide sample BiasDB data for testing."""
    return {
        "receptor_subtype": ["5HT2C receptor", "D2 receptor", "beta2-adrenoceptor"],
        "ligand_name": ["Lorcaserin", "Aripiprazole", "Isoproterenol"],
        "smiles": [
            "Clc1ccc2[nH]cc(c2c1)-c1ccc(Br)cc1",
            "Clc1ccc2c(c1)nccc2-c1ncccn1",
            "CC(C)[C@@H](NC(=O)C(C)C)CC1=CC(=C(O)C=C1)O",
        ],
        "bias_category": ["G protein-biased", "Balanced", "G protein-biased"],
        "publication_title": ["Title1", "Title2", "Title3"],
    }


@pytest.fixture
def sample_pdb_file(tmp_path):
    """Create a minimal mock PDB file for testing."""
    pdb_content = """HEADER    TEST PROTEIN                        31-OCT-24   TEST
TITLE     TEST STRUCTURE
ATOM      1  N   ALA A   1      20.154  16.967  26.689  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.042  15.984  26.720  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.682  16.617  27.074  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.375  17.764  26.763  1.00 20.00           O  
END
"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)
    return str(pdb_file)


@pytest.fixture
def sample_config_file(tmp_path):
    """Create a temporary config.yaml file for testing."""
    config_content = """paths:
  raw_data: data/raw
  processed_data: data/processed
  interim_data: data/interim
  models: results/models
  figures: results/figures
  reports: results/reports
  biasdb_input: data/raw/biasdb_data.csv
  pdb_summary: data/pdb

data_collection:
  max_pdb_files_per_receptor: 10
  enable_chembl: false

preprocessing:
  activity_threshold_nm: 1000
  
docking:
  exhaustiveness: 4
  num_modes: 3
  
ml_model:
  test_size: 0.2
  random_state: 42
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment before each test."""
    # Set test environment variable
    os.environ["TESTING"] = "1"
    yield
    # Cleanup
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
