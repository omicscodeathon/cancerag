#!/usr/bin/env python3
"""
Script to download and prepare receptor structures for the inference app.

This script downloads receptor PDB structures and prepares them for use
in the inference application. Since receptors are excluded from git,
this script must be run locally to set up the required receptor files.

Usage:
    python scripts/download_receptors.py

Requirements:
    - Python dependencies installed
    - Internet connection for downloading from PDB
    - AutoDock Vina and OpenBabel (optional, for docking features)
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from cancerag.data_collection.receptor_retriever import ReceptorRetriever
from cancerag.preprocessing.receptor_preprocessor import ReceptorPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from config.yaml."""
    config_path = project_root / "configs" / "config.yaml"
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_receptors_from_biasdb(config):
    """Extract unique receptor names from BiasDB data."""
    biasdb_path = Path(config["paths"]["biasdb_input"])
    
    if not biasdb_path.exists():
        logger.warning(f"BiasDB file not found: {biasdb_path}")
        logger.info("You may need to run the data collection pipeline first:")
        logger.info("  python -m src.cancerag.data_collection.biasdb_retriever")
        return []

    import pandas as pd
    df = pd.read_csv(biasdb_path)
    
    # Extract unique receptors (adjust column name as needed)
    receptor_col = None
    for col in ["receptor", "Receptor", "target", "Target", "protein"]:
        if col in df.columns:
            receptor_col = col
            break
    
    if receptor_col is None:
        logger.error("Could not find receptor column in BiasDB data")
        return []
    
    receptors = df[receptor_col].unique().tolist()
    logger.info(f"Found {len(receptors)} unique receptors in BiasDB data")
    return receptors


def download_receptors(config, receptor_names):
    """Download receptor PDB structures."""
    output_dir = config["paths"]["pdb_summary"]
    max_downloads = config["data_collection"].get("max_pdb_files_per_receptor", 10)
    
    logger.info(f"Downloading receptors to: {output_dir}")
    logger.info(f"Maximum downloads per receptor: {max_downloads}")
    
    retriever = ReceptorRetriever(
        output_dir=output_dir,
        max_downloads=max_downloads,
        force_redownload=False,
        network_config=config.get("network", {}),
    )
    
    downloaded = {}
    for receptor_name in receptor_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing receptor: {receptor_name}")
        logger.info(f"{'='*60}")
        
        try:
            files = retriever._download_for_receptor(
                receptor_name, use_alphafold_fallback=True
            )
            if files:
                downloaded[receptor_name] = files
                logger.info(f"✓ Downloaded {len(files)} structures for {receptor_name}")
            else:
                logger.warning(f"✗ No structures downloaded for {receptor_name}")
        except Exception as e:
            logger.error(f"✗ Error downloading {receptor_name}: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Download complete: {len(downloaded)} receptors processed")
    logger.info(f"{'='*60}")
    
    return downloaded


def prepare_receptors(config):
    """Clean and prepare receptor structures."""
    logger.info("\nPreparing receptor structures...")
    
    preprocessor = ReceptorPreprocessor(config)
    preprocessor.run()
    
    logger.info("✓ Receptor preparation complete")


def check_binding_sites(config):
    """Check if binding_sites.json exists, create if needed."""
    binding_sites_path = Path(config["paths"]["processed_data"]) / "binding_sites.json"
    
    if binding_sites_path.exists():
        logger.info(f"✓ Binding sites file exists: {binding_sites_path}")
        with open(binding_sites_path, "r") as f:
            binding_sites = json.load(f)
        logger.info(f"  Contains {len(binding_sites)} receptors")
        return True
    else:
        logger.warning(f"⚠ Binding sites file not found: {binding_sites_path}")
        logger.info("  This file is created during the active site identification step")
        logger.info("  Run the main pipeline to generate it:")
        logger.info("    python src/cancerag/main.py")
        logger.info("  Or run active site identification:")
        logger.info("    python -m src.cancerag.features.active_site_identifier")
        return False


def verify_setup(config):
    """Verify that receptors are set up correctly."""
    logger.info("\n" + "="*60)
    logger.info("Verifying receptor setup...")
    logger.info("="*60)
    
    receptors_dir = Path(config["paths"]["processed_data"]) / "receptors"
    binding_sites_path = Path(config["paths"]["processed_data"]) / "binding_sites.json"
    
    checks = {
        "Receptors directory exists": receptors_dir.exists(),
        "Binding sites file exists": binding_sites_path.exists(),
    }
    
    if receptors_dir.exists():
        pdb_files = list(receptors_dir.glob("*.pdb"))
        checks[f"Receptor PDB files ({len(pdb_files)} found)"] = len(pdb_files) > 0
    
    if binding_sites_path.exists():
        with open(binding_sites_path, "r") as f:
            binding_sites = json.load(f)
        checks[f"Binding sites loaded ({len(binding_sites)} receptors)"] = len(binding_sites) > 0
    
    all_ok = True
    for check, status in checks.items():
        status_symbol = "✓" if status else "✗"
        logger.info(f"{status_symbol} {check}")
        if not status:
            all_ok = False
    
    return all_ok


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("CancerAg Receptor Download Script")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    
    # Check if binding sites exist
    has_binding_sites = check_binding_sites(config)
    
    # Get receptor names
    if has_binding_sites:
        # Load from binding_sites.json
        binding_sites_path = Path(config["paths"]["processed_data"]) / "binding_sites.json"
        with open(binding_sites_path, "r") as f:
            binding_sites = json.load(f)
        receptor_names = list(binding_sites.keys())
        logger.info(f"\nFound {len(receptor_names)} receptors in binding_sites.json")
    else:
        # Extract from BiasDB
        receptor_names = get_receptors_from_biasdb(config)
        if not receptor_names:
            logger.error("No receptors found. Cannot proceed.")
            sys.exit(1)
    
    # Ask user for confirmation
    logger.info(f"\nWill download structures for {len(receptor_names)} receptors:")
    for i, name in enumerate(receptor_names[:10], 1):
        logger.info(f"  {i}. {name}")
    if len(receptor_names) > 10:
        logger.info(f"  ... and {len(receptor_names) - 10} more")
    
    response = input("\nProceed with download? (y/N): ").strip().lower()
    if response != 'y':
        logger.info("Download cancelled.")
        sys.exit(0)
    
    # Download receptors
    downloaded = download_receptors(config, receptor_names)
    
    # Prepare receptors
    prepare_receptors(config)
    
    # Verify setup
    verify_setup(config)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Receptor download and preparation complete!")
    logger.info("="*60)
    logger.info("\nReceptors are now available for use in the inference app.")
    logger.info("You can now run:")
    logger.info("  python inference_app/app.py")
    logger.info("  or")
    logger.info("  docker-compose -f inference_app/docker-compose.yml up")


if __name__ == "__main__":
    main()

