"""
CancerAg Main Pipeline

Entry point for the complete GPCR signaling bias prediction pipeline.
This pipeline is idempotent - it checks for existing results and skips
completed stages automatically.

Usage:
    python -m src.cancerag.main
    python src/cancerag/main.py
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from cancerag.analysis.data_analyzer import run_data_analysis
from cancerag.data_collection import (
    biasdb_retriever,
    receptor_retriever,
)
from cancerag.data_collection.chembl_retriever import ChEMBLRetriever
from cancerag.docking import run_docking
from cancerag.features import active_site_identifier, molecular_descriptors
from cancerag.ml.dataset_assembly import run_dataset_assembly
from cancerag.ml.ml_pipeline import run_ml_pipeline
from cancerag.preprocessing import receptor_preprocessor
from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineStatus:
    """Tracks and manages pipeline stage completion status."""

    def __init__(self, config: Dict):
        self.config = config
        self.status_file = (
            Path(config["paths"].get("processed_data", "data/processed"))
            / "pipeline_status.json"
        )
        self.status = self._load_status()

    def _load_status(self) -> Dict:
        """Load existing pipeline status."""
        if self.status_file.exists():
            try:
                with open(self.status_file, "r") as f:
                    return json.load(f)
            except:
                logger.warning("Could not load pipeline status. Starting fresh.")
        return {}

    def _save_status(self):
        """Save pipeline status to disk."""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def is_stage_complete(self, stage_name: str, check_files: list = None) -> bool:
        """
        Check if a stage is complete.

        Args:
            stage_name: Name of the pipeline stage
            check_files: List of file paths that must exist

        Returns:
            True if stage is complete and all required files exist
        """
        # Check status file
        if stage_name not in self.status:
            return False

        if not self.status[stage_name].get("completed", False):
            return False

        # Check required files if specified
        if check_files:
            for file_path in check_files:
                if not Path(file_path).exists():
                    logger.info(
                        f"Stage '{stage_name}' marked complete but file missing: {file_path}"
                    )
                    return False

        return True

    def mark_complete(self, stage_name: str, **metadata):
        """Mark a stage as complete with optional metadata."""
        from datetime import datetime

        self.status[stage_name] = {
            "completed": True,
            "timestamp": datetime.now().isoformat(),
            **metadata,
        }
        self._save_status()

    def mark_in_progress(self, stage_name: str):
        """Mark a stage as in progress."""
        from datetime import datetime

        self.status[stage_name] = {
            "completed": False,
            "in_progress": True,
            "started": datetime.now().isoformat(),
        }
        self._save_status()

    def get_status_summary(self) -> str:
        """Get a formatted summary of pipeline status."""
        if not self.status:
            return "No stages completed yet."

        summary = []
        for stage, info in self.status.items():
            status = "✓" if info.get("completed") else "⧗"
            summary.append(f"  {status} {stage}")

        return "\n".join(summary)

    def reset_stage(self, stage_name: str) -> None:
        """
        Reset a specific stage to incomplete status.

        Args:
            stage_name: Name of the stage to reset
        """
        if stage_name in self.status:
            del self.status[stage_name]
            self._save_status()
            logger.info(f"Reset stage: {stage_name}")

    def reset_all_stages(self) -> None:
        """Reset all pipeline stages to incomplete status."""
        self.status = {}
        self._save_status()
        logger.info("Reset all pipeline stages")

    def recover_from_interruption(self) -> None:
        """
        Recover from interrupted pipeline run.
        Marks any 'in_progress' stages as incomplete so they can be rerun.
        """
        recovered = False
        for stage_name, stage_info in self.status.items():
            if stage_info.get("in_progress") and not stage_info.get("completed"):
                logger.warning(f"Recovering interrupted stage: {stage_name}")
                stage_info.pop("in_progress", None)
                stage_info.pop("started", None)
                recovered = True

        if recovered:
            self._save_status()
            logger.info("Pipeline recovery completed")


def run_pipeline(config_path: str, force_rerun: bool = False):
    """
    Orchestrates the entire GPCR signaling bias prediction pipeline.

    This pipeline is idempotent - it checks for existing results and skips
    completed stages automatically.

    Args:
        config_path: Path to configuration YAML file
        force_rerun: If True, re-run all stages even if complete
    """
    # 1. Load Configuration
    logger.info("=" * 80)
    logger.info("CANCERAG PIPELINE - GPCR SIGNALING BIAS PREDICTION")
    logger.info("=" * 80)

    print("\n1. Loading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config["paths"]
    network_config = config.get("network")

    # Create all necessary directories (skip file paths)
    for path_key, path_value in paths.items():
        # Skip paths that are files (contain file extensions)
        if "." in os.path.basename(path_value):
            # This is a file path, create parent directory instead
            parent_dir = os.path.dirname(path_value)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
        else:
            # This is a directory path
            os.makedirs(path_value, exist_ok=True)

    # Initialize pipeline status tracker
    pipeline_status = PipelineStatus(config)

    if not force_rerun:
        print("\nPipeline Status:")
        print(pipeline_status.get_status_summary())
        print()

    # --- STAGE 1: DATA COLLECTION ---
    stage_name = "data_collection"
    check_files = [
        paths["biasdb_input"],
        os.path.join(paths["pdb_summary"], "summary.json"),
    ]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n2. ✓ Data Collection Stage [SKIPPED - Already complete]")
        biasdb_df = pd.read_csv(paths["biasdb_input"])
        unique_receptors = biasdb_df["receptor_subtype"].dropna().unique()
    else:
        print("\n2. Starting Data Collection Stage...")
        pipeline_status.mark_in_progress(stage_name)

        biasdb_df = biasdb_retriever.download_biasdb_data(
            paths["biasdb_input"],
            network_config=network_config,
        )
        if biasdb_df.empty:
            logger.error("Failed to retrieve data from BiasDB. Halting pipeline.")
            return

        unique_receptors = biasdb_df["receptor_subtype"].dropna().unique()
        print(f"   - Found {len(unique_receptors)} unique receptors")

        retriever = receptor_retriever.ReceptorRetriever(
            output_dir=paths["pdb_summary"],
            max_downloads=config["data_collection"]["max_pdb_files_per_receptor"],
            force_redownload=False,
            network_config=network_config,
        )
        retriever.run(unique_receptors)

        pipeline_status.mark_complete(
            stage_name,
            biasdb_records=len(biasdb_df),
            unique_receptors=len(unique_receptors),
        )
        print("   ✓ Data Collection complete.")

    # --- STAGE 1B: ChEMBL AGONIST COLLECTION (Optional) ---
    chembl_enabled = config["data_collection"].get("enable_chembl", False)
    if chembl_enabled:
        stage_name = "chembl_collection"
        chembl_raw_dir = paths["chembl_raw"]
        check_files = [os.path.join(chembl_raw_dir, "chembl_summary.json")]

        if not force_rerun and pipeline_status.is_stage_complete(
            stage_name, check_files
        ):
            print("\n2B. ✓ ChEMBL Collection Stage [SKIPPED - Already complete]")
        else:
            print("\n2B. Starting ChEMBL Collection Stage...")
            pipeline_status.mark_in_progress(stage_name)

            try:
                chembl_retriever = ChEMBLRetriever(
                    output_dir=chembl_raw_dir,
                    network_config=network_config,
                )
                chembl_retriever.run(unique_receptors.tolist())
                print(f"   - ChEMBL data saved to {chembl_raw_dir}")
                pipeline_status.mark_complete(stage_name)
                print("   ✓ ChEMBL collection complete.")
            except Exception as e:
                logger.error(f"ChEMBL collection failed: {e}")
                logger.info("Continuing pipeline without ChEMBL data...")
                # Mark as complete anyway to avoid blocking pipeline
                pipeline_status.mark_complete(stage_name)
                print("   ⚠ ChEMBL collection failed, continuing without it.")

    # --- STAGE 2: LIGAND PREPROCESSING ---
    stage_name = "ligand_preprocessing"
    check_files = [os.path.join(paths["processed_data"], "unified_ligands.csv")]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n3. ✓ Ligand Preprocessing Stage [SKIPPED - Already complete]")
    else:
        print("\n3. Starting Ligand Preprocessing Stage...")
        pipeline_status.mark_in_progress(stage_name)

        ligand_processor = LigandPreprocessor(config)
        ligand_processor.run()

        pipeline_status.mark_complete(stage_name)
        print("   ✓ Ligand preprocessing complete.")

    # --- STAGE 3: RECEPTOR PREPROCESSING ---
    stage_name = "receptor_preprocessing"
    check_files = [os.path.join(paths["interim_data"], "pdbqt")]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n4. ✓ Receptor Preprocessing Stage [SKIPPED - Already complete]")
    else:
        print("\n4. Starting Receptor Preprocessing Stage...")
        pipeline_status.mark_in_progress(stage_name)

        receptor_processor = receptor_preprocessor.ReceptorPreprocessor(config)
        receptor_processor.run()

        pipeline_status.mark_complete(stage_name)
        print("   ✓ Receptor preprocessing complete.")

    # --- STAGE 4: MOLECULAR DESCRIPTORS ---
    stage_name = "molecular_descriptors"
    check_files = [
        os.path.join(paths["processed_data"], "ligands_with_descriptors.csv")
    ]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n5. ✓ Molecular Descriptors Stage [SKIPPED - Already complete]")
    else:
        print("\n5. Starting Molecular Descriptors Stage...")
        pipeline_status.mark_in_progress(stage_name)

        descriptor_calculator = molecular_descriptors.MolecularDescriptorCalculator(
            config
        )
        descriptor_calculator.run()

        pipeline_status.mark_complete(stage_name)
        print("   ✓ Molecular descriptors complete.")

    # --- STAGE 5: ACTIVE SITE IDENTIFICATION ---
    stage_name = "active_site_identification"
    check_files = [os.path.join(paths["processed_data"], "binding_sites.json")]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n6. ✓ Active Site Identification Stage [SKIPPED - Already complete]")
    else:
        print("\n6. Starting Active Site Identification Stage...")
        pipeline_status.mark_in_progress(stage_name)

        active_site_identifier_instance = active_site_identifier.ActiveSiteIdentifier(
            config
        )
        active_site_identifier_instance.run()

        pipeline_status.mark_complete(stage_name)
        print("   ✓ Active site identification complete.")

    # --- STAGE 6: MOLECULAR DOCKING ---
    stage_name = "molecular_docking"
    check_files = [
        os.path.join(paths["reports"], "docking_results", "affinity_comparison.csv")
    ]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n7. ✓ Molecular Docking Stage [SKIPPED - Already complete]")
    else:
        print("\n7. Starting Molecular Docking Stage...")
        pipeline_status.mark_in_progress(stage_name)

        run_docking.run_docking_stage(config)

        pipeline_status.mark_complete(stage_name)
        print("   ✓ Molecular docking complete.")

    # --- STAGE 7: DATASET ASSEMBLY ---
    stage_name = "dataset_assembly"
    check_files = [os.path.join(paths["processed_data"], "unified_ml_dataset.csv")]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n8. ✓ Dataset Assembly Stage [SKIPPED - Already complete]")
        dataset_summary = json.load(
            open(os.path.join(paths["processed_data"], "dataset_summary.json"))
        )
    else:
        print("\n8. Starting Dataset Assembly Stage...")
        pipeline_status.mark_in_progress(stage_name)

        unified_dataset, dataset_summary = run_dataset_assembly(config)

        pipeline_status.mark_complete(
            stage_name,
            total_samples=dataset_summary["total_samples"],
            total_features=dataset_summary["total_features"],
        )
        print(
            f"   ✓ Dataset assembly complete: {dataset_summary['total_samples']} samples, {dataset_summary['total_features']} features"
        )

    # --- STAGE 8: DATA ANALYSIS ---
    stage_name = "data_analysis"
    check_files = [
        os.path.join(paths["reports"], "data_analysis", "analysis_summary.json")
    ]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n9. ✓ Data Analysis Stage [SKIPPED - Already complete]")
    else:
        print("\n9. Starting Data Analysis Stage...")
        pipeline_status.mark_in_progress(stage_name)

        _ = run_data_analysis(config)

        pipeline_status.mark_complete(stage_name)
        print("   ✓ Data analysis complete.")

    # --- STAGE 9: MACHINE LEARNING ---
    stage_name = "machine_learning"
    check_files = [
        os.path.join(paths["models"], "random_forest.pkl"),
        os.path.join(paths["reports"], "training_results.json"),
    ]

    if not force_rerun and pipeline_status.is_stage_complete(stage_name, check_files):
        print("\n10. ✓ Machine Learning Stage [SKIPPED - Already complete]")
        # Load results for display
        with open(os.path.join(paths["reports"], "ml_pipeline_summary.json")) as f:
            ml_summary = json.load(f)
            if "best_model" in ml_summary:
                print(f"   - Best model: {ml_summary['best_model']['model_name']}")
                print(f"   - Test F1 Score: {ml_summary['best_model']['test_f1']:.4f}")
    else:
        print("\n10. Starting Machine Learning Stage...")
        pipeline_status.mark_in_progress(stage_name)

        ml_results = run_ml_pipeline(config, skip_feature_selection=True)

        if "best_model" in ml_results:
            pipeline_status.mark_complete(
                stage_name,
                best_model=ml_results["best_model"]["model_name"],
                test_accuracy=ml_results["best_model"]["test_accuracy"],
                test_f1=ml_results["best_model"]["test_f1"],
            )

            print(f"   - Best model: {ml_results['best_model']['model_name']}")
            print(
                f"   - Test Accuracy: {ml_results['best_model']['test_accuracy']:.4f}"
            )
            print(f"   - Test F1 Score: {ml_results['best_model']['test_f1']:.4f}")
        else:
            pipeline_status.mark_complete(stage_name)

        print("   ✓ Machine learning complete.")

    # Final Summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nFinal Status:")
    print(pipeline_status.get_status_summary())
    print("\nAll outputs saved to:")
    print(f"  - Data: {paths['processed_data']}")
    print(f"  - Models: {paths['models']}")
    print(f"  - Reports: {paths['reports']}")
    print(f"  - Figures: {paths['figures']}")
    print("\n" + "=" * 80)


def main():
    """Main entry point for the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CancerAg Pipeline - GPCR Signaling Bias Prediction"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force re-run all stages even if complete",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Automatically recover from interrupted pipeline run",
    )
    parser.add_argument(
        "--reset-stage",
        metavar="STAGE",
        help="Reset a specific stage to incomplete (e.g., ligand_preprocessing)",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Reset all pipeline stages to incomplete",
    )

    args = parser.parse_args()

    # Find config file
    if os.path.exists(args.config):
        config_file = args.config
    else:
        # Try relative to this script
        script_dir = Path(__file__).parent
        config_file = script_dir / ".." / ".." / args.config
        if not config_file.exists():
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)
        config_file = str(config_file)

    logger.info(f"Using configuration: {config_file}")

    # Handle recovery and reset commands
    if args.reset_all or args.reset_stage or args.recover:
        # Load config to initialize pipeline status
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
        pipeline_status = PipelineStatus(config)

        if args.reset_all:
            logger.info("Resetting all pipeline stages...")
            pipeline_status.reset_all_stages()
            logger.info("All stages have been reset.")
            sys.exit(0)
        elif args.reset_stage:
            logger.info(f"Resetting stage: {args.reset_stage}")
            pipeline_status.reset_stage(args.reset_stage)
            logger.info(f"Stage '{args.reset_stage}' has been reset.")
            sys.exit(0)
        elif args.recover:
            logger.info("Recovering from interruption...")
            pipeline_status.recover_from_interruption()
            logger.info("Recovery completed. Run pipeline again to continue.")
            sys.exit(0)

    try:
        run_pipeline(config_file, force_rerun=args.force_rerun)
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
