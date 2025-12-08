#!/usr/bin/env python3
"""
Pipeline Status Checker

This utility checks the status of the CancerAg pipeline by examining
which output files exist and which steps have been completed.

Usage:
    python pipeline_status.py
"""

import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineStatusChecker:
    """
    Checks the status of the CancerAg pipeline.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the status checker.

        Args:
            config_path (str): Path to config file. If None, uses default.
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml"
            )

        import yaml

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.paths = self.config["paths"]

    def check_file_exists(self, file_path: str, description: str) -> dict:
        """
        Checks if a file exists and returns status information.

        Args:
            file_path (str): Path to the file
            description (str): Description of what the file represents

        Returns:
            dict: Status information
        """
        exists = os.path.exists(file_path)
        size = 0
        if exists:
            size = os.path.getsize(file_path)

        return {
            "file": file_path,
            "description": description,
            "exists": exists,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2) if size > 0 else 0,
        }

    def check_directory_exists(self, dir_path: str, description: str) -> dict:
        """
        Checks if a directory exists and returns status information.

        Args:
            dir_path (str): Path to the directory
            description (str): Description of what the directory represents

        Returns:
            dict: Status information
        """
        exists = os.path.exists(dir_path)
        file_count = 0
        if exists:
            file_count = len(
                [
                    f
                    for f in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, f))
                ]
            )

        return {
            "directory": dir_path,
            "description": description,
            "exists": exists,
            "file_count": file_count,
        }

    def check_pipeline_status(self):
        """
        Checks the complete pipeline status.

        Returns:
            dict: Complete status information
        """
        logger.info("Checking pipeline status...")

        status = {
            "data_collection": {},
            "preprocessing": {},
            "feature_extraction": {},
            "docking": {},
            "machine_learning": {},
        }

        # Data Collection Stage
        status["data_collection"] = {
            "biasdb_data": self.check_file_exists(
                self.paths["biasdb_input"], "BiasDB ligand data"
            ),
            "pdb_summary": self.check_file_exists(
                os.path.join(self.paths["pdb_summary"], "summary.json"),
                "PDB download summary",
            ),
            "pdb_files": self.check_directory_exists(
                self.paths["pdb_summary"], "Downloaded PDB structures"
            ),
        }

        # Preprocessing Stage
        status["preprocessing"] = {
            "unified_ligands": self.check_file_exists(
                os.path.join(self.paths["processed_data"], "unified_ligands.csv"),
                "Unified ligand dataset",
            ),
            "cleaned_receptors": self.check_directory_exists(
                os.path.join(self.paths["processed_data"], "receptors"),
                "Cleaned receptor structures",
            ),
        }

        # Feature Extraction Stage
        status["feature_extraction"] = {
            "molecular_descriptors": self.check_file_exists(
                os.path.join(
                    self.paths["processed_data"], "ligands_with_descriptors.csv"
                ),
                "Ligands with molecular descriptors",
            ),
            "binding_sites": self.check_file_exists(
                os.path.join(self.paths["processed_data"], "binding_sites.json"),
                "Binding site coordinates",
            ),
            "structure_selection": self.check_file_exists(
                os.path.join(
                    self.paths["processed_data"], "structure_selection_summary.json"
                ),
                "Structure selection summary",
            ),
        }

        # Docking Stage
        status["docking"] = {
            "docking_results": self.check_directory_exists(
                os.path.join(self.paths["reports"], "docking_results"),
                "Docking results",
            )
        }

        # Machine Learning Stage
        status["machine_learning"] = {
            "trained_models": self.check_directory_exists(
                self.paths["models"], "Trained ML models"
            ),
            "ml_reports": self.check_directory_exists(
                self.paths["reports"], "ML analysis reports"
            ),
        }

        return status

    def print_status_report(self, status: dict):
        """
        Prints a comprehensive status report.

        Args:
            status (dict): Status information from check_pipeline_status
        """
        print("\n" + "=" * 80)
        print("CANCERAG PIPELINE STATUS REPORT")
        print("=" * 80)

        stages = [
            ("Data Collection", status["data_collection"]),
            ("Preprocessing", status["preprocessing"]),
            ("Feature Extraction", status["feature_extraction"]),
            ("Docking", status["docking"]),
            ("Machine Learning", status["machine_learning"]),
        ]

        for stage_name, stage_status in stages:
            print(f"\n{stage_name.upper()} STAGE:")
            print("-" * 40)

            for key, info in stage_status.items():
                if "file" in info:
                    # File status
                    status_icon = "✅" if info["exists"] else "❌"
                    size_info = (
                        f" ({info['size_mb']} MB)" if info["size_mb"] > 0 else ""
                    )
                    print(
                        f"  {status_icon} {info['description']}: {info['file']}{size_info}"
                    )
                elif "directory" in info:
                    # Directory status
                    status_icon = "✅" if info["exists"] else "❌"
                    file_info = (
                        f" ({info['file_count']} files)"
                        if info["file_count"] > 0
                        else ""
                    )
                    print(
                        f"  {status_icon} {info['description']}: {info['directory']}{file_info}"
                    )

        # Overall completion status
        print("\nOVERALL STATUS:")
        print("-" * 40)

        completed_stages = 0
        total_stages = len(stages)

        for stage_name, stage_status in stages:
            stage_complete = all(info["exists"] for info in stage_status.values())
            if stage_complete:
                completed_stages += 1
                print(f"  ✅ {stage_name}: Complete")
            else:
                print(f"  ❌ {stage_name}: Incomplete")

        completion_percentage = (completed_stages / total_stages) * 100
        print(
            f"\nPipeline Completion: {completed_stages}/{total_stages} stages ({completion_percentage:.1f}%)"
        )

        # Next steps
        print("\nNEXT STEPS:")
        print("-" * 40)

        if not all(info["exists"] for info in status["data_collection"].values()):
            print("  🔄 Run data collection stage")
        elif not all(info["exists"] for info in status["preprocessing"].values()):
            print("  🔄 Run preprocessing stage")
        elif not all(info["exists"] for info in status["feature_extraction"].values()):
            print("  🔄 Run feature extraction stage")
        elif not all(info["exists"] for info in status["docking"].values()):
            print("  🔄 Run docking stage")
        elif not all(info["exists"] for info in status["machine_learning"].values()):
            print("  🔄 Run machine learning stage")
        else:
            print("  🎉 Pipeline is complete!")

    def run(self):
        """
        Runs the complete pipeline status check.
        """
        status = self.check_pipeline_status()
        self.print_status_report(status)
        return status


def main():
    """
    Main function to run the pipeline status checker.
    """
    checker = PipelineStatusChecker()
    checker.run()


if __name__ == "__main__":
    main()
