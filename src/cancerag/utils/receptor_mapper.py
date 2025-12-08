#!/usr/bin/env python3
"""
Receptor Mapper Utility

This utility helps map ligands from BiasDB to their corresponding receptor structures
for the docking process. It handles the mapping between:

1. BiasDB receptor names (e.g., "5HT1A receptor", "D2 receptor")
2. PDB directory names (e.g., "5ht1a_receptor", "d2_receptor")
3. Selected PDB structures for docking

Usage:
    python receptor_mapper.py
"""

import json
import logging
import os

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReceptorMapper:
    """
    Maps ligands from BiasDB to their corresponding receptor structures.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the receptor mapper.

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
        self.biasdb_path = self.paths["biasdb_input"]
        self.pdb_summary_path = os.path.join(self.paths["pdb_summary"], "summary.json")
        self.selection_summary_path = os.path.join(
            self.paths["processed_data"], "structure_selection_summary.json"
        )

    def _normalize_receptor_name(self, receptor_name: str) -> str:
        """
        Normalizes receptor names to match PDB directory naming convention.

        Args:
            receptor_name (str): Original receptor name from BiasDB

        Returns:
            str: Normalized name matching PDB directory structure
        """
        # Convert to lowercase and replace spaces/special chars with underscores
        normalized = receptor_name.lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("-", "_")
        normalized = normalized.replace("α", "alpha")
        normalized = normalized.replace("β", "beta")
        normalized = normalized.replace("δ", "delta")
        normalized = normalized.replace("κ", "kappa")
        normalized = normalized.replace("μ", "mu")

        # Handle specific receptor naming conventions
        replacements = {
            "5ht1a_receptor": "5ht1a_receptor",
            "5ht2b_receptor": "5ht2b_receptor",
            "5ht2c_receptor": "5ht2c_receptor",
            "5ht7_receptor": "5ht7_receptor",
            "d1_receptor": "d1_receptor",
            "d2_receptor": "d2_receptor",
            "d3_receptor": "d3_receptor",
            "d4_receptor": "d4_receptor",
            "a1_receptor": "a1_receptor",
            "a3_receptor": "a3_receptor",
            "at1_receptor": "at1_receptor",
            "cb1_receptor": "cb1_receptor",
            "cb2_receptor": "cb2_receptor",
            "m1_receptor": "m1_receptor",
            "m2_receptor": "m2_receptor",
            "m3_receptor": "m3_receptor",
            "h2_receptor": "h2_receptor",
            "h3_receptor": "h3_receptor",
            "h4_receptor": "h4_receptor",
            "alpha1a_adrenoceptor": "1aadrenoceptor",
            "alpha1_adrenoceptor": "1adrenoceptor",
            "alpha2a_adrenoceptor": "2aadrenoceptor",
            "alpha2_adrenoceptor": "2adrenoceptor",
            "beta1_adrenoceptor": "1adrenoceptor",
            "beta2_adrenoceptor": "2adrenoceptor",
            "ghrelin_receptor": "ghrelin_receptor",
            "apelin_receptor": "apelin_receptor",
            "cxcr2": "cxcr2",
            "cxcr3": "cxcr3",
            "cxcr4": "cxcr4",
            "ccr2": "ccr2",
            "ccr5": "ccr5",
            "fpr1": "fpr1",
            "fpr2": "fpr2",
            "gpr84": "gpr84",
            "lpa1_receptor": "lpa1_receptor",
            "ffa1_receptor": "ffa1_receptor",
            "ffa2_receptor": "ffa2_receptor",
            "mc1_receptor": "mc1_receptor",
            "mc3_receptor": "mc3_receptor",
            "v2_receptor": "v2_receptor",
            "rxfp1": "rxfp1",
            "nop_receptor": "nop_receptor",
            "lh_receptor": "lh_receptor",
            "hca2_receptor": "hca2_receptor",
            "dp2_receptor": "dp2_receptor",
            "cas_receptor": "cas_receptor",
            "nk2_receptor": "nk2_receptor",
            "par2": "par2",
            "ep2": "ep2",
            "mrgprx2": "mrgprx2",
            "c5ar1": "c5ar1",
        }

        return replacements.get(normalized, normalized)

    def create_ligand_receptor_mapping(self):
        """
        Creates a mapping between ligands and their corresponding receptor structures.

        Returns:
            dict: Mapping of ligand info to receptor structure details
        """
        logger.info("Creating ligand-receptor mapping...")

        # Load BiasDB data
        if not os.path.exists(self.biasdb_path):
            logger.error(f"BiasDB data not found: {self.biasdb_path}")
            return None

        biasdb_df = pd.read_csv(self.biasdb_path)
        logger.info(f"Loaded {len(biasdb_df)} ligands from BiasDB")

        # Load PDB summary
        if not os.path.exists(self.pdb_summary_path):
            logger.error(f"PDB summary not found: {self.pdb_summary_path}")
            return None

        with open(self.pdb_summary_path, "r") as f:
            pdb_summary = json.load(f)

        # Load structure selection summary
        selection_summary = {}
        if os.path.exists(self.selection_summary_path):
            with open(self.selection_summary_path, "r") as f:
                selection_summary = json.load(f)

        # Create mapping
        ligand_mapping = {}
        unmapped_ligands = []

        for _, row in biasdb_df.iterrows():
            ligand_name = row["ligand_name"]
            receptor_subtype = row["receptor_subtype"]
            smiles = row["smiles"]
            bias_category = row["bias_category"]
            bias_pathway = row["bias_pathway"]

            # Normalize receptor name
            normalized_receptor = self._normalize_receptor_name(receptor_subtype)

            # Check if we have structures for this receptor
            if normalized_receptor in pdb_summary:
                # Get selected structure info
                selected_info = selection_summary.get(normalized_receptor, {})

                ligand_info = {
                    "ligand_name": ligand_name,
                    "smiles": smiles,
                    "receptor_subtype": receptor_subtype,
                    "normalized_receptor": normalized_receptor,
                    "bias_category": bias_category,
                    "bias_pathway": bias_pathway,
                    "available_structures": len(pdb_summary[normalized_receptor]),
                    "selected_pdb": selected_info.get("selected_pdb"),
                    "resolution": selected_info.get("resolution"),
                    "has_ligand": selected_info.get("has_ligand"),
                    "selection_score": selected_info.get("score"),
                    "can_dock": True,
                }

                ligand_mapping[ligand_name] = ligand_info
            else:
                # No structures available for this receptor
                ligand_info = {
                    "ligand_name": ligand_name,
                    "smiles": smiles,
                    "receptor_subtype": receptor_subtype,
                    "normalized_receptor": normalized_receptor,
                    "bias_category": bias_category,
                    "bias_pathway": bias_pathway,
                    "available_structures": 0,
                    "selected_pdb": None,
                    "resolution": None,
                    "has_ligand": False,
                    "selection_score": 0,
                    "can_dock": False,
                }

                ligand_mapping[ligand_name] = ligand_info
                unmapped_ligands.append(ligand_name)

        logger.info(
            f"Successfully mapped {len(ligand_mapping) - len(unmapped_ligands)} ligands"
        )
        logger.info(f"Unmapped ligands (no structures): {len(unmapped_ligands)}")

        return ligand_mapping, unmapped_ligands

    def analyze_mapping_coverage(self, ligand_mapping):
        """
        Analyzes the coverage of ligand-receptor mapping.

        Args:
            ligand_mapping (dict): The ligand mapping dictionary

        Returns:
            dict: Coverage statistics
        """
        logger.info("Analyzing mapping coverage...")

        total_ligands = len(ligand_mapping)
        dockable_ligands = sum(
            1 for info in ligand_mapping.values() if info["can_dock"]
        )

        # Group by receptor
        receptor_counts = {}
        receptor_dockable = {}

        for ligand_info in ligand_mapping.values():
            receptor = ligand_info["receptor_subtype"]
            receptor_counts[receptor] = receptor_counts.get(receptor, 0) + 1
            if ligand_info["can_dock"]:
                receptor_dockable[receptor] = receptor_dockable.get(receptor, 0) + 1

        # Group by bias category
        bias_category_counts = {}
        bias_category_dockable = {}

        for ligand_info in ligand_mapping.values():
            category = ligand_info["bias_category"]
            bias_category_counts[category] = bias_category_counts.get(category, 0) + 1
            if ligand_info["can_dock"]:
                bias_category_dockable[category] = (
                    bias_category_dockable.get(category, 0) + 1
                )

        coverage_stats = {
            "total_ligands": total_ligands,
            "dockable_ligands": dockable_ligands,
            "coverage_percentage": (dockable_ligands / total_ligands) * 100,
            "receptor_coverage": receptor_dockable,
            "bias_category_coverage": bias_category_dockable,
        }

        return coverage_stats

    def save_mapping_results(self, ligand_mapping, unmapped_ligands, coverage_stats):
        """
        Saves the mapping results to files.

        Args:
            ligand_mapping (dict): The ligand mapping dictionary
            unmapped_ligands (list): List of unmapped ligand names
            coverage_stats (dict): Coverage statistics
        """
        logger.info("Saving mapping results...")

        # Save full mapping
        mapping_path = os.path.join(
            self.paths["processed_data"], "ligand_receptor_mapping.json"
        )
        with open(mapping_path, "w") as f:
            json.dump(ligand_mapping, f, indent=2)
        logger.info(f"Ligand-receptor mapping saved to {mapping_path}")

        # Save unmapped ligands
        unmapped_path = os.path.join(
            self.paths["processed_data"], "unmapped_ligands.json"
        )
        with open(unmapped_path, "w") as f:
            json.dump(unmapped_ligands, f, indent=2)
        logger.info(f"Unmapped ligands saved to {unmapped_path}")

        # Save coverage statistics
        coverage_path = os.path.join(
            self.paths["processed_data"], "mapping_coverage_stats.json"
        )
        with open(coverage_path, "w") as f:
            json.dump(coverage_stats, f, indent=2)
        logger.info(f"Coverage statistics saved to {coverage_path}")

    def print_mapping_summary(self, ligand_mapping, unmapped_ligands, coverage_stats):
        """
        Prints a summary of the mapping results.

        Args:
            ligand_mapping (dict): The ligand mapping dictionary
            unmapped_ligands (list): List of unmapped ligand names
            coverage_stats (dict): Coverage statistics
        """
        print("\n" + "=" * 80)
        print("LIGAND-RECEPTOR MAPPING SUMMARY")
        print("=" * 80)

        print(f"\nTotal ligands in BiasDB: {coverage_stats['total_ligands']}")
        print(
            f"Ligands with available structures: {coverage_stats['dockable_ligands']}"
        )
        print(f"Coverage: {coverage_stats['coverage_percentage']:.1f}%")
        print(f"Unmapped ligands: {len(unmapped_ligands)}")

        print("\nReceptor Coverage:")
        for receptor, count in coverage_stats["receptor_coverage"].items():
            total = sum(
                1
                for info in ligand_mapping.values()
                if info["receptor_subtype"] == receptor
            )
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {receptor}: {count}/{total} ({percentage:.1f}%)")

        print("\nBias Category Coverage:")
        for category, count in coverage_stats["bias_category_coverage"].items():
            total = sum(
                1
                for info in ligand_mapping.values()
                if info["bias_category"] == category
            )
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {category}: {count}/{total} ({percentage:.1f}%)")

        if unmapped_ligands:
            print("\nTop 10 Unmapped Ligands:")
            for ligand in unmapped_ligands[:10]:
                ligand_info = ligand_mapping[ligand]
                print(f"  {ligand} -> {ligand_info['receptor_subtype']}")

    def run(self):
        """
        Runs the complete ligand-receptor mapping process.
        """
        logger.info("Starting ligand-receptor mapping process...")

        # Create mapping
        result = self.create_ligand_receptor_mapping()
        if result is None:
            return

        ligand_mapping, unmapped_ligands = result

        # Analyze coverage
        coverage_stats = self.analyze_mapping_coverage(ligand_mapping)

        # Save results
        self.save_mapping_results(ligand_mapping, unmapped_ligands, coverage_stats)

        # Print summary
        self.print_mapping_summary(ligand_mapping, unmapped_ligands, coverage_stats)

        logger.info("Ligand-receptor mapping process complete!")


def main():
    """
    Main function to run the receptor mapper.
    """
    mapper = ReceptorMapper()
    mapper.run()


if __name__ == "__main__":
    main()
