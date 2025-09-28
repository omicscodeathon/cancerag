#!/usr/bin/env python3
"""
Receptor Structure Analyzer

This script helps analyze and visualize the receptor structures downloaded
for the CancerAg pipeline. It provides insights into:

1. Available structures per receptor
2. Quality metrics (resolution, ligand presence)
3. Structure selection rationale
4. Binding site identification results

Usage:
    python receptor_analyzer.py
"""

import json
import os
import logging
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReceptorAnalyzer:
    """
    Analyzes receptor structures and provides detailed reports.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize the analyzer.

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
        self.pdb_summary_path = os.path.join(self.paths["pdb_summary"], "summary.json")
        self.binding_sites_path = os.path.join(
            self.paths["processed_data"], "binding_sites.json"
        )
        self.selection_summary_path = os.path.join(
            self.paths["processed_data"], "structure_selection_summary.json"
        )

    def analyze_available_structures(self):
        """
        Analyzes all available receptor structures and creates a summary report.
        """
        logger.info("Analyzing available receptor structures...")

        if not os.path.exists(self.pdb_summary_path):
            logger.error(f"PDB summary not found: {self.pdb_summary_path}")
            return None

        with open(self.pdb_summary_path, "r") as f:
            pdb_summary = json.load(f)

        analysis_results = []

        for receptor_name, pdb_files in pdb_summary.items():
            receptor_info = {
                "receptor_name": receptor_name,
                "total_structures": len(pdb_files),
                "pdb_ids": list(pdb_files.keys()),
                "has_ligand_structures": 0,
                "best_resolution": float("inf"),
                "best_pdb_id": None,
            }

            # Analyze each structure
            for pdb_id, pdb_path in pdb_files.items():
                if os.path.exists(pdb_path):
                    # Check for ligands
                    with open(pdb_path, "r") as f:
                        content = f.read()
                        if any(ligand in content for ligand in ["HETATM", "HET "]):
                            # Check if it's not just water
                            lines = content.split("\n")
                            for line in lines:
                                if line.startswith("HETATM"):
                                    res_name = line[17:20].strip()
                                    if res_name not in [
                                        "HOH",
                                        "WAT",
                                        "SO4",
                                        "GOL",
                                        "PO4",
                                        "EDO",
                                        "MG",
                                        "CA",
                                        "ZN",
                                        "MN",
                                        "CL",
                                        "NA",
                                        "K",
                                    ]:
                                        receptor_info["has_ligand_structures"] += 1
                                        break

                    # Extract resolution
                    for line in content.split("\n"):
                        if line.startswith("REMARK   2 RESOLUTION"):
                            try:
                                resolution_text = (
                                    line.split("RESOLUTION.")[1]
                                    .split("ANGSTROMS")[0]
                                    .strip()
                                )
                                resolution = float(resolution_text)
                                if resolution < receptor_info["best_resolution"]:
                                    receptor_info["best_resolution"] = resolution
                                    receptor_info["best_pdb_id"] = pdb_id
                            except:
                                pass

            analysis_results.append(receptor_info)

        return analysis_results

    def create_summary_report(self, analysis_results):
        """
        Creates a comprehensive summary report of receptor structures.
        """
        logger.info("Creating summary report...")

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(analysis_results)

        print("\n" + "=" * 80)
        print("RECEPTOR STRUCTURE ANALYSIS REPORT")
        print("=" * 80)

        print(f"\nTotal receptors analyzed: {len(df)}")
        print(f"Total structures available: {df['total_structures'].sum()}")
        print(f"Average structures per receptor: {df['total_structures'].mean():.1f}")

        print(
            f"\nReceptors with co-crystallized ligands: {len(df[df['has_ligand_structures'] > 0])}"
        )
        print(f"Receptors without ligands: {len(df[df['has_ligand_structures'] == 0])}")

        print(
            f"\nBest resolution range: {df['best_resolution'].min():.2f} - {df['best_resolution'].max():.2f} Å"
        )
        print(f"Average best resolution: {df['best_resolution'].mean():.2f} Å")

        print("\n" + "-" * 80)
        print("DETAILED RECEPTOR BREAKDOWN")
        print("-" * 80)

        for _, row in df.iterrows():
            print(f"\n{row['receptor_name']}:")
            print(f"  Structures: {row['total_structures']}")
            print(f"  With ligands: {row['has_ligand_structures']}")
            print(
                f"  Best resolution: {row['best_resolution']:.2f} Å ({row['best_pdb_id']})"
            )
            print(f"  PDB IDs: {', '.join(row['pdb_ids'])}")

        return df

    def analyze_binding_sites(self):
        """
        Analyzes the identified binding sites.
        """
        logger.info("Analyzing binding sites...")

        if not os.path.exists(self.binding_sites_path):
            logger.warning(
                "Binding sites file not found. Run active site identification first."
            )
            return None

        with open(self.binding_sites_path, "r") as f:
            binding_sites = json.load(f)

        print("\n" + "=" * 80)
        print("BINDING SITE ANALYSIS")
        print("=" * 80)

        print(f"\nReceptors with identified binding sites: {len(binding_sites)}")

        for receptor_name, site_info in binding_sites.items():
            print(f"\n{receptor_name}:")
            print(f"  Source PDB: {site_info.get('source_pdb', 'N/A')}")
            print(f"  Method: {site_info.get('method', 'N/A')}")
            print(f"  Ligand: {site_info.get('ligand_name', 'N/A')}")
            print(
                f"  Center: ({site_info.get('center_x', 0):.2f}, {site_info.get('center_y', 0):.2f}, {site_info.get('center_z', 0):.2f})"
            )
            print(
                f"  Size: ({site_info.get('size_x', 0):.2f}, {site_info.get('size_y', 0):.2f}, {site_info.get('size_z', 0):.2f})"
            )

        return binding_sites

    def analyze_structure_selection(self):
        """
        Analyzes the structure selection process.
        """
        logger.info("Analyzing structure selection...")

        if not os.path.exists(self.selection_summary_path):
            logger.warning(
                "Structure selection summary not found. Run enhanced active site identification first."
            )
            return None

        with open(self.selection_summary_path, "r") as f:
            selection_summary = json.load(f)

        print("\n" + "=" * 80)
        print("STRUCTURE SELECTION ANALYSIS")
        print("=" * 80)

        print(f"\nReceptors with selected structures: {len(selection_summary)}")

        # Calculate statistics
        resolutions = [info["resolution"] for info in selection_summary.values()]
        scores = [info["score"] for info in selection_summary.values()]
        ligand_counts = [
            1 if info["has_ligand"] else 0 for info in selection_summary.values()
        ]

        print(f"\nSelection Statistics:")
        print(f"  Average resolution: {sum(resolutions) / len(resolutions):.2f} Å")
        print(f"  Average selection score: {sum(scores) / len(scores):.1f}")
        print(
            f"  Structures with ligands: {sum(ligand_counts)}/{len(ligand_counts)} ({100 * sum(ligand_counts) / len(ligand_counts):.1f}%)"
        )

        print(f"\nDetailed Selection Results:")
        for receptor_name, info in selection_summary.items():
            print(f"\n{receptor_name}:")
            print(f"  Selected PDB: {info['selected_pdb']}")
            print(f"  Resolution: {info['resolution']:.2f} Å")
            print(f"  Has ligand: {info['has_ligand']}")
            print(f"  Ligand name: {info.get('ligand_name', 'N/A')}")
            print(f"  Selection score: {info['score']:.1f}")

        return selection_summary

    def run_full_analysis(self):
        """
        Runs a complete analysis of all receptor structures and binding sites.
        """
        logger.info("Starting full receptor analysis...")

        # Analyze available structures
        analysis_results = self.analyze_available_structures()
        if analysis_results:
            self.create_summary_report(analysis_results)

        # Analyze binding sites
        self.analyze_binding_sites()

        # Analyze structure selection
        self.analyze_structure_selection()

        logger.info("Analysis complete!")


def main():
    """
    Main function to run the receptor analyzer.
    """
    analyzer = ReceptorAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
