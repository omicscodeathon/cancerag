import logging
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser
from cancerag.preprocessing.receptor_preprocessor import extract_binding_site

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ActiveSiteIdentifier:
    """
    Enhanced active site identifier that selects the best receptor structure
    and identifies optimal binding sites for docking.
    """

    def __init__(self, config: dict):
        """
        Initializes the ActiveSiteIdentifier.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.paths = config["paths"]
        self.summary_path = os.path.join(self.paths["pdb_summary"], "summary.json")
        self.output_path = os.path.join(
            self.paths["processed_data"], "binding_sites.json"
        )
        self.selection_summary_path = os.path.join(
            self.paths["processed_data"], "structure_selection_summary.json"
        )
        self.parser = PDBParser(QUIET=True)

    def _evaluate_pdb_structure(self, pdb_path: str) -> dict:
        """
        Evaluates a PDB structure and returns quality metrics.

        Args:
            pdb_path (str): Path to the PDB file

        Returns:
            dict: Quality metrics including resolution, ligand presence, etc.
        """
        metrics = {
            "resolution": float("inf"),
            "has_ligand": False,
            "ligand_name": None,
            "completeness": 0.0,
            "score": 0.0,
        }

        try:
            with open(pdb_path, "r") as f:
                lines = f.readlines()

            # Extract resolution from REMARK 2
            for line in lines:
                if line.startswith("REMARK   2 RESOLUTION"):
                    try:
                        resolution_text = (
                            line.split("RESOLUTION.")[1].split("ANGSTROMS")[0].strip()
                        )
                        metrics["resolution"] = float(resolution_text)
                    except:
                        pass
                    break

            # Check for co-crystallized ligands
            ligand_residues = set()
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
                        metrics["has_ligand"] = True
                        ligand_residues.add(res_name)

            if ligand_residues:
                metrics["ligand_name"] = list(ligand_residues)[
                    0
                ]  # Use first ligand found

            # Calculate completeness (rough estimate)
            atom_count = sum(1 for line in lines if line.startswith("ATOM"))
            metrics["completeness"] = min(
                atom_count / 1000.0, 1.0
            )  # Rough completeness metric

            # Calculate overall score
            score = 0.0
            if metrics["has_ligand"]:
                score += 50.0  # High bonus for having ligand
            if metrics["resolution"] < 3.0:
                score += 30.0  # Good resolution bonus
            elif metrics["resolution"] < 2.5:
                score += 40.0  # Excellent resolution bonus
            score += metrics["completeness"] * 20.0  # Completeness bonus

            metrics["score"] = score

        except Exception as e:
            logger.warning(f"Could not evaluate PDB structure {pdb_path}: {e}")

        return metrics

    def _select_best_structure(self, receptor_name: str, pdb_files: dict) -> tuple:
        """
        Selects the best PDB structure for a receptor based on quality metrics.

        Args:
            receptor_name (str): Name of the receptor
            pdb_files (dict): Dictionary of {pdb_id: file_path}

        Returns:
            tuple: (best_pdb_id, best_pdb_path, metrics)
        """
        best_pdb_id = None
        best_pdb_path = None
        best_metrics = None
        best_score = -1

        logger.info(f"Evaluating {len(pdb_files)} structures for {receptor_name}...")

        for pdb_id, pdb_path in pdb_files.items():
            if not os.path.exists(pdb_path):
                logger.warning(f"PDB file not found: {pdb_path}")
                continue

            metrics = self._evaluate_pdb_structure(pdb_path)
            logger.info(
                f"  {pdb_id}: Resolution={metrics['resolution']:.2f}Å, "
                f"Ligand={metrics['has_ligand']}, Score={metrics['score']:.1f}"
            )

            if metrics["score"] > best_score:
                best_score = metrics["score"]
                best_pdb_id = pdb_id
                best_pdb_path = pdb_path
                best_metrics = metrics

        if best_pdb_id:
            logger.info(
                f"Selected {best_pdb_id} for {receptor_name} (score: {best_score:.1f})"
            )
        else:
            logger.warning(f"No suitable structure found for {receptor_name}")

        return best_pdb_id, best_pdb_path, best_metrics

    def _identify_binding_site(self, pdb_path: str, ligand_name: str = None) -> dict:
        """
        Identifies binding site coordinates from a PDB structure.

        Args:
            pdb_path (str): Path to the PDB file
            ligand_name (str, optional): Specific ligand name to look for

        Returns:
            dict: Binding site coordinates and metadata
        """
        binding_site = extract_binding_site(pdb_path, ligand_name)

        if binding_site:
            # Add additional metadata
            binding_site["method"] = "co_crystallized_ligand"
            binding_site["ligand_name"] = ligand_name
        else:
            logger.warning(f"Could not identify binding site for {pdb_path}")

        return binding_site

    def run(self):
        """
        Executes the enhanced active site identification pipeline.
        This method is idempotent - it will skip processing if outputs already exist.

        For each receptor:
        1. Evaluates all available PDB structures
        2. Selects the best structure based on quality metrics
        3. Identifies binding site from co-crystallized ligand
        4. Saves results to binding_sites.json
        """
        # Check if outputs already exist (idempotent behavior)
        if os.path.exists(self.output_path) and os.path.exists(
            self.selection_summary_path
        ):
            logger.info(f"Active site identification already complete. Outputs exist:")
            logger.info(f"  - Binding sites: {self.output_path}")
            logger.info(f"  - Selection summary: {self.selection_summary_path}")
            return

        logger.info("Starting enhanced active site identification...")

        if not os.path.exists(self.summary_path):
            logger.error(f"PDB summary file not found: {self.summary_path}")
            return

        with open(self.summary_path, "r") as f:
            pdb_summary = json.load(f)

        all_binding_sites = {}
        structure_selection_summary = {}

        for receptor_name, pdb_files in tqdm(
            pdb_summary.items(), desc="Processing Receptors"
        ):
            logger.info(f"\nProcessing receptor: {receptor_name}")

            # Select best structure
            best_pdb_id, best_pdb_path, best_metrics = self._select_best_structure(
                receptor_name, pdb_files
            )

            if not best_pdb_id:
                logger.warning(
                    f"Skipping {receptor_name} - no suitable structure found"
                )
                continue
                
            # Record structure selection
            structure_selection_summary[receptor_name] = {
                "selected_pdb": best_pdb_id,
                "resolution": best_metrics["resolution"],
                "has_ligand": best_metrics["has_ligand"],
                "ligand_name": best_metrics["ligand_name"],
                "score": best_metrics["score"],
            }

            # Identify binding site
            binding_site = self._identify_binding_site(
                best_pdb_path, best_metrics["ligand_name"]
            )
                
            if binding_site:
                binding_site["source_pdb"] = best_pdb_id
                binding_site["receptor_name"] = receptor_name
                all_binding_sites[receptor_name] = binding_site
                logger.info(f"Successfully identified binding site for {receptor_name}")
            else:
                logger.warning(f"Could not identify binding site for {receptor_name}")

        # Save results
        if all_binding_sites:
            logger.info(
                f"Successfully identified binding sites for {len(all_binding_sites)} receptors"
            )

            # Save binding sites
            with open(self.output_path, "w") as f:
                json.dump(all_binding_sites, f, indent=2)
            logger.info(f"Binding sites saved to {self.output_path}")

            # Save structure selection summary
            with open(self.selection_summary_path, "w") as f:
                json.dump(structure_selection_summary, f, indent=2)
            logger.info(
                f"Structure selection summary saved to {self.selection_summary_path}"
            )

        else:
            logger.error("No binding sites were identified")

    def get_receptor_structure_path(self, receptor_name: str) -> str:
        """
        Returns the path to the selected receptor structure for docking.

        Args:
            receptor_name (str): Name of the receptor

        Returns:
            str: Path to the cleaned receptor structure
        """
        if not os.path.exists(self.selection_summary_path):
            logger.error(
                "Structure selection summary not found. Run the identifier first."
            )
            return None

        with open(self.selection_summary_path, "r") as f:
            summary = json.load(f)

        if receptor_name not in summary:
            logger.error(f"Receptor {receptor_name} not found in selection summary")
            return None

        selected_pdb = summary[receptor_name]["selected_pdb"]
        cleaned_structure_path = os.path.join(
            self.paths["processed_data"], "receptors", f"{selected_pdb}.pdb"
        )

        if not os.path.exists(cleaned_structure_path):
            logger.error(f"Cleaned structure not found: {cleaned_structure_path}")
            return None

        return cleaned_structure_path
