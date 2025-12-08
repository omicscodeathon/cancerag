"""
Receptor Descriptor Calculator

Calculates comprehensive binding pocket descriptors for receptor structures.
This includes geometric properties, physicochemical characteristics, and
structural features of the binding site.

Usage:
    from cancerag.features.receptor_descriptors import ReceptorDescriptorCalculator
    calculator = ReceptorDescriptorCalculator(config)
    calculator.run()
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReceptorDescriptorCalculator:
    """
    Calculates binding pocket descriptors for receptor structures.

    Extracts geometric, physicochemical, and structural features from
    the identified binding sites to enable receptor-ligand interaction
    modeling.
    """

    def __init__(self, config: dict):
        """
        Initialize the ReceptorDescriptorCalculator.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.paths = config["paths"]
        self.binding_sites_path = os.path.join(
            self.paths["processed_data"], "binding_sites.json"
        )
        self.output_path = os.path.join(
            self.paths["processed_data"], "receptor_descriptors.csv"
        )
        self.parser = PDBParser(QUIET=True)

        # Amino acid hydrophobicity (Kyte-Doolittle scale)
        self.hydrophobicity = {
            "ALA": 1.8,
            "ARG": -4.5,
            "ASN": -3.5,
            "ASP": -3.5,
            "CYS": 2.5,
            "GLN": -3.5,
            "GLU": -3.5,
            "GLY": -0.4,
            "HIS": -3.2,
            "ILE": 4.5,
            "LEU": 3.8,
            "LYS": -3.9,
            "MET": 1.9,
            "PHE": 2.8,
            "PRO": -1.6,
            "SER": -0.8,
            "THR": -0.7,
            "TRP": -0.9,
            "TYR": -1.3,
            "VAL": 4.2,
        }

        # Charged residues
        self.charged_residues = {"ARG", "LYS", "ASP", "GLU", "HIS"}

        # Polar residues
        self.polar_residues = {"SER", "THR", "ASN", "GLN", "CYS", "MET", "TRP", "TYR"}

    def _get_pocket_residues(self, pdb_path: str, binding_site: dict) -> List[Tuple]:
        """
        Extract residues within the binding pocket.

        Args:
            pdb_path: Path to PDB file
            binding_site: Binding site dictionary with center and size

        Returns:
            List of (residue object, distance) tuples
        """
        try:
            structure = self.parser.get_structure("receptor", pdb_path)
        except Exception as e:
            logger.error(f"Could not parse PDB file {pdb_path}: {e}")
            return []

        center = np.array(
            [
                binding_site.get("center_x", 0),
                binding_site.get("center_y", 0),
                binding_site.get("center_z", 0),
            ]
        )

        # Maximum radius from center to edges of box
        size = np.array(
            [
                binding_site.get("size_x", 25),
                binding_site.get("size_y", 25),
                binding_site.get("size_z", 25),
            ]
        )
        max_distance = np.sqrt(np.sum((size / 2) ** 2)) + 5.0  # Add 5A padding

        pocket_residues = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    # Only consider amino acids
                    if residue.id[0] != " ":
                        continue

                    # Get residue center of mass
                    atoms = [atom for atom in residue.get_atoms()]
                    if not atoms:
                        continue

                    coords = np.array([atom.coord for atom in atoms])
                    residue_center = np.mean(coords, axis=0)

                    # Check if within pocket radius
                    distance = np.linalg.norm(residue_center - center)
                    if distance <= max_distance:
                        pocket_residues.append((residue, distance))

        return pocket_residues

    def _calculate_pocket_volume(self, pocket_residues: List[Tuple]) -> float:
        """
        Estimate pocket volume using alpha shapes method.

        For now, uses a simple convex hull approximation.
        More sophisticated methods can be added later.

        Args:
            pocket_residues: List of (residue, distance) tuples

        Returns:
            Estimated volume in cubic Angstroms
        """
        if not pocket_residues:
            return 0.0

        try:
            from scipy.spatial import ConvexHull, Delaunay

            # Get all atom coordinates in pocket
            all_coords = []
            for residue, _ in pocket_residues:
                for atom in residue.get_atoms():
                    all_coords.append(atom.coord)

            if len(all_coords) < 4:
                return 0.0

            coords = np.array(all_coords)

            # Calculate convex hull volume
            hull = ConvexHull(coords)
            volume = hull.volume

            return volume

        except ImportError:
            logger.warning("scipy not available, using simple volume estimation")
            # Simple approximation: bounding box volume
            coords = np.array(
                [
                    atom.coord
                    for residue, _ in pocket_residues
                    for atom in residue.get_atoms()
                ]
            )
            if len(coords) == 0:
                return 0.0
            bounding_box = np.max(coords, axis=0) - np.min(coords, axis=0)
            return np.prod(bounding_box)
        except Exception as e:
            logger.error(f"Error calculating pocket volume: {e}")
            return 0.0

    def _calculate_shape_descriptors(
        self, pocket_residues: List[Tuple]
    ) -> Dict[str, float]:
        """
        Calculate shape descriptors for the pocket.

        Args:
            pocket_residues: List of (residue, distance) tuples

        Returns:
            Dictionary of shape descriptors
        """
        if not pocket_residues:
            return {"sphericity": 0.0, "asphericity": 0.0}

        coords = np.array(
            [
                atom.coord
                for residue, _ in pocket_residues
                for atom in residue.get_atoms()
            ]
        )

        if len(coords) < 3:
            return {"sphericity": 0.0, "asphericity": 0.0}

        # Calculate principal moments
        centered = coords - np.mean(coords, axis=0)
        cov_matrix = np.cov(centered.T)
        eigenvals = np.sort(np.real(np.linalg.eigvals(cov_matrix)))[::-1]

        # Calculate sphericity (measure of roundness)
        # Normalized ratio of smallest to largest principal moment
        sphericity = eigenvals[2] / eigenvals[0] if eigenvals[0] > 0 else 0.0

        # Asphericity (deviation from spherical)
        asphericity = (
            0.5 * (eigenvals[0] - eigenvals[1]) / np.sum(eigenvals)
            if np.sum(eigenvals) > 0
            else 0.0
        )

        return {"sphericity": sphericity, "asphericity": asphericity}

    def _calculate_physicochemical_properties(
        self, pocket_residues: List[Tuple]
    ) -> Dict[str, float]:
        """
        Calculate physicochemical properties of the pocket.

        Args:
            pocket_residues: List of (residue, distance) tuples

        Returns:
            Dictionary of physicochemical properties
        """
        if not pocket_residues:
            return {
                "mean_hydrophobicity": 0.0,
                "total_charge": 0.0,
                "fraction_charged": 0.0,
                "fraction_polar": 0.0,
                "fraction_hydrophobic": 0.0,
            }

        residues_list = [residue for residue, _ in pocket_residues]

        # Calculate properties
        hydrophobicities = []
        charges = 0
        charged_count = 0
        polar_count = 0
        hydrophobic_count = 0

        for residue in residues_list:
            res_name = residue.get_resname()

            # Hydrophobicity
            if res_name in self.hydrophobicity:
                hydrophobicities.append(self.hydrophobicity[res_name])

            # Charge
            if res_name == "ARG" or res_name == "LYS":
                charges += 1
                charged_count += 1
            elif res_name == "ASP" or res_name == "GLU":
                charges -= 1
                charged_count += 1
            elif res_name == "HIS":
                charged_count += 1  # Histidine charge depends on pH

            # Classification
            if res_name in self.charged_residues:
                charged_count += 0  # Already counted
            elif res_name in self.polar_residues:
                polar_count += 1
            elif res_name in self.hydrophobicity and self.hydrophobicity[res_name] > 0:
                hydrophobic_count += 1

        n_residues = len(residues_list)
        fraction_charged = charged_count / n_residues if n_residues > 0 else 0.0
        fraction_polar = polar_count / n_residues if n_residues > 0 else 0.0
        fraction_hydrophobic = hydrophobic_count / n_residues if n_residues > 0 else 0.0
        mean_hydrophobicity = np.mean(hydrophobicities) if hydrophobicities else 0.0

        return {
            "mean_hydrophobicity": mean_hydrophobicity,
            "total_charge": charges,
            "fraction_charged": fraction_charged,
            "fraction_polar": fraction_polar,
            "fraction_hydrophobic": fraction_hydrophobic,
        }

    def _calculate_residue_composition(
        self, pocket_residues: List[Tuple]
    ) -> Dict[str, float]:
        """
        Calculate amino acid composition of the pocket.

        Args:
            pocket_residues: List of (residue, distance) tuples

        Returns:
            Dictionary with amino acid frequencies
        """
        if not pocket_residues:
            return {}

        residues_list = [residue for residue, _ in pocket_residues]

        # Count residues
        composition = {}
        total = len(residues_list)

        for residue in residues_list:
            res_name = residue.get_resname()
            composition[res_name] = composition.get(res_name, 0) + 1

        # Convert to frequencies
        composition = {f"frac_{aa}": count / total for aa, count in composition.items()}

        return composition

    def _calculate_secondary_structure(
        self, pocket_residues: List[Tuple]
    ) -> Dict[str, float]:
        """
        Estimate secondary structure in pocket.

        This is a simplified version. Full DSSP analysis can be added.

        Args:
            pocket_residues: List of (residue, distance) tuples

        Returns:
            Dictionary with secondary structure fractions
        """
        # For now, return placeholder values
        # Full implementation would use DSSP or similar
        return {
            "fraction_helix": 0.0,
            "fraction_sheet": 0.0,
            "fraction_loop": 1.0,  # Most binding sites are in loop regions
        }

    def _calculate_descriptors_for_receptor(
        self, receptor_name: str, pdb_path: str, binding_site: dict
    ) -> Dict[str, float]:
        """
        Calculate all descriptors for a single receptor.

        Args:
            receptor_name: Name of the receptor
            pdb_path: Path to PDB file
            binding_site: Binding site dictionary

        Returns:
            Dictionary of all descriptors
        """
        descriptors = {"receptor_name": receptor_name}

        try:
            # Get pocket residues
            pocket_residues = self._get_pocket_residues(pdb_path, binding_site)

            if not pocket_residues:
                logger.warning(f"No pocket residues found for {receptor_name}")
                return descriptors

            # Calculate descriptors
            descriptors["n_pocket_residues"] = len(pocket_residues)

            # Volume
            volume = self._calculate_pocket_volume(pocket_residues)
            descriptors["pocket_volume"] = volume

            # Shape
            shape_desc = self._calculate_shape_descriptors(pocket_residues)
            descriptors.update(shape_desc)

            # Physicochemical
            physchem = self._calculate_physicochemical_properties(pocket_residues)
            descriptors.update(physchem)

            # Residue composition
            composition = self._calculate_residue_composition(pocket_residues)
            descriptors.update(composition)

            # Secondary structure
            ss = self._calculate_secondary_structure(pocket_residues)
            descriptors.update(ss)

        except Exception as e:
            logger.error(f"Error calculating descriptors for {receptor_name}: {e}")

        return descriptors

    def run(self):
        """
        Execute the receptor descriptor calculation pipeline.

        This method is idempotent - it will skip if output already exists.
        """
        # Check if output already exists
        if os.path.exists(self.output_path):
            logger.info(f"Receptor descriptors already calculated: {self.output_path}")
            return

        # Check if binding sites exist
        if not os.path.exists(self.binding_sites_path):
            logger.error(f"Binding sites not found: {self.binding_sites_path}")
            logger.error("Run active site identification first.")
            return

        logger.info("Starting receptor descriptor calculation...")

        # Load binding sites
        with open(self.binding_sites_path, "r") as f:
            binding_sites = json.load(f)

        all_descriptors = []

        # Get structure selection summary to find PDB paths
        selection_summary_path = os.path.join(
            self.paths["processed_data"], "structure_selection_summary.json"
        )

        if os.path.exists(selection_summary_path):
            with open(selection_summary_path, "r") as f:
                json.load(f)
        else:
            logger.error("Structure selection summary not found")
            return

        # Process each receptor
        for receptor_name, binding_site in tqdm(
            binding_sites.items(), desc="Calculating descriptors"
        ):
            # Find PDB file path
            source_pdb = binding_site.get("source_pdb")
            if not source_pdb:
                logger.warning(f"No source PDB for {receptor_name}")
                continue

            # Construct path to PDB file
            sanitized_name = receptor_name.replace(" ", "_").lower()
            pdb_path = os.path.join(
                self.paths["pdb_summary"], sanitized_name, f"{source_pdb}.pdb"
            )

            if not os.path.exists(pdb_path):
                logger.warning(f"PDB file not found: {pdb_path}")
                continue

            # Calculate descriptors
            descriptors = self._calculate_descriptors_for_receptor(
                receptor_name, pdb_path, binding_site
            )
            all_descriptors.append(descriptors)

        # Convert to DataFrame
        df = pd.DataFrame(all_descriptors)

        # Save to CSV
        df.to_csv(self.output_path, index=False)
        logger.info(f"Receptor descriptors saved to: {self.output_path}")
        logger.info(f"Calculated descriptors for {len(df)} receptors")


def main():
    """Main entry point for standalone execution."""
    import yaml

    # Load config
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run descriptor calculation
    calculator = ReceptorDescriptorCalculator(config)
    calculator.run()


if __name__ == "__main__":
    main()
