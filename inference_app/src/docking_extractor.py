"""
Docking Feature Extraction for Inference

This module performs molecular docking for a single ligand against all available
receptors and extracts binding affinity scores to be used as features.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


class DockingFeatureExtractor:
    """
    Extracts docking affinity features for inference by docking a ligand
    against all available receptors.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the docking feature extractor.

        Args:
            base_path: Base path to project root (defaults to parent of inference_app)
        """
        if base_path is None:
            # Default to parent directory of inference_app
            base_path = Path(__file__).parent.parent.parent.parent

        self.base_path = Path(base_path)
        self.binding_sites_path = (
            self.base_path / "data" / "processed" / "binding_sites.json"
        )
        self.receptors_dir = self.base_path / "data" / "processed" / "receptors"

        # Load binding sites
        self.binding_sites = self._load_binding_sites()
        self.receptor_names = (
            list(self.binding_sites.keys()) if self.binding_sites else []
        )

        logger.info(
            f"Initialized with {len(self.receptor_names)} receptors for docking"
        )

    def _load_binding_sites(self) -> Dict:
        """Load binding sites configuration."""
        if not self.binding_sites_path.exists():
            logger.warning(
                f"Binding sites file not found: {self.binding_sites_path}. "
                "Docking features will not be available."
            )
            return {}

        try:
            with open(self.binding_sites_path, "r") as f:
                binding_sites = json.load(f)
            logger.info(f"Loaded binding sites for {len(binding_sites)} receptors")
            return binding_sites
        except Exception as e:
            logger.error(f"Failed to load binding sites: {e}")
            return {}

    def _prepare_ligand_3d(self, mol: Chem.Mol) -> Optional[str]:
        """
        Prepare ligand 3D structure and convert to PDBQT format.
        Uses the same approach as the main pipeline.

        Args:
            mol: RDKit molecule object

        Returns:
            Path to PDBQT file, or None if failed
        """
        try:
            # Generate 3D coordinates if not present
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)

            # Create temporary files
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".mol", delete=False
            ) as mol_file:
                mol_path = mol_file.name
                Chem.MolToMolFile(mol, mol_path)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdbqt", delete=False
            ) as pdbqt_file:
                pdbqt_path = pdbqt_file.name

            # Convert to PDBQT using obabel - use same flags as main pipeline
            # For ligands: -xh (add hydrogens) and --partialcharge gasteiger
            import subprocess

            result = subprocess.run(
                [
                    "obabel",
                    mol_path,
                    "-O",
                    pdbqt_path,
                    "-xh",  # Add hydrogens (for ligands)
                    "--partialcharge",
                    "gasteiger",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            # Clean up mol file
            os.unlink(mol_path)

            if result.returncode != 0:
                logger.warning(f"obabel conversion failed: {result.stderr}")
                if os.path.exists(pdbqt_path):
                    os.unlink(pdbqt_path)
                return None

            if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
                logger.warning("PDBQT file is empty or doesn't exist")
                return None

            return pdbqt_path

        except Exception as e:
            logger.error(f"Failed to prepare ligand for docking: {e}")
            return None

    def _prepare_receptor(self, receptor_name: str) -> Optional[str]:
        """
        Prepare receptor structure for docking.

        Args:
            receptor_name: Name of the receptor

        Returns:
            Path to prepared receptor PDBQT file, or None if failed
        """
        if receptor_name not in self.binding_sites:
            return None

        binding_site = self.binding_sites[receptor_name]
        pdb_id = binding_site.get("source_pdb")
        if not pdb_id:
            return None

        pdb_path = self.receptors_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            logger.warning(f"Receptor PDB file not found: {pdb_path}")
            return None

        # Check if prepared receptor already exists
        prepared_dir = (
            self.base_path / "data" / "interim" / "docking_results" / "receptors"
        )
        prepared_dir.mkdir(parents=True, exist_ok=True)
        prepared_path = prepared_dir / f"{receptor_name}.pdbqt"

        if prepared_path.exists():
            return str(prepared_path)

        # Prepare receptor using obabel
        try:
            import subprocess

            result = subprocess.run(
                [
                    "obabel",
                    str(pdb_path),
                    "-O",
                    str(prepared_path),
                    "-xr",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(
                    f"Receptor preparation failed for {receptor_name}: {result.stderr}"
                )
                return None

            if prepared_path.exists():
                return str(prepared_path)
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to prepare receptor {receptor_name}: {e}")
            return None

    def _run_docking(
        self,
        ligand_pdbqt: str,
        receptor_pdbqt: str,
        binding_site: Dict,
        receptor_name: str,
    ) -> Optional[float]:
        """
        Run docking for a single ligand-receptor pair.
        Uses the same approach as the main pipeline.

        Args:
            ligand_pdbqt: Path to ligand PDBQT file
            receptor_pdbqt: Path to receptor PDBQT file
            binding_site: Binding site configuration dictionary
            receptor_name: Name of the receptor

        Returns:
            Best binding affinity (kcal/mol), or None if failed
        """
        try:
            import subprocess

            # Create temporary output and log files
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pdbqt", delete=False
            ) as out_file:
                out_path = out_file.name

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as log_file:
                log_path = log_file.name

            # Extract binding site parameters
            center_x = binding_site.get("center_x", 0)
            center_y = binding_site.get("center_y", 0)
            center_z = binding_site.get("center_z", 0)
            size_x = binding_site.get("size_x", 20)
            size_y = binding_site.get("size_y", 20)
            size_z = binding_site.get("size_z", 20)

            # Run AutoDock Vina - use same command format as main pipeline
            # Use shell=True with single command string and redirect output
            cmd = (
                f"vina --receptor '{receptor_pdbqt}' --ligand '{ligand_pdbqt}' "
                f"--center_x {center_x} --center_y {center_y} --center_z {center_z} "
                f"--size_x {size_x} --size_y {size_y} --size_z {size_z} "
                f"--out '{out_path}' --exhaustiveness 4 --num_modes 3 --cpu 1 "
                f"> '{log_path}' 2>&1"
            )

            subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

            # Check if output file was created
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                # Read log file for error details
                error_msg = "Output file not created"
                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        log_content = f.read()
                        error_msg += f": {log_content[:500]}"
                logger.warning(f"Docking failed for {receptor_name}: {error_msg}")
                # Clean up
                if os.path.exists(out_path):
                    os.unlink(out_path)
                if os.path.exists(log_path):
                    os.unlink(log_path)
                return None

            # Parse binding affinity from log file (Vina writes to stdout which we redirected)
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log_content = f.read()
                    # Vina output format: "   1    -6.5      0.000      0.000"
                    for line in log_content.split("\n"):
                        if line.strip().startswith("1") and len(line.split()) >= 2:
                            try:
                                affinity = float(line.split()[1])
                                # Clean up files
                                if os.path.exists(out_path):
                                    os.unlink(out_path)
                                if os.path.exists(log_path):
                                    os.unlink(log_path)
                                return affinity
                            except (ValueError, IndexError):
                                continue

            logger.warning(f"Could not parse docking result for {receptor_name}")
            # Clean up
            if os.path.exists(out_path):
                os.unlink(out_path)
            if os.path.exists(log_path):
                os.unlink(log_path)
            return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Docking timeout for {receptor_name}")
            return None
        except Exception as e:
            logger.error(f"Docking error for {receptor_name}: {e}")
            return None

    def dock_single_receptor(
        self, mol: Chem.Mol, receptor_name: str
    ) -> Optional[float]:
        """
        Dock a ligand against a single receptor.

        Args:
            mol: RDKit molecule object
            receptor_name: Name of the receptor to dock against

        Returns:
            Binding affinity (kcal/mol), or None if failed
        """
        if receptor_name not in self.binding_sites:
            logger.warning(f"Receptor {receptor_name} not found in binding sites")
            return None

        # Prepare ligand
        ligand_pdbqt = self._prepare_ligand_3d(mol)
        if ligand_pdbqt is None:
            logger.warning("Failed to prepare ligand for docking")
            return None

        try:
            binding_site = self.binding_sites[receptor_name]
            receptor_pdbqt = self._prepare_receptor(receptor_name)

            if receptor_pdbqt is None:
                logger.warning(f"Receptor {receptor_name} not available")
                return None

            # Run docking
            affinity = self._run_docking(
                ligand_pdbqt, receptor_pdbqt, binding_site, receptor_name
            )

            return affinity

        finally:
            # Clean up ligand PDBQT file
            if ligand_pdbqt and os.path.exists(ligand_pdbqt):
                os.unlink(ligand_pdbqt)

    def _prepare_receptor_from_path(self, pdb_path: str) -> Optional[str]:
        """
        Prepare a receptor PDB file for docking by converting to PDBQT.

        Args:
            pdb_path: Path to receptor PDB file

        Returns:
            Path to PDBQT file, or None if failed
        """
        pdbqt_path = str(Path(pdb_path).with_suffix(".pdbqt"))

        try:
            # Convert PDB to PDBQT using obabel
            cmd = f'obabel -ipdb "{pdb_path}" -opdbqt -xr -O "{pdbqt_path}" 2>&1'
            result = os.popen(cmd).read()

            if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) < 100:
                logger.error(f"Failed to convert receptor to PDBQT: {result}")
                return None

            return pdbqt_path
        except Exception as e:
            logger.error(f"Error preparing receptor from path: {e}")
            return None

    def dock_custom_receptor_path(
        self, mol: Chem.Mol, receptor_pdb_path: str
    ) -> Optional[float]:
        """
        Dock a ligand against a custom receptor PDB file.

        Args:
            mol: RDKit molecule object
            receptor_pdb_path: Path to receptor PDB file

        Returns:
            Binding affinity (kcal/mol), or None if failed
        """
        from cancerag.preprocessing.receptor_preprocessor import extract_binding_site

        # Prepare ligand
        ligand_pdbqt = self._prepare_ligand_3d(mol)
        if ligand_pdbqt is None:
            logger.warning("Failed to prepare ligand for docking")
            return None

        receptor_pdbqt = None
        try:
            # Extract binding site from the PDB file
            binding_site = extract_binding_site(
                receptor_pdb_path, ligand_name=None, padding=5.0
            )
            if binding_site is None:
                logger.warning("Failed to extract binding site from custom receptor")
                return None

            # Prepare receptor PDBQT
            receptor_pdbqt = self._prepare_receptor_from_path(receptor_pdb_path)
            if receptor_pdbqt is None:
                logger.warning("Failed to prepare custom receptor for docking")
                return None

            # Run docking
            receptor_name = Path(receptor_pdb_path).stem
            affinity = self._run_docking(
                ligand_pdbqt, receptor_pdbqt, binding_site, receptor_name
            )

            return affinity

        finally:
            # Clean up ligand PDBQT file
            if ligand_pdbqt and os.path.exists(ligand_pdbqt):
                os.unlink(ligand_pdbqt)
            # Clean up receptor PDBQT if it was created in temp directory
            if receptor_pdbqt and os.path.exists(receptor_pdbqt):
                if "temp" in receptor_pdbqt:
                    try:
                        os.unlink(receptor_pdbqt)
                    except OSError:
                        pass

    def extract_docking_features(self, mol: Chem.Mol) -> pd.DataFrame:
        """
        Extract docking affinity features by docking against all receptors.

        Args:
            mol: RDKit molecule object

        Returns:
            DataFrame with one row containing docking affinities for all receptors
        """
        if not self.binding_sites:
            logger.warning(
                "No binding sites available. Returning empty docking features."
            )
            return pd.DataFrame()

        # Prepare ligand
        ligand_pdbqt = self._prepare_ligand_3d(mol)
        if ligand_pdbqt is None:
            logger.warning(
                "Failed to prepare ligand. Returning default docking features."
            )
            # Return default values (same as training data handling)
            docking_features = {name: -5.0 for name in self.receptor_names}
            return pd.DataFrame([docking_features])

        docking_features = {}

        try:
            # Dock against each receptor
            for receptor_name in self.receptor_names:
                binding_site = self.binding_sites[receptor_name]
                receptor_pdbqt = self._prepare_receptor(receptor_name)

                if receptor_pdbqt is None:
                    logger.warning(
                        f"Receptor {receptor_name} not available. Using default value."
                    )
                    docking_features[receptor_name] = -5.0
                    continue

                # Run docking
                affinity = self._run_docking(
                    ligand_pdbqt, receptor_pdbqt, binding_site, receptor_name
                )

                if affinity is not None:
                    docking_features[receptor_name] = affinity
                else:
                    # Use default value if docking failed
                    docking_features[receptor_name] = -5.0
                    logger.warning(
                        f"Docking failed for {receptor_name}. Using default value -5.0"
                    )

        finally:
            # Clean up ligand PDBQT file
            if ligand_pdbqt and os.path.exists(ligand_pdbqt):
                os.unlink(ligand_pdbqt)

        # Create DataFrame
        df = pd.DataFrame([docking_features])

        logger.info(f"Extracted docking features for {len(docking_features)} receptors")

        return df
