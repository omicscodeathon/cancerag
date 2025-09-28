import logging
import os
import glob
import json
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from tqdm import tqdm

logger = logging.getLogger(__name__)

# --- RECEPTOR CLEANING ---


class ReceptorPreprocessor:
    """
    Cleans raw PDB files to prepare them for docking.

    This class is responsible for:
    1. Finding all raw PDB files downloaded from the PDB.
    2. Removing all non-protein molecules (water, ligands, ions).
    3. Saving the cleaned, protein-only PDB structures to a processed directory.
    """

    def __init__(self, config: dict):
        """
        Initializes the ReceptorPreprocessor.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.paths = config["paths"]
        self.pdb_raw_dir = self.paths["pdb_summary"]
        self.pdb_processed_dir = os.path.join(self.paths["processed_data"], "receptors")
        os.makedirs(self.pdb_processed_dir, exist_ok=True)
        self.parser = PDBParser(QUIET=True)

    class ProteinSelect(Select):
        """A Bio.PDB Select class to keep only standard protein residues."""

        def accept_residue(self, residue):
            # The residue ID tuple is ('HETATM', residue_number, insertion_code) for heteroatoms
            # and (' ', residue_number, insertion_code) for standard residues.
            return residue.id[0] == " "

    def _clean_pdb_file(self, input_path: str, output_path: str):
        """
        Reads a PDB file, removes non-protein atoms, and saves the result.
        """
        try:
            structure = self.parser.get_structure("receptor", input_path)
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_path, self.ProteinSelect())
        except Exception as e:
            logger.error(f"Could not process PDB file {input_path}: {e}")

    def run(self):
        """
        Executes the full receptor cleaning pipeline.
        This method is idempotent - it will skip cleaning if output already exists.
        """
        logger.info("Starting receptor preprocessing...")

        # Find all PDB files within the subdirectories of the pdb summary path
        pdb_files = glob.glob(
            os.path.join(self.pdb_raw_dir, "**", "*.pdb"), recursive=True
        )

        if not pdb_files:
            logger.error(
                "No raw PDB files found to preprocess. Halting receptor preprocessing."
            )
            return

        logger.info(f"Found {len(pdb_files)} raw PDB files to clean.")

        # Check which files need cleaning (idempotent behavior)
        files_to_process = []
        for pdb_path in pdb_files:
            pdb_id = os.path.basename(pdb_path)
            output_path = os.path.join(self.pdb_processed_dir, pdb_id)
            if not os.path.exists(output_path):
                files_to_process.append((pdb_path, output_path))

        if not files_to_process:
            logger.info(
                "All receptor PDB files already cleaned. Skipping preprocessing."
            )
            return

        logger.info(
            f"Processing {len(files_to_process)} PDB files that need cleaning..."
        )

        for pdb_path, output_path in tqdm(
            files_to_process, desc="Cleaning Receptor PDBs"
        ):
            self._clean_pdb_file(pdb_path, output_path)

        logger.info(
            f"Receptor cleaning complete. Processed files are in {self.pdb_processed_dir}"
        )


# --- BINDING SITE EXTRACTION (for later use) ---


def extract_binding_site(pdb_file: str, ligand_name: str = None, padding: float = 5.0):
    """
    Calculates the binding site center and dimensions from a co-crystallized ligand.

    Args:
        pdb_file (str): Path to the input PDB file (the raw, not cleaned one).
        ligand_name (str, optional): The 3-letter residue name of the ligand.
                                     If None, the first non-solvent/non-ion
                                     heteroatom will be used. Defaults to None.
        padding (float, optional): Extra padding (in Angstroms) to add to the
                                   bounding box dimensions. Defaults to 5.0.

    Returns:
        dict: A dictionary containing the center and size of the binding site box.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("receptor", pdb_file)
    except Exception as e:
        logger.error(f"Could not parse PDB file {pdb_file}: {e}")
        return None

    ligand_atoms = []
    IGNORE_LIST = [
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
    ]

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith("H_"):
                    res_name = residue.get_resname().strip()
                    if ligand_name and res_name == ligand_name:
                        ligand_atoms.extend(list(residue.get_atoms()))
                        break
                    elif not ligand_name and res_name not in IGNORE_LIST:
                        logger.info(
                            f"Auto-detecting ligand in {pdb_file}. Found '{res_name}'."
                        )
                        ligand_atoms.extend(list(residue.get_atoms()))
                        break
            if ligand_atoms:
                break
        if ligand_atoms:
            break

    if not ligand_atoms:
        logger.warning(
            f"No suitable ligand found in {pdb_file}. Cannot define binding site."
        )
        return None

    coords = np.array([atom.get_coord() for atom in ligand_atoms])
    center = np.mean(coords, axis=0)
    min_coords, max_coords = np.min(coords, axis=0), np.max(coords, axis=0)
    size = (max_coords - min_coords) + (2 * padding)

    binding_site = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }
    logger.info(
        f"Calculated binding site for {pdb_file}: Center {center.round(2)}, Size {size.round(2)}"
    )
    return binding_site
