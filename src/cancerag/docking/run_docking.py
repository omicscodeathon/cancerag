import logging
import os
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from .pipeline import DockingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_ligands_for_docking(config: dict) -> str | None:
    """
    Prepares a single SDF file containing 3D ligand structures.

    Reads the processed ligands CSV, generates 3D conformers, and saves them
    to an SDF file, which is the required input format for the DockingPipeline.

    Args:
        config (dict): The project's configuration dictionary.

    Returns:
        str | None: The path to the generated SDF file, or None if it fails.
    """
    paths = config["paths"]
    input_csv = os.path.join(paths["processed_data"], "drug_like_ligands_clean.csv")
    output_sdf = os.path.join(paths["interim_data"], "ligands_for_docking.sdf")
    os.makedirs(paths["interim_data"], exist_ok=True)

    if not os.path.exists(input_csv):
        logger.error(f"Ligand input file not found: {input_csv}")
        return None

    # Check if SDF already exists and is newer than input CSV
    if os.path.exists(output_sdf):
        input_mtime = os.path.getmtime(input_csv)
        output_mtime = os.path.getmtime(output_sdf)
        if output_mtime > input_mtime:
            logger.info(f"Using existing SDF file: {output_sdf}")
            return output_sdf

    logger.info(f"Preparing 3D structures for ligands from {input_csv}...")
    ligands_df = pd.read_csv(input_csv)

    writer = Chem.SDWriter(output_sdf)

    for _, row in ligands_df.iterrows():
        smiles = row["canonical_smiles_standardized"]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            try:
                AllChem.UFFOptimizeMolecule(mol)
                # Carry over important properties to the SDF file
                mol.SetProp("_Name", str(row.get("ligand_name", "N/A")))
                mol.SetProp("receptor_subtype", str(row.get("receptor_subtype", "N/A")))
                writer.write(mol)
            except Exception as e:
                logger.warning(
                    f"Could not generate 3D structure for SMILES {smiles}: {e}"
                )

    writer.close()
    logger.info(f"3D ligand structures saved to {output_sdf}")
    return output_sdf


def run_docking_stage(config: dict):
    """
    Main function to set up and execute the entire docking pipeline.
    """
    logger.info("--- Setting up Docking Pipeline ---")
    paths = config["paths"]

    # 1. Prepare the ligand SDF file
    ligand_sdf_path = prepare_ligands_for_docking(config)
    if not ligand_sdf_path:
        logger.error("Halting docking stage due to ligand preparation failure.")
        return

    # 2. Load receptor and binding site data
    cleaned_receptors_dir = os.path.join(paths["processed_data"], "receptors")
    binding_sites_path = os.path.join(paths["processed_data"], "binding_sites.json")

    if not os.path.exists(binding_sites_path):
        logger.error(f"Binding sites file not found: {binding_sites_path}. Halting.")
        return

    with open(binding_sites_path, "r") as f:
        binding_sites = json.load(f)

    # Create the receptor_structures dictionary expected by the pipeline
    receptor_structures = {}
    for receptor_name in binding_sites.keys():
        # The binding site keys are the sanitized dir names
        pdb_id = binding_sites[receptor_name]["source_pdb"]
        pdb_path = os.path.join(cleaned_receptors_dir, f"{pdb_id}.pdb")
        if os.path.exists(pdb_path):
            receptor_structures[receptor_name] = pdb_path
        else:
            logger.warning(
                f"Cleaned PDB file not found for {receptor_name} ({pdb_id}). It will be skipped."
            )

    if not receptor_structures:
        logger.error(
            "No valid, cleaned receptor structures found for docking. Halting."
        )
        return

    # 3. Initialize and run the pipeline
    docking_pipeline = DockingPipeline(
        ligand_file=ligand_sdf_path,
        receptor_structures=receptor_structures,
        binding_sites=binding_sites,
        output_dir=os.path.join(config["paths"]["reports"], "docking_results"),
        num_cpu=config["docking"].get("num_cpu"),  # Use num_cpu from config
    )

    docking_pipeline.run_pipeline()
    logger.info("--- Docking Pipeline Stage Finished ---")
