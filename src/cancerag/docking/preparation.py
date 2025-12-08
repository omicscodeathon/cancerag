import os
import subprocess

from rdkit import Chem
from tqdm import tqdm


def prepare_receptors(receptor_structures: dict, output_dir: str) -> dict:
    """
    Prepare receptor PDB files by converting them to PDBQT format.

    Args:
        receptor_structures (dict): Maps receptor names to PDB file paths.
        output_dir (str): Directory to save the PDBQT files.

    Returns:
        dict: Maps receptor names to their new PDBQT file paths.
    """
    prepared_receptors = {}
    print("Preparing receptors for docking...")

    for receptor_name, pdb_file in receptor_structures.items():
        output_pdbqt = os.path.join(output_dir, f"{receptor_name}.pdbqt")

        if not os.path.exists(pdb_file):
            print(f"  - ERROR: Receptor PDB file not found: {pdb_file}")
            continue

        # Check if PDBQT already exists and is newer than source PDB
        if os.path.exists(output_pdbqt):
            pdb_mtime = os.path.getmtime(pdb_file)
            pdbqt_mtime = os.path.getmtime(output_pdbqt)
            if pdbqt_mtime > pdb_mtime:
                print(f"  - Using existing PDBQT: {receptor_name}")
                prepared_receptors[receptor_name] = output_pdbqt
                continue

        try:
            cmd = [
                "obabel",
                pdb_file,
                "-O",
                output_pdbqt,
                "-xr",
            ]  # Add hydrogens and compute partial charges
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            prepared_receptors[receptor_name] = output_pdbqt
            print(f"  - Prepared {receptor_name} -> {os.path.basename(output_pdbqt)}")
        except subprocess.CalledProcessError as e:
            print(f"  - ERROR preparing {receptor_name}: {e.stderr.decode().strip()}")
            continue

    return prepared_receptors


def prepare_ligands(ligands: list, output_dir: str) -> list:
    """
    Prepare ligand RDKit Mol objects by converting them to PDBQT format.

    Args:
        ligands (list): A list of RDKit Mol objects.
        output_dir (str): Directory to save the individual PDBQT files.

    Returns:
        list: A list of dictionaries, each containing info about a prepared ligand.
    """
    ligand_dir = os.path.join(output_dir, "ligands")
    os.makedirs(ligand_dir, exist_ok=True)

    prepared_ligands = []
    print("Preparing ligands for docking...")

    for idx, mol in enumerate(tqdm(ligands, desc="Preparing Ligands")):
        # Skip None molecules (failed to load from SMILES)
        if mol is None:
            continue

        # Additional safety check
        try:
            # Prefer the SDF record name set as _Name during preparation; fall back to ChEMBL_ID; else index-based name
            if mol.HasProp("_Name"):
                mol_name = mol.GetProp("_Name")
            elif mol.HasProp("ChEMBL_ID"):
                mol_name = mol.GetProp("ChEMBL_ID")
            else:
                mol_name = f"ligand_{idx}"
        except AttributeError:
            print(f"  - Skipping ligand {idx}: Invalid molecule object")
            continue

        mol_file = os.path.join(ligand_dir, f"{mol_name}.mol")
        pdbqt_file = os.path.join(ligand_dir, f"{mol_name}.pdbqt")

        # Check if ligand PDBQT already exists
        if os.path.exists(pdbqt_file):
            prepared_ligands.append(
                {"mol_idx": idx, "name": mol_name, "pdbqt_file": pdbqt_file}
            )
            continue

        Chem.MolToMolFile(mol, mol_file)

        try:
            cmd = [
                "obabel",
                mol_file,
                "-O",
                pdbqt_file,
                "-xh",
                "--partialcharge",
                "gasteiger",
            ]
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            prepared_ligands.append(
                {"mol_idx": idx, "name": mol_name, "pdbqt_file": pdbqt_file}
            )
        except subprocess.CalledProcessError as e:
            print(f"  - ERROR preparing ligand {mol_name}: {e.stderr.decode().strip()}")
            continue

    print(f"Successfully prepared {len(prepared_ligands)} ligands.")
    return prepared_ligands
