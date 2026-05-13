"""
Molecular descriptors and fingerprints.

Owns:
- ``MolecularDescriptorCalculator`` — the legacy 200+ RDKit 2D descriptor block.
- Morgan / MACCS fingerprint extractors (Stage 06 — the README claimed they
  were here but they weren't).
- 3D descriptors from an embedded conformer (Stage 06).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Descriptors3D, MACCSkeys
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MolecularDescriptorCalculator:
    """
    Calculates a comprehensive set of molecular descriptors for a list of ligands.
    """

    def __init__(self, config: dict):
        """
        Initializes the MolecularDescriptorCalculator.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.paths = config["paths"]
        self.input_path = os.path.join(
            self.paths["processed_data"], "unified_ligands.csv"
        )
        self.output_path = os.path.join(
            self.paths["processed_data"], "ligands_with_descriptors.csv"
        )

        # Get the list of all available 2D descriptors from RDKit
        self.descriptor_list = [desc[0] for desc in Descriptors._descList]
        logger.info(
            f"Initialized with {len(self.descriptor_list)} available RDKit descriptors."
        )

    def _calculate_descriptors(self, mol: Chem.Mol) -> list:
        """
        Calculates all registered RDKit descriptors for a single molecule.
        """
        if mol is None:
            return [None] * len(self.descriptor_list)

        try:
            # Calculate all descriptors in the list
            return [func(mol) for name, func in Descriptors._descList]
        except Exception as e:
            logger.warning(f"Could not calculate descriptors for a molecule: {e}")
            return [None] * len(self.descriptor_list)

    def run(self):
        """
        Executes the full descriptor calculation pipeline.
        This method is idempotent - it will skip processing if output already exists.

        It loads the processed ligands, calculates ~200 molecular descriptors for each,
        and saves the augmented dataset.
        """
        # Check if output already exists (idempotent behavior)
        if os.path.exists(self.output_path):
            logger.info(
                f"Molecular descriptors already exist at {self.output_path}. Skipping calculation."
            )
            return

        logger.info(f"Loading processed ligands from {self.input_path}...")
        if not os.path.exists(self.input_path):
            logger.error(
                f"Input file not found: {self.input_path}. Halting feature extraction."
            )
            return

        ligands_df = pd.read_csv(self.input_path)

        # Use the standardized SMILES for descriptor calculation
        smiles_column = "canonical_smiles_standardized"
        if smiles_column not in ligands_df.columns:
            logger.error(
                f"Required column '{smiles_column}' not found in the input file. Halting."
            )
            return

        logger.info(f"Calculating descriptors for {len(ligands_df)} ligands...")

        all_descriptors = []
        for smiles in tqdm(
            ligands_df[smiles_column], desc="Calculating Molecular Descriptors"
        ):
            mol = Chem.MolFromSmiles(smiles)
            descriptors = self._calculate_descriptors(mol)
            all_descriptors.append(descriptors)

        # Create a new DataFrame with the descriptor data
        descriptors_df = pd.DataFrame(
            all_descriptors, columns=self.descriptor_list, index=ligands_df.index
        )

        # Combine the original data with the new descriptor data
        final_df = pd.concat([ligands_df, descriptors_df], axis=1)

        # Drop rows where descriptors could not be calculated
        final_df.dropna(subset=self.descriptor_list, how="all", inplace=True)

        logger.info(
            f"Saving {len(final_df)} ligands with descriptors to {self.output_path}..."
        )
        final_df.to_csv(self.output_path, index=False)
        logger.info("Molecular descriptor calculation complete.")


# -------------------------------------------------------------- fingerprints


def _bitvect_to_array(bv) -> np.ndarray:
    arr = np.zeros((len(bv),), dtype=np.uint8)
    for i in bv.GetOnBits():
        arr[i] = 1
    return arr


def morgan_fp(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a Morgan (ECFP-like) fingerprint as a uint8 numpy array.

    Returns an all-zero array if RDKit cannot parse the SMILES.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.zeros((n_bits,), dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return _bitvect_to_array(bv)


def maccs_fp(smiles: str) -> np.ndarray:
    """Return the 167-bit MACCS keys fingerprint as uint8."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return np.zeros((167,), dtype=np.uint8)
    bv = MACCSkeys.GenMACCSKeys(mol)
    return _bitvect_to_array(bv)


def morgan_dataframe(
    smiles: list[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    prefix: str = "morgan",
) -> pd.DataFrame:
    arr = np.stack([morgan_fp(s, radius=radius, n_bits=n_bits) for s in smiles])
    cols = [f"{prefix}_{i}" for i in range(n_bits)]
    return pd.DataFrame(arr, columns=cols)


def maccs_dataframe(
    smiles: list[str], *, prefix: str = "maccs"
) -> pd.DataFrame:
    arr = np.stack([maccs_fp(s) for s in smiles])
    cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


# ----------------------------------------------------------- 3D descriptors


def embed_3d(smiles: str, *, seed: int = 42) -> Chem.Mol | None:
    """Generate a single 3D conformer (ETKDG + MMFF94) for ``smiles``.

    Returns None if parsing/embedding/optimization fails.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        # Embedding succeeded but optimization failed; still usable.
        pass
    return mol


def descriptors_3d_from_mol(mol: Chem.Mol | None) -> dict:
    """Compute the standard RDKit 3D descriptor block."""
    if mol is None or mol.GetNumConformers() == 0:
        return {
            "Asphericity": float("nan"),
            "Eccentricity": float("nan"),
            "InertialShapeFactor": float("nan"),
            "NPR1": float("nan"),
            "NPR2": float("nan"),
            "PMI1": float("nan"),
            "PMI2": float("nan"),
            "PMI3": float("nan"),
            "RadiusOfGyration": float("nan"),
            "SpherocityIndex": float("nan"),
        }
    return {
        "Asphericity": float(Descriptors3D.Asphericity(mol)),
        "Eccentricity": float(Descriptors3D.Eccentricity(mol)),
        "InertialShapeFactor": float(Descriptors3D.InertialShapeFactor(mol)),
        "NPR1": float(Descriptors3D.NPR1(mol)),
        "NPR2": float(Descriptors3D.NPR2(mol)),
        "PMI1": float(Descriptors3D.PMI1(mol)),
        "PMI2": float(Descriptors3D.PMI2(mol)),
        "PMI3": float(Descriptors3D.PMI3(mol)),
        "RadiusOfGyration": float(Descriptors3D.RadiusOfGyration(mol)),
        "SpherocityIndex": float(Descriptors3D.SpherocityIndex(mol)),
    }


def descriptors_3d_from_smiles(smiles: str, *, seed: int = 42) -> dict:
    return descriptors_3d_from_mol(embed_3d(smiles, seed=seed))


def is_nan_block(features: dict) -> bool:
    """True if every value in a 3D-descriptor block is NaN (embedding failed)."""
    return all(
        isinstance(v, float) and (v != v) for v in features.values()  # NaN check
    )
