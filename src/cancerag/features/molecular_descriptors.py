import logging
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
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
