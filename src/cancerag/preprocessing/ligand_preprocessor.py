import logging
import os
import glob
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LigandPreprocessor:
    """
    Preprocesses and unifies ligand data from BiasDB and ChEMBL sources.

    This class is responsible for:
    1. Loading ligand data from raw source files.
    2. Unifying the data into a single, consistent format.
    3. Cleaning and standardizing molecules (e.g., removing salts, neutralizing).
    4. Applying drug-likeness filters (Lipinski, PAINS, etc.) based on project configuration.
    5. Saving the processed, analysis-ready dataset.
    """

    def __init__(self, config: dict):
        """
        Initializes the LigandPreprocessor with the project configuration.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.config = config
        self.paths = config["paths"]
        self.params = config["preprocessing"]

        # Initialize PAINS filter
        pains_params = FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        self.pains_filter = FilterCatalog(pains_params)

        # Standardizer
        self.unchoarger = rdMolStandardize.Uncharger()

    def _standardize_mol(self, mol: Chem.Mol) -> Chem.Mol | None:
        """Standardizes a molecule by removing salts and neutralizing it."""
        try:
            # Remove salts and fragments, keeping the largest fragment
            mol = rdMolStandardize.Cleanup(mol)
            parent = rdMolStandardize.FragmentParent(mol)
            # Neutralize charges
            neutral = self.unchoarger.uncharge(parent)
            return neutral
        except Exception as e:
            logger.warning(f"Could not standardize molecule: {e}")
            return None

    def _load_biasdb_data(self) -> pd.DataFrame:
        """Loads and formats data from the BiasDB CSV file."""
        logger.info("Loading BiasDB data...")
        df = pd.read_csv(self.paths["biasdb_input"])

        # Select the required columns
        required_columns = [
            "smiles",
            "ligand_name",
            "receptor_subtype",
            "bias_category",
        ]
        df = df[required_columns].copy()

        # Rename smiles to canonical_smiles for consistency
        df.rename(columns={"smiles": "canonical_smiles"}, inplace=True)
        df["source"] = "BiasDB"

        logger.info(f"Loaded {len(df)} records from BiasDB.")
        return df

    def _load_chembl_data(self) -> pd.DataFrame:
        """Loads and consolidates data from all ChEMBL CSV files."""
        logger.info("Loading ChEMBL data...")

        # Check if ChEMBL directory exists
        if not os.path.exists(self.paths["chembl_raw"]):
            logger.info(
                "ChEMBL directory does not exist. Skipping ChEMBL data loading."
            )
            return pd.DataFrame()

        chembl_files = glob.glob(os.path.join(self.paths["chembl_raw"], "*.csv"))
        if not chembl_files:
            logger.info("No ChEMBL files found. Skipping ChEMBL data loading.")
            return pd.DataFrame()

        all_chembl_df = []
        for f in chembl_files:
            receptor_name = os.path.basename(f).replace("_agonists.csv", "")
            df = pd.read_csv(f)
            df = df[["canonical_smiles", "molecule_chembl_id"]].copy()
            df["receptor_subtype"] = receptor_name  # Add receptor context
            df["ligand_name"] = df["molecule_chembl_id"]  # Use ChEMBL ID as ligand name
            all_chembl_df.append(df)

        if not all_chembl_df:
            return pd.DataFrame()

        consolidated_df = pd.concat(all_chembl_df, ignore_index=True)
        consolidated_df["source"] = "ChEMBL"
        consolidated_df["bias_category"] = "Agonist"  # Label as pure agonist

        # Drop the molecule_chembl_id column to match BiasDB structure
        consolidated_df = consolidated_df.drop(columns=["molecule_chembl_id"])

        logger.info(
            f"Loaded {len(consolidated_df)} records from {len(chembl_files)} ChEMBL files."
        )
        return consolidated_df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all configured filters to the unified dataframe."""
        logger.info(f"Applying filters to {len(df)} molecules...")

        # --- Pre-calculation of properties ---
        properties = []
        for mol in tqdm(df["mol_standardized"], desc="Calculating Properties"):
            try:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                tpsa = Descriptors.TPSA(mol)
                rot_bonds = Descriptors.NumRotatableBonds(mol)
                has_pains = self.pains_filter.HasMatch(mol)

                violations = (
                    (1 if mw > 500 else 0)
                    + (1 if logp > 5 else 0)
                    + (1 if hbd > 5 else 0)
                    + (1 if hba > 10 else 0)
                )

                properties.append(
                    [mw, logp, hbd, hba, tpsa, rot_bonds, violations, has_pains]
                )
            except Exception:
                properties.append([None] * 8)

        # Create properties DataFrame with proper index alignment
        props_df = pd.DataFrame(
            properties,
            columns=[
                "MW",
                "LogP",
                "HBD",
                "HBA",
                "TPSA",
                "Rotatable_Bonds",
                "Lipinski_Violations",
                "Has_PAINS",
            ],
        )

        # Ensure both DataFrames have the same length and reset indices
        if len(df) != len(props_df):
            logger.warning(f"Length mismatch: df={len(df)}, props_df={len(props_df)}")
            # Truncate to the shorter length
            min_len = min(len(df), len(props_df))
            df = df.iloc[:min_len].reset_index(drop=True)
            props_df = props_df.iloc[:min_len].reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
            props_df = props_df.reset_index(drop=True)

        df = pd.concat([df, props_df], axis=1)

        # Remove rows where any property calculation failed
        df = df.dropna(subset=["MW", "LogP", "HBD", "HBA", "TPSA", "Rotatable_Bonds"])

        # --- Filtering ---
        start_count = len(df)

        # PAINS filter - ensure boolean values
        df = df[df["Has_PAINS"] == False]
        logger.info(f"PAINS Filter: {len(df)} molecules remain.")

        # Lipinski filter
        max_violations = 0 if self.params["lipinski_strict"] else 1
        df = df[df["Lipinski_Violations"] <= max_violations]
        logger.info(f"Lipinski Filter: {len(df)} molecules remain.")

        # TPSA filter
        df = df[df["TPSA"] <= self.params["tpsa_max"]]
        logger.info(f"TPSA Filter: {len(df)} molecules remain.")

        # Rotatable bonds filter
        df = df[df["Rotatable_Bonds"] <= self.params["rotatable_bonds_max"]]
        logger.info(f"Rotatable Bonds Filter: {len(df)} molecules remain.")

        logger.info(f"Filtering complete. {start_count - len(df)} molecules removed.")
        return df

    def run(self):
        """
        Executes the full ligand preprocessing pipeline.
        This method is idempotent - it will skip processing if output already exists.
        """
        output_path = os.path.join(self.paths["processed_data"], "unified_ligands.csv")

        # Check if output already exists (idempotent behavior)
        if os.path.exists(output_path):
            logger.info(
                f"Unified ligands already exist at {output_path}. Skipping preprocessing."
            )
            return

        # 1. Load data
        biasdb_df = self._load_biasdb_data()
        chembl_df = self._load_chembl_data()

        # 2. Unify data
        if not chembl_df.empty:
            unified_df = pd.concat([biasdb_df, chembl_df], ignore_index=True)
        else:
            unified_df = biasdb_df.copy()

        unified_df.drop_duplicates(
            subset=["canonical_smiles"], keep="first", inplace=True
        )
        unified_df.dropna(subset=["canonical_smiles"], inplace=True)
        logger.info(f"Unified dataset contains {len(unified_df)} unique molecules.")

        # 3. Clean and Standardize Molecules
        mols, standardized_mols = [], []
        for smiles in tqdm(
            unified_df["canonical_smiles"], desc="Standardizing Molecules"
        ):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                standardized_mol = self._standardize_mol(mol)
                if standardized_mol:
                    mols.append(mol)
                    standardized_mols.append(standardized_mol)
                    continue
            mols.append(None)
            standardized_mols.append(None)

        unified_df["mol"] = mols
        unified_df["mol_standardized"] = standardized_mols
        unified_df.dropna(subset=["mol_standardized"], inplace=True)

        # Update SMILES to the standardized version
        unified_df["canonical_smiles_standardized"] = unified_df[
            "mol_standardized"
        ].apply(Chem.MolToSmiles)

        logger.info(f"{len(unified_df)} molecules remain after standardization.")

        # 4. Apply filters
        filtered_df = self._apply_filters(unified_df)

        # 5. Save results
        output_path = os.path.join(self.paths["processed_data"], "unified_ligands.csv")
        # Drop non-serializable columns before saving
        filtered_df.drop(columns=["mol", "mol_standardized"], inplace=True)

        os.makedirs(self.paths["processed_data"], exist_ok=True)
        filtered_df.to_csv(output_path, index=False)
        logger.info(
            f"Successfully saved {len(filtered_df)} processed ligands to {output_path}"
        )
