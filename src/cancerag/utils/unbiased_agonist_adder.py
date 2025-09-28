#!/usr/bin/env python3
"""
Unbiased Agonist Adder

This utility adds unbiased agonists (pure agonists) to the dataset as a separate class.
It retrieves known agonists from ChEMBL for each receptor and ensures they don't
duplicate existing biased ligands from BiasDB.

The unbiased agonists will form a new class: "Pure Agonist" for each receptor,
providing a baseline for the machine learning models to learn from.

Usage:
    python unbiased_agonist_adder.py
"""

import os
import json
import logging
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UnbiasedAgonistAdder:
    """
    Adds unbiased agonists from ChEMBL to supplement the biased dataset.
    """

    def __init__(self, config: dict):
        """
        Initialize the UnbiasedAgonistAdder.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.paths = config["paths"]
        self.biasdb_path = self.paths["biasdb_input"]
        self.output_path = os.path.join(
            self.paths["processed_data"], "unbiased_agonists.csv"
        )
        self.max_ligands_per_receptor = config["data_collection"].get(
            "max_unbiased_ligands_per_receptor", 500
        )

        # Initialize ChEMBL client
        self.target_api = new_client.target
        self.activity_api = new_client.activity

    def _get_target_chembl_id(self, receptor_name: str) -> str | None:
        """
        Searches for a target receptor and returns its ChEMBL ID.
        Prioritizes single protein targets from Homo sapiens.
        """
        try:
            # Sanitize name for searching
            search_name = receptor_name.replace("-", " ")
            targets = self.target_api.search(search_name)

            if not targets:
                logger.warning(f"No ChEMBL target found for '{receptor_name}'.")
                return None

            # Filter for human, single protein targets
            hs_targets = [
                t
                for t in targets
                if t["organism"] == "Homo sapiens"
                and t["target_type"] == "SINGLE PROTEIN"
            ]

            if not hs_targets:
                logger.warning(
                    f"No human single protein target found for '{receptor_name}'."
                )
                return None

            # Return the first match (could be improved with additional criteria)
            return hs_targets[0]["chembl_id"]

        except Exception as e:
            logger.error(f"Error searching for target '{receptor_name}': {e}")
            return None

    def _get_agonist_activities(
        self, target_chembl_id: str, receptor_name: str
    ) -> list:
        """
        Retrieves agonist activities for a given target.

        Args:
            target_chembl_id (str): ChEMBL ID of the target
            receptor_name (str): Name of the receptor for logging

        Returns:
            list: List of activity records
        """
        try:
            logger.info(
                f"Fetching agonist activities for {receptor_name} (ChEMBL: {target_chembl_id})..."
            )

            # Search for agonist activities
            activities = self.activity_api.filter(
                target_chembl_id=target_chembl_id,
                assay_type="B",  # Binding assay
                activity_type="IC50",  # Potency measurement
                standard_type="IC50",
            ).only(
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_value",
                "standard_units",
                "assay_description",
                "molecule_pref_name",
            )

            # Convert to list and filter for reasonable activities
            activity_list = []
            for activity in activities:
                try:
                    # Filter for reasonable IC50 values (nM range)
                    if (
                        activity["standard_value"]
                        and activity["standard_units"] == "nM"
                    ):
                        ic50_value = float(activity["standard_value"])
                        if 1 <= ic50_value <= 10000:  # 1 nM to 10 μM range
                            activity_list.append(
                                {
                                    "molecule_chembl_id": activity[
                                        "molecule_chembl_id"
                                    ],
                                    "canonical_smiles": activity["canonical_smiles"],
                                    "standard_value": ic50_value,
                                    "standard_units": activity["standard_units"],
                                    "assay_description": activity["assay_description"],
                                    "molecule_pref_name": activity[
                                        "molecule_pref_name"
                                    ],
                                }
                            )
                except (ValueError, TypeError):
                    continue

            logger.info(
                f"Found {len(activity_list)} agonist activities for {receptor_name}"
            )
            return activity_list

        except Exception as e:
            logger.error(f"Error fetching activities for {receptor_name}: {e}")
            return []

    def _filter_duplicate_smiles(self, new_ligands: list, existing_smiles: set) -> list:
        """
        Filters out ligands that have SMILES already present in the biased dataset.

        Args:
            new_ligands (list): List of new ligand records
            existing_smiles (set): Set of existing SMILES from BiasDB

        Returns:
            list: Filtered list without duplicates
        """
        filtered_ligands = []
        duplicates_removed = 0

        for ligand in new_ligands:
            smiles = ligand["canonical_smiles"]
            if smiles and smiles not in existing_smiles:
                filtered_ligands.append(ligand)
            else:
                duplicates_removed += 1

        logger.info(f"Removed {duplicates_removed} duplicate SMILES")
        return filtered_ligands

    def _create_unbiased_ligand_record(
        self, ligand_data: dict, receptor_name: str
    ) -> dict:
        """
        Creates a standardized ligand record for unbiased agonists.

        Args:
            ligand_data (dict): Raw ligand data from ChEMBL
            receptor_name (str): Name of the target receptor

        Returns:
            dict: Standardized ligand record
        """
        return {
            "ligand_name": ligand_data["molecule_pref_name"]
            or f"ChEMBL_{ligand_data['molecule_chembl_id']}",
            "smiles": ligand_data["canonical_smiles"],
            "smiles_duplicate": ligand_data[
                "canonical_smiles"
            ],  # Same as smiles for unbiased
            "receptor_family": "ChEMBL",  # Mark as ChEMBL source
            "receptor": receptor_name,
            "receptor_subtype": receptor_name,
            "bias_category": "Pure Agonist",  # New class for unbiased agonists
            "bias_pathway": "Balanced",  # Balanced signaling
            "reference_ligand": "ChEMBL",
            "assay_1": ligand_data["assay_description"] or "ChEMBL agonist assay",
            "assay_2": "IC50",
            "publication_title": "ChEMBL Database",
            "author": "ChEMBL",
            "doi": "N/A",
            "pmid": "N/A",
            "year": 2024,  # Current year
            "molecular_weight": None,  # Will be calculated later
            "logp": None,  # Will be calculated later
            "hba": None,  # Will be calculated later
            "hbd": None,  # Will be calculated later
            "rings": None,  # Will be calculated later
            "tpsa": None,  # Will be calculated later
            "chembl_id": ligand_data["molecule_chembl_id"],
            "ic50_nm": ligand_data["standard_value"],
            "source": "ChEMBL",
        }

    def run(self):
        """
        Executes the unbiased agonist addition pipeline.

        Process:
        1. Load existing biased ligands from BiasDB
        2. Extract unique receptor names
        3. For each receptor, fetch agonist activities from ChEMBL
        4. Filter out duplicates (SMILES already in BiasDB)
        5. Create standardized records with "Pure Agonist" class
        6. Save to CSV file
        """
        logger.info("Starting unbiased agonist addition process...")

        # Load existing biased ligands
        if not os.path.exists(self.biasdb_path):
            logger.error(f"BiasDB data not found: {self.biasdb_path}")
            return

        biasdb_df = pd.read_csv(self.biasdb_path)
        logger.info(f"Loaded {len(biasdb_df)} biased ligands from BiasDB")

        # Extract existing SMILES to avoid duplicates
        existing_smiles = set(biasdb_df["smiles"].dropna().tolist())
        logger.info(f"Found {len(existing_smiles)} unique SMILES in biased dataset")

        # Get unique receptors
        unique_receptors = biasdb_df["receptor_subtype"].dropna().unique()
        logger.info(f"Found {len(unique_receptors)} unique receptors")

        # Collect unbiased agonists
        all_unbiased_ligands = []

        for receptor_name in tqdm(unique_receptors, desc="Processing Receptors"):
            logger.info(f"\nProcessing receptor: {receptor_name}")

            # Get ChEMBL target ID
            target_chembl_id = self._get_target_chembl_id(receptor_name)
            if not target_chembl_id:
                logger.warning(f"Skipping {receptor_name} - no ChEMBL target found")
                continue

            # Get agonist activities
            activities = self._get_agonist_activities(target_chembl_id, receptor_name)
            if not activities:
                logger.warning(f"No agonist activities found for {receptor_name}")
                continue

            # Filter duplicates
            filtered_activities = self._filter_duplicate_smiles(
                activities, existing_smiles
            )

            # Limit number of ligands per receptor
            if len(filtered_activities) > self.max_ligands_per_receptor:
                # Sort by potency (lower IC50 = more potent)
                filtered_activities.sort(key=lambda x: x["standard_value"])
                filtered_activities = filtered_activities[
                    : self.max_ligands_per_receptor
                ]
                logger.info(
                    f"Limited to top {self.max_ligands_per_receptor} most potent ligands"
                )

            # Create standardized records
            for activity in filtered_activities:
                ligand_record = self._create_unbiased_ligand_record(
                    activity, receptor_name
                )
                all_unbiased_ligands.append(ligand_record)

            logger.info(
                f"Added {len(filtered_activities)} unbiased agonists for {receptor_name}"
            )

        # Save results
        if all_unbiased_ligands:
            unbiased_df = pd.DataFrame(all_unbiased_ligands)
            unbiased_df.to_csv(self.output_path, index=False)

            logger.info(
                f"Successfully added {len(all_unbiased_ligands)} unbiased agonists"
            )
            logger.info(f"Unbiased agonists saved to {self.output_path}")

            # Print summary
            self._print_summary(unbiased_df, biasdb_df)
        else:
            logger.warning("No unbiased agonists were added")

    def _print_summary(self, unbiased_df: pd.DataFrame, biasdb_df: pd.DataFrame):
        """
        Prints a summary of the unbiased agonist addition.

        Args:
            unbiased_df (pd.DataFrame): DataFrame of unbiased agonists
            biasdb_df (pd.DataFrame): DataFrame of biased ligands
        """
        print("\n" + "=" * 80)
        print("UNBIASED AGONIST ADDITION SUMMARY")
        print("=" * 80)

        print(f"\nOriginal biased ligands: {len(biasdb_df)}")
        print(f"Added unbiased agonists: {len(unbiased_df)}")
        print(f"Total ligands: {len(biasdb_df) + len(unbiased_df)}")

        print(f"\nReceptor Coverage:")
        receptor_counts = unbiased_df["receptor_subtype"].value_counts()
        for receptor, count in receptor_counts.items():
            print(f"  {receptor}: {count} unbiased agonists")

        print(f"\nClass Distribution:")
        print(f"  Biased ligands: {len(biasdb_df)}")
        print(f"  Pure agonists: {len(unbiased_df)}")

        print(f"\nSample Unbiased Agonists:")
        sample_ligands = unbiased_df.head(5)
        for _, row in sample_ligands.iterrows():
            print(
                f"  {row['ligand_name']} -> {row['receptor_subtype']} (IC50: {row['ic50_nm']:.1f} nM)"
            )


def main():
    """
    Main function to run the unbiased agonist adder.
    """
    import yaml

    # Load config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "configs", "config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    adder = UnbiasedAgonistAdder(config)
    adder.run()


if __name__ == "__main__":
    main()
