import os
import logging
import pandas as pd
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChEMBLRetriever:
    """
    Handles the retrieval of agonist data for specific receptors from the ChEMBL database.

    This class automates:
    1. Searching for a target receptor in ChEMBL.
    2. Filtering for activities related to agonists.
    3. Fetching and saving the relevant molecule data.
    """

    def __init__(self, output_dir: str):
        """
        Initializes the ChEMBLRetriever.

        Args:
            output_dir (str): The directory to save the downloaded ChEMBL data.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.target_api = new_client.target
        self.activity_api = new_client.activity

    def _get_target_chembl_id(self, receptor_name: str) -> str | None:
        """
        Searches for a target receptor and returns its ChEMBL ID.
        Prioritizes single protein targets from Homo sapiens.
        """
        try:
            # Sanitize name for searching (e.g., "alpha-1A adrenoceptor" -> "alpha 1A adrenoceptor")
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
                    f"No 'Homo sapiens' SINGLE PROTEIN target found for '{receptor_name}'."
                )
                return None

            # Return the first and most relevant hit
            return hs_targets[0]["target_chembl_id"]

        except Exception as e:
            logger.error(
                f"An error occurred while searching for target '{receptor_name}': {e}"
            )
            return None

    def _fetch_agonists(self, target_id: str) -> pd.DataFrame:
        """
        Fetches activities for a given target, filtering for agonists.
        """
        try:
            # Query for activities with 'agonist' in the description
            activities = self.activity_api.filter(
                target_chembl_id=target_id,
                assay_type="B",  # Binding assays
                assay_description__icontains="agonist",
            )

            if not activities:
                return pd.DataFrame()

            return pd.DataFrame.from_records(activities)
        except Exception as e:
            logger.error(f"Failed to fetch activities for target {target_id}: {e}")
            return pd.DataFrame()

    def run(self, receptor_list: list[str]):
        """
        Executes the ChEMBL retrieval process for a list of receptors.

        For each receptor, it finds the corresponding ChEMBL target, fetches
        activities associated with agonists, and saves the data to a CSV file.

        Args:
            receptor_list (list[str]): A list of receptor names to process.
        """
        logger.info(
            f"Starting ChEMBL data retrieval for {len(receptor_list)} receptors."
        )

        for receptor_name in tqdm(receptor_list, desc="Fetching ChEMBL Data"):
            sanitized_name = receptor_name.replace(" ", "_").lower()
            output_path = os.path.join(
                self.output_dir, f"{sanitized_name}_agonists.csv"
            )

            if os.path.exists(output_path):
                logger.info(f"Data for '{receptor_name}' already exists. Skipping.")
                continue

            target_id = self._get_target_chembl_id(receptor_name)
            if not target_id:
                continue

            agonists_df = self._fetch_agonists(target_id)

            if not agonists_df.empty:
                agonists_df.to_csv(output_path, index=False)
                logger.info(
                    f"Saved {len(agonists_df)} agonist activities for '{receptor_name}' to {output_path}"
                )
            else:
                logger.warning(
                    f"No agonist activity data found for '{receptor_name}' (Target ID: {target_id})."
                )

        logger.info("ChEMBL data retrieval process complete.")


if __name__ == "__main__":
    # Example usage
    test_receptors = ["5-ht1a receptor", "ADRB2", "OPRD1"]
    retriever = ChEMBLRetriever(output_dir="data/raw/chembl_test")
    retriever.run(test_receptors)
