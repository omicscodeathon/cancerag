import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

from cancerag.utils.network import NetworkRetrier, NetworkRetrySettings

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

    def __init__(self, output_dir: str, network_config: dict | None = None):
        """
        Initializes the ChEMBLRetriever.

        Args:
            output_dir (str): The directory to save the downloaded ChEMBL data.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.target_api = new_client.target
        self.activity_api = new_client.activity
        self.mechanism_api = new_client.mechanism
        self.molecule_api = new_client.molecule
        self.retry_settings = NetworkRetrySettings.from_config(network_config)
        self.network_retrier = NetworkRetrier(self.retry_settings, logger=logger)

    def _run_with_retry(
        self,
        description: str,
        func,
        exceptions: tuple[type[BaseException], ...] = (
            requests.exceptions.RequestException,
        ),
    ):
        return self.network_retrier.run(description, func, exceptions)

    def _get_target_chembl_id(self, receptor_name: str) -> str | None:
        """
        Searches for a target receptor and returns its ChEMBL ID.
        Prioritizes single protein targets from Homo sapiens.
        """
        try:
            # Sanitize name for searching (e.g., "alpha-1A adrenoceptor" -> "alpha 1A adrenoceptor")
            search_name = receptor_name.replace("-", " ")

            def _search_targets():
                return list(self.target_api.search(search_name))

            targets = self._run_with_retry(
                f"ChEMBL target search for '{receptor_name}'",
                _search_targets,
            )

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

        except requests.exceptions.RequestException as e:
            logger.error(
                f"An error occurred while searching for target '{receptor_name}': {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error while searching for target '{receptor_name}': {e}"
            )
            return None

    def _get_molecule_smiles(self, chembl_id: str) -> Optional[str]:
        """Fetch canonical SMILES for a given ChEMBL molecule ID."""
        try:

            def _fetch():
                return self.molecule_api.get(chembl_id)

            record = self._run_with_retry(
                f"Molecule fetch for {chembl_id}",
                _fetch,
            )
            if not record:
                return None
            structures = record.get("molecule_structures") or {}
            return structures.get("canonical_smiles")
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch molecule %s: %s", chembl_id, exc)
            return None
        except Exception as exc:
            logger.error("Unexpected error fetching molecule %s: %s", chembl_id, exc)
            return None

    def _get_agonist_molecule_ids(self, target_id: str) -> List[str]:
        """Retrieve ChEMBL molecule IDs whose mechanism is annotated as agonist."""
        try:

            def _fetch():
                return list(
                    self.mechanism_api.filter(
                        target_chembl_id=target_id,
                        action_type="AGONIST",
                    )
                )

            records = self._run_with_retry(
                f"Mechanism lookup for target {target_id}",
                _fetch,
            )
            ids = []
            for record in records:
                chembl_id = record.get("molecule_chembl_id")
                if chembl_id:
                    ids.append(chembl_id)
            return ids
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch mechanisms for target %s: %s", target_id, exc)
            return []
        except Exception as exc:
            logger.error(
                "Unexpected error fetching mechanisms for target %s: %s",
                target_id,
                exc,
            )
            return []

    def _fetch_agonists(self, target_id: str, receptor_name: str) -> pd.DataFrame:
        """
        Fetches molecules annotated as agonists for the given target and returns
        canonical SMILES records ready for downstream processing.
        """
        molecule_ids = self._get_agonist_molecule_ids(target_id)
        if not molecule_ids:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for chembl_id in molecule_ids:
            smiles = self._get_molecule_smiles(chembl_id)
            if smiles:
                rows.append(
                    {
                        "canonical_smiles": smiles,
                        "ligand_name": chembl_id,
                        "molecule_chembl_id": chembl_id,
                        "receptor_subtype": receptor_name,
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["source"] = "ChEMBL"
        df["bias_category"] = "Agonist"
        return df

    def run(self, receptor_list: list[str]):
        """
        Executes the ChEMBL retrieval process for a list of receptors.

        For each receptor, it finds the corresponding ChEMBL target, fetches
        activities associated with agonists, and saves the data to a CSV file.

        Args:
            receptor_list (list[str]): A list of receptor names to process.
        """
        summary = {
            "total_receptors": len(receptor_list),
            "total_records": 0,
            "receptors": [],
        }

        logger.info(
            f"Starting ChEMBL data retrieval for {len(receptor_list)} receptors."
        )

        for receptor_name in tqdm(receptor_list, desc="Fetching ChEMBL Data"):
            sanitized_name = (
                receptor_name.replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                .lower()
            )
            output_path = os.path.join(
                self.output_dir, f"{sanitized_name}_agonists.csv"
            )

            if os.path.exists(output_path):
                logger.info(f"Data for '{receptor_name}' already exists. Skipping.")
                continue

            target_id = self._get_target_chembl_id(receptor_name)
            if not target_id:
                summary["receptors"].append(
                    {
                        "receptor": receptor_name,
                        "status": "no_target",
                        "records": 0,
                    }
                )
                continue

            agonists_df = self._fetch_agonists(target_id, receptor_name)

            if not agonists_df.empty:
                agonists_df.to_csv(output_path, index=False)
                logger.info(
                    f"Saved {len(agonists_df)} agonist activities for '{receptor_name}' to {output_path}"
                )
                summary["total_records"] += len(agonists_df)
                summary["receptors"].append(
                    {
                        "receptor": receptor_name,
                        "target_id": target_id,
                        "status": "saved",
                        "records": int(len(agonists_df)),
                        "file": output_path,
                    }
                )
            else:
                logger.warning(
                    f"No agonist activity data found for '{receptor_name}' (Target ID: {target_id})."
                )
                summary["receptors"].append(
                    {
                        "receptor": receptor_name,
                        "target_id": target_id,
                        "status": "no_data",
                        "records": 0,
                    }
                )

        summary_path = os.path.join(self.output_dir, "chembl_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("ChEMBL summary saved to %s", summary_path)
        logger.info("ChEMBL data retrieval process complete.")


if __name__ == "__main__":
    # Example usage
    test_receptors = ["5-ht1a receptor", "ADRB2", "OPRD1"]
    retriever = ChEMBLRetriever(output_dir="data/raw/chembl_test")
    retriever.run(test_receptors)
