import json
import os
import time
import re
import requests
import shutil
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReceptorRetriever:
    """
    Handles the retrieval of receptor PDB structures from the RCSB PDB database.

    This class automates the process of:
    1. Searching for relevant PDB IDs based on receptor names.
    2. Downloading the corresponding PDB files.
    3. Organizing files into receptor-specific directories.
    4. Maintaining a summary JSON file to track downloaded files and avoid redundant downloads.
    """

    def __init__(
        self, output_dir: str, max_downloads: int, force_redownload: bool = False
    ):
        """
        Initializes the ReceptorRetriever.

        Args:
            output_dir (str): The root directory to save PDB files and the summary.
            max_downloads (int): The maximum number of PDB files to download per receptor.
            force_redownload (bool, optional): If True, deletes the entire output directory
                                               before starting. Defaults to False.
        """
        self.output_dir = output_dir
        self.max_downloads = max_downloads
        self.force_redownload = force_redownload
        self.summary_path = os.path.join(self.output_dir, "summary.json")
        self.session = requests.Session()  # Use a session for connection pooling

    def _sanitize_search_term(self, term: str) -> str:
        """Cleans up a receptor name for a more effective PDB search query."""
        term = (
            term.replace("α", "alpha")
            .replace("β", "beta")
            .replace("δ", "delta")
            .replace("κ", "kappa")
            .replace("μ", "mu")
        )
        term = re.sub(r"[-/]", " ", term)
        term = term.replace("receptor", "").replace("adrenoceptor", "")
        return " ".join(term.split())

    def _sanitize_dir_name(self, term: str) -> str:
        """Sanitizes a search term to create a valid directory name."""
        return re.sub(r"[^a-zA-Z0-9_]", "", term.replace(" ", "_")).lower()

    def _search_pdb_ids(self, query_text: str) -> list[str]:
        """Searches the PDB for structures matching the query text."""
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                            "operator": "contains_phrase",
                            "value": "Homo sapiens",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "struct.title",
                            "operator": "contains_words",
                            "value": query_text,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_entry_info.structure_determination_methodology",
                            "operator": "exact_match",
                            "value": "experimental",
                        },
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {
                "scoring_strategy": "combined",
                "sort": [{"sort_by": "score", "direction": "desc"}],
            },
        }
        try:
            response = self.session.post(search_url, json=query, timeout=30)
            response.raise_for_status()
            result_json = response.json()
            return [hit["identifier"] for hit in result_json.get("result_set", [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Search for '{query_text}' failed: {e}")
            return []
        except json.JSONDecodeError:
            logger.error(
                f"Search for '{query_text}' failed: Could not decode JSON from response."
            )
            return []

    def _download_pdb_file(self, pdb_id: str, target_dir: str) -> str | None:
        """Downloads a single PDB file."""
        os.makedirs(target_dir, exist_ok=True)
        output_file = os.path.join(target_dir, f"{pdb_id}.pdb")
        if os.path.exists(output_file):
            return output_file  # Already exists

        download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            response = self.session.get(download_url, timeout=30)
            response.raise_for_status()
            with open(output_file, "wb") as f:
                f.write(response.content)
            return output_file
        except requests.exceptions.RequestException as e:
            logger.warning(f"Download for {pdb_id} failed: {e}")
            return None

    def _download_for_receptor(self, search_term: str) -> dict[str, str]:
        """Searches for and downloads PDB structures for a single receptor."""
        sanitized_query = self._sanitize_search_term(search_term)
        logger.info(
            f"Searching for structures matching '{search_term}' (sanitized to '{sanitized_query}')..."
        )

        pdb_ids = self._search_pdb_ids(sanitized_query)
        if not pdb_ids:
            logger.warning(f"No potential structures found for '{search_term}'.")
            return {}

        receptor_specific_dir = os.path.join(
            self.output_dir, self._sanitize_dir_name(search_term)
        )
        logger.info(
            f"Found {len(pdb_ids)} potential structures. Attempting to download up to {self.max_downloads}..."
        )

        downloaded_files = {}
        pbar = tqdm(
            pdb_ids[: self.max_downloads],
            desc=f"Downloading for {search_term}",
            leave=False,
        )
        for pdb_id in pbar:
            time.sleep(0.2)  # Respect API rate limits
            file_path = self._download_pdb_file(pdb_id, receptor_specific_dir)
            if file_path:
                downloaded_files[pdb_id] = file_path

        logger.info(
            f"Downloaded {len(downloaded_files)} PDB files to {receptor_specific_dir}"
        )
        return downloaded_files

    def _load_summary(self) -> dict:
        """Loads the existing summary JSON file."""
        if os.path.exists(self.summary_path):
            with open(self.summary_path, "r") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    logger.warning(
                        "Could not parse existing summary.json. A new one will be created."
                    )
        return {}

    def _update_summary_file(self, summary: dict, new_data: dict):
        """Updates the summary JSON file with new receptor data."""
        summary.update(new_data)
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def run(self, receptor_list: list[str]):
        """
        Executes the PDB retrieval process for a list of receptors.

        Args:
            receptor_list (list[str]): A list of receptor names to process.
        """
        if self.force_redownload and os.path.exists(self.output_dir):
            logger.info(
                f"Force re-download enabled. Removing existing directory: {self.output_dir}"
            )
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        summary = self._load_summary()
        all_newly_downloaded = {}

        for receptor_name in tqdm(receptor_list, desc="Processing Receptors"):
            sanitized_name = self._sanitize_dir_name(receptor_name)
            if sanitized_name in summary and not self.force_redownload:
                logger.info(f"Skipping '{receptor_name}', already processed.")
                continue

            downloaded_files = self._download_for_receptor(receptor_name)
            if downloaded_files:
                all_newly_downloaded[sanitized_name] = downloaded_files

        if all_newly_downloaded:
            self._update_summary_file(summary, all_newly_downloaded)
            logger.info(f"Process complete. Summary updated at {self.summary_path}")
        else:
            logger.info("No new receptor structures were downloaded.")
