import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from cancerag.utils.network import (
    NetworkRetrier,
    NetworkRetrySettings,
    create_retry_session,
)

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
        self,
        output_dir: str,
        max_downloads: int,
        force_redownload: bool = False,
        max_retries: int = 3,
        timeout: int = 60,
        min_resolution: float = 3.5,
        network_config: Optional[dict] = None,
    ):
        """
        Initializes the ReceptorRetriever.

        Args:
            output_dir (str): The root directory to save PDB files and the summary.
            max_downloads (int): The maximum number of PDB files to download per receptor.
            force_redownload (bool, optional): If True, deletes the entire output directory
                                               before starting. Defaults to False.
            max_retries (int): Maximum number of retries for failed requests.
            timeout (int): Timeout in seconds for HTTP requests.
            min_resolution (float): Minimum resolution (in Å) for structure quality filtering.
        """
        self.output_dir = output_dir
        self.max_downloads = max_downloads
        self.force_redownload = force_redownload
        self.max_retries = max_retries
        self.timeout = timeout
        self.min_resolution = min_resolution
        self.summary_path = os.path.join(self.output_dir, "summary.json")
        self.errors_path = os.path.join(self.output_dir, "download_errors.json")

        # Network resiliency
        self.network_config = network_config
        self.retry_settings = NetworkRetrySettings.from_config(network_config)
        self.network_retrier = NetworkRetrier(self.retry_settings, logger=logger)

        # Configure session with retry strategy
        self.session = self._create_robust_session()

        # Track statistics
        self.stats = {
            "total_searched": 0,
            "total_downloaded": 0,
            "total_failed": 0,
            "total_skipped": 0,
        }

    def _create_robust_session(self) -> requests.Session:
        """
        Creates a requests session with automatic retry logic.

        Returns:
            Configured requests.Session with retry strategy
        """
        return create_retry_session(
            self.retry_settings,
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
        )

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Perform an HTTP request with the configured retry strategy.
        """
        kwargs.setdefault("timeout", self.timeout)
        return self.network_retrier.run(
            f"{method} {url}",
            lambda: self.session.request(method, url, **kwargs),
            (requests.exceptions.RequestException,),
        )

    def _validate_pdb_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validates a downloaded PDB file for quality and completeness.

        Args:
            file_path: Path to PDB file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"

            file_size = os.path.getsize(file_path)
            if file_size < 1000:  # PDB files should be at least 1KB
                return False, f"File too small ({file_size} bytes)"

            # Check file content
            with open(file_path, "r") as f:
                content = f.read(1000)  # Read first 1KB

                # Check for HTML error pages
                if "<html>" in content.lower() or "<!doctype html>" in content.lower():
                    return False, "Downloaded HTML error page instead of PDB"

                # Check for PDB header
                if not content.startswith("HEADER") and "ATOM" not in content[:500]:
                    return False, "Invalid PDB format"

            return True, None

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _get_structure_quality(self, pdb_id: str) -> Optional[Dict]:
        """
        Fetches structure quality metrics from RCSB PDB API.

        Args:
            pdb_id: PDB identifier

        Returns:
            Dictionary with quality metrics or None if unavailable
        """
        try:
            api_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = self._request_with_retry("GET", api_url)
            response.raise_for_status()

            data = response.json()

            quality = {
                "pdb_id": pdb_id,
                "resolution": data.get("rcsb_entry_info", {}).get(
                    "resolution_combined", [None]
                )[0],
                "method": data.get("exptl", [{}])[0].get("method", "Unknown"),
                "release_date": data.get("rcsb_accession_info", {}).get(
                    "initial_release_date"
                ),
                "completeness": data.get("refine", [{}])[0].get("ls_percent_reflns_obs")
                if data.get("refine")
                else None,
            }

            return quality

        except Exception as e:
            logger.debug(f"Could not fetch quality metrics for {pdb_id}: {e}")
            return None

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

    def _search_pdb_ids(
        self, query_text: str, with_resolution_filter: bool = True
    ) -> List[str]:
        """
        Searches the PDB for structures matching the query text.

        Args:
            query_text: Search term for PDB
            with_resolution_filter: Whether to filter by resolution

        Returns:
            List of PDB IDs
        """
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        # Build query nodes
        query_nodes = [
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
        ]

        # Add resolution filter if requested
        # Numeric fields require 'range' service instead of 'text'
        if with_resolution_filter and self.min_resolution:
            query_nodes.append(
                {
                    "type": "terminal",
                    "service": "range",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": self.min_resolution,
                    },
                }
            )

        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": query_nodes,
            },
            "return_type": "entry",
            "request_options": {
                "scoring_strategy": "combined",
                "sort": [{"sort_by": "score", "direction": "desc"}],
                "pager": {"start": 0, "rows": 100},  # Limit to 100 results
            },
        }

        try:
            response = self._request_with_retry("POST", search_url, json=query)
            response.raise_for_status()
            result_json = response.json()
            pdb_ids = [hit["identifier"] for hit in result_json.get("result_set", [])]
            self.stats["total_searched"] += len(pdb_ids)
            logger.info(f"Found {len(pdb_ids)} structures for '{query_text}'")
            return pdb_ids

        except requests.exceptions.Timeout:
            logger.error(f"Search for '{query_text}' timed out after {self.timeout}s")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Search for '{query_text}' failed: {e}")
            # Retry without resolution filter if it was enabled
            if with_resolution_filter:
                logger.info("Retrying search without resolution filter...")
                return self._search_pdb_ids(query_text, with_resolution_filter=False)
            return []
        except json.JSONDecodeError:
            logger.error(
                f"Search for '{query_text}' failed: Could not decode JSON from response."
            )
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search for '{query_text}': {e}")
            return []

    def _download_pdb_file(
        self, pdb_id: str, target_dir: str, retry_count: int = 0
    ) -> Optional[Tuple[str, Dict]]:
        """
        Downloads a single PDB file with validation and quality metrics.

        Args:
            pdb_id: PDB identifier
            target_dir: Directory to save file
            retry_count: Current retry attempt

        Returns:
            Tuple of (file_path, quality_metrics) or None if failed
        """
        os.makedirs(target_dir, exist_ok=True)
        output_file = os.path.join(target_dir, f"{pdb_id}.pdb")

        # Check if file already exists and is valid
        if os.path.exists(output_file):
            is_valid, error_msg = self._validate_pdb_file(output_file)
            if is_valid:
                logger.debug(f"{pdb_id} already exists and is valid")
                self.stats["total_skipped"] += 1
                quality = self._get_structure_quality(pdb_id)
                return output_file, quality or {}
            else:
                logger.warning(
                    f"Existing file {pdb_id} is invalid: {error_msg}. Re-downloading..."
                )
                os.remove(output_file)

        download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

        try:
            response = self._request_with_retry("GET", download_url, stream=True)
            response.raise_for_status()

            # Download with progress tracking
            total_size = int(response.headers.get("content-length", 0))
            with open(output_file, "wb") as f:
                if total_size:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                else:
                    f.write(response.content)

            # Validate downloaded file
            is_valid, error_msg = self._validate_pdb_file(output_file)
            if not is_valid:
                os.remove(output_file)
                raise ValueError(f"Downloaded file validation failed: {error_msg}")

            # Get quality metrics
            quality = self._get_structure_quality(pdb_id)

            # Check resolution if available
            if quality and quality.get("resolution"):
                resolution = quality["resolution"]
                if resolution > self.min_resolution:
                    logger.warning(
                        f"{pdb_id} resolution ({resolution}Å) exceeds threshold ({self.min_resolution}Å)"
                    )

            self.stats["total_downloaded"] += 1
            logger.info(f"Successfully downloaded and validated {pdb_id}")

            return output_file, quality or {}

        except requests.exceptions.Timeout:
            logger.error(f"Download for {pdb_id} timed out after {self.timeout}s")
            self.stats["total_failed"] += 1
            self._log_download_error(pdb_id, "Timeout", retry_count)
            return None

        except requests.exceptions.RequestException as e:
            logger.warning(f"Download for {pdb_id} failed: {e}")

            # Retry logic
            if retry_count < self.max_retries:
                logger.info(
                    f"Retrying download for {pdb_id} (attempt {retry_count + 1}/{self.max_retries})..."
                )
                time.sleep(2**retry_count)  # Exponential backoff
                return self._download_pdb_file(pdb_id, target_dir, retry_count + 1)

            self.stats["total_failed"] += 1
            self._log_download_error(pdb_id, str(e), retry_count)
            return None

        except Exception as e:
            logger.error(f"Unexpected error downloading {pdb_id}: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            self.stats["total_failed"] += 1
            self._log_download_error(pdb_id, str(e), retry_count)
            return None

    def _log_download_error(self, pdb_id: str, error: str, retry_count: int) -> None:
        """
        Logs download errors to a JSON file for later analysis.

        Args:
            pdb_id: PDB identifier
            error: Error message
            retry_count: Number of retries attempted
        """
        error_entry = {
            "pdb_id": pdb_id,
            "error": error,
            "retry_count": retry_count,
            "timestamp": datetime.now().isoformat(),
        }

        errors = []
        if os.path.exists(self.errors_path):
            try:
                with open(self.errors_path, "r") as f:
                    errors = json.load(f)
            except:
                pass

        errors.append(error_entry)

        with open(self.errors_path, "w") as f:
            json.dump(errors, f, indent=2)

    def _download_for_receptor(
        self, search_term: str, use_alphafold_fallback: bool = True
    ) -> dict[str, str]:
        """
        Searches for and downloads PDB structures for a single receptor.

        Args:
            search_term: Receptor name to search
            use_alphafold_fallback: If True, use AlphaFold when PDB search fails

        Returns:
            Dict mapping structure IDs to file paths
        """
        sanitized_query = self._sanitize_search_term(search_term)
        logger.info(
            f"Searching for structures matching '{search_term}' (sanitized to '{sanitized_query}')..."
        )

        pdb_ids = self._search_pdb_ids(sanitized_query)
        if not pdb_ids:
            logger.warning(f"No PDB structures found for '{search_term}'.")

            # Try AlphaFold as fallback
            if use_alphafold_fallback:
                logger.info(f"Attempting AlphaFold fallback for '{search_term}'...")
                return self._try_alphafold_fallback(search_term)

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

    def _try_alphafold_fallback(self, receptor_name: str) -> dict[str, str]:
        """
        Try to download AlphaFold structure when PDB search fails.

        Args:
            receptor_name: Receptor name

        Returns:
            Dict mapping AlphaFold ID to file path
        """
        try:
            from cancerag.data_collection.alphafold_retriever import AlphaFoldRetriever

            # Initialize AlphaFold retriever
            alphafold_retriever = AlphaFoldRetriever(
                output_dir=self.output_dir,
                max_retries=self.max_retries,
                timeout=self.timeout,
                network_config=self.network_config,
            )

            # Get UniProt ID
            uniprot_id = alphafold_retriever.get_uniprot_id(receptor_name)
            if not uniprot_id:
                logger.warning(f"No UniProt mapping for '{receptor_name}'")
                return {}

            # Try to download structure
            pdb_path = alphafold_retriever.download_structure(
                uniprot_id, receptor_name, force_download=False
            )

            if pdb_path:
                # Return in same format as PDB downloads
                return {"alphafold": pdb_path}
            else:
                logger.warning(f"AlphaFold fallback failed for '{receptor_name}'")
                return {}

        except ImportError:
            logger.error(
                "AlphaFold retriever not available. Install required dependencies."
            )
            return {}
        except Exception as e:
            logger.error(f"AlphaFold fallback error for '{receptor_name}': {e}")
            return {}

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
