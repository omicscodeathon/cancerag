"""
AlphaFold Structure Retriever

Downloads predicted protein structures from the AlphaFold Database for receptors
that don't have experimental PDB structures available.

AlphaFold DB: https://alphafold.ebi.ac.uk/
API Documentation: https://alphafold.ebi.ac.uk/api-docs
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

from cancerag.utils.network import (
    NetworkRetrier,
    NetworkRetrySettings,
    create_retry_session,
)

logger = logging.getLogger(__name__)


class AlphaFoldRetriever:
    """
    Downloads AlphaFold predicted structures for GPCRs and other receptors.

    This class:
    1. Maps receptor names to UniProt IDs
    2. Downloads AlphaFold PDB predictions
    3. Saves structures with metadata (pLDDT scores, model version)
    4. Maintains compatibility with existing PDB pipeline
    """

    # Mapping of common receptor names to UniProt IDs (Human)
    RECEPTOR_UNIPROT_MAP = {
        # Serotonin receptors
        "5HT1A receptor": "P08908",  # HTR1A
        "5HT1B receptor": "P28222",  # HTR1B
        "5HT2B receptor": "P41595",  # HTR2B
        "5HT2C receptor": "P28335",  # HTR2C
        "5HT7 receptor": "P34969",  # HTR7
        # Adenosine receptors
        "A2B receptor": "P29275",  # ADORA2B
        # Sphingosine receptors
        "S1P1 receptor": "P21453",  # S1PR1
        # Neurotensin receptors
        "NTS1 receptor": "P30989",  # NTSR1
        # Melanocortin receptors
        "MC4 receptor": "P32245",  # MC4R
        "MC5 receptor": "P33032",  # MC5R
        # Other receptors
        "OXE receptor": "Q8TDS5",  # OXER1
        # Additional common receptors (for future use)
        "D1 receptor": "P21728",  # DRD1
        "D2 receptor": "P14416",  # DRD2
        "D3 receptor": "P35462",  # DRD3
        "D4 receptor": "P21917",  # DRD4
        "β1-adrenoceptor": "P08588",  # ADRB1
        "β2-adrenoceptor": "P07550",  # ADRB2
        "α1A-adrenoceptor": "P35348",  # ADRA1A
        "α2A-adrenoceptor": "P08913",  # ADRA2A
        "α2C-adrenoceptor": "P18825",  # ADRA2C
        "M1 receptor": "P11229",  # CHRM1
        "M2 receptor": "P08172",  # CHRM2
        "M3 receptor": "P20309",  # CHRM3
        "CB1 receptor": "P21554",  # CNR1
        "CB2 receptor": "P34972",  # CNR2
        "H2 receptor": "P25021",  # HRH2
        "H3 receptor": "Q9Y5N1",  # HRH3
        "H4 receptor": "Q9H3N8",  # HRH4
        "μ receptor": "P35372",  # OPRM1 (mu opioid)
        "δ receptor": "P41143",  # OPRD1 (delta opioid)
        "κ receptor": "P41145",  # OPRK1 (kappa opioid)
    }

    def __init__(
        self,
        output_dir: str,
        max_retries: int = 3,
        timeout: int = 60,
        cache_dir: Optional[str] = None,
        network_config: Optional[dict] = None,
    ):
        """
        Initialize the AlphaFold retriever.

        Args:
            output_dir: Directory to save AlphaFold structures
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            cache_dir: Optional cache directory for metadata
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = max_retries
        self.timeout = timeout
        self.cache_dir = (
            Path(cache_dir) if cache_dir else self.output_dir / "alphafold_cache"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Network resiliency
        self.retry_settings = NetworkRetrySettings.from_config(network_config)
        self.network_retrier = NetworkRetrier(self.retry_settings, logger=logger)

        # AlphaFold DB API endpoints
        self.api_base = "https://alphafold.ebi.ac.uk/api"
        self.files_base = "https://alphafold.ebi.ac.uk/files"

        # Summary tracking
        self.summary_path = self.output_dir / "alphafold_summary.json"
        self.summary = self._load_summary()

        # Configure session with retry
        self.session = self._create_session()

        # Statistics
        self.stats = {
            "total_searched": 0,
            "total_downloaded": 0,
            "total_failed": 0,
            "total_skipped": 0,
        }

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        return create_retry_session(
            self.retry_settings,
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Execute HTTP request with retry/backoff support."""
        kwargs.setdefault("timeout", self.timeout)
        return self.network_retrier.run(
            f"{method} {url}",
            lambda: self.session.request(method, url, **kwargs),
            (requests.exceptions.RequestException,),
        )

    def _load_summary(self) -> Dict:
        """Load existing summary of downloaded structures."""
        if self.summary_path.exists():
            try:
                with open(self.summary_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load AlphaFold summary: {e}")
        return {}

    def _save_summary(self):
        """Save summary of downloaded structures."""
        with open(self.summary_path, "w") as f:
            json.dump(self.summary, f, indent=2)

    def get_uniprot_id(self, receptor_name: str) -> Optional[str]:
        """
        Get UniProt ID for a receptor name.

        Args:
            receptor_name: Common receptor name

        Returns:
            UniProt ID or None if not found
        """
        # Direct lookup
        uniprot_id = self.RECEPTOR_UNIPROT_MAP.get(receptor_name)
        if uniprot_id:
            return uniprot_id

        # Try case-insensitive lookup
        receptor_lower = receptor_name.lower()
        for name, uid in self.RECEPTOR_UNIPROT_MAP.items():
            if name.lower() == receptor_lower:
                return uid

        # Try partial match
        for name, uid in self.RECEPTOR_UNIPROT_MAP.items():
            if receptor_lower in name.lower() or name.lower() in receptor_lower:
                logger.info(f"Partial match for '{receptor_name}' -> '{name}' ({uid})")
                return uid

        logger.warning(f"No UniProt ID found for receptor: {receptor_name}")
        return None

    def get_structure_metadata(self, uniprot_id: str) -> Optional[Dict]:
        """
        Fetch metadata about an AlphaFold structure.

        Args:
            uniprot_id: UniProt accession ID

        Returns:
            Metadata dict or None if not found
        """
        try:
            url = f"{self.api_base}/prediction/{uniprot_id}"
            response = self._request_with_retry("GET", url)

            if response.status_code == 404:
                logger.warning(f"No AlphaFold prediction found for {uniprot_id}")
                return None

            response.raise_for_status()
            metadata = response.json()

            return metadata

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch metadata for {uniprot_id}: {e}")
            return None

    def download_structure(
        self, uniprot_id: str, receptor_name: str, force_download: bool = False
    ) -> Optional[str]:
        """
        Download AlphaFold structure for a UniProt ID.

        Args:
            uniprot_id: UniProt accession ID
            receptor_name: Human-readable receptor name
            force_download: Re-download even if file exists

        Returns:
            Path to downloaded PDB file or None
        """
        # Check if already downloaded
        receptor_dir = self.output_dir / receptor_name.replace(" ", "_")
        pdb_filename = f"AF-{uniprot_id}-F1-model_v4.pdb"
        pdb_path = receptor_dir / pdb_filename

        if pdb_path.exists() and not force_download:
            logger.info(f"AlphaFold structure already exists for {receptor_name}")
            self.stats["total_skipped"] += 1
            return str(pdb_path)

        # Create receptor directory
        receptor_dir.mkdir(parents=True, exist_ok=True)

        # Fetch metadata first
        metadata = self.get_structure_metadata(uniprot_id)
        if not metadata:
            self.stats["total_failed"] += 1
            return None

        # Download PDB file
        try:
            pdb_url = f"{self.files_base}/{pdb_filename}"
            logger.info(f"Downloading AlphaFold structure: {pdb_url}")

            response = self._request_with_retry("GET", pdb_url)
            response.raise_for_status()

            # Save PDB file
            with open(pdb_path, "wb") as f:
                f.write(response.content)

            # Validate file
            if not self._validate_alphafold_pdb(pdb_path):
                logger.error(f"Downloaded file is invalid: {pdb_path}")
                pdb_path.unlink()
                self.stats["total_failed"] += 1
                return None

            # Extract quality metrics
            quality_info = self._extract_quality_metrics(pdb_path, metadata)

            # Update summary
            if receptor_name not in self.summary:
                self.summary[receptor_name] = []

            self.summary[receptor_name].append(
                {
                    "source": "AlphaFold",
                    "uniprot_id": uniprot_id,
                    "file": str(pdb_path.relative_to(self.output_dir)),
                    "model_version": "v4",
                    "mean_plddt": quality_info.get("mean_plddt"),
                    "organism": "Homo sapiens",
                    "download_date": time.strftime("%Y-%m-%d"),
                    "metadata": quality_info,
                }
            )

            self._save_summary()
            self.stats["total_downloaded"] += 1

            logger.info(
                f"✓ Downloaded AlphaFold structure for {receptor_name} (pLDDT: {quality_info.get('mean_plddt', 'N/A'):.1f})"
            )

            return str(pdb_path)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download structure for {uniprot_id}: {e}")
            self.stats["total_failed"] += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading {uniprot_id}: {e}")
            self.stats["total_failed"] += 1
            return None

    def _validate_alphafold_pdb(self, pdb_path: Path) -> bool:
        """
        Validate an AlphaFold PDB file.

        Args:
            pdb_path: Path to PDB file

        Returns:
            True if valid
        """
        if not pdb_path.exists():
            return False

        if pdb_path.stat().st_size < 1000:
            logger.error(f"PDB file too small: {pdb_path}")
            return False

        # Check for valid PDB content
        try:
            with open(pdb_path, "r") as f:
                content = f.read(2000)

                # Should have ATOM records
                if "ATOM" not in content:
                    logger.error(f"No ATOM records in PDB: {pdb_path}")
                    return False

                # AlphaFold files have specific headers
                if "ALPHAFOLD" not in content.upper():
                    logger.warning(f"May not be an AlphaFold file: {pdb_path}")

                return True

        except Exception as e:
            logger.error(f"Error validating PDB file {pdb_path}: {e}")
            return False

    def _extract_quality_metrics(self, pdb_path: Path, metadata: Dict) -> Dict:
        """
        Extract quality metrics from AlphaFold structure.

        AlphaFold provides pLDDT (predicted Local Distance Difference Test) scores
        in the B-factor column of the PDB file.

        Args:
            pdb_path: Path to PDB file
            metadata: Metadata from API

        Returns:
            Quality metrics dict
        """
        quality = {
            "mean_plddt": None,
            "min_plddt": None,
            "max_plddt": None,
            "sequence_length": None,
        }

        try:
            plddt_scores = []

            with open(pdb_path, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        # B-factor column contains pLDDT score
                        plddt = float(line[60:66].strip())
                        plddt_scores.append(plddt)

            if plddt_scores:
                quality["mean_plddt"] = sum(plddt_scores) / len(plddt_scores)
                quality["min_plddt"] = min(plddt_scores)
                quality["max_plddt"] = max(plddt_scores)
                quality["sequence_length"] = len(plddt_scores)

                # Quality interpretation
                if quality["mean_plddt"] > 90:
                    quality["confidence"] = "Very High"
                elif quality["mean_plddt"] > 70:
                    quality["confidence"] = "High"
                elif quality["mean_plddt"] > 50:
                    quality["confidence"] = "Medium"
                else:
                    quality["confidence"] = "Low"

            # Add metadata from API
            if metadata:
                quality["gene_name"] = metadata.get("geneName")
                quality["organism"] = metadata.get("organismScientificName")
                quality["sequence_version"] = metadata.get("sequenceVersionDate")
                quality["model_created"] = metadata.get("modelCreatedDate")

        except Exception as e:
            logger.warning(f"Could not extract quality metrics: {e}")

        return quality

    def download_for_receptors(
        self, receptor_names: List[str], force_download: bool = False
    ) -> Dict[str, Optional[str]]:
        """
        Download AlphaFold structures for multiple receptors.

        Args:
            receptor_names: List of receptor names
            force_download: Re-download existing files

        Returns:
            Dict mapping receptor names to file paths (None if failed)
        """
        results = {}

        logger.info(
            f"Downloading AlphaFold structures for {len(receptor_names)} receptors..."
        )

        for receptor_name in tqdm(receptor_names, desc="AlphaFold Downloads"):
            self.stats["total_searched"] += 1

            # Get UniProt ID
            uniprot_id = self.get_uniprot_id(receptor_name)
            if not uniprot_id:
                logger.warning(f"No UniProt mapping for: {receptor_name}")
                results[receptor_name] = None
                continue

            # Download structure
            pdb_path = self.download_structure(
                uniprot_id, receptor_name, force_download
            )
            results[receptor_name] = pdb_path

            # Rate limiting
            time.sleep(0.5)

        # Print statistics
        logger.info("\n" + "=" * 60)
        logger.info("AlphaFold Download Summary:")
        logger.info(f"  Total searched: {self.stats['total_searched']}")
        logger.info(f"  Downloaded: {self.stats['total_downloaded']}")
        logger.info(f"  Skipped (existing): {self.stats['total_skipped']}")
        logger.info(f"  Failed: {self.stats['total_failed']}")
        logger.info("=" * 60)

        return results

    def add_uniprot_mapping(self, receptor_name: str, uniprot_id: str):
        """
        Add a custom receptor -> UniProt mapping.

        Args:
            receptor_name: Receptor name
            uniprot_id: UniProt accession ID
        """
        self.RECEPTOR_UNIPROT_MAP[receptor_name] = uniprot_id
        logger.info(f"Added mapping: {receptor_name} -> {uniprot_id}")


def main():
    """Example usage of AlphaFoldRetriever."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download AlphaFold structures for GPCRs"
    )
    parser.add_argument(
        "--output-dir", default="data/alphafold", help="Output directory"
    )
    parser.add_argument("--receptors", nargs="+", help="Receptor names to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")

    args = parser.parse_args()

    retriever = AlphaFoldRetriever(output_dir=args.output_dir)

    # Default receptors if none specified
    if args.receptors:
        receptors = args.receptors
    else:
        # Download for commonly missing receptors
        receptors = [
            "5HT1A receptor",
            "5HT1B receptor",
            "5HT2B receptor",
            "5HT7 receptor",
            "A2B receptor",
            "S1P1 receptor",
            "NTS1 receptor",
            "MC4 receptor",
            "MC5 receptor",
            "OXE receptor",
        ]

    results = retriever.download_for_receptors(receptors, force_download=args.force)

    # Print results
    print("\nDownload Results:")
    for receptor, path in results.items():
        status = "✓ Success" if path else "✗ Failed"
        print(f"  {status}: {receptor}")


if __name__ == "__main__":
    main()
