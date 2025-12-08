"""
Receptor Processing for Inference App

Reuses code from the main pipeline to handle:
- PDB ID fetching
- File upload processing
- Active site identification
- Receptor preprocessing
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from cancerag.data_collection.receptor_retriever import ReceptorRetriever
from cancerag.features.active_site_identifier import ActiveSiteIdentifier
from cancerag.preprocessing.receptor_preprocessor import (
    ReceptorPreprocessor,
    extract_binding_site,
)

logger = logging.getLogger(__name__)


class ReceptorProcessor:
    """
    Processes receptors for inference app.
    Handles PDB ID fetching, file uploads, and preprocessing.
    """

    def __init__(self, base_path: str):
        """
        Initialize the receptor processor.

        Args:
            base_path: Base path for the inference app
        """
        self.base_path = Path(base_path)
        self.temp_dir = self.base_path / "temp_receptors"
        self.temp_dir.mkdir(exist_ok=True)

        # Create minimal config for pipeline components
        self.config = {
            "paths": {
                "pdb_summary": str(self.temp_dir),
                "processed_data": str(self.temp_dir / "processed"),
            }
        }

        # Initialize pipeline components
        self.retriever = ReceptorRetriever(
            output_dir=str(self.temp_dir),
            max_downloads=1,  # Only need one structure
            force_redownload=False,
            max_retries=3,
            timeout=60,
            min_resolution=3.5,
        )

        self.preprocessor = ReceptorPreprocessor(self.config)
        self.active_site_identifier = ActiveSiteIdentifier(self.config)

    def fetch_pdb_by_id(
        self, pdb_id: str, progress_callback=None
    ) -> Tuple[Optional[str], Optional[Dict], str]:
        """
        Fetch a PDB structure by ID.

        Args:
            pdb_id: PDB identifier (e.g., "1F88")
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (pdb_path, binding_site_info, status_message)
        """
        try:
            pdb_id = pdb_id.upper().strip()

            if progress_callback:
                progress_callback(0.2, desc=f"Searching for PDB ID: {pdb_id}...")

            # Download PDB file
            result = self.retriever._download_pdb_file(
                pdb_id=pdb_id,
                target_dir=str(self.temp_dir / pdb_id),
                retry_count=0,
            )

            if result is None:
                return None, None, f"❌ Failed to download PDB structure {pdb_id}"

            pdb_path, quality_metrics = result

            if progress_callback:
                progress_callback(0.5, desc="Preprocessing structure...")

            # Preprocess the structure
            processed_dir = self.temp_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / f"{pdb_id}.pdb"

            self.preprocessor._clean_pdb_file(str(pdb_path), str(processed_path))

            if progress_callback:
                progress_callback(0.7, desc="Identifying active site...")

            # Extract binding site
            binding_site = extract_binding_site(
                pdb_file=str(pdb_path), ligand_name=None, padding=5.0
            )

            if progress_callback:
                progress_callback(1.0, desc="Complete!")

            binding_info = {
                "center": binding_site.get("center", [0, 0, 0])
                if binding_site
                else [0, 0, 0],
                "size": binding_site.get("size", [20, 20, 20])
                if binding_site
                else [20, 20, 20],
                "pdb_id": pdb_id,
                "source": "PDB",
            }

            return (
                str(processed_path),
                binding_info,
                f"✅ Successfully processed {pdb_id}",
            )

        except Exception as e:
            logger.error(f"Error fetching PDB {pdb_id}: {e}", exc_info=True)
            return None, None, f"❌ Error: {str(e)}"

    def process_uploaded_file(
        self, uploaded_file_path: str, progress_callback=None
    ) -> Tuple[Optional[str], Optional[Dict], str]:
        """
        Process an uploaded PDB file.

        Args:
            uploaded_file_path: Path to uploaded PDB file
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (pdb_path, binding_site_info, status_message)
        """
        try:
            if progress_callback:
                progress_callback(0.2, desc="Reading uploaded file...")

            # Copy to temp directory
            uploaded_path = Path(uploaded_file_path)
            temp_pdb_path = self.temp_dir / f"uploaded_{uploaded_path.stem}.pdb"

            # Read and validate
            with open(uploaded_path, "r") as f:
                content = f.read()

            if "ATOM" not in content and "HETATM" not in content:
                return (
                    None,
                    None,
                    "❌ Invalid PDB file: No ATOM or HETATM records found",
                )

            with open(temp_pdb_path, "w") as f:
                f.write(content)

            if progress_callback:
                progress_callback(0.5, desc="Preprocessing structure...")

            # Preprocess
            processed_dir = self.temp_dir / "processed"
            processed_dir.mkdir(exist_ok=True)
            processed_path = processed_dir / f"uploaded_{uploaded_path.stem}.pdb"

            self.preprocessor._clean_pdb_file(str(temp_pdb_path), str(processed_path))

            if progress_callback:
                progress_callback(0.7, desc="Identifying active site...")

            # Extract binding site
            binding_site = extract_binding_site(
                pdb_file=str(temp_pdb_path), ligand_name=None, padding=5.0
            )

            if progress_callback:
                progress_callback(1.0, desc="Complete!")

            binding_info = {
                "center": binding_site.get("center", [0, 0, 0])
                if binding_site
                else [0, 0, 0],
                "size": binding_site.get("size", [20, 20, 20])
                if binding_site
                else [20, 20, 20],
                "pdb_id": uploaded_path.stem,
                "source": "uploaded",
            }

            return (
                str(processed_path),
                binding_info,
                "✅ Successfully processed uploaded structure",
            )

        except Exception as e:
            logger.error(f"Error processing uploaded file: {e}", exc_info=True)
            return None, None, f"❌ Error: {str(e)}"

    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            import shutil

            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Error cleaning temp files: {e}")
