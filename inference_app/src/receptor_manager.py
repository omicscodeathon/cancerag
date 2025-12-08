"""
Receptor Manager for Inference App

Manages available receptors and their metadata for the interactive interface.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ReceptorManager:
    """Manages receptor information and structures."""

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize receptor manager.

        Args:
            base_path: Base path to project root
        """
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
        else:
            base_path = Path(base_path)

        self.base_path = base_path
        self.binding_sites_path = (
            base_path / "data" / "processed" / "binding_sites.json"
        )
        self.receptors_dir = base_path / "data" / "processed" / "receptors"

        self.binding_sites = self._load_binding_sites()
        self.receptors = self._process_receptors()

    def _load_binding_sites(self) -> Dict:
        """Load binding sites configuration."""
        if not self.binding_sites_path.exists():
            logger.warning(f"Binding sites file not found: {self.binding_sites_path}")
            return {}

        try:
            with open(self.binding_sites_path, "r") as f:
                binding_sites = json.load(f)
            logger.info(f"Loaded {len(binding_sites)} receptors")
            return binding_sites
        except Exception as e:
            logger.error(f"Failed to load binding sites: {e}")
            return {}

    def _process_receptors(self) -> List[Dict]:
        """Process receptors into a list with metadata."""
        receptors = []

        for receptor_name, binding_site in self.binding_sites.items():
            pdb_id = binding_site.get("source_pdb", "")
            pdb_path = self.receptors_dir / f"{pdb_id}.pdb"

            receptor_info = {
                "name": receptor_name,
                "display_name": self._format_receptor_name(receptor_name),
                "pdb_id": pdb_id,
                "pdb_path": str(pdb_path) if pdb_path.exists() else None,
                "binding_site": binding_site,
                "available": pdb_path.exists(),
            }

            receptors.append(receptor_info)

        # Sort by display name
        receptors.sort(key=lambda x: x["display_name"])
        return receptors

    def _format_receptor_name(self, name: str) -> str:
        """Format receptor name for display."""
        # Convert snake_case to Title Case
        formatted = name.replace("_", " ").title()

        # Handle special cases
        replacements = {
            "A1": "α1",
            "A3": "α3",
            "At1": "AT1",
            "Cb1": "CB1",
            "Cb2": "CB2",
            "5ht": "5-HT",
            "H2": "H₂",
            "H3": "H₃",
            "H4": "H₄",
            "M1": "M₁",
            "M2": "M₂",
            "M3": "M₃",
            "D1": "D₁",
            "D2": "D₂",
            "D3": "D₃",
            "D4": "D₄",
        }

        for old, new in replacements.items():
            if formatted.startswith(old):
                formatted = formatted.replace(old, new, 1)
                break

        return formatted

    def get_available_receptors(self) -> List[Dict]:
        """Get list of available receptors."""
        return [r for r in self.receptors if r["available"]]

    def get_receptor_by_name(self, name: str) -> Optional[Dict]:
        """Get receptor information by name."""
        for receptor in self.receptors:
            if receptor["name"] == name:
                return receptor
        return None

    def get_receptor_pdb_path(self, receptor_name: str) -> Optional[str]:
        """Get PDB file path for a receptor."""
        receptor = self.get_receptor_by_name(receptor_name)
        if receptor and receptor["available"]:
            return receptor["pdb_path"]
        return None

    def get_binding_site(self, receptor_name: str) -> Optional[Dict]:
        """Get binding site configuration for a receptor."""
        return self.binding_sites.get(receptor_name)
