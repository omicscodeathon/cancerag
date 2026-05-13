"""
Receptor Manager for Inference App.

Manages available receptors and their metadata for the interactive interface,
keyed against the post-rebuild Stage-04 ``binding_sites.json`` schema:
``{'binding_sites': [ { 'uniprot': ..., 'pdb_id': ..., 'biasdb_name': ...,
'center_x/y/z': ..., 'size_x/y/z': ..., 'method': ..., 'confidence': ...,
... }, ... ], ...}``.

Receptor PDBs are looked up by **UniProt accession** (one prepared receptor
per UniProt) at ``data/processed/receptors/<UNIPROT>.pdb``. The prepared
PDBQTs live at ``data/processed/receptors_pdbqt/<UNIPROT>.pdbqt``
(consolidated from the per-job Stage-04 ``.redock_work/`` directories so a
single Docker COPY suffices).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ReceptorManager:
    """Manages receptor information and structures (post-rebuild schema)."""

    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent.parent
        else:
            base_path = Path(base_path)

        self.base_path = base_path
        self.binding_sites_path = (
            base_path / "data" / "processed" / "binding_sites.json"
        )
        self.receptors_dir = base_path / "data" / "processed" / "receptors"
        self.receptors_pdbqt_dir = (
            base_path / "data" / "processed" / "receptors_pdbqt"
        )

        # Backward-compat fallbacks for legacy deployments where PDBQTs
        # were stored under different paths.
        self._pdbqt_legacy_dirs = [
            base_path / "data" / "processed" / ".redock_work",  # source-of-truth
            base_path / "data" / "processed" / "receptors_prepared",
        ]

        self._raw_payload = self._load_binding_sites()
        self._sites_by_uniprot, self._sites_by_biasdb_name = (
            self._index_sites()
        )
        self.receptors = self._process_receptors()

    # ------------------------------------------------------------------ load

    def _load_binding_sites(self) -> Dict:
        if not self.binding_sites_path.exists():
            logger.warning(
                "Binding sites file not found: %s", self.binding_sites_path
            )
            return {"binding_sites": []}
        try:
            with open(self.binding_sites_path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.error("Failed to load binding sites: %s", exc)
            return {"binding_sites": []}
        # Accept either the rebuilt schema (top-level dict with
        # 'binding_sites' list) OR a bare list (some test fixtures).
        if isinstance(payload, list):
            payload = {"binding_sites": payload}
        elif isinstance(payload, dict) and "binding_sites" not in payload:
            # Legacy schema: dict keyed by receptor_name. Convert to list.
            converted = []
            for name, site in payload.items():
                if isinstance(site, dict):
                    site = {**site, "biasdb_name": name}
                    if "uniprot" not in site:
                        site["uniprot"] = name  # best-effort fallback
                    converted.append(site)
            payload = {"binding_sites": converted}
        return payload

    def _index_sites(self):
        by_uniprot: Dict[str, Dict] = {}
        by_name: Dict[str, Dict] = {}
        for site in self._raw_payload.get("binding_sites", []):
            uni = site.get("uniprot")
            name = site.get("biasdb_name")
            if uni:
                by_uniprot[uni] = site
            if name:
                by_name[name] = site
        return by_uniprot, by_name

    # --------------------------------------------------------------- process

    def _process_receptors(self) -> List[Dict]:
        receptors: List[Dict] = []
        for site in self._raw_payload.get("binding_sites", []):
            uniprot = site.get("uniprot", "")
            biasdb_name = site.get("biasdb_name", uniprot or "unknown")
            pdb_id = site.get("pdb_id", "")
            pdb_path = self.receptors_dir / f"{uniprot}.pdb"
            pdbqt_path = self._resolve_pdbqt_path(uniprot)
            available = pdb_path.exists()
            receptors.append({
                "name": biasdb_name,
                "display_name": self._format_receptor_name(biasdb_name),
                "uniprot": uniprot,
                "uniprot_id": uniprot,
                "pdb_id": pdb_id,
                "pdb_path": str(pdb_path) if available else None,
                "pdbqt_path": str(pdbqt_path) if pdbqt_path else None,
                "binding_site": site,
                "confidence": site.get("confidence"),
                "method": site.get("method"),
                "structure_source": site.get("structure_source"),
                "available": available,
            })
        receptors.sort(key=lambda x: x["display_name"])
        n_avail = sum(1 for r in receptors if r["available"])
        logger.info(
            "ReceptorManager: %d binding-site entries, %d with PDB on disk, "
            "%d with PDBQT packaged",
            len(receptors), n_avail,
            sum(1 for r in receptors if r["pdbqt_path"]),
        )
        return receptors

    def _resolve_pdbqt_path(self, uniprot: str) -> Optional[Path]:
        # Primary: consolidated receptors_pdbqt/ (Docker-friendly)
        primary = self.receptors_pdbqt_dir / f"{uniprot}.pdbqt"
        if primary.exists() and primary.stat().st_size > 100:
            return primary
        # Fallback: source-of-truth Stage-04 work directory
        for legacy_dir in self._pdbqt_legacy_dirs:
            candidates = [
                legacy_dir / uniprot / "receptor.pdbqt",
                legacy_dir / f"{uniprot}.pdbqt",
            ]
            for cand in candidates:
                if cand.exists() and cand.stat().st_size > 100:
                    return cand
        return None

    # -------------------------------------------------------------- formatter

    def _format_receptor_name(self, name: str) -> str:
        if not name:
            return ""
        # Already camel/space-separated (e.g., "5HT1A receptor")? Leave it.
        if " " in name and not name.islower():
            return name
        formatted = name.replace("_", " ").title()
        replacements = {
            "A1": "α1", "A3": "α3",
            "At1": "AT1", "Cb1": "CB1", "Cb2": "CB2",
            "5ht": "5-HT",
            "H2": "H₂", "H3": "H₃", "H4": "H₄",
            "M1": "M₁", "M2": "M₂", "M3": "M₃",
            "D1": "D₁", "D2": "D₂", "D3": "D₃", "D4": "D₄",
        }
        for old, new in replacements.items():
            if formatted.startswith(old):
                formatted = formatted.replace(old, new, 1)
                break
        return formatted

    # ----------------------------------------------------------- public API

    def get_available_receptors(self) -> List[Dict]:
        return [r for r in self.receptors if r["available"]]

    def get_receptor_by_name(self, name: str) -> Optional[Dict]:
        # Match against biasdb_name OR uniprot OR display_name (UI tolerance)
        for r in self.receptors:
            if r["name"] == name or r["uniprot"] == name or r["display_name"] == name:
                return r
        return None

    def get_receptor_by_uniprot(self, uniprot: str) -> Optional[Dict]:
        for r in self.receptors:
            if r["uniprot"] == uniprot:
                return r
        return None

    def get_receptor_pdb_path(self, receptor_name: str) -> Optional[str]:
        r = self.get_receptor_by_name(receptor_name)
        return r["pdb_path"] if r and r["available"] else None

    def get_receptor_pdbqt_path(self, receptor_name: str) -> Optional[str]:
        r = self.get_receptor_by_name(receptor_name)
        return r["pdbqt_path"] if r else None

    def get_binding_site(self, receptor_name: str) -> Optional[Dict]:
        r = self.get_receptor_by_name(receptor_name)
        return r["binding_site"] if r else None
