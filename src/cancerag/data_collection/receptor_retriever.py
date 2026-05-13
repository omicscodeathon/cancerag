"""
UniProt-anchored receptor PDB retriever.

Stage 03 fix — see CURATION_JOURNEY.md.

The legacy retriever did free-text RCSB searches on receptor names like
"5HT1A receptor" or "μ receptor". That produced two failure modes:

1. **Wrong-protein hits.** The text search returned old PDBs of unrelated
   proteins that happened to mention the receptor name. (e.g. `1ATZ` got
   stored under the A3 receptor folder; it isn't an adenosine A3
   structure.)
2. **Empty results.** Receptors with non-canonical names (Greek letters,
   no-hyphen variants) got zero hits — 21 of our 61 receptors had no
   cached structure at all.

This rewrite anchors retrieval to **UniProt accessions**:

1. Walk the canonical receptor registry (``data/registry/receptors.tsv``).
2. For each UniProt, query EBI's UniProt API
   (``https://www.ebi.ac.uk/proteins/api/proteins/<UNIPROT>``).
3. Extract every ``dbReferences[type=PDB]`` entry. Those are the PDBs
   that UniProt itself cross-references — guaranteed to be real
   structures of the right protein.
4. Filter by experimental method and resolution cap.
5. Download to ``data/pdb/<UNIPROT>/<PDB_ID>.pdb`` (UniProt-keyed folder).
6. Emit per-PDB and per-UniProt ``.meta.json`` provenance.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from cancerag.data_collection.provenance import write_meta
from cancerag.data_collection.registry import ReceptorRegistry
from cancerag.utils.network import (
    NetworkRetrier,
    NetworkRetrySettings,
    create_retry_session,
)

logger = logging.getLogger(__name__)


UNIPROT_URL = "https://www.ebi.ac.uk/proteins/api/proteins/{uniprot}"
RCSB_PDB_FILE_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

DEFAULT_MAX_PDBS_PER_RECEPTOR = 10
DEFAULT_RESOLUTION_CUTOFF_ANGSTROM = 4.0  # cryo-EM friendly
DEFAULT_ALLOWED_METHODS = ("X-ray", "EM", "Other")


class ReceptorRetrievalError(RuntimeError):
    """Raised when a receptor PDB retrieval fails after retries."""


class _NotFoundError(Exception):
    """Internal marker for HTTP 404 — a permanent, do-not-retry condition.

    Subclassed from Exception (not RequestException) so the NetworkRetrier
    does not loop on it: the retrier only retries the exception types we
    explicitly tell it to retry on.
    """


class ReceptorRetriever:
    """Fetch real PDB structures for every receptor in the registry,
    anchored to UniProt accessions (deterministic, reproducible)."""

    def __init__(
        self,
        output_dir: str | Path,
        registry: ReceptorRegistry,
        *,
        max_pdbs_per_receptor: int = DEFAULT_MAX_PDBS_PER_RECEPTOR,
        resolution_cutoff: float = DEFAULT_RESOLUTION_CUTOFF_ANGSTROM,
        allowed_methods: tuple[str, ...] = DEFAULT_ALLOWED_METHODS,
        network_config: dict | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.max_pdbs_per_receptor = int(max_pdbs_per_receptor)
        self.resolution_cutoff = float(resolution_cutoff)
        self.allowed_methods = tuple(allowed_methods)
        self.retry_settings = NetworkRetrySettings.from_config(network_config)
        self.retrier = NetworkRetrier(self.retry_settings, logger=logger)
        self.session = create_retry_session(
            self.retry_settings, allowed_methods=["GET"]
        )

    @classmethod
    def from_config(
        cls,
        output_dir: str | Path,
        config: dict,
        *,
        registry: ReceptorRegistry | None = None,
    ) -> "ReceptorRetriever":
        if registry is None:
            registry = ReceptorRegistry.load()
        dc = config.get("data_collection", {})
        return cls(
            output_dir=output_dir,
            registry=registry,
            max_pdbs_per_receptor=int(
                dc.get("max_pdb_files_per_receptor", DEFAULT_MAX_PDBS_PER_RECEPTOR)
            ),
            resolution_cutoff=float(
                dc.get("max_resolution_angstrom", DEFAULT_RESOLUTION_CUTOFF_ANGSTROM)
            ),
            network_config=config.get("network"),
        )

    # -------------------------------------------------------- UniProt lookup

    def _fetch_uniprot_record(self, uniprot: str) -> dict[str, Any]:
        url = UNIPROT_URL.format(uniprot=uniprot)

        def _do():
            r = self.session.get(
                url, headers={"Accept": "application/json"}, timeout=60
            )
            r.raise_for_status()
            return r.json()

        try:
            return self.retrier.run(
                f"UniProt fetch {uniprot}", _do,
                (requests.exceptions.RequestException,),
            )
        except requests.exceptions.RequestException as exc:
            raise ReceptorRetrievalError(
                f"UniProt fetch failed for {uniprot}: {exc}"
            ) from exc

    @staticmethod
    def parse_pdb_refs(uniprot_record: dict) -> list[dict]:
        """Extract and normalize PDB cross-references from a UniProt record."""
        out: list[dict] = []
        for ref in uniprot_record.get("dbReferences", []) or []:
            if ref.get("type") != "PDB":
                continue
            props = ref.get("properties", {}) or {}
            method = str(props.get("method", "?"))
            res_token = str(props.get("resolution", "")).split()[0] if props.get("resolution") else ""
            try:
                resolution = float(res_token)
            except (ValueError, IndexError):
                resolution = float("inf")
            chains = str(props.get("chains", ""))
            out.append({
                "pdb_id": ref["id"],
                "method": method,
                "resolution": resolution,
                "chains": chains,
            })
        return out

    def filter_and_rank(self, refs: list[dict]) -> list[dict]:
        """Apply quality filters and rank by resolution (best first)."""
        keep = []
        for r in refs:
            method_ok = any(m in r["method"] for m in self.allowed_methods)
            res_ok = r["resolution"] <= self.resolution_cutoff
            if method_ok and res_ok:
                keep.append(r)
        keep.sort(key=lambda x: x["resolution"])
        return keep[: self.max_pdbs_per_receptor]

    # ------------------------------------------------------------- download

    def _download_pdb(self, pdb_id: str, dest: Path) -> bool:
        if dest.exists() and dest.stat().st_size > 0:
            return True
        url = RCSB_PDB_FILE_URL.format(pdb_id=pdb_id)

        def _do():
            r = self.session.get(url, timeout=120)
            # 404 is a *permanent* "this PDB does not exist" — do not let
            # the NetworkRetrier loop on it (the project default is
            # max_retries=null = retry forever, which would block the run
            # on any obsolete or never-released PDB id).
            if r.status_code == 404:
                raise _NotFoundError(f"PDB {pdb_id} not in RCSB (404)")
            r.raise_for_status()
            return r.text

        try:
            text = self.retrier.run(
                f"PDB download {pdb_id}", _do,
                (requests.exceptions.RequestException,),
            )
        except _NotFoundError as exc:
            logger.info("Skipping unavailable PDB %s: %s", pdb_id, exc)
            return False
        except requests.exceptions.RequestException as exc:
            logger.warning("PDB download failed for %s: %s", pdb_id, exc)
            return False
        dest.write_text(text)
        return True

    # ----------------------------------------------------------------- run

    def run(self, restrict_to: set[str] | None = None) -> dict:
        """Retrieve PDBs for every UniProt in the registry.

        Args:
            restrict_to: optional set of UniProt accessions to limit the
                run to (e.g. only the receptors that appear in
                ``unified_ligands.csv``). If None, all registry rows are
                processed.
        """
        summary: dict = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "max_pdbs_per_receptor": self.max_pdbs_per_receptor,
                "resolution_cutoff_angstrom": self.resolution_cutoff,
                "allowed_methods": list(self.allowed_methods),
            },
            "receptors": [],
            "totals": {
                "registry_rows_seen": 0,
                "uniprots_processed": 0,
                "pdbs_listed": 0,
                "pdbs_downloaded": 0,
                "uniprots_with_zero_pdbs": 0,
                "uniprot_fetch_failures": 0,
            },
        }

        for _, row in self.registry.dataframe.iterrows():
            summary["totals"]["registry_rows_seen"] += 1
            uniprot = str(row["uniprot"])
            biasdb_name = str(row["biasdb_name"])
            if restrict_to and uniprot not in restrict_to:
                continue
            summary["totals"]["uniprots_processed"] += 1

            try:
                rec = self._fetch_uniprot_record(uniprot)
            except ReceptorRetrievalError as exc:
                summary["totals"]["uniprot_fetch_failures"] += 1
                summary["receptors"].append({
                    "uniprot": uniprot, "biasdb_name": biasdb_name,
                    "status": "uniprot_fetch_failed", "error": str(exc),
                })
                logger.error("%s", exc)
                continue

            all_refs = self.parse_pdb_refs(rec)
            kept = self.filter_and_rank(all_refs)
            summary["totals"]["pdbs_listed"] += len(kept)

            uniprot_dir = self.output_dir / uniprot
            uniprot_dir.mkdir(parents=True, exist_ok=True)

            downloaded: list[dict] = []
            for r in kept:
                pdb_id = r["pdb_id"]
                dest = uniprot_dir / f"{pdb_id}.pdb"
                if self._download_pdb(pdb_id, dest):
                    summary["totals"]["pdbs_downloaded"] += 1
                    write_meta(
                        dest,
                        source_url=RCSB_PDB_FILE_URL.format(pdb_id=pdb_id),
                        source_version="rcsb-pdb",
                        query_params={"uniprot": uniprot, "pdb_id": pdb_id},
                        row_count=1,
                        extra={
                            "uniprot": uniprot,
                            "biasdb_name": biasdb_name,
                            "method": r["method"],
                            "resolution_angstrom": (
                                r["resolution"]
                                if r["resolution"] != float("inf") else None
                            ),
                            "chains": r["chains"],
                        },
                    )
                    downloaded.append(r)

            uniprot_meta = uniprot_dir / "_uniprot_summary.json"
            uniprot_meta.write_text(json.dumps({
                "uniprot": uniprot,
                "biasdb_name": biasdb_name,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "all_pdb_refs_in_uniprot": [r["pdb_id"] for r in all_refs],
                "kept_after_filter": [r["pdb_id"] for r in downloaded],
                "filters": {
                    "max_pdbs_per_receptor": self.max_pdbs_per_receptor,
                    "resolution_cutoff_angstrom": self.resolution_cutoff,
                    "allowed_methods": list(self.allowed_methods),
                },
            }, indent=2))

            if not downloaded:
                summary["totals"]["uniprots_with_zero_pdbs"] += 1
            summary["receptors"].append({
                "uniprot": uniprot, "biasdb_name": biasdb_name,
                "status": "ok" if downloaded else "no_pdbs_after_filter",
                "n_uniprot_refs": len(all_refs),
                "n_kept": len(downloaded),
                "kept_pdbs": [r["pdb_id"] for r in downloaded],
            })

        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary_path = self.output_dir / "retrieval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(
            "Retrieval done: %d UniProts processed, %d PDBs downloaded, "
            "%d UniProts with zero PDBs, %d fetch failures",
            summary["totals"]["uniprots_processed"],
            summary["totals"]["pdbs_downloaded"],
            summary["totals"]["uniprots_with_zero_pdbs"],
            summary["totals"]["uniprot_fetch_failures"],
        )
        return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    registry = ReceptorRegistry.load()
    retriever = ReceptorRetriever.from_config(
        output_dir="data/pdb", config=config, registry=registry
    )
    retriever.run()
