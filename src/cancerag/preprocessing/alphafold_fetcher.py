"""
AlphaFold model fetcher with pLDDT-aware quality gating.

Stage 03 fallback: when no acceptable PDB structure is cached for a
receptor, download the AlphaFold prediction from EBI and verify the
structure is high-confidence enough to dock against.

EBI AlphaFold endpoints (model_v6 is the current production version as of
2026-04):
    PDB:  https://alphafold.ebi.ac.uk/files/AF-<UNIPROT>-F1-model_v6.pdb
    PAE:  https://alphafold.ebi.ac.uk/files/AF-<UNIPROT>-F1-predicted_aligned_error_v6.json

The PDB file's per-residue pLDDT score is stored in the B-factor field of
each ATOM record. We gate on the *mean pLDDT of the 7TM-bundle residues*:
those are the residues that line the orthosteric pocket. If they're high
confidence, we accept the model; otherwise we reject it.

Without per-receptor GPCRdb pocket-residue lists this module gates on the
mean pLDDT across the full chain (which for class A GPCRs is dominated by
the well-predicted 7TM bundle and is usually > 80). A future enhancement
will gate on a per-receptor pocket-residue subset.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests

from cancerag.data_collection.provenance import write_meta

logger = logging.getLogger(__name__)


AF_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v6.pdb"
DEFAULT_PLDDT_THRESHOLD = 70.0


class AlphaFoldFetchError(RuntimeError):
    """Raised when an AlphaFold model cannot be retrieved or fails the
    pLDDT gate."""


def _mean_plddt_from_pdb(pdb_text: str) -> tuple[float, int]:
    """Compute the mean pLDDT (B-factor) across all CA atoms.

    AlphaFold writes the per-residue pLDDT into columns 61-66 of every ATOM
    record. We restrict to CA atoms so we get one number per residue.
    Returns (mean_plddt, n_residues_counted).
    """
    plddts: list[float] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        # Atom name is at columns 13-16 (1-indexed); 12-16 in 0-indexed.
        if line[12:16].strip() != "CA":
            continue
        try:
            plddts.append(float(line[60:66]))
        except ValueError:
            continue
    if not plddts:
        return 0.0, 0
    return sum(plddts) / len(plddts), len(plddts)


def fetch_alphafold_pdb(
    uniprot: str,
    output_dir: Path | str,
    *,
    plddt_threshold: float = DEFAULT_PLDDT_THRESHOLD,
    timeout: float = 60.0,
) -> dict:
    """Download the AlphaFold model for ``uniprot`` and gate on pLDDT.

    Returns a metadata dict with the local path, mean pLDDT, threshold,
    and whether the model passed the gate. Writes a `.meta.json` sidecar
    alongside the downloaded PDB.

    Raises:
        AlphaFoldFetchError: when the download fails (HTTP error, no model
            available) or when the PDB body is malformed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdb = output_dir / f"AF-{uniprot}-F1.pdb"

    if out_pdb.exists():
        # Idempotent: re-use the cached download but re-evaluate the gate.
        text = out_pdb.read_text()
        mean_plddt, n_res = _mean_plddt_from_pdb(text)
        return {
            "uniprot": uniprot,
            "output_pdb": str(out_pdb),
            "mean_plddt": mean_plddt,
            "n_residues": n_res,
            "plddt_threshold": plddt_threshold,
            "passes_gate": mean_plddt >= plddt_threshold,
            "from_cache": True,
        }

    url = AF_PDB_URL.format(uniprot=uniprot)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            raise AlphaFoldFetchError(
                f"No AlphaFold model exists for UniProt {uniprot} "
                f"(EBI returned 404 for {url})"
            ) from e
        raise AlphaFoldFetchError(
            f"AlphaFold fetch failed for {uniprot}: HTTP {resp.status_code}"
        ) from e
    except requests.exceptions.RequestException as e:
        raise AlphaFoldFetchError(
            f"AlphaFold fetch failed for {uniprot}: {e}"
        ) from e

    text = resp.text
    if not text.startswith("HEADER") and "ATOM" not in text[:1024]:
        raise AlphaFoldFetchError(
            f"AlphaFold response for {uniprot} does not look like a PDB file"
        )
    out_pdb.write_text(text)

    mean_plddt, n_res = _mean_plddt_from_pdb(text)
    passes = mean_plddt >= plddt_threshold

    write_meta(
        out_pdb,
        source_url=url,
        source_version="alphafold-v6",
        query_params={"uniprot": uniprot},
        row_count=n_res,
        extra={
            "mean_plddt": mean_plddt,
            "plddt_threshold": plddt_threshold,
            "passes_gate": passes,
            "structure_source": "alphafold",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    return {
        "uniprot": uniprot,
        "output_pdb": str(out_pdb),
        "mean_plddt": mean_plddt,
        "n_residues": n_res,
        "plddt_threshold": plddt_threshold,
        "passes_gate": passes,
        "from_cache": False,
    }
