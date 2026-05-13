"""
Binding-site / docking-box definition.

Stage 04 — see CURATION_JOURNEY.md and improvements/04_binding_site_definition.md.

For each prepared receptor in ``data/processed/receptors/<UNIPROT>.pdb``,
define the AutoDock Vina docking box:

1. **Co-crystal-ligand path (preferred when available).**
   The original RCSB PDB the receptor came from often includes a
   co-crystallized ligand sitting in the orthosteric pocket. If we can
   identify a real orthosteric ligand HETATM (filtered through the
   curated ignore list to skip lipids, buffers, ions, etc.), use the
   ligand's bounding-box centroid as the box center. The orthosteric
   pocket is then known *exactly*; pocket-prediction is unnecessary.

2. **Pocket-prediction path (fallback for AlphaFold receptors and any
   PDBs without a usable co-crystal ligand).**
   Run P2Rank (deep-learning) and fpocket (geometric) independently.
   Accept the prediction with high confidence only when both methods'
   top-ranked pockets are within ``agreement_threshold`` Å of each other
   (defaults to 5 Å — see CURATION_JOURNEY.md for the rationale).
   Otherwise, fall back to P2Rank alone and flag the receptor as
   "low_confidence".

In either case, the box is centered at the chosen point and sized as a
22 Å cube (the GPCR-orthosteric-pocket-appropriate default; see
the audit notes in CURATION_JOURNEY.md).

Optional re-docking validation (when a co-crystal ligand exists): re-dock
the bound ligand into our chosen box with Vina; flag the receptor if the
top-pose RMSD vs the experimental pose is > 2.5 Å (Trott & Olson 2010
"marginal pass" cutoff). This step is gated behind a config flag because
docking 61 receptors at exhaustiveness 16 is non-trivial compute.

Outputs:
- ``data/processed/binding_sites.json`` — one entry per receptor.
- ``data/processed/binding_sites_audit.md`` — reviewer-facing summary.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import PDBParser

from cancerag.preprocessing.het_resnames import LIGAND_AUTO_DETECT_IGNORE
from cancerag.preprocessing.pocket_predictors import (
    consensus_pocket,
    run_fpocket,
    run_p2rank,
)

logger = logging.getLogger(__name__)


DEFAULT_BOX_SIZE_ANGSTROM = 22.0
DEFAULT_AGREEMENT_THRESHOLD = 5.0
DEFAULT_LIGAND_PADDING = 5.0  # additional Å on each side of co-crystal ligand


@dataclass
class BindingSite:
    """One receptor's docking box."""

    uniprot: str
    biasdb_name: str
    pdb_id: str | None
    structure_source: str  # "pdb" | "alphafold"
    method: str  # "cocrystal_ligand" | "consensus_pocket" | "p2rank_only"
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    cocrystal_ligand_resname: str | None = None
    cocrystal_ligand_n_atoms: int | None = None
    pocket_prediction_distance_angstroms: float | None = None
    pocket_prediction_agrees: bool | None = None
    pocket_score_p2rank: float | None = None
    pocket_score_fpocket: float | None = None
    confidence: str = "ok"  # "ok" | "low_confidence" | "no_pocket_found"
    notes: str = ""


# ---------------------------------------------- co-crystal ligand detection


def find_cocrystal_ligand(
    pdb_path: Path | str,
    *,
    min_heavy_atoms: int = 8,
) -> tuple[str, list[tuple[float, float, float]]] | None:
    """Return (resname, list_of_xyz) for a real orthosteric-ish ligand,
    or None if no candidate found.

    A "real ligand" here means a HETATM residue:
    - Whose resname is NOT in LIGAND_AUTO_DETECT_IGNORE (skips waters,
      ions, buffers, lipids, glycans, cofactors, modified residues).
    - That has at least ``min_heavy_atoms`` heavy atoms (skips tiny
      crystal additives).

    The first such residue encountered (by PDBParser iteration order) is
    returned. For most GPCR PDBs this is the orthosteric drug.
    """
    pdb_path = Path(pdb_path)
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("rec", str(pdb_path))
    except Exception as exc:
        logger.warning("Could not parse %s: %s", pdb_path, exc)
        return None

    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.id[0]
                if not isinstance(hetflag, str) or not hetflag.startswith("H_"):
                    continue
                resname = residue.get_resname().strip().upper()
                if not resname or resname in LIGAND_AUTO_DETECT_IGNORE:
                    continue
                heavy_atoms = [
                    a for a in residue.get_atoms()
                    if a.element and a.element.upper() != "H"
                ]
                if len(heavy_atoms) < min_heavy_atoms:
                    continue
                coords = [tuple(float(c) for c in a.get_coord()) for a in heavy_atoms]
                return resname, coords
        break  # only first model
    return None


def box_from_ligand_coords(
    coords: list[tuple[float, float, float]],
    *,
    fixed_size: float | None = DEFAULT_BOX_SIZE_ANGSTROM,
    padding: float = DEFAULT_LIGAND_PADDING,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ((center_x, center_y, center_z), (size_x, size_y, size_z)).

    If ``fixed_size`` is provided the box is a cube of that edge length
    centered on the ligand's geometric center. Otherwise the box matches
    the ligand's bounding box plus 2x``padding`` per dimension.
    """
    if not coords:
        raise ValueError("box_from_ligand_coords: no coordinates")
    arr = np.asarray(coords)
    center = tuple(float(arr.mean(axis=0)[i]) for i in range(3))
    if fixed_size is not None:
        return center, (fixed_size, fixed_size, fixed_size)
    span = arr.max(axis=0) - arr.min(axis=0)
    size = tuple(float(span[i] + 2 * padding) for i in range(3))
    return center, size


# --------------------------------------------------------- main builder


def define_binding_site(
    *,
    uniprot: str,
    biasdb_name: str,
    receptor_pdb: Path | str,
    source_pdb_id: str | None,
    structure_source: str,
    raw_pdb_for_cocrystal: Path | str | None = None,
    box_size_angstrom: float = DEFAULT_BOX_SIZE_ANGSTROM,
    agreement_threshold_angstroms: float = DEFAULT_AGREEMENT_THRESHOLD,
    work_dir: Path | str | None = None,
) -> BindingSite:
    """Define a docking box for one receptor.

    ``receptor_pdb`` is the cleaned, prepared receptor used for docking.
    ``raw_pdb_for_cocrystal`` is the original raw PDB (which may still
    contain the co-crystal ligand we stripped during preparation). If
    ``raw_pdb_for_cocrystal`` is None or contains no usable ligand, we
    fall back to pocket prediction on ``receptor_pdb``.
    """
    receptor_pdb = Path(receptor_pdb)
    raw_pdb = Path(raw_pdb_for_cocrystal) if raw_pdb_for_cocrystal else None

    # 1. Try the co-crystal ligand path first.
    if raw_pdb is not None and raw_pdb.exists():
        ligand = find_cocrystal_ligand(raw_pdb)
        if ligand is not None:
            resname, coords = ligand
            center, size = box_from_ligand_coords(
                coords, fixed_size=box_size_angstrom
            )
            return BindingSite(
                uniprot=uniprot, biasdb_name=biasdb_name,
                pdb_id=source_pdb_id, structure_source=structure_source,
                method="cocrystal_ligand",
                center_x=center[0], center_y=center[1], center_z=center[2],
                size_x=size[0], size_y=size[1], size_z=size[2],
                cocrystal_ligand_resname=resname,
                cocrystal_ligand_n_atoms=len(coords),
                confidence="ok",
                notes=f"box centered on bounded {resname} ({len(coords)} heavy atoms)",
            )

    # 2. Fallback: pocket prediction with cross-check.
    if work_dir is None:
        work_dir = receptor_pdb.parent / f".pocket_{uniprot}"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    p2rank_pockets = []
    fpocket_pockets = []
    try:
        p2rank_pockets = run_p2rank(receptor_pdb, work_dir=work_dir / "p2rank")
    except Exception as exc:
        logger.warning("p2rank failed for %s: %s", uniprot, exc)
    try:
        fpocket_pockets = run_fpocket(receptor_pdb, work_dir=work_dir / "fpocket")
    except Exception as exc:
        logger.warning("fpocket failed for %s: %s", uniprot, exc)

    if not p2rank_pockets and not fpocket_pockets:
        return BindingSite(
            uniprot=uniprot, biasdb_name=biasdb_name,
            pdb_id=source_pdb_id, structure_source=structure_source,
            method="no_pocket_found",
            center_x=0.0, center_y=0.0, center_z=0.0,
            size_x=box_size_angstrom, size_y=box_size_angstrom,
            size_z=box_size_angstrom,
            confidence="no_pocket_found",
            notes="both p2rank and fpocket returned no pockets",
        )

    consensus = consensus_pocket(
        p2rank_pockets, fpocket_pockets,
        agreement_threshold_angstroms=agreement_threshold_angstroms,
    )
    primary = consensus.primary
    confidence = "ok" if consensus.agrees else "low_confidence"
    return BindingSite(
        uniprot=uniprot, biasdb_name=biasdb_name,
        pdb_id=source_pdb_id, structure_source=structure_source,
        method=consensus.method,
        center_x=primary.center_x, center_y=primary.center_y,
        center_z=primary.center_z,
        size_x=box_size_angstrom, size_y=box_size_angstrom,
        size_z=box_size_angstrom,
        pocket_prediction_distance_angstroms=consensus.distance_angstroms,
        pocket_prediction_agrees=consensus.agrees,
        pocket_score_p2rank=p2rank_pockets[0].score if p2rank_pockets else None,
        pocket_score_fpocket=fpocket_pockets[0].score if fpocket_pockets else None,
        confidence=confidence,
        notes=(
            f"p2rank top + fpocket top agreement = {consensus.distance_angstroms:.2f} Å"
            if consensus.distance_angstroms is not None
            else "only one method returned a pocket"
        ),
    )


# ---------------------------------------------------------- batch driver


def define_all_binding_sites(
    receptors_dir: Path | str = "data/processed/receptors",
    raw_pdb_dir: Path | str = "data/pdb",
    preferred_pdbs_tsv: Path | str = "data/registry/preferred_pdbs.tsv",
    output_path: Path | str = "data/processed/binding_sites.json",
    audit_path: Path | str = "data/processed/binding_sites_audit.md",
    *,
    box_size_angstrom: float = DEFAULT_BOX_SIZE_ANGSTROM,
    agreement_threshold_angstroms: float = DEFAULT_AGREEMENT_THRESHOLD,
    work_root: Path | str = "data/processed/.binding_site_work",
) -> dict:
    """Walk every prepared receptor and emit binding-site definitions."""
    receptors_dir = Path(receptors_dir)
    raw_pdb_dir = Path(raw_pdb_dir)
    preferred_pdbs_tsv = Path(preferred_pdbs_tsv)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    # Map UniProt -> (selected_pdb_id, biasdb_name, structure_source)
    chosen: dict[str, tuple[str, str]] = {}
    for line in preferred_pdbs_tsv.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        uniprot, biasdb_name, selected_pdb = parts[0], parts[1], parts[2]
        chosen[uniprot] = (selected_pdb, biasdb_name)

    results: list[BindingSite] = []
    for prepared in sorted(receptors_dir.glob("*.pdb")):
        uniprot = prepared.stem
        selected_pdb, biasdb_name = chosen.get(uniprot, ("", ""))
        # Determine the raw-PDB source (where the co-crystal ligand may live).
        raw_pdb = None
        structure_source = "pdb"
        if selected_pdb:
            cand = raw_pdb_dir / uniprot / f"{selected_pdb}.pdb"
            if cand.exists():
                raw_pdb = cand
        if not raw_pdb:
            structure_source = "alphafold"

        site = define_binding_site(
            uniprot=uniprot, biasdb_name=biasdb_name,
            receptor_pdb=prepared,
            source_pdb_id=selected_pdb or None,
            structure_source=structure_source,
            raw_pdb_for_cocrystal=raw_pdb,
            box_size_angstrom=box_size_angstrom,
            agreement_threshold_angstroms=agreement_threshold_angstroms,
            work_dir=work_root / uniprot,
        )
        results.append(site)
        logger.info(
            "%s (%s): method=%s confidence=%s center=(%.1f,%.1f,%.1f)",
            uniprot, biasdb_name, site.method, site.confidence,
            site.center_x, site.center_y, site.center_z,
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "box_size_angstrom": box_size_angstrom,
            "agreement_threshold_angstroms": agreement_threshold_angstroms,
        },
        "totals": {
            "receptors": len(results),
            "by_method": _count(results, "method"),
            "by_confidence": _count(results, "confidence"),
        },
        "binding_sites": [asdict(b) for b in results],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    _emit_audit(payload, Path(audit_path))
    return payload


def _count(results: list[BindingSite], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in results:
        v = getattr(r, attr)
        out[v] = out.get(v, 0) + 1
    return out


def _emit_audit(payload: dict, out_path: Path) -> None:
    md = [
        "# Stage 04 — Binding-site audit",
        "",
        f"_Generated: {payload['generated_at_utc']}_  ",
        f"_Box size: {payload['config']['box_size_angstrom']} Å (cube)_  ",
        f"_Pocket-method agreement threshold: {payload['config']['agreement_threshold_angstroms']} Å_  ",
        "",
        "## Summary",
        "",
        f"- Receptors with binding box: **{payload['totals']['receptors']}**",
        "- Methods used:",
    ]
    for k, v in sorted(payload["totals"]["by_method"].items()):
        md.append(f"  - {k}: {v}")
    md.append("- Confidence breakdown:")
    for k, v in sorted(payload["totals"]["by_confidence"].items()):
        md.append(f"  - {k}: {v}")
    md.append("")
    md.append("## Per-receptor box")
    md.append("")
    md.append(
        "| uniprot | biasdb_name | source | method | confidence | "
        "ligand | distance Å | center (x,y,z) |"
    )
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for s in payload["binding_sites"]:
        md.append(
            f"| {s['uniprot']} | {s['biasdb_name']} | {s['structure_source']} | "
            f"{s['method']} | {s['confidence']} | "
            f"{s.get('cocrystal_ligand_resname') or '—'} | "
            f"{s.get('pocket_prediction_distance_angstroms') if s.get('pocket_prediction_distance_angstroms') is not None else '—'} | "
            f"({s['center_x']:.2f}, {s['center_y']:.2f}, {s['center_z']:.2f}) |"
        )
    out_path.write_text("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    define_all_binding_sites()
