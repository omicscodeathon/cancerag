"""Ballesteros-Weinstein generic numbering for ProLIF interaction columns.

ProLIF names each contact after the residue as numbered in the prepared
structure (``ifp_ASP86.A_VdWContact``). That number is a property of the PDB
file, not of biology: the prolif receptors are renumbered from 1, and each
construct has a different N-terminal length, so the *same* functional residue
lands on a different number in every receptor.

The consequence, measured on the raw ProLIF block (docs/BW_NUMBERING_AUDIT.md):
362 of 384 contact columns fire for exactly one receptor, 295 fire in fewer
than 5 rows, and the median column fires in 2. The conserved TM3 aspartate that
anchors every aminergic ligand is split across five columns (ASP80/83/84/86/91).
Remapping to generic positions gives 122 contact columns, 42 of them rebuilt
from more than one original column.

Ballesteros-Weinstein numbering labels a residue by its structural position —
helix number, then offset from that helix's most conserved residue, which is
defined as 50. D3.32 is the same position in every class A GPCR regardless of
construct, so contacts become comparable across receptors and knowledge learned
on one receptor transfers to another.

Mapping chain (GPCRdb is the authority for the generic numbers):

    prolif residue index  --[sequence alignment]-->  UniProt position
    UniProt position      --[GPCRdb residue table]-->  BW generic number

**Always build the map from the ``.prolif_H.pdb`` file, never the prepared
``.pdb``.** ProLIF writes column names against the protonated structure, which
is renumbered from 1, while the prepared structure keeps the crystal numbering
(kappa-opioid runs 57..339). Reading ``ifp_ASP82`` against the prepared file
resolves to a residue 28 A outside a 22 A docking box; against the prolif file
it is D3.32, the conserved salt-bridge anchor. Both lookups succeed silently,
so the mistake shows up as implausible pharmacology rather than an error.

Class A only. Receptors GPCRdb does not cover, or whose mapping fails the
anchor gate (e.g. the class C calcium-sensing receptor, which GPCRdb numbers
but not with class A anchors), keep their original per-structure column names
and are reported.

The featurisation pipeline (``featurization.build_interaction_fingerprints``)
applies :func:`remap_interaction_fingerprints` to the raw ProLIF block and
writes ``bw_column_map.json``, ``bw_maps.json`` and ``bw_map_reports.json``
next to the fingerprints.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GPCRDB = "https://gpcrdb.org/services"
CACHE_DIR = Path("data/raw/gpcrdb")

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # protonation-state variants emitted by structure-prep tools
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "CYM": "C",
    "ASH": "D", "GLH": "E", "LYN": "K", "TYM": "Y", "ARN": "R",
}


# ------------------------------------------------------------------ GPCRdb


def _get_json(url: str, cache_name: str, *, pause: float = 0.34):
    """Fetch JSON with an on-disk cache (GPCRdb is a shared public service)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / cache_name
    if cache.exists():
        return json.loads(cache.read_text())
    time.sleep(pause)
    with urllib.request.urlopen(url, timeout=60) as fh:
        payload = json.loads(fh.read().decode())
    cache.write_text(json.dumps(payload))
    return payload


def _is_absent(exc: Exception) -> bool:
    """True when GPCRdb answered that the record does not exist.

    Only a 404 means "GPCRdb does not cover this receptor". A timeout, DNS
    failure or 5xx means "we could not ask", and must not be treated as an
    answer: swallowing it would silently leave every receptor on its raw
    per-structure names when the pipeline runs offline without the cache,
    producing a fingerprint block that looks valid and is not the one shipped.
    """
    return isinstance(exc, urllib.error.HTTPError) and exc.code == 404


def fetch_protein(accession: str) -> dict | None:
    """GPCRdb protein record: entry_name + canonical sequence.

    Returns None when GPCRdb has no record (HTTP 404). Any other failure to
    reach GPCRdb is raised.
    """
    try:
        return _get_json(f"{GPCRDB}/protein/accession/{accession}/",
                         f"protein_{accession}.json")
    except Exception as exc:
        if not _is_absent(exc):
            raise
        logger.warning("GPCRdb has no protein record for %s (%s)", accession, exc)
        return None


def fetch_residues(entry_name: str) -> list | None:
    """GPCRdb residue table: sequence_number, amino_acid, display_generic_number.

    Returns None when GPCRdb has no table (HTTP 404). Any other failure to
    reach GPCRdb is raised.
    """
    try:
        return _get_json(f"{GPCRDB}/residues/{entry_name}/",
                         f"residues_{entry_name}.json")
    except Exception as exc:
        if not _is_absent(exc):
            raise
        logger.warning("GPCRdb has no residue table for %s (%s)", entry_name, exc)
        return None


_GENERIC_RE = re.compile(r"^(\d+)\.(\d+)")


def clean_generic(display_generic_number: str | None) -> str | None:
    """``'3.32x32'`` -> ``'3.32'``.

    GPCRdb concatenates the Ballesteros-Weinstein number with its own
    structure-based scheme. We keep the BW part, which is what the
    pharmacology literature uses.
    """
    if not display_generic_number:
        return None
    m = _GENERIC_RE.match(display_generic_number)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def uniprot_position_to_bw(accession: str) -> tuple[dict[int, str], str]:
    """{UniProt sequence position -> BW number} plus the canonical sequence."""
    prot = fetch_protein(accession)
    if not prot:
        return {}, ""
    residues = fetch_residues(prot["entry_name"])
    if not residues:
        return {}, prot.get("sequence", "")
    mapping = {}
    for r in residues:
        bw = clean_generic(r.get("display_generic_number"))
        if bw:
            mapping[int(r["sequence_number"])] = bw
    return mapping, prot.get("sequence", "")


# ------------------------------------------------------------- structures


def structure_residues(pdb_path: Path | str) -> list[tuple[int, str]]:
    """[(residue number, one-letter code)] in file order, ATOM records only."""
    seen: dict[int, str] = {}
    order: list[int] = []
    for line in Path(pdb_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip().upper()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        if resseq in seen:
            continue
        aa = THREE_TO_ONE.get(resname)
        if aa is None:
            continue
        seen[resseq] = aa
        order.append(resseq)
    return [(n, seen[n]) for n in order]


def align_structure_to_sequence(
    struct: list[tuple[int, str]], canonical: str,
) -> tuple[dict[int, int], float]:
    """Map structure residue number -> UniProt position via alignment.

    Semi-global: end gaps are free, so N/C-terminal truncation costs nothing,
    while internal gaps are penalised — the right shape for a construct that is
    a contiguous slice of the receptor, possibly with disordered loops missing
    or a fusion partner spliced into ICL3.
    """
    from Bio.Align import PairwiseAligner

    if not struct or not canonical:
        return {}, 0.0
    query = "".join(aa for _, aa in struct)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.target_end_gap_score = 0.0   # canonical may extend past the construct
    aligner.query_end_gap_score = 0.0

    aln = aligner.align(canonical, query)[0]
    mapping: dict[int, int] = {}
    matches = 0
    aligned = 0
    for (t0, t1), (q0, q1) in zip(*aln.aligned):
        for offset in range(t1 - t0):
            uni_pos = t0 + offset + 1            # UniProt is 1-based
            struct_num = struct[q0 + offset][0]
            mapping[struct_num] = uni_pos
            aligned += 1
            if canonical[t0 + offset] == query[q0 + offset]:
                matches += 1
    # Identity over ALIGNED pairs, not over the whole construct. GPCR structures
    # routinely carry a fusion partner (T4-lysozyme, BRIL) spliced into ICL3 —
    # beta2 here is 443 residues against a 413-residue receptor. Those residues
    # correctly align to nothing; scoring them as mismatches would reject every
    # fusion construct in the set.
    identity = matches / max(1, aligned)
    return mapping, identity


# The most conserved residue of each helix — the anchors the whole scheme is
# built on. If an alignment is sound these land on the right amino acid; if it
# has slipped, they do not.
BW_ANCHORS: dict[str, str] = {
    "1.50": "N", "2.50": "D", "3.50": "R", "6.50": "P", "7.50": "P",
}


def _anchor_agreement(
    bw_map: dict[int, str], struct: list[tuple[int, str]],
) -> tuple[int, int]:
    """(anchors landing on the expected residue, anchors present at all)."""
    residue = dict(struct)
    inverse = {v: k for k, v in bw_map.items()}
    present = ok = 0
    for generic, expected in BW_ANCHORS.items():
        num = inverse.get(generic)
        if num is None:
            continue
        present += 1
        if residue.get(num) == expected:
            ok += 1
    return ok, present


def receptor_bw_map(
    accession: str,
    pdb_path: Path | str,
    *,
    min_identity: float = 0.50,
    min_anchors: int = 3,
) -> tuple[dict[int, str], dict]:
    """{structure residue number -> BW number} for one receptor, plus a report.

    Validated by anchor agreement rather than by sequence identity alone.
    Identity is a poor gate here: seven aminergic receptors in this set (D2, D3,
    D4, M1, M2, alpha-2A, alpha-2C) align to their own canonical sequence at
    only 0.64-0.77 — non-human orthologs or engineered constructs — yet their
    mappings are structurally perfect, placing 3.32 on the aspartate that
    anchors every aminergic ligand and 6.50 on the CWxP proline. An 0.80
    identity cut discarded exactly the family that BW numbering helps most.

    Anchor agreement separates cleanly on this dataset: 54 receptors match all
    five anchors, two match four, and only the class C calcium-sensing receptor
    matches none — which is correct, as class C does not use class A numbering.
    """
    pos_to_bw, canonical = uniprot_position_to_bw(accession)
    struct = structure_residues(pdb_path)
    report = {
        "accession": accession, "n_structure_residues": len(struct),
        "n_bw_positions": len(pos_to_bw), "identity": 0.0,
        "n_mapped": 0, "anchors_ok": 0, "anchors_present": 0, "status": "ok",
    }
    if not pos_to_bw or not canonical:
        report["status"] = "no_gpcrdb_record"
        return {}, report

    struct_to_uni, identity = align_structure_to_sequence(struct, canonical)
    report["identity"] = round(identity, 3)
    if identity < min_identity:
        report["status"] = f"low_identity({identity:.2f})"
        return {}, report

    out = {n: pos_to_bw[u] for n, u in struct_to_uni.items() if u in pos_to_bw}
    ok, present = _anchor_agreement(out, struct)
    report["anchors_ok"], report["anchors_present"] = ok, present
    if ok < min_anchors:
        report["status"] = f"anchor_mismatch({ok}/{present})"
        return {}, report

    report["n_mapped"] = len(out)
    return out, report


# ---------------------------------------------------- column-name rewriting

COLUMN_RE = re.compile(r"^ifp_([A-Za-z]{3})(\d+)\.([A-Za-z0-9])_(.+)$")


def parse_column(col: str) -> tuple[str, int, str, str] | None:
    """``ifp_ASP86.A_VdWContact`` -> ``('ASP', 86, 'A', 'VdWContact')``."""
    m = COLUMN_RE.match(col)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2)), m.group(3), m.group(4)


def bw_column_name(bw: str, interaction: str) -> str:
    """``('3.32', 'Cationic')`` -> ``ifp_BW3.32_Cationic``.

    Named by position, not residue identity: the amino acid at a generic
    position differs between families, and embedding it would keep apart the
    very fragments the remap exists to merge.
    """
    return f"ifp_BW{bw}_{interaction}"


def unmapped_column_name(accession: str, col: str) -> str:
    """``('P41145', 'ifp_GLU150.A_Hydrophobic')`` -> ``ifp_P41145_GLU150.A_Hydrophobic``.

    A raw ProLIF column names a residue by its *construct* number, which is a
    property of one PDB entry and means nothing across receptors. Where a BW
    position is available that number is replaced by one; where it is not, the
    column has to carry the receptor instead, or two receptors that happen to
    share a residue number share a feature.

    Measured 2026-10-25: Glu150 is an unmapped contact in both P41145
    (kappa-opioid) and Q9H3N8 (H4), and 8 of 235 unmapped columns fired for
    more than one receptor. Merging them invents a pattern that spans two
    unrelated positions, and under receptor-grouped CV it ties a held-out
    receptor to a training one through a shared column.
    """
    stem = col[4:] if col.startswith("ifp_") else col
    return f"ifp_{accession}_{stem}"


# ------------------------------------------------------- pipeline transform

IFP_META_COLS: tuple[str, ...] = (
    "pair_id", "inchikey", "receptor_uniprot",
    "n_residues_contacted", "n_total_contacts",
)


def build_receptor_maps(
    accessions: Iterable[str],
    receptors_dir: Path | str,
    *,
    min_identity: float = 0.50,
    min_anchors: int = 3,
) -> tuple[dict[str, dict[int, str]], dict[str, dict]]:
    """Run :func:`receptor_bw_map` for every receptor.

    Reads ``<receptors_dir>/<accession>.prolif_H.pdb`` — the structure ProLIF
    numbered its columns against (see the module docstring for why the
    prepared ``.pdb`` gives silently wrong answers). A receptor without that
    file gets an empty map and status ``no_prolif_structure``.

    Returns ``(maps, reports)``, both keyed by accession. A receptor whose map
    is empty keeps its raw column names downstream.
    """
    receptors_dir = Path(receptors_dir)
    maps: dict[str, dict[int, str]] = {}
    reports: dict[str, dict] = {}
    for acc in sorted(set(accessions)):
        pdb = receptors_dir / f"{acc}.prolif_H.pdb"
        if not pdb.exists():
            maps[acc] = {}
            reports[acc] = {"accession": acc, "status": "no_prolif_structure",
                            "n_mapped": 0}
            logger.warning("BW: no %s; %s keeps raw column names", pdb.name, acc)
            continue
        maps[acc], reports[acc] = receptor_bw_map(
            acc, pdb, min_identity=min_identity, min_anchors=min_anchors,
        )
        if not maps[acc]:
            logger.warning("BW: %s keeps raw column names (%s)",
                           acc, reports[acc]["status"])
    return maps, reports


def remap_interaction_fingerprints(
    raw: pd.DataFrame,
    maps: Mapping[str, Mapping[int, str]],
    *,
    meta_cols: Iterable[str] = IFP_META_COLS,
    receptor_col: str = "receptor_uniprot",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Rename raw ProLIF bit columns to Ballesteros-Weinstein positions.

    For each row, a raw column ``ifp_<RES><num>.<chain>_<Interaction>`` that
    fires is moved to ``ifp_BW<pos>_<Interaction>`` when that row's receptor
    map assigns residue ``num`` a generic position. Bits landing on the same
    BW column are merged with logical OR, so two construct numbers that are
    the same structural position (D80 in one receptor, D86 in another) become
    one column. A bit whose receptor has no map (class C, failed anchor gate,
    no GPCRdb record) or whose residue has no generic number (loops, termini)
    goes to ``ifp_<accession>_<RES><num>.<chain>_<Interaction>`` instead — the
    construct number alone is not comparable across receptors, so the column
    has to name the receptor it belongs to. That also keeps the two paths
    disjoint: a raw column that fired for both a mapped and an unmapped
    receptor splits cleanly rather than one shadowing the other. Columns that
    end up firing in no row are not emitted.

    Meta columns pass through unchanged, including ``n_residues_contacted``
    and ``n_total_contacts``, which count ProLIF residues and contacts before
    merging. Output columns are the meta columns, then the bit columns sorted
    by name, as uint8.

    Returns ``(fingerprints, column_map)``, where ``column_map`` sends each BW
    column to the sorted raw columns merged into it.
    """
    meta = [c for c in meta_cols if c in raw.columns]
    bit_cols = [c for c in raw.columns if c not in meta]
    receptors = raw[receptor_col].astype(str).to_numpy()

    merged: dict[str, np.ndarray] = {}
    column_map: dict[str, set[str]] = {}
    n = len(raw)
    for col in bit_cols:
        values = raw[col].to_numpy()
        fired = np.flatnonzero(values != 0)
        if fired.size == 0:
            continue
        parsed = parse_column(col)
        for acc in np.unique(receptors[fired]):
            rows = fired[receptors[fired] == acc]
            bw = maps.get(acc, {}).get(parsed[1]) if parsed else None
            target = (bw_column_name(bw, parsed[3]) if bw
                      else unmapped_column_name(acc, col))
            arr = merged.get(target)
            if arr is None:
                arr = merged[target] = np.zeros(n, dtype=np.uint8)
            arr[rows] = 1
            if bw:
                column_map.setdefault(target, set()).add(col)

    bits = pd.DataFrame(
        {c: merged[c] for c in sorted(merged)}, index=raw.index, dtype=np.uint8,
    )
    out = pd.concat([raw[meta], bits], axis=1)
    return out, {k: sorted(v) for k, v in sorted(column_map.items())}


def write_bw_artifacts(
    output_dir: Path | str,
    *,
    column_map: Mapping[str, list[str]],
    maps: Mapping[str, Mapping[int, str]],
    reports: Mapping[str, dict],
) -> dict[str, Path]:
    """Persist the remap so it can be audited without re-running it.

    - ``bw_column_map.json`` — BW column -> raw columns merged into it.
    - ``bw_maps.json`` — accession -> {structure residue number -> BW}.
    - ``bw_map_reports.json`` — accession -> identity, anchors, status.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "column_map": output_dir / "bw_column_map.json",
        "maps": output_dir / "bw_maps.json",
        "reports": output_dir / "bw_map_reports.json",
    }
    paths["column_map"].write_text(json.dumps(dict(column_map), indent=2,
                                              sort_keys=True))
    paths["maps"].write_text(json.dumps(
        {acc: {str(k): v for k, v in sorted(m.items())}
         for acc, m in sorted(maps.items())},
        indent=2,
    ))
    paths["reports"].write_text(json.dumps(dict(sorted(reports.items())),
                                           indent=2))
    return paths
