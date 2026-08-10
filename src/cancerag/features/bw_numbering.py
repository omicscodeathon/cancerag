"""Ballesteros-Weinstein generic numbering for ProLIF interaction columns.

ProLIF names each contact after the residue as numbered in the prepared
structure (``ifp_ASP86.A_VdWContact``). That number is a property of the PDB
file, not of biology: the prolif receptors are renumbered from 1, and each
construct has a different N-terminal length, so the *same* functional residue
lands on a different number in every receptor.

The consequence, measured on the 443-row training set: 305 of 386 contact
columns fire for exactly one receptor, the median column fires in 2 rows, and
the whole block carries 1.3% of model attribution. The conserved TM3 aspartate
that anchors every aminergic ligand is split across five columns
(ASP80/83/84/86/91), holding 20/12/13/27/6 rows instead of one column holding
78.

Ballesteros-Weinstein numbering labels a residue by its structural position —
helix number, then offset from that helix's most conserved residue, which is
defined as 50. D3.32 is the same position in every class A GPCR regardless of
construct, so contacts become comparable across receptors and knowledge learned
on one receptor transfers to another.

Mapping chain (GPCRdb is the authority for the generic numbers):

    prolif residue index  --[sequence alignment]-->  UniProt position
    UniProt position      --[GPCRdb residue table]-->  BW generic number

Class A only. Receptors GPCRdb does not cover (e.g. the class C calcium-sensing
receptor) keep their original per-structure column names and are reported.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from pathlib import Path

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


def fetch_protein(accession: str) -> dict | None:
    """GPCRdb protein record: entry_name + canonical sequence."""
    try:
        return _get_json(f"{GPCRDB}/protein/accession/{accession}/",
                         f"protein_{accession}.json")
    except Exception as exc:
        logger.warning("GPCRdb has no protein record for %s (%s)", accession, exc)
        return None


def fetch_residues(entry_name: str) -> list | None:
    """GPCRdb residue table: sequence_number, amino_acid, display_generic_number."""
    try:
        return _get_json(f"{GPCRDB}/residues/{entry_name}/",
                         f"residues_{entry_name}.json")
    except Exception as exc:
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


def receptor_bw_map(
    accession: str, pdb_path: Path | str, *, min_identity: float = 0.80,
) -> tuple[dict[int, str], dict]:
    """{structure residue number -> BW number} for one receptor, plus a report."""
    pos_to_bw, canonical = uniprot_position_to_bw(accession)
    struct = structure_residues(pdb_path)
    report = {
        "accession": accession, "n_structure_residues": len(struct),
        "n_bw_positions": len(pos_to_bw), "identity": 0.0,
        "n_mapped": 0, "status": "ok",
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


def bw_column_name(resname: str, bw: str, interaction: str) -> str:
    """``('ASP', '3.32', 'Cationic')`` -> ``ifp_ASP3.32_Cationic``."""
    return f"ifp_{resname}{bw}_{interaction}"
