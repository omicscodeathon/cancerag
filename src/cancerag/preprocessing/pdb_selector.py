"""
Per-receptor PDB selection.

Stage 03 — see improvements/03_receptor_curation.md.

Given a candidate PDB file, score it for suitability as the docking receptor:
- Resolution (continuous penalty above 2.0 A).
- Presence of an orthosteric ligand (HETATM not in the curated ignore list).
- Active-state hint: presence of a G-protein α-subunit chain
  (GNAS / GNAI / GNAQ / GNA12 / mini-Gα fusions are all G-protein chains
  named with ``GNA*`` in the SEQRES / COMPND records).
- Inactive-state bias: presence of T4L / BRIL / nanobody fusion proteins
  (T4L = lysozyme, fused to ICL3 to stabilize inactive state in many
  early GPCR crystals; BRIL = apocytochrome b562RIL; common nanobodies
  like Nb6, Nb80, Nb35 stabilize active state — these are notoriously
  context-dependent so we treat fusion presence as a weak negative).

Output: a numeric score; higher is better. The selector keeps the
top-scoring PDB per receptor and writes the choice to
``data/registry/preferred_pdbs.tsv``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from cancerag.preprocessing.het_resnames import LIGAND_AUTO_DETECT_IGNORE

logger = logging.getLogger(__name__)


# Fusion-protein resnames frequently found in GPCR crystal constructs.
# Presence is a weak signal that the construct was engineered (often
# inactive-state stabilization).
FUSION_PROTEIN_HINTS: frozenset[str] = frozenset(
    {
        "T4L",  # T4 lysozyme — single most common ICL3 fusion
        "BRIL",  # apocytochrome b562RIL — N-term fusion
        "FLAV",  # flavodoxin — rare
    }
)

# G-protein α-subunit gene-symbol prefixes; presence in a chain implies
# the construct is the active-state G-protein-coupled complex.
G_ALPHA_PREFIXES: tuple[str, ...] = ("GNAS", "GNAI", "GNAO", "GNAQ", "GNA12", "GNA13")


@dataclass
class PDBCandidate:
    pdb_id: str
    path: Path
    resolution: float = float("inf")
    has_orthosteric_ligand: bool = False
    detected_ligand_resname: str | None = None
    has_g_protein_chain: bool = False
    has_fusion_protein: bool = False
    n_chains: int = 0
    longest_chain_residues: int = 0
    score: float = float("-inf")
    reason: str = ""
    detected_state: str = "unknown"  # "active" | "inactive_likely" | "unknown"


# Class A GPCRs are ~280-450 residues. PDBs whose longest chain is much
# shorter than this are usually fragments — N-terminal extracellular
# domains, isolated TM bundles, or other partial constructs. The picker
# should refuse to choose them as the docking receptor (a fragment can't
# host the orthosteric pocket properly).
MIN_LONGEST_CHAIN_RESIDUES = 200


def _parse_resolution(text: str) -> float:
    """Pull resolution from a PDB REMARK 2 line, or +inf if not found.
    Handles X-ray and cryo-EM resolution-statement formats."""
    m = re.search(r"REMARK\s+2\s+RESOLUTION\.\s+([\d.]+)\s*ANGSTROM", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("inf")


def _scan_pdb(path: Path) -> PDBCandidate:
    """Lightweight first-pass scan of a PDB file: parse REMARK 2 for
    resolution; walk HETATM and ATOM records to detect ligand / G-protein /
    fusion / chain count.
    """
    cand = PDBCandidate(pdb_id=path.stem.upper(), path=path)
    chains: set[str] = set()
    het_resnames: set[str] = set()
    g_alpha_chain_seen = False
    fusion_seen = False

    try:
        text = path.read_text(errors="ignore")
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)
        cand.reason = f"unreadable: {e}"
        return cand

    cand.resolution = _parse_resolution(text)

    # COMPND records sometimes name the protein component (helps catch
    # mini-Gα fusions that don't carry GNA* in the chain name).
    compnd_lines = [ln for ln in text.splitlines() if ln.startswith("COMPND")]
    compnd_blob = " ".join(compnd_lines).upper()
    if any(p in compnd_blob for p in G_ALPHA_PREFIXES) or "G-ALPHA" in compnd_blob:
        g_alpha_chain_seen = True
    if any(f in compnd_blob for f in FUSION_PROTEIN_HINTS):
        fusion_seen = True
    if "T4 LYSOZYME" in compnd_blob or "APOCYTOCHROME B562" in compnd_blob:
        fusion_seen = True

    for line in text.splitlines():
        if line.startswith("ATOM") and len(line) >= 22:
            chains.add(line[21])
        elif line.startswith("HETATM") and len(line) >= 22:
            chains.add(line[21])
            res_name = line[17:20].strip().upper()
            if res_name and res_name not in LIGAND_AUTO_DETECT_IGNORE:
                het_resnames.add(res_name)
            if res_name in FUSION_PROTEIN_HINTS:
                fusion_seen = True

    # Per-chain residue counts. Used by the scorer to reject fragment
    # constructs (e.g. CXCR2 4Q3H is only the 90-residue N-terminal
    # extracellular domain, not the full 360-residue receptor).
    chain_residue_counts: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("ATOM") and len(line) >= 22:
            ch = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            key = (ch, resnum)
            # Track unique residues per chain via the seen-residues set.
            if "_seen_residues" not in chain_residue_counts:
                chain_residue_counts.setdefault("_seen_residues", set())
            seen = chain_residue_counts["_seen_residues"]
            if (ch, resnum) not in seen:
                seen.add((ch, resnum))
                chain_residue_counts[ch] = chain_residue_counts.get(ch, 0) + 1
    chain_residue_counts.pop("_seen_residues", None)
    cand.longest_chain_residues = (
        max(chain_residue_counts.values()) if chain_residue_counts else 0
    )

    cand.n_chains = len(chains)
    cand.has_orthosteric_ligand = bool(het_resnames)
    cand.detected_ligand_resname = (
        sorted(het_resnames)[0] if het_resnames else None
    )
    cand.has_g_protein_chain = g_alpha_chain_seen
    cand.has_fusion_protein = fusion_seen

    if g_alpha_chain_seen:
        cand.detected_state = "active"
    elif fusion_seen:
        cand.detected_state = "inactive_likely"
    return cand


def score_candidate(cand: PDBCandidate, *, prefer_state: str = "active") -> float:
    """Continuous score; higher is better.

    +50 for the presence of a co-crystal orthosteric ligand (essential for
       defining the docking box without needing a pocket predictor).
    Linear resolution penalty above 2.0 A (so 2.0 A beats 2.8 A beats 3.5 A).
    +30 if the structure looks active-state and prefer_state=="active".
    -10 for fusion proteins (engineering artifact).
    -3 per chain above 1 (favors clean single-receptor structures over
       multi-chain complexes — though G-protein chains are excluded
       implicitly since the +30 for active-state offsets this).
    """
    # Hard reject fragments — picking a 90-residue N-terminal domain of a
    # 360-residue receptor and trying to dock against it is meaningless.
    if cand.longest_chain_residues and cand.longest_chain_residues < MIN_LONGEST_CHAIN_RESIDUES:
        return float("-inf")

    s = 0.0
    if cand.has_orthosteric_ligand:
        s += 50.0
    s -= 10.0 * max(0.0, cand.resolution - 2.0)
    if cand.detected_state == "active" and prefer_state == "active":
        s += 30.0
    if cand.detected_state == "inactive_likely" and prefer_state == "inactive":
        s += 30.0
    if cand.has_fusion_protein:
        s -= 10.0
    s -= 3.0 * max(0, cand.n_chains - 1)
    return s


def select_best_pdb(
    candidates_dir: Path | str,
    *,
    prefer_state: str = "active",
) -> PDBCandidate | None:
    """Score every ``.pdb`` in ``candidates_dir`` and return the top one.

    Returns None if no PDB files are present.
    """
    candidates_dir = Path(candidates_dir)
    if not candidates_dir.exists():
        return None
    pdbs = sorted(candidates_dir.glob("*.pdb"))
    if not pdbs:
        return None
    scored: list[PDBCandidate] = []
    for p in pdbs:
        c = _scan_pdb(p)
        c.score = score_candidate(c, prefer_state=prefer_state)
        c.reason = (
            f"res={c.resolution:.2f}A; ligand={c.detected_ligand_resname}; "
            f"state={c.detected_state}; chains={c.n_chains}; "
            f"longest_chain={c.longest_chain_residues}; "
            f"fusion={c.has_fusion_protein}"
        )
        scored.append(c)
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[0]


def candidates_to_records(cands: list[PDBCandidate]) -> list[dict]:
    return [
        {
            "pdb_id": c.pdb_id,
            "path": str(c.path),
            "resolution": c.resolution if c.resolution != float("inf") else None,
            "has_orthosteric_ligand": c.has_orthosteric_ligand,
            "detected_ligand_resname": c.detected_ligand_resname,
            "detected_state": c.detected_state,
            "has_g_protein_chain": c.has_g_protein_chain,
            "has_fusion_protein": c.has_fusion_protein,
            "n_chains": c.n_chains,
            "score": c.score,
            "reason": c.reason,
        }
        for c in cands
    ]


def write_preferred_tsv(
    rows: list[dict],
    output_path: Path | str,
) -> Path:
    """Write a TSV of selected PDBs, one per receptor.

    Columns: uniprot, biasdb_name, selected_pdb, resolution, state, score,
    reason, manual_override (blank — for hand-edits).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        "uniprot",
        "biasdb_name",
        "selected_pdb",
        "resolution",
        "state",
        "ligand_resname",
        "score",
        "reason",
        "manual_override",
    )
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append(
            "\t".join(
                str(r.get(c, "")) if r.get(c) is not None else ""
                for c in cols
            )
        )
    output_path.write_text("\n".join(lines) + "\n")
    return output_path
