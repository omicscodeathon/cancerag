"""Load a docked Vina pose into an RDKit Mol with proper bond perception.

Vina output is `out.pdbqt` — a multi-MODEL PDBQT file. The first MODEL is the
top pose. PDBQT differs from PDB by including a `BRANCH`/`ROOT` torsion tree
and AutoDock atom types in the element column, neither of which RDKit reads.

This module extracts MODEL 1, strips the AutoDock-specific bits, and runs
RDKit bond perception against the original ligand SMILES (template) so atom
identities/bond orders are correct — needed for both 3D descriptors and
ProLIF interaction perception.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)


_TWO_LETTER_ELEMENTS = frozenset({
    "Cl", "Br", "Fe", "Mg", "Mn", "Zn", "Cu", "Ca", "Na", "Si", "Se",
    "Co", "Ni", "Cd", "Hg", "Pb", "Al", "As", "Sb", "Sn", "Li", "Be",
    "Ba", "Sr", "Cr", "Mo", "Pt", "Au", "Ag",
})

# AutoDock / Meeko atom-type prefix -> element. Order matters: longer
# prefixes are tried first via a length-sorted iteration.
_AUTODOCK_TYPE_TO_ELEMENT = {
    # standard AutoDock4 / Vina
    "A": "C",   # aromatic C
    "C": "C", "N": "N", "O": "O", "S": "S", "P": "P", "H": "H",
    "F": "F", "I": "I",
    "Cl": "Cl", "Br": "Br",
    # AutoDock charged / acceptor variants
    "NA": "N", "NS": "N", "NX": "N",
    "OA": "O", "OS": "O", "OX": "O",
    "SA": "S",
    "HD": "H", "HS": "H",
    # Meeko extended (cyclic carbons, generic graph nodes)
    "CG": "C", "CG0": "C", "CG1": "C", "CG2": "C", "CG3": "C",
    "Cg": "C", "Cg0": "C", "Cg1": "C", "Cg2": "C", "Cg3": "C",
    "G": "C", "G0": "C", "G1": "C", "G2": "C", "G3": "C",
    # metals (rare in ligands but possible)
    "Mg": "Mg", "Zn": "Zn", "Ca": "Ca", "Fe": "Fe", "Mn": "Mn",
    "Cu": "Cu", "Co": "Co", "Ni": "Ni",
}


def _element_from_atom_name(atom_name: str) -> str:
    """Derive a real element symbol from a PDB/PDBQT atom name.

    Strategy: strip whitespace and digits from the atom name, then look up
    the resulting string in our AutoDock-aware table. If unknown, fall back
    to the first character (capitalized) — which is the element for normal
    PDB atom names like "C1", "N2", "OD1".
    """
    nm = atom_name.strip()
    if not nm:
        return "C"  # last-resort default
    # Strip trailing digits (atom name → atom-type prefix), e.g. "CG21" → "CG"
    prefix = "".join(ch for ch in nm if not ch.isdigit())
    if not prefix:
        return "C"
    # Try the AutoDock map (case-insensitive on the first-char-upper form)
    candidates = [prefix, prefix.capitalize(), prefix.upper()]
    for cand in candidates:
        if cand in _AUTODOCK_TYPE_TO_ELEMENT:
            return _AUTODOCK_TYPE_TO_ELEMENT[cand]
    # Two-letter element fallback
    if len(prefix) >= 2:
        cand = prefix[0].upper() + prefix[1].lower()
        if cand in _TWO_LETTER_ELEMENTS:
            return cand
    # Single-letter fallback
    return prefix[0].upper()


def _extract_first_model_pdb(pdbqt_text: str) -> str:
    """Pull MODEL 1 from a Vina out.pdbqt and re-emit as plain PDB lines.

    Drops AutoDock-only records (BRANCH/ENDBRANCH/ROOT/ENDROOT/TORSDOF) and
    rewrites the trailing element column from AutoDock atom types (which
    can be ``A``, ``OA``, ``NA``, ``Cg``, etc.) to plain element symbols
    derived from the atom name. RDKit's PDB parser rejects unknown elements
    like ``Cg``, so this step is mandatory.
    """
    out: list[str] = ["MODEL 1"]
    in_model = False
    seen_first_model = False
    for raw in pdbqt_text.splitlines():
        if raw.startswith("MODEL"):
            if seen_first_model:
                break
            seen_first_model = True
            in_model = True
            continue
        if not in_model:
            continue
        if raw.startswith("ENDMDL"):
            break
        if raw.startswith(("BRANCH", "ENDBRANCH", "ROOT", "ENDROOT",
                           "TORSDOF", "REMARK")):
            continue
        if raw.startswith(("ATOM", "HETATM")):
            atom_name = raw[12:16] if len(raw) >= 16 else ""
            element = _element_from_atom_name(atom_name)
            # Pad to col 76, then write element right-justified in cols 77-78.
            base = raw[:76].ljust(76)
            line = base + f"{element:>2}"
            out.append(line)
    out.append("ENDMDL")
    out.append("END")
    return "\n".join(out)


def _load_pose_via_meeko(pdbqt_path: Path) -> Chem.Mol | None:
    """Reconstruct the pose with Meeko, which wrote the PDBQT in the first place.

    Meeko records the ligand's chemistry in the PDBQT it emits and can read
    it straight back, so nothing has to be re-derived from coordinates. This
    is the only route that is correct by construction.
    """
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except ImportError:
        return None
    try:
        pmol = PDBQTMolecule(
            pdbqt_path.read_text(), is_dlg=False, skip_typing=True
        )
        mols = [m for m in RDKitMolCreate.from_pdbqt_mol(pmol) if m is not None]
    except Exception as exc:
        logger.debug("meeko round-trip failed for %s: %s", pdbqt_path, exc)
        return None
    if not mols or mols[0].GetNumConformers() == 0:
        return None
    mol = mols[0]
    # Keep only the top-ranked pose, so the returned Mol has the same shape
    # as the template route: one conformer, the best-scoring one.
    top = Chem.Mol(mol)
    top.RemoveAllConformers()
    top.AddConformer(mol.GetConformer(0), assignId=True)
    Chem.SanitizeMol(top)
    return top


def load_pose_mol(
    pdbqt_path: Path | str,
    template_smiles: str,
) -> Chem.Mol | None:
    """Load the top Vina pose as a chemically correct RDKit Mol.

    Returns a Mol with explicit Hs and one 3D conformer matching the docked
    pose, or ``None``.

    Two routes, in order. Meeko reads back the PDBQT it wrote, which needs no
    inference at all. Failing that, bond orders are assigned from the ligand's
    own SMILES.

    There is deliberately no third route. The previous one sanitised the
    raw PDB-parsed pose with ``SANITIZE_KEKULIZE`` switched off and returned
    that, with a docstring claiming it remained "usable by ProLIF for
    hydrophobic / VdW / pi-stacking perception". It was not: disabling
    kekulisation leaves every ring non-aromatic, so PiStacking, CationPi and
    PiCation cannot fire at all, and the molecule also carries whatever
    connectivity RDKit guessed from interatomic distances. It fired on
    **112 of 455** poses (24.6%) and logged at ``debug``, so the features it
    produced were wrong and nobody saw it. A pose we cannot read correctly is
    now dropped loudly instead of silently degraded.
    """
    pdbqt_path = Path(pdbqt_path)
    if not pdbqt_path.exists() or pdbqt_path.stat().st_size == 0:
        return None

    mol = _load_pose_via_meeko(pdbqt_path)
    if mol is not None:
        return mol

    template = Chem.MolFromSmiles(template_smiles) if template_smiles else None
    if template is not None:
        try:
            pdb_text = _extract_first_model_pdb(pdbqt_path.read_text())
            pose = Chem.MolFromPDBBlock(pdb_text, removeHs=False, sanitize=False)
            if pose is not None:
                out = AllChem.AssignBondOrdersFromTemplate(template, pose)
                Chem.SanitizeMol(out)
                return out
        except Exception as exc:
            logger.debug(
                "template bond-order assignment failed for %s: %s",
                pdbqt_path, exc,
            )

    logger.error(
        "Could not load %s as a chemically correct molecule (meeko and "
        "template routes both failed). The pair is dropped rather than "
        "featurised from a guessed structure.",
        pdbqt_path,
    )
    return None
