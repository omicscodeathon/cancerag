"""ProLIF-derived protein-ligand interaction fingerprints.

Each (ligand-pose, receptor) pair is reduced to a binary vector indexed by
(pocket-residue × interaction-type). These are the only true *pair-level*
features in the pipeline — they vary with both the ligand and the receptor
and directly probe the contacts that biased agonism literature implicates
(W6.48 toggle, D3.32 ionic, ECL2 contacts, etc.).

Standard interaction types tracked (ProLIF defaults):
    Hydrophobic, HBDonor, HBAcceptor, PiStacking, Anionic, Cationic,
    CationPi, PiCation, VdWContact

Per-pair output is a sparse-friendly Series:
    {"ifp_LEU101_Hydrophobic": 1, "ifp_ASP113_Cationic": 1, ...}
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

logger = logging.getLogger(__name__)

# ProLIF emits a thicket of MDAnalysis deprecation warnings on import; quiet
# them at module import time rather than per-call.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="MDAnalysis")


@dataclass(frozen=True)
class IFPResult:
    pair_id: str
    ifp_bits: dict[str, int]
    n_residues_contacted: int
    n_total_contacts: int
    error: str | None = None


_RECEPTOR_CACHE: dict[Path, "object"] = {}


def _load_receptor_for_prolif(receptor_pdb: Path):
    """Load a prepared receptor PDB into a ProLIF Molecule.

    Stage 03 prep strips hydrogens (normal for RCSB PDB files), but ProLIF's
    H-bond perception needs them explicit. We use PDBFixer to add Hs at
    pH 7.4 once per receptor, cache the H-added PDB, and load that.
    """
    import MDAnalysis as mda
    import prolif as plf

    if receptor_pdb in _RECEPTOR_CACHE:
        return _RECEPTOR_CACHE[receptor_pdb]

    h_added = receptor_pdb.with_suffix(".prolif_H.pdb")
    if not h_added.exists():
        try:
            from openmm.app import PDBFile
            from pdbfixer import PDBFixer
            fixer = PDBFixer(filename=str(receptor_pdb))
            fixer.findMissingResidues()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(pH=7.4)
            with h_added.open("w") as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f)
        except Exception as exc:
            logger.warning("PDBFixer H-add failed for %s: %s", receptor_pdb, exc)
            # Fall through to using the raw PDB; ProLIF will likely fail.
            h_added = receptor_pdb

    u = mda.Universe(str(h_added))
    protein = u.select_atoms("protein")
    rec = plf.Molecule.from_mda(protein)
    _RECEPTOR_CACHE[receptor_pdb] = rec
    return rec


def _ligand_to_prolif(mol: Chem.Mol):
    """RDKit Mol with explicit Hs + 3D coords -> ProLIF Molecule."""
    import prolif as plf
    return plf.Molecule(mol)


def compute_ifp_for_pair(
    pair_id: str,
    ligand_mol: Chem.Mol,
    receptor_pdb: Path,
) -> IFPResult:
    """Run ProLIF for one (pose, receptor) pair.

    Args:
        pair_id: short identifier, e.g. "ABCDEFGHIJKLMN__P12345".
        ligand_mol: RDKit Mol with one 3D conformer (the docked pose) and
            explicit Hs (load via :mod:`cancerag.features.pose_loader`).
        receptor_pdb: path to the prepared receptor PDB (chain-resolved,
            water-stripped — typically `data/processed/receptors/<UNIPROT>.pdb`).
    """
    import prolif as plf

    if ligand_mol is None:
        return IFPResult(pair_id, {}, 0, 0, error="ligand_mol is None")
    if not receptor_pdb.exists():
        return IFPResult(pair_id, {}, 0, 0,
                         error=f"receptor pdb missing: {receptor_pdb}")

    try:
        receptor = _load_receptor_for_prolif(receptor_pdb)
        ligand = _ligand_to_prolif(ligand_mol)
        fp = plf.Fingerprint()  # default 9 interactions
        fp.run_from_iterable([ligand], receptor)
        df = fp.to_dataframe()
    except Exception as exc:
        return IFPResult(pair_id, {}, 0, 0, error=f"prolif failed: {exc}")

    if df.empty:
        return IFPResult(pair_id, {}, 0, 0)

    # ProLIF dataframe columns are MultiIndex: (ligand_id, residue, interaction)
    bits: dict[str, int] = {}
    residues_seen: set[str] = set()
    n_contacts = 0
    row = df.iloc[0]
    for col, value in row.items():
        if not value:
            continue
        # col is a 3-tuple (ligand, residue, interaction)
        try:
            _lig, residue, interaction = col
        except (ValueError, TypeError):
            continue
        residue_str = str(residue)
        residues_seen.add(residue_str)
        n_contacts += 1
        bits[f"ifp_{residue_str}_{interaction}"] = 1
    return IFPResult(
        pair_id=pair_id,
        ifp_bits=bits,
        n_residues_contacted=len(residues_seen),
        n_total_contacts=n_contacts,
    )


def prolif_default_interaction_names() -> list[str]:
    import prolif as plf
    return list(plf.Fingerprint().interactions.keys())
