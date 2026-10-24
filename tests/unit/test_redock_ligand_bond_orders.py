"""A co-crystal ligand's bond orders come from its component, not its geometry.

A PDB file carries coordinates and no bond orders, so RDKit infers
connectivity from interatomic distances and makes every bond single. An
aromatic ring then reads as six rotatable bonds that do not exist, and the
rotatable-bond count is what decides how hard Vina's search is -- hence the
re-dock RMSD, which is this pipeline's own quality gate.

Measured over the 53 co-crystal ligands before the fix: 39 had the wrong
TORSDOF, 37 of them inflated, median +2 and up to +8 (HAL-3777).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from cancerag.preprocessing.redock_validation import (
    fetch_component_template,
    residue_code_of,
)

# Two residues' worth of a real HETATM record: the component code lives in
# columns 18-20, which is the only field this parser reads.
_LIGAND_PDB = """\
HETATM    1  C1  U99 A 900     130.597 124.466  98.781  1.00  0.00           C
HETATM    2  C2  U99 A 900     131.220 122.847  97.206  1.00  0.00           C
END
"""


def test_residue_code_is_read_from_columns_18_to_20(tmp_path: Path):
    p = tmp_path / "lig.pdb"
    p.write_text(_LIGAND_PDB)
    assert residue_code_of(p) == "U99"


def test_residue_code_of_missing_file_is_none(tmp_path: Path):
    assert residue_code_of(tmp_path / "nope.pdb") is None


def test_residue_code_of_file_without_atoms_is_none(tmp_path: Path):
    p = tmp_path / "empty.pdb"
    p.write_text("REMARK nothing here\nEND\n")
    assert residue_code_of(p) is None


@pytest.mark.skipif(
    not Path("data/cache/pdb_components/U99_ideal.sdf").exists(),
    reason="component not cached and this test must not hit the network",
)
def test_cached_component_template_has_no_hydrogens():
    """The template must be heavy-atom only to match a heavy-atom crystal.

    This is the bug the first version of the fix shipped with: the RCSB ideal
    SDF is fully protonated, AssignBondOrdersFromTemplate matches on atom
    count, so every match failed and the correction silently did nothing on
    51 of 53 receptors.
    """
    template = fetch_component_template("U99")

    assert template is not None
    assert all(a.GetAtomicNum() != 1 for a in template.GetAtoms()), (
        "template still carries explicit hydrogens; it will never match a "
        "heavy-atom crystal ligand"
    )
    assert template.GetNumAtoms() == template.GetNumHeavyAtoms()


@pytest.mark.skipif(
    not Path("data/cache/pdb_components/U99_ideal.sdf").exists(),
    reason="component not cached and this test must not hit the network",
)
def test_template_restores_aromaticity_the_pdb_lost():
    """U99 is the kappa-opioid showcase ligand: 9 torsions shipped, 6 real."""
    template = fetch_component_template("U99")

    assert template is not None
    assert sum(1 for a in template.GetAtoms() if a.GetIsAromatic()) > 0, (
        "a template with no aromatic atoms cannot correct an inflated "
        "rotatable-bond count"
    )
    Chem.SanitizeMol(template)  # must not raise
