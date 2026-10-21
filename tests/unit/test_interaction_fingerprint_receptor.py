"""The receptor reaching ProLIF must be a bonded molecule, not loose atoms.

Regression cover for the defect in HAL-3770: the receptor was loaded through
MDAnalysis, whose PDB parser reads bonds from ``CONECT`` records only. OpenMM
writes those for disulfides alone, so a 6672-atom receptor arrived with one
bond and no aromaticity, and six of ProLIF's nine interaction types could
never fire. Nothing raised; the fingerprint was simply wrong.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from cancerag.features.interaction_fingerprint import assert_receptor_is_bonded

# Two residues of real backbone, enough for RDKit's templates to bond and to
# perceive the phenylalanine ring as aromatic.
_PHE_ALA_PDB = """\
ATOM      1  N   PHE A   1      -1.204   1.398   0.000  1.00  0.00           N
ATOM      2  CA  PHE A   1       0.000   0.577   0.000  1.00  0.00           C
ATOM      3  C   PHE A   1       1.247   1.441   0.000  1.00  0.00           C
ATOM      4  O   PHE A   1       1.176   2.668   0.000  1.00  0.00           O
ATOM      5  CB  PHE A   1       0.021  -0.302  -1.253  1.00  0.00           C
ATOM      6  CG  PHE A   1       0.043  -1.787  -0.987  1.00  0.00           C
ATOM      7  CD1 PHE A   1       1.247  -2.457  -0.828  1.00  0.00           C
ATOM      8  CD2 PHE A   1      -1.139  -2.508  -0.896  1.00  0.00           C
ATOM      9  CE1 PHE A   1       1.269  -3.826  -0.583  1.00  0.00           C
ATOM     10  CE2 PHE A   1      -1.117  -3.877  -0.651  1.00  0.00           C
ATOM     11  CZ  PHE A   1       0.087  -4.536  -0.494  1.00  0.00           C
ATOM     12  N   ALA A   2       2.433   0.834   0.000  1.00  0.00           N
ATOM     13  CA  ALA A   2       3.680   1.573   0.000  1.00  0.00           C
ATOM     14  C   ALA A   2       4.883   0.639   0.000  1.00  0.00           C
ATOM     15  O   ALA A   2       4.741  -0.585   0.000  1.00  0.00           O
ATOM     16  CB  ALA A   2       3.730   2.452  -1.253  1.00  0.00           C
TER
END
"""


def _mol_from(block: str, sanitize: bool = True) -> Chem.Mol:
    mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=sanitize)
    assert mol is not None, "fixture failed to parse"
    return mol


def test_rdkit_pdb_reader_bonds_and_aromatises_the_backbone():
    """The loader's parser must recover connectivity and ring aromaticity."""
    mol = _mol_from(_PHE_ALA_PDB)

    assert mol.GetNumBonds() >= mol.GetNumAtoms() - 1, (
        "a peptide is a covalent chain; near-zero bonds means template "
        "perception did not run"
    )
    assert sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) == 6, (
        "the phenylalanine ring must be perceived as aromatic or PiStacking "
        "and CationPi can never fire"
    )


def test_guard_accepts_a_properly_bonded_receptor():
    assert_receptor_is_bonded(_mol_from(_PHE_ALA_PDB), "fixture")


def test_guard_rejects_atoms_without_bonds():
    """The exact shape of the shipped defect: atoms present, bonds absent."""
    unbonded = Chem.RWMol(_mol_from(_PHE_ALA_PDB))
    for bond in list(unbonded.GetBonds()):
        unbonded.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    with pytest.raises(ValueError, match="no connectivity"):
        assert_receptor_is_bonded(unbonded.GetMol(), "fixture")


def test_guard_rejects_a_receptor_with_no_aromatic_atoms():
    mol = _mol_from(_PHE_ALA_PDB)
    for atom in mol.GetAtoms():
        atom.SetIsAromatic(False)

    with pytest.raises(ValueError, match="no aromatic atoms"):
        assert_receptor_is_bonded(mol, "fixture")


def test_mdanalysis_route_would_have_been_caught():
    """The old loader's output fails the guard, which is the whole point.

    MDAnalysis is not imported here -- its failure mode is reproduced
    directly, because what matters is that a bond-free molecule cannot pass.
    """
    mda_like = Chem.RWMol(_mol_from(_PHE_ALA_PDB))
    for bond in list(mda_like.GetBonds())[1:]:  # leave one, as a disulfide would
        mda_like.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    with pytest.raises(ValueError):
        assert_receptor_is_bonded(mda_like.GetMol(), "fixture")
