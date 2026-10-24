"""Capping groups and modified residues are not co-crystal ligands.

Regression cover for the defect that produced two unusable receptors.

`active_site_identifier` and the PDB selector auto-detect "the co-crystal
orthosteric ligand" from HETATM records, and anything not in the ignore list
is a candidate. Peptide chain terminators are HETATM records attached to the
protein, and they are small, so they won here:

    CCR2  (P41597)  picked ACE -- an N-terminal acetyl cap
    CCR5  (P51681)  picked TYS -- a sulfotyrosine residue

A 22 A docking box was then centred on each, landing outside any pocket. Every
pose for those receptors scored about 0 kcal/mol. Both had already been through
one repair pass that swapped the PDB in response to the chain-identity gate
without re-checking what the box was built from, so the repair changed the
structure and left the fault in place.

Rebuilt from a real ligand the same pairs give -7.321 (CCR2, 6GPS) and -6.833
(CCR5, 4MBS with the maraviroc box).

TYS carries extra weight: sulfotyrosine is characteristic of chemokine-receptor
N-termini, so the failure was concentrated on that family rather than scattered.
"""

from __future__ import annotations

import pytest

from cancerag.preprocessing.het_resnames import (
    LIGAND_AUTO_DETECT_IGNORE,
    is_ignorable_het,
)

# The exact residues that were mistaken for ligands, plus the other common
# terminators that would fail the same way.
CAPS_AND_MODIFIED = ["ACE", "NH2", "NME", "FOR", "TYS"]

# Real co-crystal ligands from this dataset's receptors. The ignore list must
# not grow so eagerly that it starts discarding these.
REAL_LIGANDS = ["MRV", "WEC", "WE9", "U99", "IXO", "Q6Q", "DGV", "MK6", "8NU"]


@pytest.mark.parametrize("resname", CAPS_AND_MODIFIED)
def test_caps_and_modified_residues_are_not_ligands(resname: str):
    assert is_ignorable_het(resname), (
        f"{resname} is a capping group or modified residue, not an orthosteric "
        "ligand; auto-detection would centre a docking box on it"
    )


@pytest.mark.parametrize("resname", REAL_LIGANDS)
def test_real_cocrystal_ligands_survive(resname: str):
    assert not is_ignorable_het(resname), (
        f"{resname} is a genuine co-crystal ligand and must remain available "
        "for binding-site detection"
    )


def test_is_ignorable_het_is_case_insensitive_and_null_safe():
    assert is_ignorable_het("ace")
    assert is_ignorable_het(" TYS ")
    assert is_ignorable_het("")
    assert is_ignorable_het(None)  # type: ignore[arg-type]


def test_ignore_list_has_not_collapsed():
    """Guard against an edit that empties or truncates the set."""
    assert len(LIGAND_AUTO_DETECT_IGNORE) > 100
    for essential in ("HOH", "MSE", "SO4", "NAG", "CLR"):
        assert essential in LIGAND_AUTO_DETECT_IGNORE
