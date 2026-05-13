"""Tests for cancerag.preprocessing.het_resnames."""

from __future__ import annotations

import pytest

from cancerag.preprocessing.het_resnames import (
    LIGAND_AUTO_DETECT_IGNORE,
    is_ignorable_het,
)


@pytest.mark.unit
class TestIgnoreList:
    def test_includes_lipids(self):
        for r in ("CLR", "CHS", "OLA", "OLC", "PLM"):
            assert r in LIGAND_AUTO_DETECT_IGNORE, f"missing lipid {r}"

    def test_includes_glycans(self):
        for r in ("NAG", "MAN", "BMA", "FUC"):
            assert r in LIGAND_AUTO_DETECT_IGNORE, f"missing glycan {r}"

    def test_includes_detergents(self):
        for r in ("LMT", "DDM", "BOG", "OG", "MES", "HEPES"):
            assert r in LIGAND_AUTO_DETECT_IGNORE, f"missing buffer/detergent {r}"

    def test_includes_cofactors(self):
        for r in ("GTP", "GDP", "ATP", "NAD"):
            assert r in LIGAND_AUTO_DETECT_IGNORE, f"missing cofactor {r}"

    def test_legacy_minimum_set_still_present(self):
        for r in (
            "HOH", "WAT", "SO4", "GOL", "PO4", "EDO",
            "MG", "CA", "ZN", "MN", "CL", "NA", "K",
        ):
            assert r in LIGAND_AUTO_DETECT_IGNORE

    def test_real_drug_resnames_NOT_in_ignore(self):
        # Three random orthosteric ligand resnames from real GPCR PDBs:
        #   6CM4 (DRD2): 8NU - risperidone
        #   3SN6 (ADRB2): P0G - BI-167107
        #   4DKL (mu-opioid): 4VO - beta-funaltrexamine
        for r in ("8NU", "P0G", "4VO"):
            assert r not in LIGAND_AUTO_DETECT_IGNORE

    def test_helper_function(self):
        assert is_ignorable_het("CLR") is True
        assert is_ignorable_het("clr") is True
        assert is_ignorable_het("8NU") is False
        assert is_ignorable_het("") is True
        assert is_ignorable_het(" hoh ") is True
