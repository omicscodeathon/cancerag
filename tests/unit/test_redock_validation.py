"""Tests for cancerag.preprocessing.redock_validation.

Pure-function tests for the parsers and RMSD computation. The Vina /
obabel subprocess calls are exercised at the integration / real-data
level, not here."""

from __future__ import annotations

from pathlib import Path

import pytest

from cancerag.preprocessing.redock_validation import (
    _heavy_atom_rmsd,
    _read_crystal_heavy_coords,
    _read_pdbqt_top_pose,
)


@pytest.mark.unit
class TestHeavyAtomRmsd:
    def test_identical_coords_returns_zero(self):
        c = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        assert _heavy_atom_rmsd(c, c) == pytest.approx(0.0)

    def test_one_angstrom_shift(self):
        c = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        p = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        assert _heavy_atom_rmsd(c, p) == pytest.approx(1.0)

    def test_mismatched_atom_count_returns_none(self):
        c = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        p = [(0.0, 0.0, 0.0)]
        assert _heavy_atom_rmsd(c, p) is None

    def test_empty_returns_none(self):
        assert _heavy_atom_rmsd([], []) is None


@pytest.mark.unit
class TestReadPdbqtTopPose:
    def test_extracts_only_first_model(self, tmp_path: Path):
        # Two MODEL blocks; we should get only the first plus its affinity.
        p = tmp_path / "out.pdbqt"
        p.write_text(
            "MODEL 1\n"
            "REMARK VINA RESULT:    -7.20    0.000    0.000\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  C   LIG A   1       1.500   0.000   0.000  1.00  0.00           C\n"
            "ENDMDL\n"
            "MODEL 2\n"
            "REMARK VINA RESULT:    -6.50    1.234    2.345\n"
            "ATOM      1  C   LIG A   1       9.000   9.000   9.000  1.00  0.00           C\n"
            "ENDMDL\n"
        )
        coords, aff = _read_pdbqt_top_pose(p)
        assert aff == pytest.approx(-7.20)
        assert coords == [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)]

    def test_skips_hydrogens(self, tmp_path: Path):
        p = tmp_path / "out.pdbqt"
        p.write_text(
            "MODEL 1\n"
            "REMARK VINA RESULT:    -7.20    0.000    0.000\n"
            "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  H   LIG A   1       1.000   0.000   0.000  1.00  0.00           H\n"
            "ENDMDL\n"
        )
        coords, _ = _read_pdbqt_top_pose(p)
        assert coords == [(0.0, 0.0, 0.0)]


@pytest.mark.unit
class TestReadCrystalHeavyCoords:
    def test_skips_hydrogens(self, tmp_path: Path):
        p = tmp_path / "lig.pdb"
        p.write_text(
            "HETATM    1  C   LIG A 900       1.000   2.000   3.000  1.00  0.00           C\n"
            "HETATM    2  H   LIG A 900       2.000   2.000   3.000  1.00  0.00           H\n"
            "HETATM    3  N   LIG A 900       4.000   5.000   6.000  1.00  0.00           N\n"
            "END\n"
        )
        coords = _read_crystal_heavy_coords(p)
        assert coords == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

    def test_empty_pdb(self, tmp_path: Path):
        p = tmp_path / "lig.pdb"
        p.write_text("HEADER nothing\nEND\n")
        assert _read_crystal_heavy_coords(p) == []
