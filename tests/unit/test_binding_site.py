"""Tests for cancerag.preprocessing.binding_site."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cancerag.preprocessing import binding_site
from cancerag.preprocessing.binding_site import (
    BindingSite,
    box_from_ligand_coords,
    define_binding_site,
    find_cocrystal_ligand,
)
from cancerag.preprocessing.pocket_predictors import Pocket


def _write_pdb_with_het(path: Path, *, resname: str, n_atoms: int) -> None:
    """Tiny PDB with a single HETATM residue of `n_atoms` heavy atoms."""
    lines: list[str] = []
    serial = 1
    for i in range(n_atoms):
        # Place atoms in a small spiral — non-degenerate bounding box.
        x = float(i) * 0.5
        y = float(i % 3) * 0.7
        z = float(i // 3) * 1.1
        lines.append(
            f"HETATM{serial:>5}  C{i:<2} {resname:>3} A{900:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.unit
class TestFindCocrystalLigand:
    def test_finds_real_ligand(self, tmp_path: Path):
        pdb = tmp_path / "rec.pdb"
        _write_pdb_with_het(pdb, resname="8NU", n_atoms=20)
        result = find_cocrystal_ligand(pdb)
        assert result is not None
        resname, coords = result
        assert resname == "8NU"
        assert len(coords) == 20

    def test_skips_lipid(self, tmp_path: Path):
        pdb = tmp_path / "rec.pdb"
        _write_pdb_with_het(pdb, resname="CLR", n_atoms=20)
        # Cholesterol should be skipped via LIGAND_AUTO_DETECT_IGNORE
        assert find_cocrystal_ligand(pdb) is None

    def test_skips_too_small(self, tmp_path: Path):
        pdb = tmp_path / "rec.pdb"
        _write_pdb_with_het(pdb, resname="8NU", n_atoms=4)
        # Only 4 heavy atoms < default min_heavy_atoms=8 -> skipped
        assert find_cocrystal_ligand(pdb) is None

    def test_no_ligand_at_all(self, tmp_path: Path):
        pdb = tmp_path / "rec.pdb"
        pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        assert find_cocrystal_ligand(pdb) is None


@pytest.mark.unit
class TestBoxFromLigand:
    def test_centered_on_geometric_center(self):
        coords = [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)]
        center, size = box_from_ligand_coords(coords, fixed_size=22.0)
        assert center == pytest.approx((1.0, 1.0, 1.0))
        assert size == (22.0, 22.0, 22.0)

    def test_variable_size_with_padding(self):
        coords = [(0, 0, 0), (10, 5, 3)]
        center, size = box_from_ligand_coords(
            coords, fixed_size=None, padding=4.0
        )
        # span = (10, 5, 3); + 8 padding -> (18, 13, 11)
        assert size == pytest.approx((18.0, 13.0, 11.0))
        # center = midpoint -> (5, 2.5, 1.5)
        assert center == pytest.approx((5.0, 2.5, 1.5))


@pytest.mark.unit
class TestDefineBindingSite:
    def test_cocrystal_path_when_ligand_present(self, tmp_path: Path):
        receptor = tmp_path / "P14416.pdb"
        receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        raw = tmp_path / "raw.pdb"
        _write_pdb_with_het(raw, resname="8NU", n_atoms=20)

        site = define_binding_site(
            uniprot="P14416", biasdb_name="D2 receptor",
            receptor_pdb=receptor, source_pdb_id="6CM4",
            structure_source="pdb", raw_pdb_for_cocrystal=raw,
        )
        assert site.method == "cocrystal_ligand"
        assert site.cocrystal_ligand_resname == "8NU"
        assert site.cocrystal_ligand_n_atoms == 20
        assert site.confidence == "ok"
        assert site.size_x == 22.0

    def test_pocket_prediction_path_when_no_cocrystal(self, tmp_path: Path):
        """When raw_pdb has no usable ligand, fall through to pocket
        prediction. We mock both wrappers."""
        receptor = tmp_path / "AF.pdb"
        receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")

        with patch.object(binding_site, "run_p2rank",
                          return_value=[
                              Pocket(rank=1, score=30.0, center_x=10.0,
                                     center_y=10.0, center_z=10.0,
                                     method="p2rank")
                          ]), \
             patch.object(binding_site, "run_fpocket",
                          return_value=[
                              Pocket(rank=1, score=0.9, center_x=11.0,
                                     center_y=10.0, center_z=10.0,
                                     method="fpocket")
                          ]):
            site = define_binding_site(
                uniprot="P51681", biasdb_name="CCR5",
                receptor_pdb=receptor, source_pdb_id=None,
                structure_source="alphafold", raw_pdb_for_cocrystal=None,
            )
        assert site.method == "consensus_pocket".replace("_pocket", "") or site.method == "consensus"
        assert site.confidence == "ok"
        assert site.center_x == pytest.approx(10.0)
        assert site.pocket_prediction_distance_angstroms == pytest.approx(1.0)
        assert site.pocket_prediction_agrees is True

    def test_pocket_prediction_disagreement_low_confidence(self, tmp_path: Path):
        receptor = tmp_path / "AF.pdb"
        receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        with patch.object(binding_site, "run_p2rank",
                          return_value=[
                              Pocket(rank=1, score=30.0, center_x=10.0,
                                     center_y=10.0, center_z=10.0,
                                     method="p2rank")
                          ]), \
             patch.object(binding_site, "run_fpocket",
                          return_value=[
                              Pocket(rank=1, score=0.9, center_x=40.0,
                                     center_y=40.0, center_z=40.0,
                                     method="fpocket")
                          ]):
            site = define_binding_site(
                uniprot="P51681", biasdb_name="CCR5",
                receptor_pdb=receptor, source_pdb_id=None,
                structure_source="alphafold",
            )
        assert site.confidence == "low_confidence"
        assert site.pocket_prediction_agrees is False
        # Falls back to P2Rank center
        assert site.center_x == pytest.approx(10.0)

    def test_no_pocket_found(self, tmp_path: Path):
        receptor = tmp_path / "AF.pdb"
        receptor.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
        with patch.object(binding_site, "run_p2rank", return_value=[]), \
             patch.object(binding_site, "run_fpocket", return_value=[]):
            site = define_binding_site(
                uniprot="X1", biasdb_name="weird",
                receptor_pdb=receptor, source_pdb_id=None,
                structure_source="alphafold",
            )
        assert site.confidence == "no_pocket_found"
        assert site.method == "no_pocket_found"
