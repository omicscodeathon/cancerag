"""Tests for cancerag.docking.runner — per-job dock-meta sidecars."""

from __future__ import annotations

from pathlib import Path

import pytest

from cancerag.docking.runner import read_dock_meta, write_dock_meta


@pytest.mark.unit
class TestDockMeta:
    def test_round_trip(self, tmp_path: Path):
        artifact = tmp_path / "out.pdbqt"
        artifact.write_text("MODEL 1\nENDMDL\n")
        write_dock_meta(
            artifact,
            vina_version="1.2.5",
            exhaustiveness=16,
            num_modes=9,
            ligand_inchikey="ABCDEFG",
            receptor_pdb_id="6CM4",
            receptor_uniprot="P14416",
            box_center=(1.0, 2.0, 3.0),
            box_size=(22.0, 22.0, 22.0),
            wall_seconds=87.3,
        )
        meta = read_dock_meta(artifact)
        assert meta["vina_version"] == "1.2.5"
        assert meta["exhaustiveness"] == 16
        assert meta["num_modes"] == 9
        assert meta["box_center"] == [1.0, 2.0, 3.0]
        assert "docked_at_utc" in meta

    def test_missing_meta_raises(self, tmp_path: Path):
        artifact = tmp_path / "out.pdbqt"
        artifact.write_text("")
        with pytest.raises(FileNotFoundError):
            read_dock_meta(artifact)
