"""Tests for cancerag.docking.production_docking.

Pure-function tests for the parts of Stage 05 that don't require Vina:
- ligand 3D embedding (_smiles_to_3d_pdb)
- job-list construction (build_job_list)
- feature CSV emission with confidence-flag joins (emit_features_csv)
- audit markdown shape (emit_audit)

The Vina subprocess path is exercised by the real-data smoke run, not here."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from cancerag.docking.production_docking import (
    DockingResult,
    _smiles_to_3d_pdb,
    build_job_list,
    emit_audit,
    emit_features_csv,
)


@pytest.mark.unit
class TestSmilesTo3dPdb:
    def test_simple_molecule(self, tmp_path: Path):
        out = tmp_path / "lig.pdb"
        ok = _smiles_to_3d_pdb("CCO", out, seed=7)
        assert ok is True
        text = out.read_text()
        assert "ATOM" in text or "HETATM" in text
        # ethanol = 3 heavy atoms, plus added Hs
        n_atoms = sum(
            1 for line in text.splitlines()
            if line.startswith(("ATOM", "HETATM"))
        )
        assert n_atoms >= 3

    def test_aspirin_embeds(self, tmp_path: Path):
        out = tmp_path / "asa.pdb"
        ok = _smiles_to_3d_pdb(
            "CC(=O)Oc1ccccc1C(=O)O", out, seed=42
        )
        assert ok is True and out.exists() and out.stat().st_size > 0

    def test_invalid_smiles_returns_false(self, tmp_path: Path):
        out = tmp_path / "bad.pdb"
        ok = _smiles_to_3d_pdb("not_a_smiles_!!!", out)
        assert ok is False

    def test_seed_is_reproducible(self, tmp_path: Path):
        a = tmp_path / "a.pdb"
        b = tmp_path / "b.pdb"
        assert _smiles_to_3d_pdb("CCO", a, seed=123)
        assert _smiles_to_3d_pdb("CCO", b, seed=123)
        # Same seed -> identical coordinates section.
        assert a.read_text() == b.read_text()


@pytest.mark.unit
class TestBuildJobList:
    @staticmethod
    def _write_dataset(
        tmp: Path,
        *,
        rows: list[dict],
        sites: list[dict],
        receptors_present: set[str],
    ) -> tuple[Path, Path, Path]:
        unified = tmp / "unified.csv"
        pd.DataFrame(rows).to_csv(unified, index=False)
        sites_path = tmp / "sites.json"
        sites_path.write_text(json.dumps({"binding_sites": sites}))
        receptors_dir = tmp / "receptors"
        receptors_dir.mkdir()
        for u in receptors_present:
            (receptors_dir / f"{u}.pdb").write_text("HEADER stub\nEND\n")
        return unified, sites_path, receptors_dir

    def test_unique_pairs_only(self, tmp_path: Path):
        # Two duplicate (inchikey, receptor_uniprot) rows -> one job.
        rows = [
            {
                "inchikey": "AAAAAAAAAAAAAA-FFFFFFFFFF",
                "inchikey14": "AAAAAAAAAAAAAA",
                "receptor_uniprot": "P00001",
                "canonical_smiles_std": "CCO",
            },
            {  # duplicate pair, different bias_pathway upstream
                "inchikey": "AAAAAAAAAAAAAA-FFFFFFFFFF",
                "inchikey14": "AAAAAAAAAAAAAA",
                "receptor_uniprot": "P00001",
                "canonical_smiles_std": "CCO",
            },
            {
                "inchikey": "BBBBBBBBBBBBBB-GGGGGGGGGG",
                "inchikey14": "BBBBBBBBBBBBBB",
                "receptor_uniprot": "P00001",
                "canonical_smiles_std": "CCN",
            },
        ]
        sites = [{
            "uniprot": "P00001",
            "center_x": 1.0, "center_y": 2.0, "center_z": 3.0,
            "size_x": 22.0, "size_y": 22.0, "size_z": 22.0,
            "confidence": "ok",
        }]
        u, s, r = self._write_dataset(
            tmp_path, rows=rows, sites=sites, receptors_present={"P00001"}
        )
        jobs = build_job_list(
            unified_csv=u, binding_sites_json=s, receptors_dir=r,
            work_root=tmp_path / "work",
        )
        assert len(jobs) == 2
        assert {j.canonical_smiles for j in jobs} == {"CCO", "CCN"}
        assert all(j.box_center == (1.0, 2.0, 3.0) for j in jobs)
        assert all(j.box_size == (22.0, 22.0, 22.0) for j in jobs)

    def test_skips_receptor_without_binding_site(self, tmp_path: Path):
        rows = [{
            "inchikey": "X" * 27,
            "inchikey14": "X" * 14,
            "receptor_uniprot": "PNOSITE",
            "canonical_smiles_std": "CCO",
        }]
        u, s, r = self._write_dataset(
            tmp_path, rows=rows, sites=[], receptors_present={"PNOSITE"}
        )
        jobs = build_job_list(
            unified_csv=u, binding_sites_json=s, receptors_dir=r,
            work_root=tmp_path / "work",
        )
        assert jobs == []

    def test_skips_receptor_without_pdb_file(self, tmp_path: Path):
        rows = [{
            "inchikey": "X" * 27,
            "inchikey14": "X" * 14,
            "receptor_uniprot": "PNOPDB",
            "canonical_smiles_std": "CCO",
        }]
        sites = [{
            "uniprot": "PNOPDB",
            "center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
            "size_x": 22.0, "size_y": 22.0, "size_z": 22.0,
            "confidence": "ok",
        }]
        u, s, r = self._write_dataset(
            tmp_path, rows=rows, sites=sites, receptors_present=set()
        )
        jobs = build_job_list(
            unified_csv=u, binding_sites_json=s, receptors_dir=r,
            work_root=tmp_path / "work",
        )
        assert jobs == []

    def test_pair_id_and_work_dir_format(self, tmp_path: Path):
        rows = [{
            "inchikey": "ABCDEFGHIJKLMN-OPQRSTUVWX",
            "inchikey14": "ABCDEFGHIJKLMN",
            "receptor_uniprot": "P12345",
            "canonical_smiles_std": "CCO",
        }]
        sites = [{
            "uniprot": "P12345",
            "center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
            "size_x": 22.0, "size_y": 22.0, "size_z": 22.0,
            "confidence": "ok",
        }]
        u, s, r = self._write_dataset(
            tmp_path, rows=rows, sites=sites, receptors_present={"P12345"}
        )
        work_root = tmp_path / "work"
        jobs = build_job_list(
            unified_csv=u, binding_sites_json=s, receptors_dir=r,
            work_root=work_root,
        )
        assert len(jobs) == 1
        j = jobs[0]
        assert j.pair_id == "ABCDEFGHIJKLMN__P12345"
        assert Path(j.work_dir) == work_root / j.pair_id


@pytest.mark.unit
class TestEmitFeaturesCsv:
    @staticmethod
    def _result(
        uniprot: str, *, success: bool = True, aff: float | None = -8.0
    ) -> dict:
        return asdict(DockingResult(
            pair_id=f"PAIR__{uniprot}",
            inchikey="K" * 27,
            receptor_uniprot=uniprot,
            success=success,
            n_poses=9 if success else 0,
            vina_affinity_best=aff,
            vina_affinity_mean_top3=(aff - 0.2) if aff is not None else None,
            vina_affinity_gap_1_2=0.4 if success else None,
            vina_pose_diversity_rmsd=2.1 if success else None,
            vina_n_distinct_clusters=3 if success else 0,
            wall_seconds=12.3,
        ))

    @staticmethod
    def _write_inputs(
        tmp: Path,
        *,
        sites: list[dict],
        redock: list[dict] | None,
        gnina: list[dict] | None,
    ) -> tuple[Path, Path, Path]:
        sites_path = tmp / "sites.json"
        sites_path.write_text(json.dumps({"binding_sites": sites}))
        redock_path = tmp / "redock.json"
        if redock is not None:
            redock_path.write_text(json.dumps({"redock_results": redock}))
        gnina_path = tmp / "gnina.json"
        if gnina is not None:
            gnina_path.write_text(json.dumps({"rescore_results": gnina}))
        return sites_path, redock_path, gnina_path

    def test_high_confidence_when_rmsd_pass_and_cnn_pass(self, tmp_path: Path):
        sites = [{"uniprot": "P1", "confidence": "ok"}]
        redock = [{"uniprot": "P1", "rmsd_angstrom": 1.5}]
        gnina = [{"uniprot": "P1", "top_pose_cnn_score": 0.85}]
        s, rd, gn = self._write_inputs(
            tmp_path, sites=sites, redock=redock, gnina=gnina
        )
        out = tmp_path / "features.csv"
        df = emit_features_csv(
            [self._result("P1")],
            binding_sites_json=s, redock_validation_json=rd,
            gnina_rescore_json=gn, output_path=out,
        )
        assert df.iloc[0]["docking_confidence"] == "high"
        assert df.iloc[0]["redock_rmsd_angstrom"] == pytest.approx(1.5)
        assert df.iloc[0]["gnina_cnn_score"] == pytest.approx(0.85)
        # round-trips to disk
        assert out.exists()

    def test_marginal_when_only_cnn_passes(self, tmp_path: Path):
        sites = [{"uniprot": "P2", "confidence": "ok"}]
        redock = [{"uniprot": "P2", "rmsd_angstrom": 4.5}]  # fail (>2.5)
        gnina = [{"uniprot": "P2", "top_pose_cnn_score": 0.75}]  # pass (>=0.7)
        s, rd, gn = self._write_inputs(
            tmp_path, sites=sites, redock=redock, gnina=gnina
        )
        df = emit_features_csv(
            [self._result("P2")],
            binding_sites_json=s, redock_validation_json=rd,
            gnina_rescore_json=gn, output_path=tmp_path / "f.csv",
        )
        assert df.iloc[0]["docking_confidence"] == "marginal"

    def test_marginal_when_cnn_in_middle_band(self, tmp_path: Path):
        sites = [{"uniprot": "P3", "confidence": "ok"}]
        redock = [{"uniprot": "P3", "rmsd_angstrom": 5.0}]
        gnina = [{"uniprot": "P3", "top_pose_cnn_score": 0.5}]  # 0.4..0.7
        s, rd, gn = self._write_inputs(
            tmp_path, sites=sites, redock=redock, gnina=gnina
        )
        df = emit_features_csv(
            [self._result("P3")],
            binding_sites_json=s, redock_validation_json=rd,
            gnina_rescore_json=gn, output_path=tmp_path / "f.csv",
        )
        assert df.iloc[0]["docking_confidence"] == "marginal"

    def test_low_when_both_fail(self, tmp_path: Path):
        sites = [{"uniprot": "P4", "confidence": "ok"}]
        redock = [{"uniprot": "P4", "rmsd_angstrom": 8.0}]
        gnina = [{"uniprot": "P4", "top_pose_cnn_score": 0.1}]
        s, rd, gn = self._write_inputs(
            tmp_path, sites=sites, redock=redock, gnina=gnina
        )
        df = emit_features_csv(
            [self._result("P4")],
            binding_sites_json=s, redock_validation_json=rd,
            gnina_rescore_json=gn, output_path=tmp_path / "f.csv",
        )
        assert df.iloc[0]["docking_confidence"] == "low"

    def test_alphafold_no_redock_falls_back_to_marginal(self, tmp_path: Path):
        # Receptor has no redock or gnina entry (e.g. AlphaFold model).
        sites = [{"uniprot": "PAF", "confidence": "ok"}]
        s, rd, gn = self._write_inputs(
            tmp_path, sites=sites, redock=[], gnina=[]
        )
        df = emit_features_csv(
            [self._result("PAF")],
            binding_sites_json=s, redock_validation_json=rd,
            gnina_rescore_json=gn, output_path=tmp_path / "f.csv",
        )
        assert df.iloc[0]["docking_confidence"] == "marginal"

    def test_empty_results_writes_empty_csv(self, tmp_path: Path):
        sites = [{"uniprot": "P1", "confidence": "ok"}]
        s, rd, gn = self._write_inputs(
            tmp_path, sites=sites, redock=[], gnina=[]
        )
        out = tmp_path / "f.csv"
        df = emit_features_csv(
            [], binding_sites_json=s, redock_validation_json=rd,
            gnina_rescore_json=gn, output_path=out,
        )
        assert df.empty
        assert out.exists()


@pytest.mark.unit
class TestEmitAudit:
    def test_summary_lines_and_per_receptor_table(self, tmp_path: Path):
        df = pd.DataFrame([
            {
                "receptor_uniprot": "P1", "success": True,
                "vina_affinity_best": -8.0,
                "docking_confidence": "high",
            },
            {
                "receptor_uniprot": "P1", "success": True,
                "vina_affinity_best": -7.5,
                "docking_confidence": "high",
            },
            {
                "receptor_uniprot": "P2", "success": False,
                "vina_affinity_best": None,
                "docking_confidence": "low",
            },
        ])
        out = tmp_path / "audit.md"
        emit_audit(df, audit_path=out)
        text = out.read_text()
        assert "Total docking jobs: 3" in text
        assert "Successful: 2" in text
        assert "Failed: 1" in text
        assert "## By docking confidence" in text
        assert "high: 2" in text
        assert "low: 1" in text
        assert "## Per-receptor" in text
        # P1 has 2 jobs, mean affinity -7.75
        assert "| P1 | 2 | 2 |" in text
        assert "-7.75" in text

    def test_handles_empty_df(self, tmp_path: Path):
        out = tmp_path / "audit.md"
        emit_audit(pd.DataFrame(), audit_path=out)
        text = out.read_text()
        assert "Total docking jobs: 0" in text
