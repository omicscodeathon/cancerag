"""Ballesteros-Weinstein remap of ProLIF interaction columns.

Every test here is offline: GPCRdb is replaced by an in-memory residue table,
and receptors are tiny synthetic PDB files.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cancerag.features import bw_numbering as bw
from cancerag.features import featurization

META = {
    "inchikey": "X", "n_residues_contacted": 1, "n_total_contacts": 1,
}

ONE_TO_THREE = {
    "A": "ALA", "D": "ASP", "E": "GLU", "G": "GLY", "K": "LYS",
    "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
}


def _write_pdb(path: Path, sequence: str, start: int = 1) -> Path:
    """One CA atom per residue, numbered from ``start``."""
    lines = []
    for i, aa in enumerate(sequence):
        num = start + i
        lines.append(
            f"ATOM  {i + 1:5d}  CA  {ONE_TO_THREE[aa]:3s} A{num:4d}    "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


# Canonical receptor: the five class A anchors at UniProt positions 2,4,6,8,10.
CANONICAL = "ANADARAPAP"
POS_TO_BW = {2: "1.50", 4: "2.50", 6: "3.50", 8: "6.50", 10: "7.50",
             5: "3.32"}


@pytest.fixture
def fake_gpcrdb(monkeypatch):
    monkeypatch.setattr(
        bw, "uniprot_position_to_bw", lambda acc: (dict(POS_TO_BW), CANONICAL)
    )


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).fillna(0)
    meta = list(bw.IFP_META_COLS)
    bits = sorted(c for c in df.columns if c not in meta)
    for c in bits:
        df[c] = df[c].astype(np.uint8)
    return df[meta + bits]


# ------------------------------------------------------------------ remap


@pytest.mark.unit
class TestRemap:
    def test_two_construct_numbers_merge_into_one_bw_column_by_or(self):
        # The same structural position (3.32) is residue 80 in one construct
        # and 86 in another. Each fires in one row; the BW column fires in both.
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1",
             "ifp_ASP80.A_VdWContact": 1, **META},
            {"pair_id": "b", "receptor_uniprot": "R2",
             "ifp_ASP86.A_VdWContact": 1, **META},
            {"pair_id": "c", "receptor_uniprot": "R2", **META},
        ])
        maps = {"R1": {80: "3.32"}, "R2": {86: "3.32"}}
        out, cmap = bw.remap_interaction_fingerprints(raw, maps)

        bits = [c for c in out.columns if c not in bw.IFP_META_COLS]
        assert bits == ["ifp_BW3.32_VdWContact"]
        assert out["ifp_BW3.32_VdWContact"].tolist() == [1, 1, 0]
        assert out["ifp_BW3.32_VdWContact"].dtype == np.uint8
        assert cmap == {"ifp_BW3.32_VdWContact": [
            "ifp_ASP80.A_VdWContact", "ifp_ASP86.A_VdWContact"]}

    def test_or_within_one_row_does_not_double_count(self):
        # Two raw residues of one receptor that map to the same position
        # (an alignment can do this across a fusion junction) still give 1.
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1",
             "ifp_ASP80.A_VdWContact": 1, "ifp_GLU81.A_VdWContact": 1, **META},
        ])
        out, _ = bw.remap_interaction_fingerprints(
            raw, {"R1": {80: "3.32", 81: "3.32"}})
        assert out["ifp_BW3.32_VdWContact"].tolist() == [1]

    def test_interaction_type_is_kept_apart(self):
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1",
             "ifp_ASP80.A_VdWContact": 1, "ifp_ASP80.A_HBDonor": 1, **META},
        ])
        out, _ = bw.remap_interaction_fingerprints(raw, {"R1": {80: "3.32"}})
        assert {"ifp_BW3.32_VdWContact", "ifp_BW3.32_HBDonor"} <= set(out.columns)

    def test_class_c_receptor_keeps_raw_names(self):
        # The calcium-sensing receptor fails the anchor gate, so its map is
        # empty and its columns pass through untouched.
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "P41180",
             "ifp_PHE149.A_VdWContact": 1, **META},
            {"pair_id": "b", "receptor_uniprot": "R1",
             "ifp_ASP80.A_VdWContact": 1, **META},
        ])
        out, cmap = bw.remap_interaction_fingerprints(
            raw, {"P41180": {}, "R1": {80: "3.32"}})
        assert out["ifp_PHE149.A_VdWContact"].tolist() == [1, 0]
        assert out["ifp_BW3.32_VdWContact"].tolist() == [0, 1]
        assert "ifp_PHE149.A_VdWContact" not in sum(cmap.values(), [])

    def test_shared_raw_column_is_split_by_receptor(self):
        # Residue 80 is 3.32 in R1, but R2 has no map: the raw column must
        # survive holding only R2's row, and R1's row moves to the BW column.
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1",
             "ifp_ASP80.A_VdWContact": 1, **META},
            {"pair_id": "b", "receptor_uniprot": "R2",
             "ifp_ASP80.A_VdWContact": 1, **META},
        ])
        out, _ = bw.remap_interaction_fingerprints(
            raw, {"R1": {80: "3.32"}, "R2": {}})
        assert out["ifp_ASP80.A_VdWContact"].tolist() == [0, 1]
        assert out["ifp_BW3.32_VdWContact"].tolist() == [1, 0]

    def test_residue_without_generic_number_keeps_raw_name(self):
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1",
             "ifp_LYS17.A_VdWContact": 1, **META},
        ])
        out, cmap = bw.remap_interaction_fingerprints(raw, {"R1": {80: "3.32"}})
        assert out["ifp_LYS17.A_VdWContact"].tolist() == [1]
        assert cmap == {}

    def test_meta_passes_through_and_column_order(self):
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1", "inchikey": "K",
             "n_residues_contacted": 7, "n_total_contacts": 9,
             "ifp_ASP80.A_VdWContact": 1, "ifp_ALA5.A_VdWContact": 1},
        ])
        out, _ = bw.remap_interaction_fingerprints(raw, {"R1": {80: "3.32"}})
        assert list(out.columns) == [
            *bw.IFP_META_COLS, "ifp_ALA5.A_VdWContact", "ifp_BW3.32_VdWContact"]
        assert out.loc[0, "n_residues_contacted"] == 7
        assert out.loc[0, "n_total_contacts"] == 9

    def test_columns_that_never_fire_are_dropped(self):
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1",
             "ifp_ASP80.A_VdWContact": 0, **META},
        ])
        out, _ = bw.remap_interaction_fingerprints(raw, {"R1": {80: "3.32"}})
        assert list(out.columns) == list(bw.IFP_META_COLS)

    def test_bw_column_name_is_by_position_not_residue(self):
        assert bw.bw_column_name("3.32", "HBDonor") == "ifp_BW3.32_HBDonor"


# ------------------------------------------------------------- anchor gate


@pytest.mark.unit
class TestAnchorGate:
    def test_all_anchors_match(self, tmp_path, fake_gpcrdb):
        pdb = _write_pdb(tmp_path / "R.prolif_H.pdb", CANONICAL)
        mapping, report = bw.receptor_bw_map("R", pdb)
        assert report["status"] == "ok"
        assert report["anchors_ok"] == 5
        assert mapping[5] == "3.32"

    def test_renumbered_construct_still_maps(self, tmp_path, fake_gpcrdb):
        # prolif files are renumbered: the same sequence starting at 101 must
        # map residue 105 (not 5) to 3.32.
        pdb = _write_pdb(tmp_path / "R.prolif_H.pdb", CANONICAL, start=101)
        mapping, report = bw.receptor_bw_map("R", pdb)
        assert report["status"] == "ok"
        assert mapping[105] == "3.32"
        assert 5 not in mapping

    def test_rejects_alignment_whose_anchors_land_wrong(
        self, tmp_path, fake_gpcrdb,
    ):
        # Identity is exactly the 0.50 floor (the five alanines match), so only
        # the anchor gate stands between this and a wrong map: every anchor
        # position holds the wrong residue.
        pdb = _write_pdb(tmp_path / "R.prolif_H.pdb", "AQAEAKAGAG")
        mapping, report = bw.receptor_bw_map("R", pdb)
        assert report["identity"] == pytest.approx(0.5)
        assert mapping == {}
        assert report["status"] == "anchor_mismatch(0/5)"

    def test_three_of_five_anchors_is_enough(self, tmp_path, fake_gpcrdb):
        pdb = _write_pdb(tmp_path / "R.prolif_H.pdb", "ANADARAGAG")
        mapping, report = bw.receptor_bw_map("R", pdb)
        assert report["anchors_ok"] == 3
        assert report["status"] == "ok"
        assert mapping

    def test_two_of_five_anchors_is_rejected(self, tmp_path, fake_gpcrdb):
        pdb = _write_pdb(tmp_path / "R.prolif_H.pdb", "ANADAKAGAG")
        mapping, report = bw.receptor_bw_map("R", pdb)
        assert report["anchors_ok"] == 2
        assert mapping == {}
        assert report["status"] == "anchor_mismatch(2/5)"

    def test_class_c_numbering_has_no_class_a_anchors(
        self, tmp_path, monkeypatch,
    ):
        # GPCRdb numbers class C receptors, but not at the class A anchors.
        monkeypatch.setattr(
            bw, "uniprot_position_to_bw",
            lambda acc: ({3: "3.40", 5: "5.47"}, CANONICAL),
        )
        pdb = _write_pdb(tmp_path / "P41180.prolif_H.pdb", CANONICAL)
        mapping, report = bw.receptor_bw_map("P41180", pdb)
        assert mapping == {}
        assert report["status"] == "anchor_mismatch(0/0)"

    def test_missing_structure_keeps_raw_names(self, tmp_path, fake_gpcrdb):
        maps, reports = bw.build_receptor_maps(["R"], tmp_path)
        assert maps == {"R": {}}
        assert reports["R"]["status"] == "no_prolif_structure"


# ---------------------------------------------------------------- GPCRdb


@pytest.mark.unit
class TestGpcrdbFailures:
    def test_404_means_no_record(self, monkeypatch):
        def raise_404(*a, **k):
            raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)
        monkeypatch.setattr(bw, "_get_json", raise_404)
        assert bw.fetch_protein("Q00000") is None

    def test_network_failure_is_raised_not_swallowed(self, monkeypatch):
        # Swallowing this would leave every receptor on raw names offline.
        def raise_offline(*a, **k):
            raise urllib.error.URLError("offline")
        monkeypatch.setattr(bw, "_get_json", raise_offline)
        with pytest.raises(urllib.error.URLError):
            bw.fetch_protein("P07550")


# -------------------------------------------------------------- pipeline


@pytest.mark.unit
class TestPipelineWiring:
    def test_apply_bw_remap_writes_artifacts(self, tmp_path, fake_gpcrdb):
        receptors = tmp_path / "receptors"
        receptors.mkdir()
        _write_pdb(receptors / "R1.prolif_H.pdb", CANONICAL, start=101)
        _write_pdb(receptors / "R2.prolif_H.pdb", CANONICAL, start=201)
        raw = _frame([
            {"pair_id": "a", "receptor_uniprot": "R1",
             "ifp_ALA105.A_VdWContact": 1, **META},
            {"pair_id": "b", "receptor_uniprot": "R2",
             "ifp_ALA205.A_VdWContact": 1, **META},
            {"pair_id": "c", "receptor_uniprot": "P41180",
             "ifp_PHE149.A_VdWContact": 1, **META},
        ])
        out = featurization.apply_bw_remap(raw, receptors, tmp_path)

        assert out["ifp_BW3.32_VdWContact"].tolist() == [1, 1, 0]
        assert out["ifp_PHE149.A_VdWContact"].tolist() == [0, 0, 1]
        cmap = json.loads((tmp_path / "bw_column_map.json").read_text())
        assert cmap == {"ifp_BW3.32_VdWContact": [
            "ifp_ALA105.A_VdWContact", "ifp_ALA205.A_VdWContact"]}
        maps = json.loads((tmp_path / "bw_maps.json").read_text())
        assert maps["R1"]["105"] == "3.32"
        assert maps["P41180"] == {}
        reports = json.loads((tmp_path / "bw_map_reports.json").read_text())
        assert reports["R1"]["status"] == "ok"
        assert reports["P41180"]["status"] == "no_prolif_structure"

    def test_build_interaction_fingerprints_remaps_by_default(
        self, tmp_path, monkeypatch, fake_gpcrdb,
    ):
        receptors = tmp_path / "receptors"
        receptors.mkdir()
        _write_pdb(receptors / "R1.prolif_H.pdb", CANONICAL, start=101)
        (receptors / "R1.pdb").write_text("END\n")
        pd.DataFrame({"inchikey": ["K1"], "canonical_smiles_std": ["C"]}).to_csv(
            tmp_path / "ulig.csv", index=False)
        pd.DataFrame({"pair_id": ["K1__R1"], "inchikey": ["K1"],
                      "receptor_uniprot": ["R1"], "success": [True]}).to_csv(
            tmp_path / "dock.csv", index=False)

        monkeypatch.setattr(featurization, "load_pose_mol", lambda p, s: object())

        def fake_ifp(pid, mol, pdb):
            from cancerag.features.interaction_fingerprint import IFPResult
            return IFPResult(pid, {"ifp_ALA105.A_VdWContact": 1}, 1, 1)
        monkeypatch.setattr(featurization, "compute_ifp_for_pair", fake_ifp)

        out_path = tmp_path / "out" / "interaction_fingerprints.parquet"
        out_path.parent.mkdir()
        featurization.build_interaction_fingerprints(
            unified_csv=tmp_path / "ulig.csv",
            docking_features_csv=tmp_path / "dock.csv",
            work_root=tmp_path / "work", receptors_dir=receptors,
            output_path=out_path,
        )
        shipped = pd.read_parquet(out_path)
        raw = pd.read_parquet(out_path.with_name(
            "interaction_fingerprints_raw.parquet"))
        assert "ifp_BW3.32_VdWContact" in shipped.columns
        assert "ifp_ALA105.A_VdWContact" in raw.columns
        assert (out_path.parent / "bw_column_map.json").exists()
