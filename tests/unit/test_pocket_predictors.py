"""Tests for cancerag.preprocessing.pocket_predictors.

The fpocket / P2Rank wrappers themselves shell out to external binaries;
those subprocess calls are not exercised here. We test the parsers
(consume known fpocket / P2Rank output formats) and the consensus logic
(pure function over Pocket dataclasses)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cancerag.preprocessing.pocket_predictors import (
    Pocket,
    _fpocket_pocket_center,
    _parse_fpocket_info,
    _parse_p2rank_predictions,
    consensus_pocket,
)


def _write_fpocket_info(path: Path) -> None:
    """Reproduce the format fpocket emits in <name>_info.txt."""
    path.write_text(
        "Pocket 1 :\n"
        "\tScore : \t0.152\n"
        "\tDruggability Score : \t0.941\n"
        "\tNumber of Alpha Spheres : \t154\n"
        "\tVolume : \t1150.729\n"
        "\n"
        "Pocket 2 :\n"
        "\tScore : \t0.139\n"
        "\tDruggability Score : \t0.902\n"
        "\tVolume : \t505.351\n"
    )


def _write_fpocket_vert(path: Path, coords: list[tuple[float, float, float]]) -> None:
    """Voronoi-vertex PQR — one ATOM record per vertex."""
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        lines.append(
            f"ATOM  {i:>5}  C   X   X{i:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    path.write_text("\n".join(lines) + "\n")


def _write_p2rank_csv(path: Path) -> None:
    """Reproduce the format P2Rank emits in <name>_predictions.csv."""
    path.write_text(
        "name     ,  rank,   score, probability, sas_points, surf_atoms,   "
        "center_x,   center_y,   center_z, residue_ids, surf_atom_ids\n"
        "pocket1  ,     1,   36.25,       0.955,        124,         65,     "
        "9.3287,     5.8816,   -10.3730, A_100 A_110 A_114, 411 413 436\n"
        "pocket2  ,     2,    3.32,       0.117,         31,         31,    "
        "16.9421,     5.8133,     5.1099, A_122 A_125, 648 650 651\n"
    )


@pytest.mark.unit
class TestFpocketParsers:
    def test_info_parser(self, tmp_path: Path):
        info = tmp_path / "info.txt"
        _write_fpocket_info(info)
        recs = _parse_fpocket_info(info)
        assert len(recs) == 2
        assert recs[0]["rank"] == 1
        assert recs[0]["druggability_score"] == 0.941
        assert recs[1]["rank"] == 2
        assert recs[1]["druggability_score"] == 0.902

    def test_vert_centroid(self, tmp_path: Path):
        v = tmp_path / "v.pqr"
        _write_fpocket_vert(v, [(0, 0, 0), (4, 0, 0), (0, 4, 0), (0, 0, 4)])
        cx, cy, cz = _fpocket_pocket_center(v)
        assert (cx, cy, cz) == pytest.approx((1.0, 1.0, 1.0))

    def test_vert_centroid_empty(self, tmp_path: Path):
        v = tmp_path / "v.pqr"
        v.write_text("HEADER no atoms here\n")
        assert _fpocket_pocket_center(v) is None


@pytest.mark.unit
class TestP2RankParser:
    def test_csv_parser(self, tmp_path: Path):
        csv = tmp_path / "x.csv"
        _write_p2rank_csv(csv)
        ps = _parse_p2rank_predictions(csv)
        assert len(ps) == 2
        assert ps[0].rank == 1
        assert ps[0].score == pytest.approx(36.25)
        assert ps[0].center_x == pytest.approx(9.3287)
        assert ps[0].method == "p2rank"
        assert ps[0].residue_ids == ["A_100", "A_110", "A_114"]
        assert ps[0].raw["probability"] == pytest.approx(0.955)


def _pkt(rank, x, y, z, score=1.0, method="p2rank"):
    return Pocket(
        rank=rank, score=score,
        center_x=x, center_y=y, center_z=z, method=method,
    )


@pytest.mark.unit
class TestConsensus:
    def test_agreement_within_threshold(self):
        c = consensus_pocket(
            [_pkt(1, 10, 10, 10), _pkt(2, 30, 30, 30)],
            [_pkt(1, 11, 10, 10, method="fpocket")],
            agreement_threshold_angstroms=5.0,
        )
        assert c.agrees is True
        assert c.method == "consensus"
        assert c.distance_angstroms == pytest.approx(1.0)

    def test_disagreement_marks_low_confidence(self):
        c = consensus_pocket(
            [_pkt(1, 10, 10, 10)],
            [_pkt(1, 30, 30, 30, method="fpocket")],
            agreement_threshold_angstroms=5.0,
        )
        assert c.agrees is False
        assert c.method == "primary_only"
        assert c.distance_angstroms > 5.0

    def test_only_fpocket(self):
        c = consensus_pocket(
            [],
            [_pkt(1, 10, 10, 10, method="fpocket")],
        )
        assert c.primary.method == "fpocket"
        assert c.method == "primary_only"
        assert c.secondary is None

    def test_only_p2rank(self):
        c = consensus_pocket(
            [_pkt(1, 10, 10, 10)],
            [],
        )
        assert c.primary.method == "p2rank"
        assert c.method == "primary_only"

    def test_both_empty_raises(self):
        with pytest.raises(ValueError):
            consensus_pocket([], [])

    def test_picks_nearest_fpocket(self):
        # P2Rank says (10,10,10); fpocket has two pockets — one nearby
        # at (12,10,10), one far at (40,40,40). Consensus should pair
        # P2Rank with the nearby fpocket pocket.
        c = consensus_pocket(
            [_pkt(1, 10, 10, 10)],
            [_pkt(1, 40, 40, 40, method="fpocket"),
             _pkt(2, 12, 10, 10, method="fpocket")],
            agreement_threshold_angstroms=5.0,
        )
        assert c.secondary.center_x == pytest.approx(12)
        assert c.agrees is True
