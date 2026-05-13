"""Tests for cancerag.preprocessing.gnina_rescore.

Pure-function tests for the score-only output parser. The Gnina binary
itself is exercised by the real-data run, not here."""

from __future__ import annotations

import pytest

from cancerag.preprocessing.gnina_rescore import parse_gnina_score_only


@pytest.mark.unit
class TestParseGninaScoreOnly:
    def test_single_pose(self):
        out = (
            "Affinity:    -7.20  (kcal/mol)\n"
            "CNNscore:    0.7421\n"
            "CNNaffinity: 6.2845\n"
        )
        poses = parse_gnina_score_only(out)
        assert len(poses) == 1
        assert poses[0].rank == 1
        assert poses[0].vina_affinity == pytest.approx(-7.20)
        assert poses[0].cnn_pose_score == pytest.approx(0.7421)
        assert poses[0].cnn_affinity == pytest.approx(6.2845)

    def test_multiple_poses_blocks(self):
        out = (
            "Affinity:    -7.20  (kcal/mol)\n"
            "CNNscore:    0.7421\n"
            "CNNaffinity: 6.2845\n"
            "Affinity:    -6.50  (kcal/mol)\n"
            "CNNscore:    0.5511\n"
            "CNNaffinity: 5.8200\n"
            "Affinity:    -5.80  (kcal/mol)\n"
            "CNNscore:    0.3010\n"
            "CNNaffinity: 5.1100\n"
        )
        poses = parse_gnina_score_only(out)
        assert len(poses) == 3
        assert [p.rank for p in poses] == [1, 2, 3]
        assert poses[0].cnn_pose_score == pytest.approx(0.7421)
        assert poses[2].cnn_pose_score == pytest.approx(0.3010)
        assert poses[2].vina_affinity == pytest.approx(-5.80)

    def test_skips_incomplete_block(self):
        # Block with only Affinity but no CNN scores -> not flushed
        out = (
            "Affinity:    -7.20  (kcal/mol)\n"
            "Affinity:    -6.50  (kcal/mol)\n"
            "CNNscore:    0.5511\n"
            "CNNaffinity: 5.8200\n"
        )
        poses = parse_gnina_score_only(out)
        assert len(poses) == 1
        assert poses[0].vina_affinity == pytest.approx(-6.50)

    def test_empty_output(self):
        assert parse_gnina_score_only("") == []
        assert parse_gnina_score_only("nothing here") == []

    def test_high_confidence_threshold(self):
        from cancerag.preprocessing.gnina_rescore import GninaPoseScore

        p = GninaPoseScore(rank=1, cnn_pose_score=0.55, cnn_affinity=6.0)
        assert p.is_high_confidence(score_threshold=0.5) is True
        assert p.is_high_confidence(score_threshold=0.7) is False
