"""Tests for cancerag.docking.analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from cancerag.docking.analysis import (
    Pose,
    cluster_poses,
    parse_vina_pdbqt,
    pose_ensemble_features,
)


def _pose(affinity: float, coords: list[list[float]]) -> Pose:
    return Pose(affinity=affinity, coords=np.asarray(coords, dtype=float))


@pytest.mark.unit
class TestPoseEnsembleFeatures:
    def test_empty_returns_nan_block(self):
        feats = pose_ensemble_features([])
        for k in (
            "vina_affinity_best",
            "vina_affinity_mean_top3",
            "vina_affinity_gap_1_2",
            "vina_pose_diversity_rmsd",
        ):
            assert math.isnan(feats[k])
        assert feats["vina_n_distinct_clusters"] == 0
        assert feats["vina_n_poses_returned"] == 0

    def test_single_pose(self):
        p = _pose(-7.5, [[0, 0, 0], [1, 0, 0]])
        feats = pose_ensemble_features([p])
        assert feats["vina_affinity_best"] == pytest.approx(-7.5)
        assert feats["vina_affinity_mean_top3"] == pytest.approx(-7.5)
        assert feats["vina_affinity_gap_1_2"] == pytest.approx(0.0)
        assert feats["vina_pose_diversity_rmsd"] == pytest.approx(0.0)
        assert feats["vina_n_distinct_clusters"] == 1

    def test_top3_average_uses_at_most_three(self):
        poses = [_pose(a, [[0, 0, 0]]) for a in (-10, -9, -8, -7, -6)]
        feats = pose_ensemble_features(poses)
        assert feats["vina_affinity_mean_top3"] == pytest.approx(
            np.mean([-10, -9, -8])
        )

    def test_gap_between_rank_1_and_2(self):
        poses = [_pose(-9.0, [[0, 0, 0]]), _pose(-7.0, [[0, 0, 0]])]
        feats = pose_ensemble_features(poses)
        assert feats["vina_affinity_gap_1_2"] == pytest.approx(2.0)

    def test_diversity_uses_rmsd_to_top(self):
        poses = [
            _pose(-9.0, [[0, 0, 0], [1, 0, 0]]),
            _pose(-8.5, [[10, 0, 0], [11, 0, 0]]),
        ]
        feats = pose_ensemble_features(poses)
        assert feats["vina_pose_diversity_rmsd"] == pytest.approx(10.0)

    def test_cluster_count_collapses_neighbors(self):
        poses = [
            _pose(-9.0, [[0, 0, 0], [1, 0, 0]]),
            _pose(-8.9, [[0.1, 0, 0], [1.1, 0, 0]]),
            _pose(-8.0, [[10, 0, 0], [11, 0, 0]]),
        ]
        feats = pose_ensemble_features(poses)
        assert feats["vina_n_distinct_clusters"] == 2

    def test_cluster_poses_threshold_strict(self):
        a = _pose(-9.0, [[0.0, 0.0, 0.0]])
        b = _pose(-8.9, [[1.9, 0.0, 0.0]])
        c = _pose(-8.5, [[2.1, 0.0, 0.0]])
        assert cluster_poses([a, b, c]) == 2


@pytest.mark.unit
class TestParseVinaPdbqt:
    def test_two_models(self):
        pdbqt = """\
MODEL 1
REMARK VINA RESULT:    -9.4    0.000    0.000
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
MODEL 2
REMARK VINA RESULT:    -8.1    1.234    2.345
ATOM      1  C   LIG A   1       0.500   0.000   0.000  1.00  0.00     0.000 C
ATOM      2  C   LIG A   1       2.000   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
"""
        poses = parse_vina_pdbqt(pdbqt)
        assert len(poses) == 2
        assert poses[0].affinity == pytest.approx(-9.4)
        assert poses[1].affinity == pytest.approx(-8.1)
        assert poses[0].coords.shape == (2, 3)

    def test_skips_model_without_vina_remark(self):
        pdbqt = """\
MODEL 1
ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C
ENDMDL
"""
        assert parse_vina_pdbqt(pdbqt) == []


@pytest.mark.unit
class TestConfigDefaults:
    """Vina config has been restored to publication-grade settings (Stage 05).
    Lives here since the rest of the docking-stage tests do too."""

    def test_vina_publication_grade_defaults(self):
        cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
        d = cfg["docking"]
        assert d["exhaustiveness"] >= 16, "exhaustiveness must be >= 16"
        assert d["num_modes"] >= 9, "num_modes must be >= 9"
        assert "per_job_timeout_seconds" in d
        assert "vina_version" in d
