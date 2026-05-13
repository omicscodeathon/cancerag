"""
Docking-result parsing and per-result analysis.

Owns:
- Vina log-file parsing (legacy ``parse_docking_results``,
  ``compare_receptor_affinities``).
- ``Pose`` / pose-ensemble feature extraction (Stage 05) — used to expose
  energy-gap, pose diversity, and cluster count as features alongside the
  legacy single best-affinity score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _parse_log_file(log_file: str) -> list:
    """Extracts binding affinities from a Vina log file."""
    affinities = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip().startswith(
                    ("1", "2", "3", "4", "5", "6", "7", "8", "9")
                ):
                    parts = line.split()
                    if len(parts) >= 2:
                        affinities.append(float(parts[1]))
    except (FileNotFoundError, ValueError) as e:
        print(f"  - WARNING: Could not parse log file {log_file}: {e}")
    return affinities


def parse_docking_results(raw_results: list) -> list:
    """
    Parses the raw output from the docking runner.

    Args:
        raw_results (list): The list of result dictionaries from the runner.

    Returns:
        list: A sorted list of parsed results with key information.
    """
    parsed_results = []
    for result in raw_results:
        if not result["success"]:
            continue

        affinities = _parse_log_file(result["log_file"])
        if not affinities:
            continue

        parsed_results.append(
            {
                "ligand_name": result["ligand_name"],
                "best_affinity": min(affinities),
                "mean_affinity": np.mean(affinities),
                "all_affinities": affinities,
                "out_file": result["out_file"],
            }
        )

    # Sort by best binding affinity (most negative is best)
    parsed_results.sort(key=lambda x: x["best_affinity"])
    return parsed_results


def compare_receptor_affinities(all_parsed_results: dict) -> pd.DataFrame:
    """
    Compares binding affinities across all receptors to find biased ligands.

    Args:
        all_parsed_results (dict): Parsed results, keyed by receptor name.

    Returns:
        pd.DataFrame: A dataframe comparing affinities and calculating bias scores.
    """
    ligand_data = {}
    receptor_names = list(all_parsed_results.keys())

    for receptor, results in all_parsed_results.items():
        for result in results:
            ligand_name = result["ligand_name"]
            if ligand_name not in ligand_data:
                ligand_data[ligand_name] = {r: None for r in receptor_names}
            ligand_data[ligand_name][receptor] = result["best_affinity"]

    df = pd.DataFrame.from_dict(ligand_data, orient="index")
    df.index.name = "ligand_name"

    # Calculate bias scores (difference in affinity)
    if len(receptor_names) >= 2:
        for i, r1 in enumerate(receptor_names):
            for r2 in receptor_names[i + 1 :]:
                bias_col = f"bias_{r1}_vs_{r2}"
                df[bias_col] = df[r1] - df[r2]

    return df


# ----------------------------------------------------- pose-ensemble features


@dataclass(frozen=True)
class Pose:
    affinity: float  # kcal/mol (Vina mode score; lower = stronger)
    coords: np.ndarray  # shape (n_atoms, 3) — heavy-atom coordinates


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"RMSD shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return float(np.sqrt((diff * diff).sum() / a.shape[0]))


def cluster_poses(poses: list[Pose], rmsd_threshold: float = 2.0) -> int:
    """Greedy single-link clustering by RMSD; return number of clusters.

    Used as a "how many distinct binding modes did Vina find" feature.
    """
    if not poses:
        return 0
    clusters: list[Pose] = [poses[0]]
    for p in poses[1:]:
        if any(_rmsd(p.coords, c.coords) < rmsd_threshold for c in clusters):
            continue
        clusters.append(p)
    return len(clusters)


def pose_ensemble_features(poses: list[Pose]) -> dict:
    """Compute pose-ensemble descriptors from a sorted-by-affinity pose list.

    Convention: poses must be sorted ascending by ``affinity`` (Vina
    convention — best/most-negative first).
    """
    if not poses:
        return {
            "vina_affinity_best": float("nan"),
            "vina_affinity_mean_top3": float("nan"),
            "vina_affinity_gap_1_2": float("nan"),
            "vina_pose_diversity_rmsd": float("nan"),
            "vina_n_distinct_clusters": 0,
            "vina_n_poses_returned": 0,
        }
    affinities = np.array([p.affinity for p in poses])
    top3 = float(np.mean(affinities[: min(3, len(affinities))]))
    gap = float(affinities[1] - affinities[0]) if len(affinities) > 1 else 0.0
    if len(poses) > 1:
        rmsds = [_rmsd(p.coords, poses[0].coords) for p in poses[1:]]
        diversity = float(np.mean(rmsds))
    else:
        diversity = 0.0
    return {
        "vina_affinity_best": float(affinities[0]),
        "vina_affinity_mean_top3": top3,
        "vina_affinity_gap_1_2": gap,
        "vina_pose_diversity_rmsd": diversity,
        "vina_n_distinct_clusters": cluster_poses(poses, rmsd_threshold=2.0),
        "vina_n_poses_returned": len(poses),
    }


def parse_vina_pdbqt(pdbqt_text: str) -> list[Pose]:
    """Parse a Vina output ``.pdbqt`` payload into a sorted list of ``Pose``.

    Vina formats each model as::

        MODEL <i>
        REMARK VINA RESULT:    <affinity>   <rmsd_lb>  <rmsd_ub>
        ATOM ...
        ENDMDL
    """
    poses: list[Pose] = []
    current_affinity: float | None = None
    current_atoms: list[tuple[float, float, float]] = []
    for raw in pdbqt_text.splitlines():
        if raw.startswith("MODEL"):
            current_affinity = None
            current_atoms = []
        elif raw.startswith("REMARK VINA RESULT:"):
            try:
                current_affinity = float(raw.split()[3])
            except (ValueError, IndexError):
                current_affinity = None
        elif raw.startswith(("ATOM", "HETATM")):
            try:
                x = float(raw[30:38])
                y = float(raw[38:46])
                z = float(raw[46:54])
                current_atoms.append((x, y, z))
            except ValueError:
                continue
        elif raw.startswith("ENDMDL"):
            if current_affinity is not None and current_atoms:
                poses.append(
                    Pose(
                        affinity=current_affinity,
                        coords=np.asarray(current_atoms, dtype=float),
                    )
                )
            current_affinity = None
            current_atoms = []
    poses.sort(key=lambda p: p.affinity)
    return poses
