"""
Pocket-prediction wrappers for fpocket and P2Rank.

Stage 04 dependency. Both tools take a PDB file and return a ranked list
of predicted pockets with center coordinates. We use them in
combination: P2Rank as the primary predictor (deep-learning based,
better accuracy on GPCRs in published benchmarks), fpocket as an
independent cross-check (geometric alpha-sphere method, completely
different algorithm). A pocket is accepted with high confidence only
when both methods point to the same region of the receptor (centers
within `agreement_threshold` Å of each other).
"""

from __future__ import annotations

import csv
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Pocket:
    """A predicted binding pocket."""

    rank: int
    score: float
    center_x: float
    center_y: float
    center_z: float
    method: str  # "fpocket" or "p2rank"
    residue_ids: list[str] | None = None  # e.g. ["A_100", "A_110", ...]
    raw: dict | None = None  # method-specific extras

    def center(self) -> tuple[float, float, float]:
        return (self.center_x, self.center_y, self.center_z)

    def distance_to(self, other: "Pocket") -> float:
        a, b = self.center(), other.center()
        return sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


# ----------------------------------------------------------------- fpocket


def _fpocket_executable() -> str:
    exe = shutil.which("fpocket") or "/snap/bin/fpocket"
    return exe


def _parse_fpocket_info(info_path: Path) -> list[dict]:
    """Parse `<name>_info.txt` into a list of per-pocket score dicts."""
    pockets: list[dict] = []
    current: dict | None = None
    with info_path.open() as f:
        for line in f:
            m = re.match(r"^Pocket\s+(\d+)\s*:", line)
            if m:
                if current:
                    pockets.append(current)
                current = {"rank": int(m.group(1))}
                continue
            m = re.match(r"^\s*([^:]+?)\s*:\s*(\S+)", line)
            if m and current is not None:
                key = m.group(1).strip().lower().replace(" ", "_").replace(".", "")
                try:
                    current[key] = float(m.group(2))
                except ValueError:
                    current[key] = m.group(2)
    if current:
        pockets.append(current)
    return pockets


def _fpocket_pocket_center(vert_pqr: Path) -> tuple[float, float, float] | None:
    """Average the x/y/z of all ATOM records in the voronoi-vertex PQR."""
    xs, ys, zs = [], [], []
    with vert_pqr.open() as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                xs.append(float(line[30:38]))
                ys.append(float(line[38:46]))
                zs.append(float(line[46:54]))
            except (ValueError, IndexError):
                continue
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs))


def run_fpocket(
    pdb_path: Path | str,
    *,
    work_dir: Path | str | None = None,
    timeout_seconds: int = 120,
    keep_top_k: int = 5,
) -> list[Pocket]:
    """Run fpocket on a PDB file; return the top-k predicted pockets.

    fpocket writes its output as a sibling directory next to the input
    PDB. We copy the input to `work_dir` first so we never mutate the
    canonical receptor cache.
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    if work_dir is None:
        work_dir = pdb_path.parent / f".fpocket_{pdb_path.stem}"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    local_pdb = work_dir / pdb_path.name
    if not local_pdb.exists() or local_pdb.read_bytes() != pdb_path.read_bytes():
        local_pdb.write_bytes(pdb_path.read_bytes())

    cmd = [_fpocket_executable(), "-f", str(local_pdb)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False
    )
    if proc.returncode != 0:
        logger.warning("fpocket non-zero exit on %s: %s", pdb_path, proc.stderr[:200])

    out_dir = work_dir / f"{pdb_path.stem}_out"
    info_file = out_dir / f"{pdb_path.stem}_info.txt"
    pockets_dir = out_dir / "pockets"
    if not info_file.exists():
        logger.warning("fpocket produced no info file for %s", pdb_path)
        return []

    raw_records = _parse_fpocket_info(info_file)
    out: list[Pocket] = []
    for rec in raw_records[:keep_top_k]:
        rank = rec["rank"]
        vert = pockets_dir / f"pocket{rank}_vert.pqr"
        if not vert.exists():
            continue
        center = _fpocket_pocket_center(vert)
        if center is None:
            continue
        out.append(Pocket(
            rank=rank,
            score=float(rec.get("druggability_score", rec.get("score", 0.0)) or 0.0),
            center_x=center[0], center_y=center[1], center_z=center[2],
            method="fpocket",
            raw=rec,
        ))
    return out


# ------------------------------------------------------------------ P2Rank


def _p2rank_executable() -> str:
    exe = shutil.which("prank")
    if exe:
        return exe
    candidate = Path.home() / "tools" / "p2rank_2.5" / "prank"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError("prank executable not found on PATH or in ~/tools/p2rank_2.5/")


def _parse_p2rank_predictions(csv_path: Path) -> list[Pocket]:
    out: list[Pocket] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            row = [c.strip() for c in row]
            try:
                pocket = Pocket(
                    rank=int(row[idx["rank"]]),
                    score=float(row[idx["score"]]),
                    center_x=float(row[idx["center_x"]]),
                    center_y=float(row[idx["center_y"]]),
                    center_z=float(row[idx["center_z"]]),
                    method="p2rank",
                    residue_ids=row[idx["residue_ids"]].split() if "residue_ids" in idx else None,
                    raw={
                        "name": row[idx["name"]],
                        "probability": float(row[idx["probability"]]),
                    },
                )
            except (KeyError, ValueError, IndexError) as e:
                logger.warning("skipping malformed P2Rank row in %s: %s", csv_path, e)
                continue
            out.append(pocket)
    return out


def run_p2rank(
    pdb_path: Path | str,
    *,
    work_dir: Path | str | None = None,
    timeout_seconds: int = 120,
    keep_top_k: int = 5,
) -> list[Pocket]:
    """Run P2Rank on a PDB file; return the top-k predicted pockets."""
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)

    if work_dir is None:
        work_dir = pdb_path.parent / f".p2rank_{pdb_path.stem}"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    local_pdb = work_dir / pdb_path.name
    if not local_pdb.exists() or local_pdb.read_bytes() != pdb_path.read_bytes():
        local_pdb.write_bytes(pdb_path.read_bytes())
    out_subdir = work_dir / "out"

    cmd = [_p2rank_executable(), "predict", "-f", str(local_pdb), "-o", str(out_subdir)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_seconds, check=False
    )
    if proc.returncode != 0:
        logger.warning("p2rank non-zero exit on %s: %s", pdb_path, proc.stderr[:200])

    csv_path = out_subdir / f"{pdb_path.name}_predictions.csv"
    if not csv_path.exists():
        logger.warning("p2rank produced no predictions CSV for %s", pdb_path)
        return []
    pockets = _parse_p2rank_predictions(csv_path)
    return pockets[:keep_top_k]


# ----------------------------------------------------- consensus / agreement


@dataclass
class PocketConsensus:
    """Outcome of cross-checking fpocket and P2Rank for one receptor."""

    primary: Pocket  # the chosen pocket (P2Rank top by default)
    secondary: Pocket | None  # the matching fpocket pocket, if any
    distance_angstroms: float | None  # distance between primary and secondary centers
    agrees: bool  # True if distance <= agreement_threshold
    method: str  # "consensus" | "primary_only" | "no_pockets"


def consensus_pocket(
    p2rank_pockets: list[Pocket],
    fpocket_pockets: list[Pocket],
    *,
    agreement_threshold_angstroms: float = 5.0,
) -> PocketConsensus:
    """Pick the consensus pocket between P2Rank and fpocket.

    Strategy:
    - Primary = P2Rank's top-ranked pocket.
    - Secondary = the *closest* fpocket pocket to that primary.
    - If their centers are within `agreement_threshold_angstroms`, mark
      ``agrees=True`` and method="consensus".
    - Otherwise mark ``agrees=False`` and method="primary_only" — the
      caller can decide whether to use it or flag the receptor as
      low-confidence.
    """
    if not p2rank_pockets and not fpocket_pockets:
        raise ValueError("no pockets predicted by either method")
    if not p2rank_pockets:
        return PocketConsensus(
            primary=fpocket_pockets[0],
            secondary=None, distance_angstroms=None, agrees=False,
            method="primary_only",
        )
    primary = p2rank_pockets[0]
    if not fpocket_pockets:
        return PocketConsensus(
            primary=primary, secondary=None, distance_angstroms=None,
            agrees=False, method="primary_only",
        )
    nearest = min(fpocket_pockets, key=lambda p: primary.distance_to(p))
    dist = primary.distance_to(nearest)
    return PocketConsensus(
        primary=primary,
        secondary=nearest,
        distance_angstroms=dist,
        agrees=dist <= agreement_threshold_angstroms,
        method="consensus" if dist <= agreement_threshold_angstroms else "primary_only",
    )
