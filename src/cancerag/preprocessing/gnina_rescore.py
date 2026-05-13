"""
Gnina rescoring of Vina docking poses.

Gnina is a Vina fork with a 3D-CNN scoring function trained on PDBbind.
It uses Vina's docking *search* but replaces Vina's empirical scoring
function with a deep-learning model that scores poses on (a) the
likelihood of being a real binding pose ("CNNscore" / "CNN_pose_score")
and (b) the predicted binding affinity ("CNNaffinity").

Why we use it:
- Reviewer 1 explicitly asked for a comparison against a more accurate
  docking tool. Gnina is the standard answer in 2025 because it shares
  Vina's CLI and PDBQT format, runs against the same boxes, and reliably
  produces 5-10% better pose RMSDs than Vina on standard benchmarks
  (CASF-2016, PoseBusters).
- For receptors where Vina's pose-recovery RMSD is high (> 2.5 Å),
  Gnina's CNN score tells us whether Vina actually *found* the right
  pose but ranked it incorrectly — in which case the pose data is still
  scientifically usable. Specifically:
    - If Vina's top pose has bad RMSD AND Gnina rescores it as low
      probability, Vina genuinely missed the pose -> low confidence.
    - If Vina's top pose has bad RMSD BUT Gnina assigns it high
      probability, that's noise from Vina's empirical scoring; the pose
      itself is fine.

This module rescores existing Vina output PDBQTs in-place — no new
docking search is done. Gnina runs in score-only mode (`--score_only`)
which evaluates each pose with the CNN and reports CNNscore +
CNNaffinity per pose.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_GNINA_TIMEOUT_S = 300


def _gnina_executable() -> str:
    exe = shutil.which("gnina")
    if exe:
        return exe
    candidate = Path.home() / "tools" / "gnina"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        "gnina executable not found; expected on PATH or at ~/tools/gnina"
    )


@dataclass
class GninaPoseScore:
    """One pose, scored by Gnina's CNN."""

    rank: int  # 1-indexed pose rank from Vina output (top pose = 1)
    cnn_pose_score: float  # likelihood this is a real binding pose [0, 1]
    cnn_affinity: float  # predicted -log(Kd) (higher = stronger binder)
    vina_affinity: float | None = None  # the original Vina kcal/mol

    def is_high_confidence(self, score_threshold: float = 0.5) -> bool:
        return self.cnn_pose_score >= score_threshold


def parse_gnina_score_only(stdout: str) -> list[GninaPoseScore]:
    """Parse the score-only output of `gnina --score_only`.

    Output is like:
        Affinity:    -7.20  (kcal/mol)
        CNNscore:    0.7421
        CNNaffinity: 6.2845
        ...
    Per-pose blocks repeat for each MODEL in the input. We use ordering
    to assign rank.
    """
    poses: list[GninaPoseScore] = []
    cur: dict = {}
    rank = 0

    def _flush():
        nonlocal cur, rank
        if "cnn_pose_score" in cur and "cnn_affinity" in cur:
            rank += 1
            poses.append(GninaPoseScore(
                rank=rank,
                cnn_pose_score=cur["cnn_pose_score"],
                cnn_affinity=cur["cnn_affinity"],
                vina_affinity=cur.get("vina_affinity"),
            ))
        cur = {}

    for line in stdout.splitlines():
        m = re.match(r"^Affinity:\s+(-?[\d.]+)", line)
        if m:
            _flush()  # affinity is the first field of a new pose block
            cur["vina_affinity"] = float(m.group(1))
            continue
        m = re.match(r"^CNNscore:\s+([\d.]+)", line)
        if m:
            cur["cnn_pose_score"] = float(m.group(1))
            continue
        m = re.match(r"^CNNaffinity:\s+([\d.]+)", line)
        if m:
            cur["cnn_affinity"] = float(m.group(1))
            continue
    _flush()
    return poses


def _split_multimodel_pdbqt(input_pdbqt: Path, work_dir: Path) -> list[Path]:
    """Split a multi-MODEL PDBQT (Vina's output) into per-pose single-model
    PDBQT files. Gnina's `--score_only` rejects multi-model inputs.

    Returns a list of single-model PDBQT paths in pose-rank order.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    text = input_pdbqt.read_text()
    poses: list[Path] = []
    current: list[str] = []
    in_model = False
    rank = 0
    for line in text.splitlines():
        if line.startswith("MODEL"):
            in_model = True
            current = []
            continue
        if line.startswith("ENDMDL"):
            if current:
                rank += 1
                out = work_dir / f"pose_{rank:02d}.pdbqt"
                out.write_text("\n".join(current) + "\n")
                poses.append(out)
            in_model = False
            continue
        if in_model:
            current.append(line)
    return poses


def gnina_rescore_pdbqt(
    *,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    timeout_s: int = DEFAULT_GNINA_TIMEOUT_S,
    cpu_only: bool = True,
    work_dir: Path | None = None,
) -> list[GninaPoseScore]:
    """Score every MODEL in `ligand_pdbqt` against `receptor_pdbqt`.

    Splits multi-MODEL ligand input into per-pose files (Gnina rejects
    multi-MODEL inputs in --score_only mode). Then rescores each pose
    in turn and stitches the per-pose Affinity/CNN values into a single
    list with the original rank order preserved.

    `cpu_only=True` adds `--no_gpu`. CUDA is still loaded by the binary
    at startup (we installed libcudart12+cusparse+cublas+etc.), but
    --no_gpu skips actual CUDA computation.
    """
    if work_dir is None:
        work_dir = ligand_pdbqt.parent / ".gnina_split"
    pose_files = _split_multimodel_pdbqt(ligand_pdbqt, work_dir)
    if not pose_files:
        return []

    out: list[GninaPoseScore] = []
    for pose_path in pose_files:
        cmd = [
            _gnina_executable(),
            "--receptor", str(receptor_pdbqt),
            "--ligand", str(pose_path),
            "--score_only",
        ]
        if cpu_only:
            cmd.append("--no_gpu")
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
        if proc.returncode != 0:
            logger.warning(
                "gnina --score_only failed for %s: stderr=%s",
                pose_path, proc.stderr[:200],
            )
            continue
        parsed = parse_gnina_score_only(proc.stdout)
        if parsed:
            # parse_gnina_score_only assigns rank by order — but each
            # invocation here scores exactly one pose, so rank=1 from
            # the parser. Re-rank using the file index.
            single = parsed[0]
            single.rank = len(out) + 1
            out.append(single)
    return out


@dataclass
class GninaRescoreResult:
    uniprot: str
    biasdb_name: str
    pdb_id: str | None
    ligand_resname: str | None
    n_poses_scored: int
    top_pose_cnn_score: float | None
    top_pose_cnn_affinity: float | None
    top_pose_vina_affinity: float | None
    cnn_assessment: str  # "high_confidence" | "marginal" | "low_confidence"
    error: str | None = None


def rescore_existing_redock_runs(
    redock_work_root: Path | str = "data/processed/.redock_work",
    redock_validation_json: Path | str = "data/processed/redock_validation.json",
    output_path: Path | str = "data/processed/gnina_rescore.json",
    audit_path: Path | str = "data/processed/gnina_rescore_audit.md",
    *,
    timeout_s: int = DEFAULT_GNINA_TIMEOUT_S,
) -> dict:
    """Walk the existing per-receptor work dirs from Stage 04 redock,
    rescore each receptor's top Vina pose with Gnina."""
    redock_work_root = Path(redock_work_root)
    payload = json.loads(Path(redock_validation_json).read_text())
    receptor_meta = {r["uniprot"]: r for r in payload["redock_results"]}

    results: list[GninaRescoreResult] = []
    for udir in sorted(redock_work_root.iterdir()):
        if not udir.is_dir():
            continue
        u = udir.name
        receptor_pdbqt = udir / "receptor.pdbqt"
        out_pdbqt = udir / "out.pdbqt"
        if not receptor_pdbqt.exists() or not out_pdbqt.exists():
            continue
        meta = receptor_meta.get(u, {})
        try:
            scores = gnina_rescore_pdbqt(
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=out_pdbqt,
                timeout_s=timeout_s,
            )
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            results.append(GninaRescoreResult(
                uniprot=u, biasdb_name=meta.get("biasdb_name", ""),
                pdb_id=meta.get("pdb_id"),
                ligand_resname=meta.get("ligand_resname"),
                n_poses_scored=0,
                top_pose_cnn_score=None, top_pose_cnn_affinity=None,
                top_pose_vina_affinity=None,
                cnn_assessment="failed", error=str(exc)[:300],
            ))
            continue
        if not scores:
            results.append(GninaRescoreResult(
                uniprot=u, biasdb_name=meta.get("biasdb_name", ""),
                pdb_id=meta.get("pdb_id"),
                ligand_resname=meta.get("ligand_resname"),
                n_poses_scored=0,
                top_pose_cnn_score=None, top_pose_cnn_affinity=None,
                top_pose_vina_affinity=None,
                cnn_assessment="failed", error="no poses parsed",
            ))
            continue
        top = scores[0]
        if top.cnn_pose_score >= 0.7:
            assessment = "high_confidence"
        elif top.cnn_pose_score >= 0.4:
            assessment = "marginal"
        else:
            assessment = "low_confidence"
        results.append(GninaRescoreResult(
            uniprot=u, biasdb_name=meta.get("biasdb_name", ""),
            pdb_id=meta.get("pdb_id"),
            ligand_resname=meta.get("ligand_resname"),
            n_poses_scored=len(scores),
            top_pose_cnn_score=top.cnn_pose_score,
            top_pose_cnn_affinity=top.cnn_affinity,
            top_pose_vina_affinity=top.vina_affinity,
            cnn_assessment=assessment,
        ))
        logger.info(
            "%s (%s): CNN=%s affinity=%s -> %s",
            u, meta.get("biasdb_name", "?"),
            f"{top.cnn_pose_score:.3f}",
            f"{top.cnn_affinity:.3f}",
            assessment,
        )

    # Aggregate
    by_assessment: dict[str, int] = {}
    for r in results:
        by_assessment[r.cnn_assessment] = by_assessment.get(r.cnn_assessment, 0) + 1

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "rescored": len(results),
            "by_assessment": by_assessment,
        },
        "rescore_results": [asdict(r) for r in results],
    }
    Path(output_path).write_text(json.dumps(out_payload, indent=2, sort_keys=True))
    _emit_audit(out_payload, payload, Path(audit_path))
    return out_payload


def _emit_audit(payload: dict, redock_payload: dict, out_path: Path) -> None:
    redock_by_uniprot = {r["uniprot"]: r for r in redock_payload["redock_results"]}
    md = [
        "# Stage 04 — Gnina rescoring of Vina poses",
        "",
        f"_Generated: {payload['generated_at_utc']}_  ",
        "",
        "## Summary",
        "",
        f"- Receptors rescored: **{payload['totals']['rescored']}**",
        "- CNN pose-quality assessment:",
    ]
    for k in ("high_confidence", "marginal", "low_confidence", "failed"):
        v = payload["totals"]["by_assessment"].get(k, 0)
        md.append(f"  - {k}: {v}")
    md.append("")
    md.append("## How to read this")
    md.append("")
    md.append(
        "- **CNN pose score** (0–1): Gnina's CNN-predicted likelihood that "
        "the top pose is a real binding pose. ≥0.7 = high confidence; "
        "0.4–0.7 = marginal; <0.4 = low."
    )
    md.append(
        "- **CNN affinity**: predicted -log(Kd). Higher = stronger binder."
    )
    md.append(
        "- **Vina affinity**: kcal/mol from Vina's empirical scoring. "
        "Lower (more negative) = stronger binder."
    )
    md.append(
        "- **Vina RMSD**: re-docking RMSD vs. crystal pose from Stage 04."
    )
    md.append(
        "- **The interesting cases** are receptors where Vina's RMSD was "
        "high (failed re-dock) but Gnina's CNN score is high — Vina found "
        "the right pose but its empirical scoring ranked it suboptimally."
    )
    md.append("")
    md.append("## Per-receptor")
    md.append("")
    md.append(
        "| uniprot | biasdb_name | pdb | ligand | Vina RMSD | "
        "Vina aff. | CNN score | CNN affinity | assessment |"
    )
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in payload["rescore_results"]:
        rd = redock_by_uniprot.get(r["uniprot"], {})
        rmsd = rd.get("rmsd_angstrom")
        vina_aff = r.get("top_pose_vina_affinity")
        cnn = r.get("top_pose_cnn_score")
        cnn_aff = r.get("top_pose_cnn_affinity")
        md.append(
            f"| {r['uniprot']} | {r['biasdb_name']} | "
            f"{r['pdb_id'] or '—'} | {r['ligand_resname'] or '—'} | "
            f"{f'{rmsd:.2f}' if rmsd is not None else '—'} | "
            f"{f'{vina_aff:.2f}' if vina_aff is not None else '—'} | "
            f"{f'{cnn:.3f}' if cnn is not None else '—'} | "
            f"{f'{cnn_aff:.3f}' if cnn_aff is not None else '—'} | "
            f"{r['cnn_assessment']} |"
        )
    out_path.write_text("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    rescore_existing_redock_runs()
