"""
Stage 05 — production docking of every (ligand, receptor) pair.

For each unique (canonical_inchikey, receptor_uniprot) pair in the
curated dataset, this module:

1. Builds the ligand 3D conformation (RDKit ETKDG) from canonical SMILES.
2. Prepares ligand PDBQT with Meeko (the same prep we used in Stage 04
   re-dock validation, which gave β2 0.79 Å RMSD).
3. Reuses the receptor PDBQT from Stage 04 if available, otherwise
   prepares it via Meeko/obabel.
4. Reads the binding box from `data/processed/binding_sites.json`.
5. Runs AutoDock Vina at exhaustiveness=16, num_modes=9.
6. Extracts pose-ensemble features (top-pose affinity, mean of top-3,
   energy gap rank-1 vs rank-2, pose-diversity RMSD, distinct-cluster
   count) using the existing `docking.analysis` helpers.

Parallelism: a `ProcessPoolExecutor` runs N workers concurrently. Each
worker calls Vina with a reduced per-job thread count so the total
thread budget on the machine is respected (default: 4 workers × 2
threads = 8 threads, matching this laptop's CPU).

Outputs:
- `data/processed/docking_features.csv` — one row per docking job:
  pair_key columns + pose-ensemble features + docking_confidence flag
  inherited from the receptor's Stage 04 verdict.
- `data/processed/.docking_work/<inchikey14>__<uniprot>/` — per-job
  artifacts (input PDBQTs, Vina output PDBQT, dock.meta.json sidecar).
- `data/processed/docking_audit.md` — per-receptor summary.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cancerag.docking.analysis import (
    Pose,
    parse_vina_pdbqt,
    pose_ensemble_features,
)
from cancerag.preprocessing.redock_validation import (
    _prepare_ligand_meeko,
    _prepare_receptor_meeko,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 4
DEFAULT_THREADS_PER_JOB = 2
DEFAULT_EXHAUSTIVENESS = 16
DEFAULT_NUM_MODES = 9
DEFAULT_VINA_TIMEOUT_S = 1800


@dataclass
class DockingJob:
    pair_id: str  # short identifier: inchikey14_short + receptor_uniprot
    inchikey: str
    inchikey14: str
    receptor_uniprot: str
    canonical_smiles: str
    receptor_pdb: str  # path to data/processed/receptors/<UNIPROT>.pdb
    box_center: tuple[float, float, float]
    box_size: tuple[float, float, float]
    work_dir: str  # absolute path to per-job work directory


@dataclass
class DockingResult:
    pair_id: str
    inchikey: str
    receptor_uniprot: str
    success: bool
    n_poses: int = 0
    vina_affinity_best: float | None = None
    vina_affinity_mean_top3: float | None = None
    vina_affinity_gap_1_2: float | None = None
    vina_pose_diversity_rmsd: float | None = None
    vina_n_distinct_clusters: int = 0
    wall_seconds: float = 0.0
    error: str | None = None


def _smiles_to_3d_pdb(smiles: str, output_pdb: Path, *, seed: int = 42) -> bool:
    """SMILES -> 3D conformer -> PDB. Returns False if embedding fails.

    ETKDG seeded for reproducibility; MMFF optimization clean-up.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # optimization is nice-to-have, not required
    Chem.MolToPDBFile(mol, str(output_pdb))
    return True


def _run_vina_for_job(
    *,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    out_pdbqt: Path,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    exhaustiveness: int,
    num_modes: int,
    cpu_threads: int,
    timeout_s: int,
) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(out_pdbqt),
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--cpu", str(cpu_threads),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=False
    )
    if proc.returncode != 0 or not out_pdbqt.exists():
        raise RuntimeError(
            f"vina failed: rc={proc.returncode} "
            f"stderr={proc.stderr[:200]}"
        )


def run_one_dock(
    job_dict: dict,
    *,
    cpu_threads: int = DEFAULT_THREADS_PER_JOB,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    num_modes: int = DEFAULT_NUM_MODES,
    timeout_s: int = DEFAULT_VINA_TIMEOUT_S,
) -> dict:
    """Run one docking job — top-level function so it picklles for the
    process pool. Argument is a dict (not a DockingJob dataclass) to keep
    the multiprocessing boundary clean."""
    job = DockingJob(**job_dict)
    work_dir = Path(job.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    ligand_pdb = work_dir / "ligand.pdb"
    ligand_pdbqt = work_dir / "ligand.pdbqt"
    receptor_pdbqt = work_dir / "receptor.pdbqt"
    out_pdbqt = work_dir / "out.pdbqt"

    # Idempotent: skip if a valid output_pdbqt already exists.
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        try:
            poses = parse_vina_pdbqt(out_pdbqt.read_text())
            if poses:
                feats = pose_ensemble_features(poses)
                return asdict(DockingResult(
                    pair_id=job.pair_id, inchikey=job.inchikey,
                    receptor_uniprot=job.receptor_uniprot,
                    success=True, n_poses=len(poses),
                    vina_affinity_best=feats["vina_affinity_best"],
                    vina_affinity_mean_top3=feats["vina_affinity_mean_top3"],
                    vina_affinity_gap_1_2=feats["vina_affinity_gap_1_2"],
                    vina_pose_diversity_rmsd=feats["vina_pose_diversity_rmsd"],
                    vina_n_distinct_clusters=feats["vina_n_distinct_clusters"],
                    wall_seconds=0.0,
                    error="cached",
                ))
        except Exception:
            pass  # fall through and re-dock

    try:
        # Ligand: SMILES -> 3D -> Meeko PDBQT.
        if not _smiles_to_3d_pdb(job.canonical_smiles, ligand_pdb):
            raise RuntimeError("RDKit could not embed ligand SMILES")
        _prepare_ligand_meeko(ligand_pdb, ligand_pdbqt)

        # Receptor: reuse cached PDBQT if a Stage-04 redock already
        # produced one, otherwise prep fresh.
        cached_receptor_pdbqt = (
            Path("data/processed/.redock_work")
            / job.receptor_uniprot / "receptor.pdbqt"
        )
        if cached_receptor_pdbqt.exists():
            # symlink for traceability rather than copy (saves disk).
            if receptor_pdbqt.exists() or receptor_pdbqt.is_symlink():
                receptor_pdbqt.unlink()
            try:
                receptor_pdbqt.symlink_to(cached_receptor_pdbqt.resolve())
            except OSError:
                receptor_pdbqt.write_bytes(cached_receptor_pdbqt.read_bytes())
        else:
            _prepare_receptor_meeko(Path(job.receptor_pdb), receptor_pdbqt)

        _run_vina_for_job(
            receptor_pdbqt=receptor_pdbqt, ligand_pdbqt=ligand_pdbqt,
            out_pdbqt=out_pdbqt,
            center=tuple(job.box_center), size=tuple(job.box_size),
            exhaustiveness=exhaustiveness, num_modes=num_modes,
            cpu_threads=cpu_threads, timeout_s=timeout_s,
        )

        poses = parse_vina_pdbqt(out_pdbqt.read_text())
        if not poses:
            raise RuntimeError("vina produced no parseable poses")

        feats = pose_ensemble_features(poses)
        return asdict(DockingResult(
            pair_id=job.pair_id, inchikey=job.inchikey,
            receptor_uniprot=job.receptor_uniprot,
            success=True, n_poses=len(poses),
            vina_affinity_best=feats["vina_affinity_best"],
            vina_affinity_mean_top3=feats["vina_affinity_mean_top3"],
            vina_affinity_gap_1_2=feats["vina_affinity_gap_1_2"],
            vina_pose_diversity_rmsd=feats["vina_pose_diversity_rmsd"],
            vina_n_distinct_clusters=feats["vina_n_distinct_clusters"],
            wall_seconds=time.time() - start,
        ))
    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        return asdict(DockingResult(
            pair_id=job.pair_id, inchikey=job.inchikey,
            receptor_uniprot=job.receptor_uniprot,
            success=False, wall_seconds=time.time() - start,
            error=str(exc)[:300],
        ))


def build_job_list(
    unified_csv: Path | str = "data/processed/unified_ligands.csv",
    binding_sites_json: Path | str = "data/processed/binding_sites.json",
    receptors_dir: Path | str = "data/processed/receptors",
    work_root: Path | str = "data/processed/.docking_work",
) -> list[DockingJob]:
    """Build the unique-pair job list from the curated dataset."""
    df = pd.read_csv(unified_csv)
    sites = {
        s["uniprot"]: s
        for s in json.loads(Path(binding_sites_json).read_text())["binding_sites"]
    }
    work_root = Path(work_root)
    receptors_dir = Path(receptors_dir)

    pairs = (
        df[["inchikey", "inchikey14", "receptor_uniprot",
            "canonical_smiles_std"]]
        .drop_duplicates(subset=["inchikey", "receptor_uniprot"])
        .reset_index(drop=True)
    )

    jobs: list[DockingJob] = []
    for _, row in pairs.iterrows():
        u = str(row["receptor_uniprot"])
        site = sites.get(u)
        if site is None:
            logger.warning("no binding site for %s — skipping", u)
            continue
        receptor_pdb = receptors_dir / f"{u}.pdb"
        if not receptor_pdb.exists():
            logger.warning("no prepared receptor for %s — skipping", u)
            continue
        # Use the FULL 27-char InChIKey (not just the 14-char connectivity
        # layer) so that stereoisomers — same connectivity, different 3D —
        # get distinct work_dirs. Earlier versions used inchikey14, which
        # caused 20 stereoisomer pairs to overwrite each other.
        full_inchi = str(row["inchikey"])
        pair_id = f"{full_inchi}__{u}"
        jobs.append(DockingJob(
            pair_id=pair_id,
            inchikey=full_inchi,
            inchikey14=str(row["inchikey14"]),
            receptor_uniprot=u,
            canonical_smiles=str(row["canonical_smiles_std"]),
            receptor_pdb=str(receptor_pdb),
            box_center=(float(site["center_x"]), float(site["center_y"]),
                        float(site["center_z"])),
            box_size=(float(site["size_x"]), float(site["size_y"]),
                      float(site["size_z"])),
            work_dir=str(work_root / pair_id),
        ))
    return jobs


def run_distributed(
    jobs: list[DockingJob],
    *,
    workers: int = DEFAULT_WORKERS,
    cpu_threads_per_job: int = DEFAULT_THREADS_PER_JOB,
    exhaustiveness: int = DEFAULT_EXHAUSTIVENESS,
    num_modes: int = DEFAULT_NUM_MODES,
    timeout_s: int = DEFAULT_VINA_TIMEOUT_S,
    progress_log_every: int = 10,
) -> list[dict]:
    """Run docking jobs across `workers` parallel processes."""
    logger.info(
        "launching %d Vina jobs over %d workers × %d threads/job",
        len(jobs), workers, cpu_threads_per_job,
    )
    job_dicts = [asdict(j) for j in jobs]
    results: list[dict] = []
    started_at = time.time()

    # spawn context: avoid fork inheriting RDKit/CUDA state
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx
    ) as pool:
        futures = [
            pool.submit(
                run_one_dock, jd,
                cpu_threads=cpu_threads_per_job,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                timeout_s=timeout_s,
            )
            for jd in job_dicts
        ]
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                r = future.result()
            except Exception as exc:
                # A worker crashed before returning; record the failure.
                r = asdict(DockingResult(
                    pair_id="", inchikey="", receptor_uniprot="",
                    success=False, error=f"worker crash: {exc}"[:300],
                ))
            results.append(r)
            if i % progress_log_every == 0 or i == len(jobs):
                elapsed = time.time() - started_at
                rate = i / elapsed if elapsed else 0
                eta = (len(jobs) - i) / rate if rate else 0
                logger.info(
                    "progress %d/%d (%.1f%%)  rate=%.2f jobs/s  "
                    "elapsed=%.0f s  eta=%.0f s",
                    i, len(jobs), 100 * i / len(jobs), rate, elapsed, eta,
                )
    return results


def emit_features_csv(
    results: list[dict],
    binding_sites_json: Path | str = "data/processed/binding_sites.json",
    redock_validation_json: Path | str = "data/processed/redock_validation.json",
    gnina_rescore_json: Path | str = "data/processed/gnina_rescore.json",
    output_path: Path | str = "data/processed/docking_features.csv",
) -> pd.DataFrame:
    """Stitch per-job results with per-receptor docking-confidence flags
    derived from Stage 04 (re-dock RMSD + Gnina CNN). Emit one row per
    successful docking with all features the model needs."""
    df = pd.DataFrame(results)
    if df.empty:
        df.to_csv(output_path, index=False)
        return df

    # Per-receptor confidence: combine Stage-04 verdicts.
    binding = json.loads(Path(binding_sites_json).read_text())
    site_conf = {s["uniprot"]: s["confidence"] for s in binding["binding_sites"]}
    redock_rmsd = {}
    if Path(redock_validation_json).exists():
        redock_rmsd = {
            r["uniprot"]: r["rmsd_angstrom"]
            for r in json.loads(Path(redock_validation_json).read_text())["redock_results"]
        }
    gnina_cnn = {}
    if Path(gnina_rescore_json).exists():
        gnina_cnn = {
            r["uniprot"]: r["top_pose_cnn_score"]
            for r in json.loads(Path(gnina_rescore_json).read_text())["rescore_results"]
        }

    def _confidence(uniprot: str) -> str:
        rmsd = redock_rmsd.get(uniprot)
        cnn = gnina_cnn.get(uniprot)
        site = site_conf.get(uniprot, "ok")
        # AlphaFold/no-redock receptors: fall back to binding-site confidence.
        if rmsd is None and cnn is None:
            return "marginal" if site == "ok" else site
        rmsd_pass = rmsd is not None and rmsd <= 2.5
        cnn_pass = cnn is not None and cnn >= 0.7
        cnn_marg = cnn is not None and cnn >= 0.4
        if rmsd_pass and cnn_pass:
            return "high"
        if rmsd_pass or cnn_pass:
            return "marginal"
        if cnn_marg:
            return "marginal"
        return "low"

    df["docking_confidence"] = df["receptor_uniprot"].map(_confidence)
    df["redock_rmsd_angstrom"] = df["receptor_uniprot"].map(redock_rmsd)
    df["gnina_cnn_score"] = df["receptor_uniprot"].map(gnina_cnn)

    df.to_csv(output_path, index=False)
    return df


def emit_audit(
    df: pd.DataFrame,
    audit_path: Path | str = "data/processed/docking_audit.md",
) -> None:
    n_total = len(df)
    n_ok = int(df["success"].sum()) if "success" in df.columns else 0
    md = [
        "# Stage 05 — Production docking audit",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_  ",
        "",
        "## Summary",
        "",
        f"- Total docking jobs: {n_total}",
        f"- Successful: {n_ok} ({100 * n_ok / n_total:.1f}%)" if n_total else "- (no jobs)",
        f"- Failed: {n_total - n_ok}",
        "",
    ]
    if "docking_confidence" in df.columns:
        md.append("## By docking confidence")
        md.append("")
        for k, v in df["docking_confidence"].value_counts().items():
            md.append(f"- {k}: {v}")
        md.append("")
    if "receptor_uniprot" in df.columns and not df.empty:
        md.append("## Per-receptor")
        md.append("")
        md.append("| uniprot | n_jobs | n_success | mean_top_affinity (kcal/mol) |")
        md.append("| --- | --- | --- | --- |")
        for u, sub in df.groupby("receptor_uniprot"):
            n = len(sub)
            s = int(sub["success"].sum())
            mean_aff = sub["vina_affinity_best"].mean()
            md.append(
                f"| {u} | {n} | {s} | "
                f"{f'{mean_aff:.2f}' if pd.notna(mean_aff) else '—'} |"
            )
    Path(audit_path).write_text("\n".join(md))


def main(
    *,
    workers: int = DEFAULT_WORKERS,
    cpu_threads_per_job: int = DEFAULT_THREADS_PER_JOB,
    limit: int | None = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    jobs = build_job_list()
    if limit:
        jobs = jobs[:limit]
    logger.info("built %d docking jobs", len(jobs))
    results = run_distributed(
        jobs, workers=workers, cpu_threads_per_job=cpu_threads_per_job,
    )
    df = emit_features_csv(results)
    emit_audit(df)
    logger.info("STAGE_05_DONE")


if __name__ == "__main__":
    main()
