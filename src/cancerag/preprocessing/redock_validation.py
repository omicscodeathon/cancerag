"""
Re-docking validation for the binding-site definitions.

Stage 04 sub-step. For each receptor whose binding box was anchored on a
co-crystal ligand, take that ligand back out, dock it into the box with
AutoDock Vina, and check that Vina recovers the experimental pose
(top-pose RMSD < 2.5 Å — Trott & Olson 2010 marginal-pass threshold).

Receptors that pass: high-confidence binding box.
Receptors that fail: flagged in the per-receptor record so downstream
features derived from those dockings can be down-weighted.

This is the validation Reviewer 1 asked for in their original critique
("docking quality assessment").

Pipeline per receptor:
1. Extract the chosen co-crystal ligand atoms (heavy only) from the raw PDB.
2. Write ligand_crystal.pdb — the experimental pose, atom order preserved.
3. obabel: ligand_crystal.pdb -> ligand.pdbqt (Gasteiger charges).
4. obabel: receptor.pdb -> receptor.pdbqt (Gasteiger charges).
5. Run vina at exhaustiveness=16, num_modes=9 against the box.
6. Parse top pose from out.pdbqt; compute heavy-atom RMSD vs crystal pose.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from typing import Optional

from Bio.PDB import PDBParser

from cancerag.preprocessing.het_resnames import LIGAND_AUTO_DETECT_IGNORE

logger = logging.getLogger(__name__)


DEFAULT_RMSD_PASS_THRESHOLD = 2.5
DEFAULT_VINA_EXHAUSTIVENESS = 16
DEFAULT_VINA_NUM_MODES = 9
DEFAULT_VINA_TIMEOUT_S = 600


@dataclass
class RedockResult:
    uniprot: str
    biasdb_name: str
    pdb_id: str | None
    ligand_resname: str | None
    n_heavy_atoms: int | None
    rmsd_angstrom: float | None
    top_pose_affinity: float | None  # kcal/mol
    passes: bool
    confidence: str  # "ok" | "marginal" | "failed"
    error: str | None = None


def _extract_ligand_pdb(
    raw_pdb: Path, ligand_resname: str, output_pdb: Path
) -> int:
    """Write a PDB containing only the heavy atoms of the named ligand."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", str(raw_pdb))
    written = 0
    serial = 1
    with output_pdb.open("w") as f:
        for model in structure:
            for chain in model:
                for residue in chain:
                    hetflag = residue.id[0]
                    if not isinstance(hetflag, str) or not hetflag.startswith("H_"):
                        continue
                    resname = residue.get_resname().strip().upper()
                    if resname != ligand_resname.upper():
                        continue
                    for atom in residue.get_atoms():
                        elem = (atom.element or "").strip().upper()
                        if elem == "H":
                            continue
                        x, y, z = (float(c) for c in atom.get_coord())
                        f.write(
                            f"HETATM{serial:>5}  {atom.get_name():<4}{resname:>3} "
                            f"A{900:>4}    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                            f"{elem:>2}\n"
                        )
                        serial += 1
                        written += 1
                    if written:
                        break
                if written:
                    break
            if written:
                break
        f.write("END\n")
    return written


def _read_pdbqt_top_pose(pdbqt_path: Path) -> tuple[list[tuple[float, float, float]], float | None]:
    """Parse the top MODEL from a Vina output PDBQT.

    Returns (heavy_atom_coords, top_affinity_kcal_per_mol)."""
    coords: list[tuple[float, float, float]] = []
    affinity: float | None = None
    in_first_model = False
    found_first_model = False
    for line in pdbqt_path.read_text().splitlines():
        if line.startswith("MODEL"):
            if found_first_model:
                break
            found_first_model = True
            in_first_model = True
            continue
        if line.startswith("ENDMDL"):
            if in_first_model:
                break
        if not in_first_model:
            continue
        if line.startswith("REMARK VINA RESULT") and affinity is None:
            try:
                affinity = float(line.split()[3])
            except (ValueError, IndexError):
                pass
        if line.startswith(("ATOM", "HETATM")):
            try:
                # skip hydrogens
                element = (line[76:78].strip().upper()
                           if len(line) >= 78 else "")
                atom_name = line[12:16].strip()
                if element == "H" or atom_name.startswith("H"):
                    continue
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
            except (ValueError, IndexError):
                continue
    return coords, affinity


def _read_crystal_heavy_coords(crystal_pdb: Path) -> list[tuple[float, float, float]]:
    coords = []
    for line in crystal_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            try:
                element = line[76:78].strip().upper() if len(line) >= 78 else ""
                if element == "H":
                    continue
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except (ValueError, IndexError):
                continue
    return coords


def _heavy_atom_rmsd(
    crystal: list[tuple[float, float, float]],
    pose: list[tuple[float, float, float]],
) -> float | None:
    """Order-preserved heavy-atom RMSD. Returns None if atom counts differ.

    Used as a fallback when symmetry-aware RMSD via RDKit is unavailable.
    """
    if not crystal or not pose:
        return None
    if len(crystal) != len(pose):
        return None
    s = 0.0
    for (x1, y1, z1), (x2, y2, z2) in zip(crystal, pose):
        s += (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    return sqrt(s / len(crystal))


def _symmetry_aware_rmsd(
    crystal_pdb: Path, pose_pdbqt: Path
) -> float | None:
    """RDKit-based best-match RMSD between two ligand poses.

    Order-independent: tries every symmetry-equivalent atom mapping and
    returns the minimum. This is the right RMSD for ligand re-docking
    because Vina may emit atoms in a different order than the crystal,
    and chemically symmetric atoms (e.g. the two oxygens of a carboxylate,
    the three methyls of a tert-butyl) shouldn't be penalised for swapping.

    Returns None if RDKit can't read either file or atom counts disagree
    after parsing.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    # Read crystal pose (PDB)
    crystal = Chem.MolFromPDBFile(str(crystal_pdb), removeHs=True, sanitize=False)
    if crystal is None:
        return None

    # Vina's output is a PDBQT (multi-MODEL). Extract the first MODEL
    # and convert to PDB so RDKit can read it.
    pdbqt_text = pose_pdbqt.read_text()
    first_model_lines: list[str] = []
    in_model = False
    for line in pdbqt_text.splitlines():
        if line.startswith("MODEL"):
            in_model = True
            continue
        if line.startswith("ENDMDL"):
            break
        if in_model and line.startswith(("ATOM", "HETATM")):
            # Strip the AD4 charge/atom-type columns (cols 70-end);
            # standard PDB ATOM ends at col 78. PDBQT puts charge at
            # 70-76 and AD4 atom type at 77-78. Just truncate to 80
            # cols and replace cols 77-78 with the element symbol.
            atom_name = line[12:16].strip()
            element = "".join(c for c in atom_name if not c.isdigit())[:2]
            first_model_lines.append(line[:76] + f"{element:>2}")
    if not first_model_lines:
        return None
    pose_pdb = pose_pdbqt.with_suffix(".first_model.pdb")
    pose_pdb.write_text("\n".join(first_model_lines) + "\nEND\n")
    pose = Chem.MolFromPDBFile(str(pose_pdb), removeHs=True, sanitize=False)
    if pose is None:
        return None

    # GetBestRMS handles symmetry-equivalent atoms automatically.
    try:
        rmsd = AllChem.GetBestRMS(crystal, pose)
        return float(rmsd)
    except Exception as exc:
        logger.warning("RDKit GetBestRMS failed: %s; falling back to "
                       "order-preserved RMSD", exc)
        return None


def _run_obabel_to_pdbqt(
    input_pdb: Path, output_pdbqt: Path, *, mode: str
) -> None:
    """Open Babel fallback for PDB→PDBQT conversion.

    Used only when Meeko is unavailable. obabel produces functional
    PDBQT but with notable quality limitations vs Meeko (generic atom
    typing, no hydrogen optimisation, no protonation at pH).

    For some chimeric receptors with unusual aromatic systems (e.g.
    BRIL fusions, engineered nanobody complexes) Gasteiger charge
    perception aborts mid-molecule and obabel writes a 0-byte file with
    returncode=0 — so we retry without partial charges if the first
    attempt produces an empty output. Vina can run on uncharged input.
    """
    base_cmd = ["obabel", str(input_pdb), "-O", str(output_pdbqt)]
    if mode == "receptor":
        base_cmd.append("-xr")

    def _exec(extra: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            base_cmd + extra, capture_output=True, text=True, timeout=120
        )

    proc = _exec(["--partialcharge", "gasteiger"])
    if (
        proc.returncode == 0
        and output_pdbqt.exists()
        and output_pdbqt.stat().st_size > 0
    ):
        return

    # Empty output (Gasteiger failed silently) — retry without charges.
    logger.warning(
        "obabel with gasteiger produced empty output for %s; "
        "retrying without partial charges",
        input_pdb,
    )
    proc = _exec([])
    if (
        proc.returncode != 0
        or not output_pdbqt.exists()
        or output_pdbqt.stat().st_size == 0
    ):
        raise RuntimeError(
            f"obabel failed for {input_pdb} -> {output_pdbqt}: "
            f"stderr={proc.stderr[:300]}"
        )


def _strip_problem_hetatms(input_pdb: Path, output_pdb: Path) -> None:
    """Strip HETATM lines (waters, ions, ligands) and write a protein-only
    PDB. Meeko's strict residue templates can't handle Na⁺/Mg²⁺ ions that
    we kept around in Stage 03 for biological correctness. For docking
    prep we drop them — the loss to Vina's scoring is negligible since
    Vina doesn't model ions explicitly anyway."""
    keep_lines = []
    for line in input_pdb.read_text().splitlines():
        if line.startswith(("HETATM", "ANISOU", "CONECT")):
            continue
        keep_lines.append(line)
    output_pdb.write_text("\n".join(keep_lines) + "\n")


def _pdbfixer_clean(input_pdb: Path, output_pdb: Path, *, ph: float = 7.4) -> bool:
    """Pre-clean a receptor PDB with PDBFixer before Meeko sees it.

    PDBFixer fixes the issues that cause Meeko's template matcher to
    reject our Stage-03-prepared receptors:
    - Adds missing heavy atoms in partially-resolved residues (the
      "ARG with 6 missing heavy atoms" failure on β2-adrenoceptor).
    - Removes unusual heterogens (waters, ions, crystallisation buffers).
    - Replaces non-standard residues with their standard equivalents
      (MSE → MET, etc.).
    - Adds missing hydrogens at the specified pH.
    - Optionally caps chain breaks.

    Returns True on success (output written), False on failure (caller
    falls back to the strip-HETATM path).
    """
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError:
        logger.warning("PDBFixer not installed; skipping fixer step")
        return False
    try:
        fixer = PDBFixer(filename=str(input_pdb))
        # Drop everything that isn't a standard amino acid; this is what
        # docking expects (Vina ignores ions/waters anyway).
        fixer.findMissingResidues()
        # Don't model in missing terminal residues (those are rarely the
        # docking pocket and modelling them in unconstrained can produce
        # garbage geometry). Keep only internal gaps.
        chains = list(fixer.topology.chains())
        keys = list(fixer.missingResidues.keys())
        for k in keys:
            chain = chains[k[0]]
            chain_residues = list(chain.residues())
            if k[1] == 0 or k[1] == len(chain_residues):
                del fixer.missingResidues[k]
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(ph)
        with output_pdb.open("w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        return True
    except Exception as exc:
        logger.warning("PDBFixer failed for %s: %s", input_pdb, exc)
        return False


def _prepare_receptor_meeko(
    input_pdb: Path, output_pdbqt: Path, *, timeout_s: int = 180
) -> None:
    """Meeko receptor prep with multiple-fallback chain.

    Strategy:
    1. Try mk_prepare_receptor.py on the input as-is.
    2. If that fails (Na⁺ template error, partial-ARG, etc.), strip
       all HETATMs and retry.
    3. If that still fails, fall back to obabel.
    """
    output_basename = output_pdbqt.with_suffix("")

    def _try_meeko(pdb: Path) -> bool:
        cmd = [
            "mk_prepare_receptor.py",
            "--read_pdb", str(pdb),
            "-o", str(output_basename),
            "-p",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s
        )
        written = output_basename.with_suffix(".pdbqt")
        if proc.returncode != 0 or not written.exists():
            return False
        if written != output_pdbqt:
            written.rename(output_pdbqt)
        return True

    # Pass 1: as-is.
    if _try_meeko(input_pdb):
        return

    # Pass 2: PDBFixer clean (adds missing atoms, replaces nonstandard
    # residues, adds Hs at pH 7.4). This is the most thorough cleanup
    # and addresses the dominant failure mode (partial residues) of
    # the existing Stage 03 prep on most receptors.
    fixed = output_pdbqt.parent / f"{input_pdb.stem}_pdbfixer.pdb"
    if _pdbfixer_clean(input_pdb, fixed):
        if _try_meeko(fixed):
            return
        logger.info("Meeko still failed after PDBFixer; trying HETATM strip")

    # Pass 3: strip HETATMs (drops Na⁺, glycans, etc. that confuse Meeko's
    # template library when PDBFixer didn't already)
    stripped = output_pdbqt.parent / f"{input_pdb.stem}_stripped.pdb"
    _strip_problem_hetatms(input_pdb, stripped)
    if _try_meeko(stripped):
        return

    # Pass 4: obabel fallback (last resort).
    logger.warning(
        "mk_prepare_receptor failed for %s after PDBFixer + HETATM strip; "
        "falling back to obabel",
        input_pdb,
    )
    _run_obabel_to_pdbqt(input_pdb, output_pdbqt, mode="receptor")

    # Post-condition: every fallback can fail silently (Meeko writes a
    # 0-byte file on errors; obabel returncode=0 doesn't always imply
    # output). Verify the file is non-empty AND has at least one atom
    # line — otherwise downstream Vina runs will dock into vacuum.
    if not output_pdbqt.exists() or output_pdbqt.stat().st_size == 0:
        raise RuntimeError(
            f"_prepare_receptor_meeko: produced empty PDBQT for {input_pdb}"
        )
    n_atoms = sum(
        1 for ln in output_pdbqt.read_text().splitlines()
        if ln.startswith(("ATOM", "HETATM"))
    )
    if n_atoms == 0:
        raise RuntimeError(
            f"_prepare_receptor_meeko: PDBQT for {input_pdb} has 0 atom lines"
        )


def _prepare_ligand_meeko(
    input_pdb: Path, output_pdbqt: Path
) -> None:
    """Use Meeko's Python API to convert ligand PDB → PDBQT.

    Pipeline:
    1. RDKit reads the PDB (heavy atoms only, no Hs because we extracted
       just the experimental positions).
    2. RDKit adds hydrogens with `addCoords=True` so the H positions are
       inferred from geometry, not random.
    3. Meeko's `MoleculePreparation` assigns AD4 atom types, identifies
       rotatable bonds, computes Gasteiger charges.
    4. `PDBQTWriterLegacy` emits the standard PDBQT format.

    Falls back to obabel if Meeko fails (e.g. exotic chemistry).
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
    except ImportError:
        _run_obabel_to_pdbqt(input_pdb, output_pdbqt, mode="ligand")
        return

    mol = Chem.MolFromPDBFile(str(input_pdb), removeHs=False)
    if mol is None:
        logger.warning(
            "RDKit could not parse ligand %s; falling back to obabel", input_pdb
        )
        _run_obabel_to_pdbqt(input_pdb, output_pdbqt, mode="ligand")
        return
    mol = Chem.AddHs(mol, addCoords=True)
    try:
        prep = MoleculePreparation()
        prep.prepare(mol)
        pdbqt_string, is_valid, err = PDBQTWriterLegacy.write_string(prep.setup)
        if not is_valid or not pdbqt_string:
            raise RuntimeError(f"meeko writer error: {err}")
        output_pdbqt.write_text(pdbqt_string)
    except Exception as exc:
        logger.warning(
            "Meeko ligand prep failed for %s (%s); falling back to obabel",
            input_pdb, exc,
        )
        _run_obabel_to_pdbqt(input_pdb, output_pdbqt, mode="ligand")


def _run_vina(
    *, receptor_pdbqt: Path, ligand_pdbqt: Path, out_pdbqt: Path,
    center: tuple[float, float, float], size: tuple[float, float, float],
    exhaustiveness: int, num_modes: int, timeout_s: int,
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
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s, check=False
    )
    if proc.returncode != 0 or not out_pdbqt.exists():
        raise RuntimeError(
            f"vina failed: returncode={proc.returncode}, "
            f"stderr={proc.stderr[:300]}"
        )


def redock_one_receptor(
    *,
    uniprot: str,
    biasdb_name: str,
    receptor_pdb: Path,
    raw_pdb: Path,
    ligand_resname: str,
    box_center: tuple[float, float, float],
    box_size: tuple[float, float, float],
    work_dir: Path,
    pdb_id: str | None = None,
    rmsd_pass_threshold: float = DEFAULT_RMSD_PASS_THRESHOLD,
    exhaustiveness: int = DEFAULT_VINA_EXHAUSTIVENESS,
    num_modes: int = DEFAULT_VINA_NUM_MODES,
    timeout_s: int = DEFAULT_VINA_TIMEOUT_S,
) -> RedockResult:
    """Run the full re-dock validation for one receptor."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    crystal_lig_pdb = work_dir / "ligand_crystal.pdb"
    ligand_pdbqt = work_dir / "ligand.pdbqt"
    receptor_pdbqt = work_dir / "receptor.pdbqt"
    out_pdbqt = work_dir / "out.pdbqt"

    try:
        n_atoms = _extract_ligand_pdb(raw_pdb, ligand_resname, crystal_lig_pdb)
        if n_atoms < 4:
            return RedockResult(
                uniprot=uniprot, biasdb_name=biasdb_name, pdb_id=pdb_id,
                ligand_resname=ligand_resname, n_heavy_atoms=n_atoms,
                rmsd_angstrom=None, top_pose_affinity=None,
                passes=False, confidence="failed",
                error=f"too few ligand heavy atoms ({n_atoms})",
            )

        # Prefer Meeko (the AutoDock-Vina-Forli-lab official prep tool)
        # over obabel — meeko produces materially better PDBQT quality
        # for downstream Vina docking. Both helpers fall back to obabel
        # if their primary path fails for an exotic input.
        _prepare_ligand_meeko(crystal_lig_pdb, ligand_pdbqt)
        _prepare_receptor_meeko(receptor_pdb, receptor_pdbqt)
        _run_vina(
            receptor_pdbqt=receptor_pdbqt, ligand_pdbqt=ligand_pdbqt,
            out_pdbqt=out_pdbqt, center=box_center, size=box_size,
            exhaustiveness=exhaustiveness, num_modes=num_modes,
            timeout_s=timeout_s,
        )

        crystal_coords = _read_crystal_heavy_coords(crystal_lig_pdb)
        pose_coords, affinity = _read_pdbqt_top_pose(out_pdbqt)

        # Prefer symmetry-aware RMSD via RDKit's GetBestRMS — handles
        # chemically equivalent atom swaps (e.g. carboxylate oxygens) and
        # is robust to atom-order differences between crystal PDB and
        # Vina output PDBQT.
        rmsd = _symmetry_aware_rmsd(crystal_lig_pdb, out_pdbqt)
        if rmsd is None:
            # Fallback if RDKit couldn't parse one of the files.
            rmsd = _heavy_atom_rmsd(crystal_coords, pose_coords)

        if rmsd is None:
            confidence = "failed"
            passes = False
            error = (
                f"could not compute RMSD: crystal has {len(crystal_coords)} "
                f"heavy atoms, pose has {len(pose_coords)}"
            )
        else:
            passes = rmsd <= rmsd_pass_threshold
            if rmsd <= 2.0:
                confidence = "ok"
            elif rmsd <= rmsd_pass_threshold:
                confidence = "marginal"
            else:
                confidence = "failed"
            error = None

        return RedockResult(
            uniprot=uniprot, biasdb_name=biasdb_name, pdb_id=pdb_id,
            ligand_resname=ligand_resname, n_heavy_atoms=n_atoms,
            rmsd_angstrom=rmsd, top_pose_affinity=affinity,
            passes=passes, confidence=confidence, error=error,
        )

    except (subprocess.TimeoutExpired, RuntimeError) as exc:
        logger.warning("redock failed for %s (%s): %s", uniprot, pdb_id, exc)
        return RedockResult(
            uniprot=uniprot, biasdb_name=biasdb_name, pdb_id=pdb_id,
            ligand_resname=ligand_resname, n_heavy_atoms=None,
            rmsd_angstrom=None, top_pose_affinity=None,
            passes=False, confidence="failed", error=str(exc)[:300],
        )


def redock_all(
    binding_sites_json: Path | str = "data/processed/binding_sites.json",
    receptors_dir: Path | str = "data/processed/receptors",
    raw_pdb_dir: Path | str = "data/pdb",
    output_path: Path | str = "data/processed/redock_validation.json",
    audit_path: Path | str = "data/processed/redock_validation_audit.md",
    work_root: Path | str = "data/processed/.redock_work",
    *,
    rmsd_pass_threshold: float = DEFAULT_RMSD_PASS_THRESHOLD,
    exhaustiveness: int = DEFAULT_VINA_EXHAUSTIVENESS,
    num_modes: int = DEFAULT_VINA_NUM_MODES,
    timeout_s: int = DEFAULT_VINA_TIMEOUT_S,
    limit: int | None = None,
) -> dict:
    binding_sites_json = Path(binding_sites_json)
    receptors_dir = Path(receptors_dir)
    raw_pdb_dir = Path(raw_pdb_dir)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    payload = json.loads(binding_sites_json.read_text())
    sites = [s for s in payload["binding_sites"]
             if s["method"] == "cocrystal_ligand"
             and s.get("cocrystal_ligand_resname")]
    if limit:
        sites = sites[:limit]

    results: list[RedockResult] = []
    for s in sites:
        u = s["uniprot"]
        prepared = receptors_dir / f"{u}.pdb"
        raw = raw_pdb_dir / u / f"{s['pdb_id']}.pdb"
        if not prepared.exists() or not raw.exists():
            results.append(RedockResult(
                uniprot=u, biasdb_name=s["biasdb_name"], pdb_id=s["pdb_id"],
                ligand_resname=s["cocrystal_ligand_resname"],
                n_heavy_atoms=None, rmsd_angstrom=None, top_pose_affinity=None,
                passes=False, confidence="failed",
                error="missing receptor or raw pdb",
            ))
            continue
        r = redock_one_receptor(
            uniprot=u, biasdb_name=s["biasdb_name"],
            receptor_pdb=prepared, raw_pdb=raw,
            ligand_resname=s["cocrystal_ligand_resname"],
            box_center=(s["center_x"], s["center_y"], s["center_z"]),
            box_size=(s["size_x"], s["size_y"], s["size_z"]),
            work_dir=work_root / u, pdb_id=s["pdb_id"],
            rmsd_pass_threshold=rmsd_pass_threshold,
            exhaustiveness=exhaustiveness, num_modes=num_modes,
            timeout_s=timeout_s,
        )
        results.append(r)
        logger.info(
            "%s (%s, %s): RMSD=%s Å -> %s",
            u, s["biasdb_name"], r.ligand_resname,
            f"{r.rmsd_angstrom:.2f}" if r.rmsd_angstrom is not None else "—",
            r.confidence,
        )

    passes = sum(1 for r in results if r.passes)
    by_conf: dict[str, int] = {}
    for r in results:
        by_conf[r.confidence] = by_conf.get(r.confidence, 0) + 1

    out_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "rmsd_pass_threshold_angstrom": rmsd_pass_threshold,
            "vina_exhaustiveness": exhaustiveness,
            "vina_num_modes": num_modes,
            "vina_timeout_s": timeout_s,
        },
        "totals": {
            "attempted": len(results),
            "passed": passes,
            "by_confidence": by_conf,
        },
        "redock_results": [asdict(r) for r in results],
    }
    Path(output_path).write_text(json.dumps(out_payload, indent=2, sort_keys=True))
    _emit_audit(out_payload, Path(audit_path))
    return out_payload


def _emit_audit(payload: dict, out_path: Path) -> None:
    md = [
        "# Stage 04 — Re-docking validation",
        "",
        f"_Generated: {payload['generated_at_utc']}_  ",
        f"_Vina exhaustiveness: {payload['config']['vina_exhaustiveness']}, "
        f"num_modes: {payload['config']['vina_num_modes']}_  ",
        f"_RMSD pass threshold: {payload['config']['rmsd_pass_threshold_angstrom']} Å_  ",
        "",
        "## Summary",
        "",
        f"- Receptors attempted: **{payload['totals']['attempted']}**",
        f"- Receptors that passed (RMSD ≤ "
        f"{payload['config']['rmsd_pass_threshold_angstrom']} Å): "
        f"**{payload['totals']['passed']}**",
        "- Confidence breakdown:",
    ]
    for k in ("ok", "marginal", "failed"):
        v = payload["totals"]["by_confidence"].get(k, 0)
        md.append(f"  - {k}: {v}")
    md.append("")
    md.append("## Per-receptor")
    md.append("")
    md.append(
        "| uniprot | biasdb_name | pdb | ligand | n atoms | "
        "RMSD (Å) | affinity (kcal/mol) | confidence | error |"
    )
    md.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in payload["redock_results"]:
        rmsd = (
            f"{r['rmsd_angstrom']:.2f}"
            if r["rmsd_angstrom"] is not None else "—"
        )
        aff = (
            f"{r['top_pose_affinity']:.2f}"
            if r["top_pose_affinity"] is not None else "—"
        )
        md.append(
            f"| {r['uniprot']} | {r['biasdb_name']} | {r['pdb_id']} | "
            f"{r['ligand_resname']} | {r['n_heavy_atoms'] or '—'} | "
            f"{rmsd} | {aff} | {r['confidence']} | "
            f"{(r['error'] or '')[:60]} |"
        )
    out_path.write_text("\n".join(md))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    redock_all()
