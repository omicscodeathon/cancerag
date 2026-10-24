"""
Stage 06 — Featurization orchestrator.

Reads the curated dataset + Stage 05 docking outputs, computes:
  1. Per-ligand 2D RDKit descriptors (200+) — refresh on the new dataset.
  2. Per-ligand Morgan (2048-bit) and MACCS (167-bit) fingerprints.
  3. Per-pair RDKit ``Descriptors3D`` from the Vina-positioned top pose.
  4. Per-pair ProLIF interaction fingerprints (default 9 interaction types).

Emits:
  - ``data/processed/ligand_features.parquet`` — 1 row / unique inchikey,
    columns = canonical_smiles + 200 2D descriptors + 2048 morgan_* + 167 maccs_*.
  - ``data/processed/pose_3d_features.csv`` — 1 row / docked pair, 10 3D cols.
  - ``data/processed/interaction_fingerprints.parquet`` — 1 row / docked pair,
    columns = sparse-set IFP bits, named by Ballesteros-Weinstein generic
    position (``ifp_BW3.32_HBDonor``); receptors outside class A numbering
    keep ProLIF's per-structure names (``ifp_ASP86.A_VdWContact``).
  - ``data/processed/interaction_fingerprints_raw.parquet`` — the same block
    before the remap, ProLIF names throughout.
  - ``data/processed/bw_column_map.json``, ``bw_maps.json``,
    ``bw_map_reports.json`` — the remap: which raw columns merged into each
    BW column, the per-receptor residue maps, and each receptor's anchor and
    identity validation.
  - ``data/processed/featurization.meta.json`` — provenance: rdkit/prolif
    versions, descriptor list, fingerprint definitions, interaction types,
    UTC timestamps, source SHAs.

Each output file also gets a ``<file>.meta.json`` sidecar with the input SHA
of the source files (unified_ligands.csv, docking_features.csv) so a
downstream stage can detect staleness.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from cancerag.features.bw_numbering import (
    BW_ANCHORS,
    IFP_META_COLS,
    build_receptor_maps,
    remap_interaction_fingerprints,
    write_bw_artifacts,
)
from cancerag.features.interaction_fingerprint import (
    compute_ifp_for_pair,
    prolif_default_interaction_names,
)
from cancerag.features.molecular_descriptors import (
    descriptors_3d_from_mol,
    maccs_dataframe,
    morgan_dataframe,
)
from cancerag.features.pose_loader import load_pose_mol

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- ligand features


def _compute_2d_descriptors(smiles_list: list[str]) -> pd.DataFrame:
    """Compute the full RDKit 2D descriptor block (Descriptors._descList)."""
    desc_list = list(Descriptors._descList)  # [(name, fn), ...]
    names = [name for name, _ in desc_list]
    rows: list[list[float]] = []
    for smi in tqdm(smiles_list, desc="2D descriptors"):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            rows.append([float("nan")] * len(names))
            continue
        vals: list[float] = []
        for _, fn in desc_list:
            try:
                vals.append(float(fn(mol)))
            except Exception:
                vals.append(float("nan"))
        rows.append(vals)
    return pd.DataFrame(rows, columns=names)


def build_ligand_features(
    unified_csv: Path | str = "data/processed/unified_ligands.csv",
    docking_features_csv: Path | str = "data/processed/docking_features.csv",
    output_path: Path | str = "data/processed/ligand_features.parquet",
    *,
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    restrict_to_docked: bool = True,
) -> pd.DataFrame:
    """One row per unique InChIKey.

    By default, ligands that never docked successfully (peptides — median 26
    rotatable bonds, MW ~900) are excluded. CancerAg's modeling scope is
    non-peptide small molecules; peptides are documented as a future-work
    extension. Set ``restrict_to_docked=False`` to keep all ligands.
    """
    df = pd.read_csv(unified_csv)
    ligands = (
        df[["inchikey", "canonical_smiles_std"]]
        .drop_duplicates(subset=["inchikey"])
        .reset_index(drop=True)
    )
    if restrict_to_docked:
        dock = pd.read_csv(docking_features_csv)
        docked_inchis = set(dock[dock["success"]]["inchikey"].unique())
        before = len(ligands)
        ligands = ligands[ligands["inchikey"].isin(docked_inchis)].reset_index(drop=True)
        logger.info(
            "Excluded %d peptide ligands (never docked successfully); kept %d",
            before - len(ligands), len(ligands),
        )
    logger.info("Computing features for %d unique ligands", len(ligands))

    smi = ligands["canonical_smiles_std"].tolist()
    desc_2d = _compute_2d_descriptors(smi)
    morgan = morgan_dataframe(smi, radius=morgan_radius, n_bits=morgan_n_bits)
    maccs = maccs_dataframe(smi)

    out = pd.concat(
        [ligands.reset_index(drop=True),
         desc_2d.reset_index(drop=True),
         morgan.reset_index(drop=True),
         maccs.reset_index(drop=True)],
        axis=1,
    )
    output_path = Path(output_path)
    out.to_parquet(output_path, index=False)
    logger.info("ligand_features written: %d rows × %d cols -> %s",
                len(out), len(out.columns), output_path)
    return out


# ------------------------------------------------------------- pose 3D features


def build_pose_3d_features(
    unified_csv: Path | str = "data/processed/unified_ligands.csv",
    docking_features_csv: Path | str = "data/processed/docking_features.csv",
    work_root: Path | str = "data/processed/.docking_work",
    output_path: Path | str = "data/processed/pose_3d_features.csv",
) -> pd.DataFrame:
    """One row per (inchikey, receptor_uniprot) docked pair — 10 3D columns."""
    work_root = Path(work_root)
    ulig = pd.read_csv(unified_csv)
    smi_lookup = (
        ulig[["inchikey", "canonical_smiles_std"]]
        .drop_duplicates("inchikey")
        .set_index("inchikey")["canonical_smiles_std"]
        .to_dict()
    )
    dock = pd.read_csv(docking_features_csv)
    docked = dock[dock["success"]].reset_index(drop=True)
    logger.info("Computing 3D pose descriptors for %d docked pairs", len(docked))

    rows: list[dict] = []
    for _, row in tqdm(docked.iterrows(), total=len(docked),
                       desc="pose 3D"):
        pid = row["pair_id"]
        smi = smi_lookup.get(row["inchikey"])
        if smi is None:
            rows.append({"pair_id": pid,
                         "inchikey": row["inchikey"],
                         "receptor_uniprot": row["receptor_uniprot"],
                         **descriptors_3d_from_mol(None)})
            continue
        pdbqt = work_root / pid / "out.pdbqt"
        mol = load_pose_mol(pdbqt, smi)
        rows.append({
            "pair_id": pid,
            "inchikey": row["inchikey"],
            "receptor_uniprot": row["receptor_uniprot"],
            **descriptors_3d_from_mol(mol),
        })
    out = pd.DataFrame(rows)
    output_path = Path(output_path)
    out.to_csv(output_path, index=False)
    logger.info("pose_3d_features written: %d rows -> %s", len(out), output_path)
    return out


# ------------------------------------------------------ interaction fingerprints


BW_MIN_IDENTITY = 0.50
BW_MIN_ANCHORS = 3


def apply_bw_remap(
    raw: pd.DataFrame,
    receptors_dir: Path | str = "data/processed/receptors",
    artifacts_dir: Path | str | None = "data/processed",
    *,
    min_identity: float = BW_MIN_IDENTITY,
    min_anchors: int = BW_MIN_ANCHORS,
) -> pd.DataFrame:
    """Remap a raw ProLIF block to Ballesteros-Weinstein columns.

    Each receptor's map is built from its ``.prolif_H.pdb`` and kept only if
    at least ``min_anchors`` of the five class A anchors (N1.50, D2.50, R3.50,
    P6.50, P7.50) land on the expected residue and aligned-pair identity is at
    least ``min_identity``. Receptors that fail keep their raw names; the
    class C calcium-sensing receptor fails by design. Bits merging onto one
    BW column are OR-ed. When ``artifacts_dir`` is set, writes
    ``bw_column_map.json``, ``bw_maps.json`` and ``bw_map_reports.json`` there.
    """
    maps, reports = build_receptor_maps(
        raw["receptor_uniprot"].astype(str).unique(), receptors_dir,
        min_identity=min_identity, min_anchors=min_anchors,
    )
    out, column_map = remap_interaction_fingerprints(raw, maps)
    n_raw = sum(c not in IFP_META_COLS for c in raw.columns)
    n_out = sum(c not in IFP_META_COLS for c in out.columns)
    logger.info(
        "BW remap: %d/%d receptors mapped; %d raw -> %d contact columns "
        "(%d BW, %d multi-source)",
        sum(1 for m in maps.values() if m), len(maps), n_raw, n_out,
        len(column_map), sum(1 for v in column_map.values() if len(v) > 1),
    )
    if artifacts_dir is not None:
        write_bw_artifacts(artifacts_dir, column_map=column_map, maps=maps,
                           reports=reports)
    return out


def build_interaction_fingerprints(
    unified_csv: Path | str = "data/processed/unified_ligands.csv",
    docking_features_csv: Path | str = "data/processed/docking_features.csv",
    work_root: Path | str = "data/processed/.docking_work",
    receptors_dir: Path | str = "data/processed/receptors",
    output_path: Path | str = "data/processed/interaction_fingerprints.parquet",
    raw_output_path: Path | str | None = None,
    *,
    bw_remap: bool = True,
    bw_artifacts_dir: Path | str | None = None,
) -> pd.DataFrame:
    """One row per docked pair, sparse IFP bits stored as wide 0/1 columns.

    ProLIF names each bit after the residue number in the receptor file, which
    differs between constructs; with ``bw_remap`` (the default, and how the
    shipped block was built) the bits are renamed to Ballesteros-Weinstein
    positions via :func:`apply_bw_remap`. The raw block is always written to
    ``raw_output_path`` (default ``<output stem>_raw.parquet``), and the remap
    artifacts to ``bw_artifacts_dir`` (default: the output's directory).
    """
    work_root = Path(work_root)
    receptors_dir = Path(receptors_dir)
    output_path = Path(output_path)
    raw_output_path = (
        Path(raw_output_path) if raw_output_path is not None
        else output_path.with_name(f"{output_path.stem}_raw.parquet")
    )
    ulig = pd.read_csv(unified_csv)
    smi_lookup = (
        ulig[["inchikey", "canonical_smiles_std"]]
        .drop_duplicates("inchikey")
        .set_index("inchikey")["canonical_smiles_std"]
        .to_dict()
    )
    dock = pd.read_csv(docking_features_csv)
    docked = dock[dock["success"]].reset_index(drop=True)
    logger.info("Computing IFPs for %d docked pairs", len(docked))

    rows: list[dict] = []
    for _, row in tqdm(docked.iterrows(), total=len(docked),
                       desc="IFP"):
        pid = row["pair_id"]
        uni = row["receptor_uniprot"]
        smi = smi_lookup.get(row["inchikey"])
        if smi is None:
            rows.append({"pair_id": pid, "inchikey": row["inchikey"],
                         "receptor_uniprot": uni,
                         "n_residues_contacted": 0, "n_total_contacts": 0})
            continue
        pdbqt = work_root / pid / "out.pdbqt"
        mol = load_pose_mol(pdbqt, smi)
        ifp = compute_ifp_for_pair(pid, mol, receptors_dir / f"{uni}.pdb")
        rec = {"pair_id": pid, "inchikey": row["inchikey"],
               "receptor_uniprot": uni,
               "n_residues_contacted": ifp.n_residues_contacted,
               "n_total_contacts": ifp.n_total_contacts}
        rec.update(ifp.ifp_bits)
        rows.append(rec)

    out = pd.DataFrame(rows).fillna(0)
    # Keep the meta cols first, then sort the IFP bit columns alphabetically.
    meta_cols = list(IFP_META_COLS)
    bit_cols = sorted(c for c in out.columns if c not in meta_cols)
    # Cast bit columns to uint8 to keep parquet compact.
    for c in bit_cols:
        out[c] = out[c].astype(np.uint8)
    out = out[meta_cols + bit_cols]
    out.to_parquet(raw_output_path, index=False)
    logger.info("raw interaction_fingerprints written: %d rows × %d cols -> %s",
                len(out), len(out.columns), raw_output_path)

    if bw_remap:
        out = apply_bw_remap(
            out, receptors_dir,
            bw_artifacts_dir if bw_artifacts_dir is not None
            else output_path.parent,
        )
    out.to_parquet(output_path, index=False)
    logger.info("interaction_fingerprints written: %d rows × %d cols -> %s",
                len(out), len(out.columns), output_path)
    return out


# ------------------------------------------------------------------- provenance


def emit_meta(
    *,
    unified_csv: Path | str = "data/processed/unified_ligands.csv",
    docking_features_csv: Path | str = "data/processed/docking_features.csv",
    output_path: Path | str = "data/processed/featurization.meta.json",
) -> None:
    import rdkit
    import prolif as plf
    desc_names = [name for name, _ in Descriptors._descList]
    meta = {
        "schema_version": 1,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rdkit_version": rdkit.__version__,
        "prolif_version": plf.__version__,
        "descriptors_2d": {
            "count": len(desc_names),
            "names": desc_names,
        },
        "fingerprints": {
            "morgan": {"radius": 2, "n_bits": 2048},
            "maccs": {"n_bits": 167},
        },
        "descriptors_3d": [
            "Asphericity", "Eccentricity", "InertialShapeFactor",
            "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
            "RadiusOfGyration", "SpherocityIndex",
        ],
        "interaction_fingerprint": {
            "tool": "ProLIF",
            "interactions": prolif_default_interaction_names(),
            "ligand_source": "vina_top_pose_pdbqt",
            "receptor_source": "stage03_prepared_pdb_with_pdbfixer_h",
            "bond_perception": "AssignBondOrdersFromTemplate(canonical_smiles)",
            "column_naming": {
                "scheme": "ballesteros_weinstein",
                "pattern": "ifp_BW<generic position>_<interaction>",
                "merge": "logical_or",
                "authority": "GPCRdb residue tables",
                "structure": "<receptor>.prolif_H.pdb",
                "gate_min_anchors": BW_MIN_ANCHORS,
                "gate_anchors": [f"{pos} {aa}" for pos, aa in BW_ANCHORS.items()],
                "gate_min_identity": BW_MIN_IDENTITY,
                "unmapped": "keep ProLIF per-structure name",
            },
        },
        "input_sha256": {
            str(unified_csv): _sha256(Path(unified_csv)),
            str(docking_features_csv): _sha256(Path(docking_features_csv)),
        },
    }
    Path(output_path).write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("meta written -> %s", output_path)


# ------------------------------------------------------------------- entrypoint


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    build_ligand_features()
    build_pose_3d_features()
    build_interaction_fingerprints()
    emit_meta()
    logger.info("STAGE_06_DONE")


if __name__ == "__main__":
    main()
