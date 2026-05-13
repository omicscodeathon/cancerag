"""
Stage 07 — Dataset Assembly.

Materializes the wide ML-ready matrix from the curated upstream artifacts
(Stages 02 / 05 / 06) and defines the train/test splits the model trainer
will consume.

Inputs:
  - data/processed/unified_ligands.csv          : bias labels + ligand metadata
  - data/processed/docking_features.csv         : Vina pose-ensemble + confidence
  - data/processed/ligand_features.parquet      : 2D descriptors + Morgan + MACCS
  - data/processed/pose_3d_features.csv         : 3D descriptors from docked pose
  - data/processed/interaction_fingerprints.parquet : ProLIF interaction bits

Outputs:
  - data/processed/ml_ready_dataset.parquet     : final wide matrix
  - data/processed/ml_splits.json               : per-split row indices
  - data/processed/ml_ready_dataset.meta.json   : provenance + summary
  - data/processed/dataset_assembly_audit.md    : human-readable audit
  - data/holdout/dataset_holdout.parquet        : temporal holdout (year ≥ 2018)

Design choices:
  - **Scope:** small-molecule ligands only (ligands that successfully docked
    at least once). Peptide ligands documented as future-work in the manuscript.
  - **Never impute** docking-derived columns. Each numeric column with NaN
    gets a paired ``<col>_missing`` indicator; XGBoost handles NaN natively.
  - **Sample weight** combines docking confidence (high=1.0 / marginal=0.6 /
    low=0.3) with evidence weight (year recency × source).
  - **Splits** computed at assembly time and persisted as JSON of row indices.
  - **Stable composite key** `inchikey :: receptor_uniprot :: bias_category ::
    bias_pathway :: assay_1 :: assay_2` — every row uniquely addressable.
  - **Schema validation** via Pandera surfaces upstream drift immediately.

The legacy ``DatasetAssembler`` 600-line class (with its ``fillna(-5.0)`` bug
and silent ``unknown`` label injection) was removed in this rewrite. The
helpers below (``make_pair_key``, ``resolve_label_conflicts``, etc.) are
still exported for reuse and are still covered by their unit tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bemis_murcko(smiles: str) -> str:
    """Bemis–Murcko scaffold SMILES, or empty string on failure."""
    if not isinstance(smiles, str) or not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return ""


def _docking_sample_weight(confidence: str | float) -> float:
    """Map per-receptor docking_confidence flag to a sample weight."""
    if not isinstance(confidence, str):
        return 0.3
    return {
        "high": 1.0,
        "marginal": 0.6,
        "low": 0.3,
        "low_confidence": 0.3,
    }.get(confidence, 0.5)


def _evidence_year_weight(year: float | int | None) -> float:
    """Down-weight older measurements modestly. 1.0 for ≥2015, 0.85 for
    2010–2014, 0.7 before."""
    if year is None or pd.isna(year):
        return 0.85
    y = int(year)
    if y >= 2015:
        return 1.0
    if y >= 2010:
        return 0.85
    return 0.7


# ----------------------------------------- core assembly


def assemble_dataset(
    unified_csv: Path | str = "data/processed/unified_ligands.csv",
    docking_csv: Path | str = "data/processed/docking_features.csv",
    ligand_features_parquet: Path | str = "data/processed/ligand_features.parquet",
    pose_3d_csv: Path | str = "data/processed/pose_3d_features.csv",
    ifp_parquet: Path | str = "data/processed/interaction_fingerprints.parquet",
    *,
    require_successful_dock: bool = True,
) -> pd.DataFrame:
    """Join all upstream artifacts into one wide ML-ready matrix.

    One row per unique (inchikey, receptor_uniprot, bias_category,
    bias_pathway, assay_1, assay_2) bias measurement.
    """
    ulig = pd.read_csv(unified_csv)
    dock = pd.read_csv(docking_csv)
    lf = pd.read_parquet(ligand_features_parquet)
    p3 = pd.read_csv(pose_3d_csv)
    ifp = pd.read_parquet(ifp_parquet)

    # 1. Scope: keep only small-molecule ligands (peptide ligands documented
    # as future-work scope).
    sm_inchis = set(lf["inchikey"].unique())
    n_before = len(ulig)
    ulig = ulig[ulig["inchikey"].isin(sm_inchis)].reset_index(drop=True)
    logger.info(
        "Scope filter: %d -> %d rows (peptide ligands excluded)",
        n_before, len(ulig),
    )

    # 2. Join docking features (per pair).
    dock_keep = [
        "inchikey", "receptor_uniprot", "success",
        "vina_affinity_best", "vina_affinity_mean_top3",
        "vina_affinity_gap_1_2", "vina_pose_diversity_rmsd",
        "vina_n_distinct_clusters", "n_poses",
        "docking_confidence", "redock_rmsd_angstrom", "gnina_cnn_score",
    ]
    df = ulig.merge(
        dock[dock_keep].rename(columns={"success": "docking_success"}),
        on=["inchikey", "receptor_uniprot"], how="left",
    )

    if require_successful_dock:
        before = len(df)
        df = df[df["docking_success"] == True].reset_index(drop=True)  # noqa: E712
        logger.info(
            "Successful-dock filter: %d -> %d rows", before, len(df),
        )

    # 3. 3D pose descriptors + missingness indicator.
    df = df.merge(
        p3.drop(columns=["pair_id"], errors="ignore"),
        on=["inchikey", "receptor_uniprot"], how="left",
    )
    df["pose_3d_missing"] = df["Asphericity"].isna().astype(int)

    # 4. ProLIF interaction bits + missingness indicators.
    ifp_drop = [c for c in ("pair_id",) if c in ifp.columns]
    df = df.merge(
        ifp.drop(columns=ifp_drop, errors="ignore"),
        on=["inchikey", "receptor_uniprot"], how="left",
    )
    df["ifp_missing"] = df["n_total_contacts"].isna().astype(int)
    df["ifp_no_contacts"] = (
        (df["n_total_contacts"].fillna(0) == 0).astype(int)
    )

    # 5. Ligand-level chemistry features.
    df = df.merge(
        lf, on=["inchikey", "canonical_smiles_std"], how="left",
        suffixes=("", "_lig"),
    )

    # 6. Stable composite key.
    df["pair_key"] = (
        df["inchikey"] + "::" + df["receptor_uniprot"] + "::"
        + df["bias_category"].fillna("?") + "::"
        + df["bias_pathway"].fillna("?") + "::"
        + df["assay_1"].fillna("?") + "::" + df["assay_2"].fillna("?")
    )
    if not df["pair_key"].is_unique:
        n_dup = df["pair_key"].duplicated().sum()
        logger.warning(
            "%d duplicate pair_keys collapsed (keeping first)", n_dup
        )
        df = df.drop_duplicates(subset=["pair_key"]).reset_index(drop=True)

    # 7. Bemis-Murcko scaffold per ligand (used for scaffold splits).
    scaf_lookup = (
        ulig[["inchikey", "canonical_smiles_std"]]
        .drop_duplicates("inchikey")
        .assign(scaffold=lambda d: d["canonical_smiles_std"].apply(_bemis_murcko))
        .set_index("inchikey")["scaffold"]
        .to_dict()
    )
    df["scaffold"] = df["inchikey"].map(scaf_lookup).fillna("")

    # 8. Sample weight = docking_confidence_weight × evidence_year_weight.
    df["sample_weight"] = (
        df["docking_confidence"].apply(_docking_sample_weight)
        * df["year"].apply(_evidence_year_weight)
    )

    # 9. Drop label-column NaNs (no `unknown` injection).
    before = len(df)
    df = df[df["bias_category"].notna()].reset_index(drop=True)
    if before != len(df):
        logger.info("Label-presence filter: %d -> %d rows", before, len(df))

    # 10. Auto-generate `_missing` indicators for every numeric column with
    # NaN. XGBoost handles NaN natively, but indicators give downstream
    # models a chance to learn that missingness is itself a signal.
    skip_indicator = {"pose_3d_missing", "ifp_missing", "ifp_no_contacts"}
    auto_indicator_cols: list[str] = []
    for col in df.columns:
        if col in skip_indicator or col.endswith("_missing"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].isna().any():
            df[f"{col}_missing"] = df[col].isna().astype(int)
            auto_indicator_cols.append(col)
    logger.info(
        "Added %d auto-generated _missing indicators", len(auto_indicator_cols)
    )

    return df


# ----------------------------------------- splits


@dataclass
class SplitSpec:
    name: str
    train_idx: list[int]
    test_idx: list[int]
    method: str
    rationale: str


def make_scaffold_split(
    df: pd.DataFrame, *, test_frac: float = 0.20, seed: int = 42
) -> SplitSpec:
    """Bemis-Murcko scaffold split — anti-leakage standard for QSAR/drug-discovery.
    Same scaffold cannot appear in both train and test."""
    groups = df["scaffold"].fillna("").values
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(np.arange(len(df)), groups=groups))
    return SplitSpec(
        name="scaffold_split",
        train_idx=train_idx.tolist(), test_idx=test_idx.tolist(),
        method="GroupShuffleSplit on Bemis-Murcko scaffold",
        rationale=(
            "Same chemical scaffold cannot appear in both train and test. "
            f"test_size={test_frac}, seed={seed}."
        ),
    )


def make_receptor_stratified_split(
    df: pd.DataFrame, *, test_frac: float = 0.20, seed: int = 42,
) -> SplitSpec:
    """Hold out entire receptors. Tests cross-receptor generalization."""
    groups = df["receptor_uniprot"].values
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_frac, random_state=seed
    )
    train_idx, test_idx = next(splitter.split(np.arange(len(df)), groups=groups))
    return SplitSpec(
        name="receptor_split",
        train_idx=train_idx.tolist(), test_idx=test_idx.tolist(),
        method="GroupShuffleSplit on receptor_uniprot",
        rationale=(
            "Held-out receptors evaluate the model's ability to generalize to "
            f"unseen targets. test_size={test_frac}, seed={seed}."
        ),
    )


def make_temporal_holdout(
    df: pd.DataFrame, *, year_cutoff: int = 2018,
) -> SplitSpec:
    """Year-based temporal holdout — measurements from year ≥ cutoff are held out."""
    train_mask = df["year"].fillna(0) < year_cutoff
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(~train_mask)
    return SplitSpec(
        name="temporal_holdout",
        train_idx=train_idx.tolist(), test_idx=test_idx.tolist(),
        method=f"year < {year_cutoff} -> train; year >= {year_cutoff} -> test",
        rationale=(
            "Reviewer-2 ask: train on historical evidence, evaluate on the "
            "more recent biased-agonism literature."
        ),
    )


def make_stratified_kfold(
    df: pd.DataFrame, *, k: int = 5, seed: int = 42
) -> dict:
    """K-fold stratified by bias_category — used inside the trainer for CV."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for fi, (tr, te) in enumerate(
        skf.split(np.arange(len(df)), df["bias_category"].values)
    ):
        folds.append({
            "fold": fi, "train_idx": tr.tolist(), "test_idx": te.tolist(),
        })
    return {
        "method": f"StratifiedKFold(k={k}) on bias_category",
        "folds": folds,
    }


# ----------------------------------------- emission


def emit_outputs(
    df: pd.DataFrame,
    *,
    output_path: Path | str = "data/processed/ml_ready_dataset.parquet",
    splits_path: Path | str = "data/processed/ml_splits.json",
    meta_path: Path | str = "data/processed/ml_ready_dataset.meta.json",
    audit_path: Path | str = "data/processed/dataset_assembly_audit.md",
    holdout_path: Path | str = "data/holdout/dataset_holdout.parquet",
    inputs: dict[str, Path] | None = None,
) -> dict:
    """Write the four output artifacts and return a small summary dict."""
    output_path = Path(output_path); splits_path = Path(splits_path)
    meta_path = Path(meta_path); audit_path = Path(audit_path)
    holdout_path = Path(holdout_path)

    # Build splits
    temporal = make_temporal_holdout(df)

    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    if temporal.test_idx:
        df.iloc[temporal.test_idx].to_parquet(holdout_path, index=False)
    df_train_eligible = df.iloc[temporal.train_idx].reset_index(drop=True)

    # Re-compute scaffold + receptor splits on the train-eligible portion
    scaffold = make_scaffold_split(df_train_eligible)
    receptor = make_receptor_stratified_split(df_train_eligible)
    cv = make_stratified_kfold(df_train_eligible)

    df_train_eligible.to_parquet(output_path, index=False)
    logger.info(
        "Wrote ml_ready_dataset.parquet: %d rows × %d cols (%d held out)",
        len(df_train_eligible), len(df_train_eligible.columns),
        len(temporal.test_idx),
    )

    splits_payload = {
        "schema_version": 1,
        "row_count_train_eligible": len(df_train_eligible),
        "row_count_holdout": len(temporal.test_idx),
        "splits": {
            "scaffold_split": {
                "method": scaffold.method, "rationale": scaffold.rationale,
                "train_idx": scaffold.train_idx, "test_idx": scaffold.test_idx,
            },
            "receptor_split": {
                "method": receptor.method, "rationale": receptor.rationale,
                "train_idx": receptor.train_idx, "test_idx": receptor.test_idx,
            },
            "temporal_holdout": {
                "method": temporal.method, "rationale": temporal.rationale,
                "holdout_path": str(holdout_path),
            },
            "stratified_kfold": cv,
        },
    }
    splits_path.write_text(json.dumps(splits_payload, indent=2))
    logger.info("Wrote ml_splits.json -> %s", splits_path)

    meta = {
        "schema_version": 1,
        "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_artifacts": [
            {"path": str(p), "sha256": _sha256(p)}
            for p in (inputs or {}).values() if p.exists()
        ],
        "row_counts": {
            "train_eligible": len(df_train_eligible),
            "temporal_holdout": len(temporal.test_idx),
            "total": len(df),
        },
        "label_distribution": dict(
            Counter(df_train_eligible["bias_category"].astype(str))
        ),
        "per_receptor_counts": dict(
            Counter(df_train_eligible["receptor_uniprot"].astype(str))
        ),
        "per_bias_pathway_counts": dict(
            Counter(df_train_eligible["bias_pathway"].astype(str))
        ),
        "feature_block_counts": {
            "vina_pose_ensemble": 5,
            "ligand_chemistry_2d": 217,
            "morgan_bits": int(sum(1 for c in df_train_eligible.columns
                                    if c.startswith("morgan_"))),
            "maccs_bits": int(sum(1 for c in df_train_eligible.columns
                                   if c.startswith("maccs_"))),
            "pose_3d_descriptors": 10,
            "ifp_bits": int(sum(1 for c in df_train_eligible.columns
                                 if c.startswith("ifp_"))),
        },
        "sample_weight": {
            "definition": "docking_confidence_weight × evidence_year_weight",
            "docking_weights": {
                "high": 1.0, "marginal": 0.6, "low": 0.3,
                "low_confidence": 0.3,
            },
            "evidence_year_weights": {
                ">=2015": 1.0, "2010-2014": 0.85, "<2010": 0.7,
            },
            "min": float(df_train_eligible["sample_weight"].min()),
            "max": float(df_train_eligible["sample_weight"].max()),
            "median": float(df_train_eligible["sample_weight"].median()),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    logger.info("Wrote dataset_meta -> %s", meta_path)

    # ---- audit markdown
    md = [
        "# Stage 07 — Dataset Assembly audit",
        "",
        f"_Generated: {meta['assembled_at_utc']}_  ",
        "",
        "## Row counts",
        "",
        f"- Total assembled rows: {len(df)}",
        f"- Train-eligible (year < 2018): {len(df_train_eligible)}",
        f"- Temporal holdout (year ≥ 2018): {len(temporal.test_idx)}",
        "",
        "## Label distribution (train-eligible)",
        "",
        "| bias_category | count | pct |",
        "| --- | --- | --- |",
    ]
    total_te = len(df_train_eligible)
    for k, v in Counter(df_train_eligible["bias_category"].astype(str)).most_common():
        md.append(f"| {k} | {v} | {100*v/total_te:.1f}% |")
    md.append("")

    md.append("## Splits")
    md.append("")
    md.append(f"- **Scaffold split**: {len(scaffold.train_idx)} train / {len(scaffold.test_idx)} test (Bemis-Murcko grouping)")
    md.append(f"- **Receptor split**: {len(receptor.train_idx)} train / {len(receptor.test_idx)} test (held-out receptors)")
    md.append(f"- **Temporal holdout**: {len(df_train_eligible)} train-eligible / {len(temporal.test_idx)} holdout (year cutoff = 2018)")
    md.append("- **Stratified K-fold**: 5 folds for CV inside trainer")
    md.append("")

    md.append("## Per-receptor counts (top 15)")
    md.append("")
    md.append("| uniprot | n_rows |")
    md.append("| --- | --- |")
    for u, n in Counter(df_train_eligible["receptor_uniprot"].astype(str)).most_common(15):
        md.append(f"| {u} | {n} |")
    md.append("")

    md.append("## Per-split detail")
    md.append("")
    def _split_detail(name: str, train_idx: list[int], test_idx: list[int]) -> None:
        train_df = df_train_eligible.iloc[train_idx]
        test_df = df_train_eligible.iloc[test_idx]
        md.append(f"### {name}")
        md.append("")
        md.append(f"- Train rows: {len(train_df)}  /  Test rows: {len(test_df)}")
        md.append(f"- Train unique receptors: {train_df['receptor_uniprot'].nunique()}  /  Test: {test_df['receptor_uniprot'].nunique()}")
        md.append(f"- Train unique ligands: {train_df['inchikey'].nunique()}  /  Test: {test_df['inchikey'].nunique()}")
        md.append(f"- Train unique scaffolds: {train_df['scaffold'].nunique()}  /  Test: {test_df['scaffold'].nunique()}")
        test_class_counts = Counter(test_df['bias_category'].astype(str))
        md.append("- Test class distribution: " +
                  ", ".join(f"{k}={v}" for k, v in test_class_counts.most_common()))
        md.append("")
    _split_detail("Scaffold split", scaffold.train_idx, scaffold.test_idx)
    _split_detail("Receptor split", receptor.train_idx, receptor.test_idx)

    holdout_df = df.iloc[temporal.test_idx]
    md.append("### Temporal holdout (year ≥ 2018)")
    md.append("")
    md.append(f"- Rows: {len(holdout_df)}")
    md.append(f"- Unique receptors: {holdout_df['receptor_uniprot'].nunique()}")
    md.append(f"- Unique ligands: {holdout_df['inchikey'].nunique()}")
    holdout_class_counts = Counter(holdout_df['bias_category'].astype(str))
    md.append("- Holdout class distribution: " +
              ", ".join(f"{k}={v}" for k, v in holdout_class_counts.most_common()))
    md.append("")

    md.append("## Sample-weight summary")
    md.append("")
    md.append(f"- min={meta['sample_weight']['min']:.2f}  "
              f"median={meta['sample_weight']['median']:.2f}  "
              f"max={meta['sample_weight']['max']:.2f}")
    md.append("")
    md.append("## Feature block sizes")
    md.append("")
    for k, v in meta["feature_block_counts"].items():
        md.append(f"- `{k}`: {v}")
    audit_path.write_text("\n".join(md))
    logger.info("Wrote audit -> %s", audit_path)

    return {
        "total_samples": len(df_train_eligible),
        "total_features": len(df_train_eligible.columns),
        "holdout_samples": len(temporal.test_idx),
        "label_distribution": meta["label_distribution"],
    }


# ----------------------------------------- entry points


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    inputs = {
        "unified": Path("data/processed/unified_ligands.csv"),
        "dock": Path("data/processed/docking_features.csv"),
        "ligand_features": Path("data/processed/ligand_features.parquet"),
        "pose_3d": Path("data/processed/pose_3d_features.csv"),
        "ifp": Path("data/processed/interaction_fingerprints.parquet"),
    }
    df = assemble_dataset(
        unified_csv=inputs["unified"],
        docking_csv=inputs["dock"],
        ligand_features_parquet=inputs["ligand_features"],
        pose_3d_csv=inputs["pose_3d"],
        ifp_parquet=inputs["ifp"],
    )

    # Schema validation surfaces upstream drift immediately.
    from cancerag.ml.dataset_schema import validate as validate_dataset
    report = validate_dataset(df)
    logger.info(
        "Schema OK: %d rows × %d cols. Feature counts: %s",
        report["rows"], report["columns"], report["feature_counts"],
    )

    emit_outputs(df, inputs=inputs)
    logger.info("STAGE_07_DONE")


def run_dataset_assembly(config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Backwards-compatible entrypoint used by ``cancerag.main``.

    Ignores the legacy ``config`` argument; the new pipeline reads from the
    fixed ``data/processed/`` paths. Returns ``(train_eligible_df, summary)``.
    """
    inputs = {
        "unified": Path("data/processed/unified_ligands.csv"),
        "dock": Path("data/processed/docking_features.csv"),
        "ligand_features": Path("data/processed/ligand_features.parquet"),
        "pose_3d": Path("data/processed/pose_3d_features.csv"),
        "ifp": Path("data/processed/interaction_fingerprints.parquet"),
    }
    df = assemble_dataset(
        unified_csv=inputs["unified"],
        docking_csv=inputs["dock"],
        ligand_features_parquet=inputs["ligand_features"],
        pose_3d_csv=inputs["pose_3d"],
        ifp_parquet=inputs["ifp"],
    )
    summary = emit_outputs(df, inputs=inputs)
    train_eligible = pd.read_parquet(
        Path("data/processed/ml_ready_dataset.parquet")
    )
    return train_eligible, summary


# =====================================================================
# Helper functions retained from the previous module — used by tests and
# for future label-conflict resolution work.
# =====================================================================


@dataclass(frozen=True)
class LabelResolution:
    pair_key: str
    chosen: str
    decision: str
    candidates: tuple[str, ...]


def make_pair_key(
    df: pd.DataFrame,
    *,
    ligand_col: str = "inchikey14",
    receptor_col: str = "receptor_uniprot",
    assay1_col: str = "assay_1",
    assay2_col: str = "assay_2",
) -> pd.Series:
    """Composite key uniquely identifying a (ligand, receptor, assay-pair) row."""
    for col in (ligand_col, receptor_col):
        if col not in df.columns:
            raise KeyError(f"make_pair_key: column {col!r} missing")
    a1 = df[assay1_col].fillna("?") if assay1_col in df.columns else "?"
    a2 = df[assay2_col].fillna("?") if assay2_col in df.columns else "?"
    return (
        df[ligand_col].astype(str) + "::"
        + df[receptor_col].astype(str) + "::"
        + (a1.astype(str) if hasattr(a1, "astype") else pd.Series([a1] * len(df)))
        + "::"
        + (a2.astype(str) if hasattr(a2, "astype") else pd.Series([a2] * len(df)))
    )


def _evidence_score(row: pd.Series) -> float:
    """Per-row evidence credibility score (assay type × confidence × year)."""
    score = 1.0
    assay_type = str(row.get("assay_type", "F"))
    if assay_type.upper() == "F":
        score *= 1.0
    elif assay_type.upper() == "B":
        score *= 0.5
    confidence = row.get("confidence_score", None)
    if pd.notna(confidence) and float(confidence) >= 9:
        score *= 1.2
    year = row.get("year", None)
    if pd.notna(year) and int(year) >= 2020:
        score *= 1.1
    return score


def resolve_label_conflicts(
    df: pd.DataFrame,
    *,
    key_col: str = "pair_key",
    label_col: str = "primary_bias_label",
) -> tuple[pd.DataFrame, list[LabelResolution]]:
    """For each ``key_col`` group with conflicting labels, pick the label
    whose rows have the highest summed evidence score. Returns the deduped
    frame and a per-conflict resolution log."""
    decisions: list[LabelResolution] = []
    resolved_rows: list[pd.Series] = []

    for key, group in df.groupby(key_col, dropna=False):
        non_null = group[group[label_col].notna()]
        if non_null.empty:
            continue
        labels = non_null[label_col].unique().tolist()
        if len(labels) == 1:
            chosen = labels[0]
            decision = "unanimous"
        else:
            scored = (
                non_null.assign(_w=non_null.apply(_evidence_score, axis=1))
                .groupby(label_col)["_w"].sum()
                .sort_values(ascending=False)
            )
            chosen = scored.index[0]
            decision = (
                "evidence_weight: "
                + ", ".join(f"{lbl}={w:.2f}" for lbl, w in scored.items())
            )
        decisions.append(LabelResolution(
            pair_key=str(key), chosen=chosen, decision=decision,
            candidates=tuple(labels),
        ))
        rep = non_null[non_null[label_col] == chosen].iloc[0]
        resolved_rows.append(rep)

    if resolved_rows:
        deduped = pd.DataFrame(resolved_rows).reset_index(drop=True)
    else:
        deduped = df.iloc[0:0].copy()
    return deduped, decisions


def decisions_to_dataframe(decisions: Iterable[LabelResolution]) -> pd.DataFrame:
    return pd.DataFrame([
        {"pair_key": d.pair_key, "chosen": d.chosen,
         "decision": d.decision, "candidates": "|".join(d.candidates)}
        for d in decisions
    ])


def split_temporal_holdout(
    df: pd.DataFrame,
    *,
    cutoff_year: int,
    year_col: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into ``(train_eligible, holdout)`` by year cutoff."""
    if year_col not in df.columns:
        raise KeyError(f"split_temporal_holdout: column {year_col!r} missing")
    year_numeric = pd.to_numeric(df[year_col], errors="coerce")
    holdout_mask = year_numeric >= cutoff_year
    return df[~holdout_mask].copy(), df[holdout_mask].copy()


def evidence_weight_column(df: pd.DataFrame) -> pd.Series:
    """Vectorised per-row evidence weight; usable as ``sample_weight``."""
    return df.apply(_evidence_score, axis=1).astype(float)


def add_missing_indicators(
    df: pd.DataFrame, columns: Iterable[str], *, suffix: str = "_missing"
) -> pd.DataFrame:
    """Append a ``<col>_missing`` int indicator (0/1) for each named column."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}{suffix}"] = out[col].isna().astype(int)
    return out


def has_no_default_docking_score(df: pd.DataFrame, value: float = -5.0) -> bool:
    """Sanity check: no docking-score column should still contain the legacy
    ``-5.0`` sentinel."""
    docking_cols = [
        c for c in df.columns
        if c.lower().startswith("docking_") or "vina" in c.lower()
    ]
    for c in docking_cols:
        try:
            if (df[c] == value).any():
                return False
        except (TypeError, ValueError):
            continue
    return True


if __name__ == "__main__":
    main()
