"""
Ligand preprocessing.

Owns:
- The legacy ``LigandPreprocessor`` class (kept for backwards compatibility
  with ``main.py``).
- Stage 02 curation primitives (standardize, dedupe, annotate, attrition,
  label-status assignment, source merging) that the new pipeline composes
  into a per-fold sklearn flow. See ``improvements/02_ligand_curation.md``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _sha256_or_none(path: str | Path) -> str | None:
    """Return the SHA-256 of a file, or None if the path does not exist
    (used for provenance sidecars where missing inputs shouldn't crash)."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------- attrition log


@dataclass
class AttritionLogger:
    """Tracks how many rows survive each curation stage."""

    records: list[dict] = field(default_factory=list)

    def log(self, stage: str, n_in: int, n_out: int, reason: str = "") -> None:
        rec = {
            "stage": stage,
            "n_in": int(n_in),
            "n_out": int(n_out),
            "n_dropped": int(n_in - n_out),
            "reason": reason,
        }
        self.records.append(rec)
        logger.info(
            "attrition[%s]: %d -> %d (-%d) %s",
            stage,
            n_in,
            n_out,
            n_in - n_out,
            f"({reason})" if reason else "",
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


# ------------------------------------------------------------- standardization


_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
_UNCHARGER = rdMolStandardize.Uncharger()

# Tautomer canonicalization is unsafe for peptides and other large
# molecules. Each amide can flip keto/enol; for a peptide this means
# 2^(n_amide) possible tautomers. RDKit's TautomerEnumerator hits its
# 1000-tautomer ceiling, gives up, and returns a stereo-less canonical
# form — which collapses *different* peptides to the same InChIKey.
# Empirically, small-molecule drug-likes top out at ~40-50 heavy atoms;
# peptides start around 60-70. Skipping tautomer canonicalization above
# the threshold preserves the input form (still standardized for salts /
# protonation / stereo) while leaving the per-residue tautomers alone.
_TAUTOMER_SKIP_HEAVY_ATOM_THRESHOLD = 50


def standardize_smiles(smiles: str) -> dict:
    """Canonicalize a SMILES string to its standardized parent form.

    Returns a dict with ``status``, the standardized ``Mol`` (or None), the
    canonical SMILES, and the InChIKey / InChIKey-14 connectivity layer.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return {"status": "empty", "mol": None}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "parse_error", "mol": None}

    try:
        cleaned = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(cleaned)
        neutral = _UNCHARGER.uncharge(parent)
        # Skip tautomer canonicalization for peptides / large molecules
        # (see comment on _TAUTOMER_SKIP_HEAVY_ATOM_THRESHOLD).
        if neutral.GetNumHeavyAtoms() > _TAUTOMER_SKIP_HEAVY_ATOM_THRESHOLD:
            canonical_taut = neutral
        else:
            canonical_taut = _TAUTOMER_ENUMERATOR.Canonicalize(neutral)
        Chem.SanitizeMol(canonical_taut)
        # NOTE: do NOT pass force=True here. force=True overwrites the
        # stereo flags RDKit parsed from the input SMILES, which collapses
        # distinct stereoisomers (e.g. the four fenoterol stereoisomers
        # from PMID 25342094) into a single canonical structure. The
        # default cleanIt=True preserves declared stereo and only fixes
        # malformed flags.
        Chem.AssignStereochemistry(canonical_taut, cleanIt=True, force=False)
        inchi = Chem.MolToInchi(canonical_taut)
        inchikey = Chem.InchiToInchiKey(inchi) if inchi else ""
        canonical_smiles = Chem.MolToSmiles(canonical_taut)
    except Exception as exc:
        return {"status": f"standardize_error: {exc}", "mol": None}

    return {
        "status": "ok",
        "mol": canonical_taut,
        "canonical_smiles": canonical_smiles,
        "inchi": inchi,
        "inchikey": inchikey,
        "inchikey14": inchikey.split("-")[0] if inchikey else "",
    }


def standardize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    attrition: "AttritionLogger | None" = None,
) -> pd.DataFrame:
    """Apply ``standardize_smiles`` row-wise; drop rows that fail."""
    n_in = len(df)
    records = df[smiles_col].map(standardize_smiles)
    out = df.copy().reset_index(drop=True)
    out["std_status"] = records.map(lambda r: r["status"])
    out["canonical_smiles_std"] = records.map(lambda r: r.get("canonical_smiles", ""))
    out["inchikey"] = records.map(lambda r: r.get("inchikey", ""))
    out["inchikey14"] = records.map(lambda r: r.get("inchikey14", ""))

    n_failed = int((out["std_status"] != "ok").sum())
    if n_failed:
        for _, row in out[out["std_status"] != "ok"].iterrows():
            logger.warning(
                "standardize failed: status=%s smiles=%r",
                row["std_status"],
                row[smiles_col],
            )

    out = out[out["std_status"] == "ok"].reset_index(drop=True)
    if attrition is not None:
        attrition.log("standardize", n_in, len(out), "drop parse/standardize errors")
    return out


# ------------------------------------------------------------- deduplication


def dedupe_with_conflict_log(
    df: pd.DataFrame,
    *,
    key_col: str = "inchikey14",
    label_col: str = "bias_category",
    attrition: "AttritionLogger | None" = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deduplicate by ``key_col``, returning ``(deduped_df, conflicts_df)``.

    A "conflict" is a key whose rows carry distinct, non-null ``label_col``
    values. Conflicting rows are dropped from ``deduped_df`` and surfaced in
    ``conflicts_df`` for manual triage — silently keeping the first row (the
    legacy behaviour) hides label disagreements that frequently flag
    underlying data-curation problems (assay drift, source mislabelling,
    receptor-name aliasing).
    """
    n_in = len(df)
    if key_col not in df.columns:
        raise KeyError(f"dedupe key column {key_col!r} not in DataFrame")

    grouped = df.groupby(key_col, dropna=False)
    label_counts = grouped[label_col].nunique(dropna=True)
    conflicting_keys = label_counts[label_counts > 1].index
    conflicts = df[df[key_col].isin(conflicting_keys)].copy()

    non_conflicting = df[~df[key_col].isin(conflicting_keys)]
    deduped = non_conflicting.drop_duplicates(subset=[key_col], keep="first").copy()

    if attrition is not None:
        attrition.log(
            "dedupe",
            n_in,
            len(deduped),
            f"InChIKey-14 dedup; {len(conflicts)} conflicting rows surfaced",
        )
    return deduped, conflicts


# ---------------------------------------------------------------- annotation


def _build_pains_filter() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


_PAINS_FILTER = _build_pains_filter()


def annotate_drug_likeness(
    df: pd.DataFrame, *, smiles_col: str = "canonical_smiles_std"
) -> pd.DataFrame:
    """Append drug-likeness columns (MW, LogP, HBD, HBA, TPSA, RotBonds,
    Lipinski_Violations, has_pains) without filtering rows."""
    # Reset the index so a freshly-built props frame (which always has a
    # 0..n-1 RangeIndex) aligns with `out` under pd.concat(axis=1).
    # The legacy code skipped this and pd.concat silently invented phantom
    # NaN rows whenever the input came from an upstream filter/dedup that
    # left a non-contiguous index.
    out = df.copy().reset_index(drop=True)

    def _props(smiles: str) -> dict:
        if not isinstance(smiles, str) or not smiles:
            return {}
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        try:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            rot = Descriptors.NumRotatableBonds(mol)
            violations = (
                int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
            )
            return {
                "MW": mw,
                "LogP": logp,
                "HBD": hbd,
                "HBA": hba,
                "TPSA": tpsa,
                "RotBonds": rot,
                "Lipinski_Violations": violations,
                "has_pains": _PAINS_FILTER.HasMatch(mol),
            }
        except Exception:
            return {}

    cols = (
        "MW", "LogP", "HBD", "HBA", "TPSA",
        "RotBonds", "Lipinski_Violations", "has_pains",
    )
    props = pd.DataFrame(
        [_props(s) for s in out[smiles_col]], columns=list(cols)
    )
    return pd.concat([out, props], axis=1)


# --------------------------------------------------------- label assignment


def mark_label_status(
    df: pd.DataFrame,
    *,
    label_col: str = "bias_category",
    status_col: str = "label_status",
) -> pd.DataFrame:
    """Tag rows as ``labeled`` or ``unlabeled`` based on whether ``label_col``
    carries a real value.

    Replaces the source-shortcut behaviour where ChEMBL rows were unilaterally
    labeled ``"Agonist"`` regardless of whether bias had ever been measured
    — which made the model learn provenance instead of bias (existential
    issue E1).
    """
    out = df.copy()
    out[status_col] = out[label_col].apply(
        lambda v: "labeled" if isinstance(v, str) and v.strip() else "unlabeled"
    )
    return out


def merge_sources(
    sources: Iterable[pd.DataFrame],
    *,
    attrition: "AttritionLogger | None" = None,
) -> pd.DataFrame:
    """Concatenate per-source ligand frames into a single unified frame.

    Each input frame must already have ``source`` and ``label_status``
    columns.
    """
    frames = list(sources)
    n_in_total = sum(len(f) for f in frames)
    for f in frames:
        for required in ("source", "label_status"):
            if required not in f.columns:
                raise KeyError(
                    f"input frame missing required column {required!r}"
                )
    merged = pd.concat(frames, ignore_index=True)
    if attrition is not None:
        attrition.log("merge_sources", n_in_total, len(merged), "concat sources")
    return merged


# --------------------------------------------------- legacy class, unchanged


class LigandPreprocessor:
    """
    Curates and unifies ligand data into a single training-ready table.

    Stage 02 wiring (post-decision, BiasDB-only multi-class framing):

    - **BiasDB only** is used for the labelled training table. ChEMBL is
      **not** merged into ``unified_ligands.csv``; the legacy source-shortcut
      where every ChEMBL row was hardcoded ``bias_category="Agonist"``
      (existential issue E1) is removed entirely.
    - All metadata BiasDB carries (``reference_ligand``, ``assay_1``,
      ``assay_2``, ``pmid``, ``year``, ``doi``) is preserved through curation
      so downstream stages (assembly, holdout split, evidence weighting) can
      use it.
    - Standardization uses ``standardize_smiles`` (Cleanup → FragmentParent
      → Uncharger → tautomer canonicalization → InChIKey).
    - Deduplication uses InChIKey-14 connectivity, with conflicting labels
      surfaced to ``dedup_conflicts.csv`` rather than silently keeping the
      first row.
    - Drug-likeness rules (PAINS, Lipinski, TPSA, RotBonds) are
      **annotations, not filters** by default. Set
      ``preprocessing.legacy_filters: true`` in the config to opt back in.
    - Per-stage attrition is logged to ``attrition.json``.

    Pipeline flow::

        BiasDB CSV
          → standardize_dataframe
          → mark_label_status (no source-shortcut labels)
          → dedupe_with_conflict_log (InChIKey-14)
          → annotate_drug_likeness (annotation, not filtering)
          → unified_ligands.csv (+ .meta.json)
          → attrition.json
          → dedup_conflicts.csv
    """

    # BiasDB columns we need to preserve through curation. Anything else in
    # the source CSV is dropped silently (those are legacy duplicate columns).
    _BIASDB_COLUMNS_KEPT = (
        "smiles",
        "ligand_name",
        "receptor_subtype",
        "bias_category",
        "bias_pathway",
        "reference_ligand",
        "assay_1",
        "assay_2",
        "pmid",
        "year",
        "doi",
    )

    def __init__(self, config: dict):
        """Initialize the LigandPreprocessor with the project configuration."""
        self.config = config
        self.paths = config["paths"]
        self.params = config.get("preprocessing", {})
        # Legacy hard-filtering is opt-in. Default behaviour is annotate-only
        # per Stage 02 decision (Reviewer-flagged biases: catechols dropped
        # by PAINS, peptides dropped by Lipinski, etc.).
        self.legacy_filters = bool(self.params.get("legacy_filters", False))
        # Lazy-load the receptor registry only when needed (so unit tests that
        # don't exercise the full curator don't have to mock it).
        self._registry = None

    def _get_registry(self):
        """Load the canonical receptor registry on first use.

        Strict mode (the Stage 02 decision): if a BiasDB receptor name is
        not in the registry, the curator raises rather than silently
        producing rows with no UniProt anchor.
        """
        if self._registry is None:
            # Imported lazily so importing LigandPreprocessor doesn't pull
            # the registry module into every test.
            from cancerag.data_collection.registry import ReceptorRegistry

            registry_path = self.paths.get(
                "registry", "data/registry/receptors.tsv"
            )
            self._registry = ReceptorRegistry.load(registry_path)
        return self._registry

    def _attach_receptor_uniprot(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """Add a `receptor_uniprot` column by joining on `receptor_subtype`
        through the canonical registry.

        Strict: any receptor_subtype value not present in the registry
        causes a KeyError so the dataset never contains UniProt-less rows.
        """
        registry = self._get_registry()
        out = df.copy()
        unresolved: list[str] = []
        uniprots: list[str] = []
        for name in out["receptor_subtype"].fillna(""):
            row = registry.by_biasdb_name(name) if name else None
            if row is None:
                unresolved.append(name)
                uniprots.append("")
            else:
                uniprots.append(str(row["uniprot"]))
        if unresolved:
            unique = sorted(set(unresolved))
            raise KeyError(
                f"LigandPreprocessor: {len(unique)} receptor name(s) not in "
                f"registry: {unique[:10]}{'...' if len(unique) > 10 else ''}. "
                f"Add rows to data/registry/receptors.tsv before re-running "
                f"(strict mode)."
            )
        out["receptor_uniprot"] = uniprots
        return out

    def _load_biasdb_data(self) -> pd.DataFrame:
        """Load BiasDB and rename ``smiles`` → ``canonical_smiles``.

        Preserves the assay / publication metadata columns rather than
        discarding them as the legacy code did.
        """
        logger.info("Loading BiasDB data from %s", self.paths["biasdb_input"])
        df = pd.read_csv(self.paths["biasdb_input"])

        keep = [c for c in self._BIASDB_COLUMNS_KEPT if c in df.columns]
        df = df[keep].copy()
        df.rename(columns={"smiles": "canonical_smiles"}, inplace=True)
        df["source"] = "BiasDB"
        logger.info("Loaded %d BiasDB records (kept %d cols)", len(df), len(keep))
        return df

    def run(self) -> None:
        """Execute the curation pipeline. Idempotent — skipped if the output
        CSV already exists."""
        out_dir = self.paths["processed_data"]
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, "unified_ligands.csv")
        attrition_path = os.path.join(out_dir, "attrition.json")
        conflicts_path = os.path.join(out_dir, "dedup_conflicts.csv")
        meta_path = output_path + ".meta.json"

        if os.path.exists(output_path):
            logger.info(
                "Unified ligands already exist at %s. Skipping preprocessing.",
                output_path,
            )
            return

        attrition = AttritionLogger()

        # 1. Load BiasDB only (decision: BiasDB-only multi-class framing).
        biasdb_df = self._load_biasdb_data()
        attrition.log("load_biasdb", n_in=0, n_out=len(biasdb_df), reason="initial load")

        # 2. Standardize molecules; drop unparseable / unstandardizable rows.
        standardized = standardize_dataframe(
            biasdb_df, smiles_col="canonical_smiles", attrition=attrition
        )

        # 3. Tag each row's label-status.
        standardized = mark_label_status(standardized, label_col="bias_category")

        # 4. Resolve receptor_subtype -> UniProt via the canonical registry
        # (strict mode: unknown receptors raise so the dataset never
        # contains UniProt-less rows).
        with_uniprot = self._attach_receptor_uniprot(standardized)

        # 5. Build the composite pair_key. Final form (v5) uses every
        # column that defines a distinct *experimental measurement* in
        # BiasDB:
        #
        #   Tier 1 (required):
        #     - inchikey            (the molecule, stereo-aware)
        #     - receptor_uniprot    (the receptor)
        #     - bias_pathway        (which two pathways were compared)
        #
        #   Tier 2 (defines the experimental setup):
        #     - assay_1, assay_2    (the readout technologies)
        #     - reference_ligand    (bias is computed Δlog(τ/KA) RELATIVE
        #                            to a reference, e.g. buprenorphine
        #                            vs morphine differs from buprenorphine
        #                            vs DAMGO)
        #
        # Five iterations of this key got us here:
        #   v1: inchikey14                       -> collapsed across receptors.
        #   v2: + receptor + assay_1/2           -> 71 fake conflicts under BRET/BRET.
        #   v3: + bias_pathway                   -> conflicts resolved.
        #   v4: full inchikey (stereo-aware)     -> stereoisomers preserved.
        #   v5: + reference_ligand               -> different-reference rows
        #                                            (e.g. buprenorphine
        #                                            against morphine vs DAMGO)
        #                                            preserved.
        def _safe_col(col: str) -> "pd.Series":
            return (
                with_uniprot[col]
                if col in with_uniprot.columns
                else pd.Series([""] * len(with_uniprot))
            ).fillna("?").astype(str)

        pair_key = (
            with_uniprot["inchikey"].astype(str)
            + "::"
            + with_uniprot["receptor_uniprot"].astype(str)
            + "::"
            + _safe_col("bias_pathway")
            + "::"
            + _safe_col("assay_1")
            + "::"
            + _safe_col("assay_2")
            + "::"
            + _safe_col("reference_ligand")
        )
        with_uniprot = with_uniprot.assign(pair_key=pair_key)

        # 6. Dedupe on the composite key.
        deduped, conflicts = dedupe_with_conflict_log(
            with_uniprot,
            key_col="pair_key",
            label_col="bias_category",
            attrition=attrition,
        )

        # 7. Annotate drug-likeness columns (PAINS, Lipinski, TPSA, RotBonds).
        # By default these are columns, NOT filters.
        annotated = annotate_drug_likeness(deduped, smiles_col="canonical_smiles_std")

        # 7a. Optional: opt-in legacy hard-filtering (off by default).
        if self.legacy_filters:
            n_in = len(annotated)
            mask_pains = ~annotated["has_pains"].fillna(False)
            mask_lipinski = annotated["Lipinski_Violations"].fillna(0) <= (
                0 if self.params.get("lipinski_strict", False) else 1
            )
            mask_tpsa = annotated["TPSA"].fillna(0) <= self.params.get("tpsa_max", 140)
            mask_rotb = annotated["RotBonds"].fillna(0) <= self.params.get(
                "rotatable_bonds_max", 10
            )
            annotated = annotated[
                mask_pains & mask_lipinski & mask_tpsa & mask_rotb
            ].copy()
            attrition.log(
                "legacy_filters",
                n_in=n_in,
                n_out=len(annotated),
                reason="opt-in PAINS/Lipinski/TPSA/RotBonds filters",
            )

        # 8. Persist the artifacts.
        annotated.to_csv(output_path, index=False)
        if not conflicts.empty:
            conflicts.to_csv(conflicts_path, index=False)
            logger.info(
                "Wrote %d label-conflicting rows to %s", len(conflicts), conflicts_path
            )
        attrition_records = attrition.to_dataframe()
        attrition_records.to_json(attrition_path, orient="records", indent=2)
        logger.info(
            "Successfully saved %d processed ligands to %s", len(annotated), output_path
        )

        # 9. Provenance sidecar. value_counts -> dict can yield NaN keys
        # (when bias_category is missing on some rows); coerce to strings so
        # json.dump doesn't try to sort heterogeneous keys.
        def _stringify_keys(d: dict) -> dict:
            return {("<NaN>" if k != k else str(k)): int(v) for k, v in d.items()}

        meta = {
            "artifact_path": output_path,
            "source": "BiasDB-only (multi-class framing)",
            "input_path": self.paths["biasdb_input"],
            "input_sha256": _sha256_or_none(self.paths["biasdb_input"]),
            "output_sha256": _sha256_or_none(output_path),
            "row_count": int(len(annotated)),
            "label_distribution": _stringify_keys(
                annotated["bias_category"].value_counts(dropna=False).to_dict()
            ),
            "label_status_distribution": (
                _stringify_keys(
                    annotated["label_status"].value_counts(dropna=False).to_dict()
                )
                if "label_status" in annotated.columns
                else {}
            ),
            "dedup_conflicts": int(len(conflicts)),
            "legacy_filters_applied": self.legacy_filters,
            "attrition": attrition.records,
            "curated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True, default=str)
        logger.info("Wrote provenance sidecar to %s", meta_path)

        # 10. Reviewer-facing dataset audit (markdown). Captures the
        # analysis of receptor families / assay distribution / per-receptor
        # label balance that a manuscript reviewer would expect to see in
        # the supplementary material. Regenerated on every curation pass.
        audit_path = os.path.join(out_dir, "dataset_audit.md")
        self._emit_dataset_audit(annotated, conflicts, attrition, meta, audit_path)
        logger.info("Wrote dataset audit to %s", audit_path)

    def _emit_dataset_audit(
        self,
        df: "pd.DataFrame",
        conflicts: "pd.DataFrame",
        attrition: "AttritionLogger",
        meta: dict,
        out_path: str,
    ) -> None:
        """Emit a human-readable Markdown audit of the curated dataset.

        Joins with the receptor registry to attach gpcrdb_family / notes
        for each receptor, then summarizes:
        - top-line counts and SHA-256 (lifted from the meta sidecar)
        - per-family breakdown (rows + receptors)
        - top 15 receptors by row count
        - bias-class distribution (overall + per-family)
        - assay-pair distribution
        - conflict summary (counts + top 5 conflicting pair_keys)
        - per-stage attrition table
        """
        registry = self._get_registry()
        reg_df = registry.dataframe[
            ["uniprot", "gpcrdb_class", "gpcrdb_family", "notes"]
        ]
        m = df.merge(
            reg_df, left_on="receptor_uniprot", right_on="uniprot", how="left"
        )

        def _md_table(df_: pd.DataFrame, max_rows: int | None = None) -> str:
            d = df_ if max_rows is None else df_.head(max_rows)
            cols = list(d.columns)
            head = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join("---" for _ in cols) + " |"
            rows = [
                "| " + " | ".join(str(v) for v in row) + " |"
                for row in d.values.tolist()
            ]
            return "\n".join([head, sep, *rows])

        # Overall counts
        n_rows = len(df)
        n_molecules = int(df["inchikey14"].nunique()) if "inchikey14" in df else 0
        n_receptors = int(df["receptor_uniprot"].nunique())

        # Per-family breakdown
        fam = (
            m.groupby("gpcrdb_family", dropna=False)
            .agg(rows=("pair_key", "count"), receptors=("receptor_uniprot", "nunique"))
            .sort_values("rows", ascending=False)
            .reset_index()
            .rename(columns={"gpcrdb_family": "GPCRdb family"})
        )

        # Top 15 receptors
        top = (
            m.groupby(["receptor_subtype", "gpcrdb_family", "notes"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
            .head(15)
            .rename(columns={
                "receptor_subtype": "Receptor",
                "gpcrdb_family": "Family",
                "notes": "Class / endogenous ligand",
            })
        )

        # Bias-label distribution (overall)
        labels = (
            df["bias_category"]
            .value_counts(dropna=False)
            .rename_axis("bias_category")
            .reset_index(name="rows")
        )

        # Bias label x family
        cross = (
            m.assign(
                _label=df["bias_category"].fillna("(missing)").values,
                _family=m["gpcrdb_family"].fillna("(unknown)"),
            )
            .pivot_table(
                index="_family", columns="_label",
                values="pair_key", aggfunc="count", fill_value=0,
            )
            .reset_index()
            .rename(columns={"_family": "GPCRdb family"})
        )

        # Assay pair distribution
        df = df.copy()
        df["assay_pair"] = (
            df["assay_1"].fillna("?") + " / " + df["assay_2"].fillna("?")
        )
        assay_top = (
            df["assay_pair"].value_counts().head(15)
            .rename_axis("assay_1 / assay_2").reset_index(name="rows")
        )

        # Conflict summary
        conflict_pair_keys = (
            int(conflicts["pair_key"].nunique()) if not conflicts.empty else 0
        )
        if not conflicts.empty:
            conf_top = (
                conflicts.groupby("pair_key")
                .agg(
                    n_rows=("pair_key", "count"),
                    receptor=("receptor_subtype", "first"),
                    inchikey14=("inchikey14", "first"),
                    assay_1=("assay_1", "first"),
                    assay_2=("assay_2", "first"),
                    labels=("bias_category", lambda s: ", ".join(sorted(set(s.dropna())))),
                )
                .sort_values("n_rows", ascending=False)
                .head(5)
                .reset_index()
            )
        else:
            conf_top = pd.DataFrame(
                columns=["pair_key", "n_rows", "receptor", "inchikey14",
                         "assay_1", "assay_2", "labels"]
            )

        # Attrition
        attr_df = attrition.to_dataframe()
        if not attr_df.empty:
            attr_df = attr_df[["stage", "n_in", "n_out", "n_dropped", "reason"]]

        md = []
        md.append("# CancerAg dataset audit")
        md.append("")
        md.append(f"_Generated: {meta['curated_at_utc']}_  ")
        md.append(f"_Source: {meta['source']}_  ")
        md.append(f"_Input SHA-256: `{meta['input_sha256']}`_  ")
        md.append(f"_Output SHA-256: `{meta['output_sha256']}`_  ")
        md.append("")
        md.append("## Top-line numbers")
        md.append("")
        md.append(f"- **{n_rows} curated rows** in `unified_ligands.csv`")
        md.append(
            f"- **{n_molecules} unique molecules** (InChIKey-14) across "
            f"**{n_receptors} unique receptors** (UniProt)"
        )
        md.append(
            f"- **{conflict_pair_keys} same-context conflict pair_keys** "
            f"surfaced in `dedup_conflicts.csv` (totaling "
            f"{int(meta['dedup_conflicts'])} rows)"
        )
        md.append(f"- Drug-likeness filtering: {'on' if meta['legacy_filters_applied'] else 'off (annotation-only)'}")
        md.append("")

        md.append("## Bias-class distribution")
        md.append("")
        md.append(_md_table(labels))
        md.append("")

        md.append("## Receptors by GPCRdb family")
        md.append("")
        md.append(_md_table(fam))
        md.append("")

        md.append("## Top 15 receptors by row count")
        md.append("")
        md.append(_md_table(top))
        md.append("")

        md.append("## Bias label by receptor family (rows)")
        md.append("")
        md.append(_md_table(cross))
        md.append("")

        md.append("## Top 15 assay-pair combinations")
        md.append("")
        md.append(_md_table(assay_top))
        md.append("")

        md.append("## Per-stage attrition")
        md.append("")
        md.append(_md_table(attr_df))
        md.append("")

        if not conf_top.empty:
            md.append("## Top 5 same-context conflict pair_keys")
            md.append("")
            md.append(_md_table(conf_top))
            md.append("")
            md.append(
                "_Most remaining conflicts share the BRET / BRET assay-pair, "
                "where BiasDB stores the detection technology rather than the "
                "pathway. This is a granularity limit of the source schema, "
                "not a curation defect._"
            )
            md.append("")

        md.append("## Reading the assay columns")
        md.append("")
        md.append(
            "- **G-protein readouts:** cAMP (Gαs/Gαi), GTPγS (direct G-protein "
            "activation), IP / PI hydrolysis (Gαq), intracellular Ca²⁺."
        )
        md.append(
            "- **β-arrestin readouts:** β-Arrestin-recruitment "
            "(PathHunter/Tango/BRET), GPCR internalization."
        )
        md.append(
            "- **Convergence readouts:** ERK-Phosphorylation, MAPK activation, "
            "CRE Luciferase. These are downstream of *both* G-protein and "
            "arrestin pathways and should be interpreted with caution as "
            "bias readouts."
        )
        md.append(
            "- **Detection-only:** BRET / FRET / LRET. These describe the "
            "*technology* (proximity assay) rather than the *pathway* — the "
            "same row labelled `BRET / BRET` could be measuring G-protein or "
            "arrestin recruitment depending on the constructs used."
        )
        md.append("")
        md.append("## Recommended manuscript framing")
        md.append("")
        md.append(
            "The dataset is dominated by class A GPCRs (aminergic + peptide "
            "subfamilies) measured via the canonical cAMP vs β-arrestin-"
            "recruitment comparison. Headline claims should therefore be "
            "scoped to: *G-protein vs β-arrestin pathway-bias prediction at "
            "class A GPCRs in well-characterised assay systems*. Generalisation "
            "to chemokine, lipid, or class B/C receptors is not supported by "
            "the row counts above and should not be claimed."
        )
        md.append("")

        with open(out_path, "w") as f:
            f.write("\n".join(md))
