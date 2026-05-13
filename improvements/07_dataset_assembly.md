# 07 — Dataset Assembly & Labelling

Module covered: `src/cancerag/ml/dataset_assembly.py`

## What the code does today

665 lines that:
1. Join unified ligand table with per-receptor docking results, receptor pocket descriptors, and BiasDB labels.
2. Compute summary columns (e.g., `receptor_count_calculated`).
3. Resolve label conflicts where the same ligand has multiple bias labels via `fillna` chains (lines 261-262, 315, 322).
4. Set `primary_bias_label` with `fillna("unknown")` (line 345).
5. Fill missing docking scores with `-5.0` (line 371).
6. Write the final CSV consumed by ML preprocessing.

## Problems

### Tier 1 — invalidate the label vector

**P7.1 (E6) `fillna(-5.0)` for missing docking scores.**
Line 371: `df[col] = df[col].fillna(-5.0)`. This is **not imputation** — it injects a plausible-looking weak-binder affinity (-5 kcal/mol is a real Vina value for low-affinity poses) where the docking actually failed. The model cannot distinguish "Vina failed" from "Vina succeeded with weak affinity." See [05_docking.md](05_docking.md) F5.3 for the fix.

**P7.2 `fillna("unknown")` on the label column.**
Line 345: `df["primary_bias_label"] = df["primary_bias_label"].fillna("unknown")`. Are "unknown" rows used in training? If yes, the model has a 6th class it never gets evaluated on. If they are filtered downstream, the location is unclear. Either way the label vector is contaminated by silent class assignment.

**P7.3 `fillna` chains for label resolution.**
Lines 261-322 cascade `fillna(other_col)` across multiple potential label columns. This means if a ligand has both `bias_pathway = "G-protein"` and `bias_category = "Antagonist"` from different sources, one wins by fillna order, not by curated rule. There is no explicit conflict-resolution policy and no log of the resolutions made.

**P7.4 `receptor_count_calculated` becomes a feature.**
Line 261-262 computes `receptor_count` as a fillna chain. If this ends up in `X` (it's not in the metadata exclusion list in `preprocessing.py:287-308` — verify), then the model has direct access to "how many receptors this ligand was tested against in BiasDB" which is a strong proxy for *being in BiasDB at all*, i.e., a label leak.

**P7.5 No external/temporal holdout enforcement.**
Reviewer 2 explicitly required this. `dataset_assembly.py` writes one CSV; the ML pipeline reads it. There is no notion of "rows added after cutoff date X go to a separate holdout file the trainer cannot see."

### Tier 2 — community-norm gaps

**P7.6 No row-level confidence weights.**
BiasDB rows differ in evidence quality (assay type, replicate count, publication year, journal). The dataset has no `weight` column to let the model down-weight low-confidence rows.

**P7.7 No `(ligand, receptor)` pair-key.**
Without a stable composite key (`inchikey14::uniprot::assay_pair`), the same biological observation can appear multiple times across CSV rows or be silently deduplicated.

**P7.8 No per-receptor sample-count audit.**
A reviewer will ask: how many ligands per receptor? How many bias-labelled vs unlabelled? This summary should be a sidecar artifact, not something to recompute by hand.

**P7.9 Continuous bias values not preserved.**
If the curation layer ([02_ligand_curation.md](02_ligand_curation.md) F2.4) preserves the continuous bias factor, dataset assembly should keep it as an alternative target column (`bias_factor_log_ratio`). Today only the categorical label survives.

**P7.10 No "negatives / decoys" set.**
For docking sanity-checking and for AUC under the right framing, a decoy set (DUD-E, or property-matched molecules from ChEMBL with no GPCR activity) is standard. Not present.

### Tier 3 — engineering

**P7.11 665 lines, no modular separation.**
Joining, label resolution, and missing-value handling are intermixed. Hard to test individually.

**P7.12 No schema for the assembled dataset.**
There is no Pydantic / dataclass defining the columns of `unified_dataset.csv`. Downstream code (ML preprocessor, inference app) infers columns dynamically. Drift breaks silently.

**P7.13 No artifact provenance.**
The output CSV has no `.meta.json` recording: which inputs (and their SHA-256), which retriever versions, when assembled.

## Standard approach

1. **Explicit conflict-resolution rules** for label collisions, with a logged decision per row.
2. **Indicator columns for every imputed value**; never silently fill.
3. **Holdout enforcement** at assembly time, not at modelling time.
4. **Schema-validated output** with Pydantic / `pandera`.
5. **Sidecar provenance** matching [01_data_collection.md](01_data_collection.md) F1.4.
6. **Confidence weights per row** based on assay metadata.

## Concrete fixes

### F7.1 Remove the -5.0 hardcode (E6)

Replace line 371 with:
```python
# DO NOT impute. Mark explicitly.
for col in docking_score_cols:
    df[f"{col}_missing"] = df[col].isna().astype(int)
# Leave NaN. LightGBM and XGBoost handle NaN natively.
# For models that don't (RF, LogReg): impute inside the CV pipeline (see 08).
```

### F7.2 Drop "unknown" labels entirely

Replace line 345 with:
```python
n_before = len(df)
df = df[df["primary_bias_label"].notna()].copy()
logger.info(f"Dropped {n_before - len(df)} rows with no bias label")
```

If the "unknown" rows came from ChEMBL via the source-shortcut (E1), they should not exist after [02_ligand_curation.md](02_ligand_curation.md) F2.1.

### F7.3 Explicit conflict-resolution policy

```python
# ml/label_resolution.py
def resolve_bias_label(group: pd.DataFrame) -> tuple[str | None, str]:
    """For a group of rows with same (inchikey14, uniprot), return (label, decision)."""
    labels = group["bias_category"].dropna().unique()
    if len(labels) == 0: return None, "no_label"
    if len(labels) == 1: return labels[0], "unanimous"
    # Weighted by recency, journal IF, assay type — defined explicitly:
    weighted = group.groupby("bias_category").apply(_score_evidence)
    return weighted.idxmax(), f"resolved_by_evidence_weight: {weighted.to_dict()}"
```

Write decisions to `data/processed/label_resolution_log.csv`.

### F7.4 Stable composite key

```python
df["pair_key"] = (df["inchikey14"] + "::" + df["receptor_uniprot"]
                  + "::" + df["assay_1"].fillna("?")
                  + "::" + df["assay_2"].fillna("?"))
assert df["pair_key"].is_unique, "duplicate pair keys present"
```

### F7.5 Holdout enforcement

```python
# ml/dataset_assembly.py
HOLDOUT_YEAR_CUTOFF = 2024  # from config

def split_train_eligible_and_holdout(df: pd.DataFrame, cutoff: int) -> tuple[...]:
    train = df[df["year"] < cutoff].copy()
    holdout = df[df["year"] >= cutoff].copy()
    return train, holdout

train.to_csv("data/processed/dataset_train.csv", index=False)
holdout.to_csv("data/holdout/dataset_holdout.csv", index=False)
```

In ML preprocessing, assert at top:
```python
HOLDOUT_PATH = Path("data/holdout")
def load_dataset(path):
    if HOLDOUT_PATH in Path(path).parents:
        raise RuntimeError("Refusing to load holdout dataset for training")
```

### F7.6 Row-level confidence weights

```python
# ml/confidence_weights.py
def evidence_weight(row) -> float:
    w = 1.0
    if row["assay_type"] == "F": w *= 1.0       # functional baseline
    if row["assay_type"] == "B": w *= 0.5       # binding-only down-weighted
    if row["confidence_score"] >= 9: w *= 1.2
    if row["year"] >= 2020: w *= 1.1
    if pd.notna(row["replicate_n"]) and row["replicate_n"] >= 3: w *= 1.1
    return w

df["sample_weight"] = df.apply(evidence_weight, axis=1)
```

Use as `sample_weight` in `model.fit(X, y, sample_weight=df["sample_weight"])` and in cross-validation scoring.

### F7.7 Schema validation with `pandera`

```python
# ml/dataset_schema.py
import pandera as pa
from pandera.typing import Series

class DatasetSchema(pa.SchemaModel):
    pair_key: Series[str] = pa.Field(unique=True)
    inchikey14: Series[str]
    receptor_uniprot: Series[str]
    primary_bias_label: Series[str] = pa.Field(isin=["G_biased","Arrestin_biased","balanced","unbiased","partial"])
    vina_affinity: Series[float] = pa.Field(nullable=True)
    vina_affinity_missing: Series[int] = pa.Field(isin=[0,1])
    sample_weight: Series[float] = pa.Field(gt=0)
    # ... declared per column

DatasetSchema.validate(df, lazy=True)  # collect all errors at once
```

### F7.8 Sidecar provenance

`data/processed/dataset_train.csv.meta.json`:
```json
{
  "assembled_at_utc": "...",
  "input_artifacts": [
    {"path": "data/processed/unified_ligands.csv", "sha256": "..."},
    {"path": "results/docking_results/scores.csv", "sha256": "..."},
    {"path": "data/processed/binding_sites.json", "sha256": "..."}
  ],
  "row_count": 504,
  "label_distribution": {"G_biased": 120, "Arrestin_biased": 95, ...},
  "per_receptor_counts": {"HTR1A": 87, "DRD2": 64, ...},
  "holdout_cutoff_year": 2024,
  "schema_hash": "...",
  "label_resolution_decisions": "data/processed/label_resolution_log.csv"
}
```

### F7.9 Per-receptor audit report

Auto-generate `results/reports/dataset_audit.html` with:
- Per-receptor sample counts (bar chart).
- Label distribution per receptor (stacked bar).
- Class imbalance ratio per receptor.
- Years covered per receptor.
- Number of ligands flagged PAINS / non-Lipinski / etc. per receptor.

This is the artifact a reviewer can interrogate quickly.

### F7.10 Decoy set

```python
# ml/decoys.py
def build_decoys(receptor_actives: pd.DataFrame, n_per_active: int = 50) -> pd.DataFrame:
    """Property-matched decoys from ChEMBL, no GPCR activity."""
    # MW, LogP, HBD, HBA, RotB, formal charge match — DUD-E protocol
    ...
```

Use only for docking-quality assessment and for downstream ROC under "active vs decoy" framing — not as part of the bias-classification training set (avoid re-introducing source confound).

## Acceptance criteria

- [ ] No `fillna(-5.0)` anywhere in the assembly module.
- [ ] No `fillna("unknown")` on the label column.
- [ ] Every imputed numeric value has a paired `_missing` indicator column.
- [ ] `pair_key` is unique across the dataset.
- [ ] `dataset_train.csv` and `dataset_holdout.csv` are written to separate paths; the trainer rejects the holdout path.
- [ ] `label_resolution_log.csv` records every conflicting label and the decision.
- [ ] `sample_weight` column exists and is used in model training.
- [ ] Pandera schema validation passes; schema is checked into the repo.
- [ ] `dataset_train.csv.meta.json` exists with full provenance.
- [ ] Auto-generated `dataset_audit.html` is produced per assembly run.
