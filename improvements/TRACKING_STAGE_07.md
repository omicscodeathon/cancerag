# Stage 07 — Dataset assembly tracking

## Original state (pre-rebuild)
`dataset_assembly.py` (665 lines) joined ligands, dockings, receptor descriptors, and bias labels with three Tier-1 defects. (1) `df[col] = df[col].fillna(-5.0)` (line 371) silently injected a plausible weak-binder Vina affinity wherever docking actually failed — the model had no way to distinguish "Vina failed" from "Vina succeeded with weak affinity." (2) `df["primary_bias_label"] = df["primary_bias_label"].fillna("unknown")` (line 345) created a phantom 6th class whose downstream fate was unclear. (3) Long `fillna` chains (lines 261-322) cascaded across potential label columns so label conflicts were resolved by fillna order, not by curated rule, with no log of resolutions. There was no stable composite `pair_key`, no row-level `sample_weight`, no Pandera schema, no temporal-holdout split, no scaffold/receptor split persistence, and no `.meta.json` provenance.

## Upgrades performed
- **Removed all `fillna(-5.0)` and `fillna("unknown")`**; missing docking values stay NaN with paired `_missing` indicator columns (LightGBM/XGBoost handle NaN natively).
- Stable `pair_key` composite (`inchikey :: receptor_uniprot :: bias_pathway :: assay_1 :: assay_2 :: reference_ligand`) asserted unique across the dataset.
- **615 rows total: 443 train-eligible + 172 temporal holdout** (year ≥ 2018 cutoff). Holdout written to `data/holdout/dataset_holdout.parquet`; the trainer raises if asked to read that path.
- **Four split strategies pre-computed** at assembly time and persisted in `ml_splits.json`: scaffold-grouped K-fold (Murcko, primary metric), receptor-grouped K-fold (cross-target generalization), stratified K-fold (in-distribution baseline), and temporal holdout.
- `sample_weight` column derived from `compute_sample_weight("balanced") × evidence_year_weight × docking_confidence_weight`.
- **Pandera schema validation** (`dataset_schema.py`) enforced on every assembled dataset; schema drift produces a loud failure.
- Reviewer-facing `dataset_assembly_audit.md` and full provenance sidecar (`ml_ready_dataset.meta.json`) recording input SHA-256s, label distribution, per-receptor counts, and holdout cutoff.

## What was NOT done (deferred / out-of-scope)
- Continuous bias-factor target column (`bias_factor_log_ratio`) preserved as alternative regression target — categorical only.
- DUD-E / property-matched decoy set for ROC-under-active-vs-decoy framing.
- Auto-generated `dataset_audit.html` interactive report.
- Per-row evidence-weighting tuned by assay type / journal IF / replicate count beyond the simple year + balance + docking-confidence product.
- Explicit per-row label-resolution log (`label_resolution_log.csv`) — current code resolves conflicts implicitly and the Stage 02 composite-key dedup left zero within-key conflicts to resolve.

## Files produced
- `data/processed/ml_ready_dataset.parquet` — 443 × 2872 (train-eligible)
- `data/holdout/dataset_holdout.parquet` — 172 × 2872 (temporal holdout, trainer-blocked)
- `data/processed/ml_splits.json` — pre-computed scaffold/receptor/stratified/temporal split assignments
- `data/processed/ml_ready_dataset.meta.json` — provenance sidecar
- `data/processed/dataset_assembly_audit.md` — reviewer-facing audit
- `src/cancerag/ml/dataset_schema.py` — Pandera schema (new)

## Tests
- `tests/unit/test_dataset_assembly.py` (186 lines)
- `tests/unit/test_schemas.py`

## Status
Complete (Done) — 615 rows joined cleanly; no leakage paths left into ML preprocessing.
