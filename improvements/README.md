# CancerAg — Pipeline Improvements

This folder contains a stage-by-stage critique of the CancerAg pipeline and the concrete improvements required to bring it to a publication-grade, reproducible, FAIR-compliant standard.

It was written in response to:
- The Scientific Reports rejection (Submission ID `680e295f-c1ea-4c8f-b59d-6cca4647f574`, 23 March 2026).
- A code-grounded internal audit of `src/cancerag/`, `inference_app/`, and `configs/`.

Each document follows the same structure:

1. **What the code does today** (with `file:line` references).
2. **Problems** (ranked by impact — Tier 1 = invalidates results; Tier 2 = compromises rigor; Tier 3 = polish).
3. **Standard approach** (what the community does).
4. **Concrete fixes** (code-level changes).
5. **Acceptance criteria** (how we know the fix landed).

## Index

| # | Stage | Document |
|---|---|---|
| 00 | Overview, risk ranking, reviewer mapping | [00_overview.md](00_overview.md) |
| 01 | Data collection (BiasDB, ChEMBL, PDB, AlphaFold) | [01_data_collection.md](01_data_collection.md) |
| 02 | Ligand curation & standardization | [02_ligand_curation.md](02_ligand_curation.md) |
| 03 | Receptor curation (PDB cleaning, structure selection) | [03_receptor_curation.md](03_receptor_curation.md) |
| 04 | Binding-site / pocket definition | [04_binding_site_definition.md](04_binding_site_definition.md) |
| 05 | Molecular docking (AutoDock Vina) | [05_docking.md](05_docking.md) |
| 06 | Featurization (ligand + receptor descriptors) | [06_featurization.md](06_featurization.md) |
| 07 | Dataset assembly & labelling | [07_dataset_assembly.md](07_dataset_assembly.md) |
| 08 | ML preprocessing (splits, imputation, scaling) | [08_ml_preprocessing.md](08_ml_preprocessing.md) |
| 09 | Feature selection (Boruta) | [09_feature_selection.md](09_feature_selection.md) |
| 10 | Model training, CV, evaluation, metrics | [10_model_training_eval.md](10_model_training_eval.md) |
| 11 | Interpretability (SHAP) | [11_interpretability.md](11_interpretability.md) |
| 12 | Inference app & deployment | [12_inference_deployment.md](12_inference_deployment.md) |

## How to read this folder

- Start with [`00_overview.md`](00_overview.md) for the executive summary, the existential issues, and the recommended order of work.
- The remaining documents can be read independently per stage. Each is self-contained with file/line references.
- Tier 1 issues across all docs are summarized in `00_overview.md` under "Existential issues to fix before any resubmission."

## Conventions

- "Standard" means established practice in cheminformatics / structural-bioinformatics literature published in journals at or above the level we're targeting (Nature family, JCIM, J. Chem. Inf. Model., Bioinformatics, JCAMD).
- "Reviewer X" refers to the Scientific Reports reviewer numbering in the rejection letter.
- All file paths are relative to repo root unless absolute.
