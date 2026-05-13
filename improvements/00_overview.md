# 00 — Overview, Risk Ranking, Reviewer Mapping

## Executive summary

CancerAg's scientific framing (biased agonism for GPCRs in oncology) is sound and the engineering surface (data collection → curation → docking → ML → inference app) is comprehensive. However, the pipeline currently has:

- **One existential scientific flaw** (the "unbiased" label is a source-assignment artifact, not a measurement) that no reviewer caught but that makes the entire ML problem ill-posed.
- **Several leakage paths** in the ML preprocessing that inflate reported performance independently of the random/scaffold split issue Reviewer 2 flagged.
- **Brittle external-data joins** (positional CSV columns, first-hit ChEMBL target search, ad-hoc PDB scoring) that break silently across source updates.
- **A docking pipeline run below default parameters** (`exhaustiveness=4`, `num_modes=3`) with no re-docking validation, against a single conformation per receptor with no activation-state filter.
- **An inference app that disables docking at predict time** (`enable_docking=False`) while the trained model expects docking-derived features — potentially a deployment-time correctness bug.

The reviewers are correct on every substantive point. This folder makes their critiques actionable and adds the deeper scientific and engineering issues they did not see.

## Mapping reviewer comments to stage documents

| Reviewer comment | Stage doc(s) |
|---|---|
| R1.1 — Introduction is biology-heavy, not method-positioned | manuscript-level (not in this folder) |
| R1.2 — Novelty / contribution unclear | [10_model_training_eval.md](10_model_training_eval.md), [00_overview.md](00_overview.md) |
| R1.3 — Descriptor-based featurization vs graph/3D | [06_featurization.md](06_featurization.md) |
| R1.4 — Docking quality not assessed | [04_binding_site_definition.md](04_binding_site_definition.md), [05_docking.md](05_docking.md) |
| R1.5 — Test set too small, CI too wide | [08_ml_preprocessing.md](08_ml_preprocessing.md), [10_model_training_eval.md](10_model_training_eval.md) |
| R1.6 — Final model selection unclear | [10_model_training_eval.md](10_model_training_eval.md), [12_inference_deployment.md](12_inference_deployment.md) |
| R1.7 — No SOTA baselines | [06_featurization.md](06_featurization.md), [10_model_training_eval.md](10_model_training_eval.md) |
| R1.8 — SHAP without scientific synthesis | [11_interpretability.md](11_interpretability.md) |
| R1.9 — Writing, typos, citation hygiene | manuscript-level |
| R2.1 — Need scaffold / temporal split | [08_ml_preprocessing.md](08_ml_preprocessing.md) |
| R2.2 — Expand test set / nested CV | [08_ml_preprocessing.md](08_ml_preprocessing.md), [10_model_training_eval.md](10_model_training_eval.md) |
| R2.3 — External validation | [01_data_collection.md](01_data_collection.md), [07_dataset_assembly.md](07_dataset_assembly.md) |
| R2.4 — Median imputation across receptor families is unsound | [07_dataset_assembly.md](07_dataset_assembly.md), [08_ml_preprocessing.md](08_ml_preprocessing.md) |
| R2.5 — Lead with macro-F1, not accuracy | [10_model_training_eval.md](10_model_training_eval.md) |
| R2.6 — Trim unsupervised clustering section | manuscript-level |
| R2.7 — Temper "precision medicine" claims | manuscript-level |

## Existential issues to fix before any resubmission

These are the issues that, if left in place, will cause re-rejection regardless of how many other fixes are made.

| ID | Issue | Stage doc |
|---|---|---|
| E1 | "Unbiased" label is a source-assignment artifact (every ChEMBL row hardcoded to `bias_category="Agonist"`). Model learns BiasDB-vs-ChEMBL provenance, not bias. | [02_ligand_curation.md](02_ligand_curation.md), [07_dataset_assembly.md](07_dataset_assembly.md) |
| E2 | Imputation, constant-column pruning, and feature selection run on the full `X` before train/test split. Textbook leakage. | [08_ml_preprocessing.md](08_ml_preprocessing.md), [09_feature_selection.md](09_feature_selection.md) |
| E3 | Random stratified split with `test_size=0.05` (26 samples) on a 5-class problem with 15:1 imbalance. | [08_ml_preprocessing.md](08_ml_preprocessing.md) |
| E4 | No scaffold / receptor-grouped split. Congeneric series straddle train and test. | [08_ml_preprocessing.md](08_ml_preprocessing.md) |
| E5 | No external validation set held out at curation time. | [01_data_collection.md](01_data_collection.md), [07_dataset_assembly.md](07_dataset_assembly.md) |
| E6 | `dataset_assembly.py:371` hardcodes `-5.0` kcal/mol as a "default docking score" for failed Vina runs — fake binding affinities silently injected. | [07_dataset_assembly.md](07_dataset_assembly.md) |
| E7 | Inference app runs with `enable_docking=False` while the model was trained with docking-derived features. Predict-time feature distribution likely does not match train-time. | [12_inference_deployment.md](12_inference_deployment.md) |
| E8 | No re-docking validation (RMSD of crystal ligand redocked into its own receptor). Docking-derived features cannot be trusted without this. | [05_docking.md](05_docking.md) |

## Risk ranking (highest impact first)

1. **E1, E6, E7** — these change *what the model is actually doing*; without fixing them no result is interpretable.
2. **E2, E4, E5** — leakage / generalization; without fixing them no metric is interpretable.
3. **E3, E8** — confidence intervals and pose quality; without these the metrics that exist are too noisy to defend.
4. Everything else in this folder — rigor, reproducibility, FAIR compliance.

## Recommended order of work

Phase 1 (unblocks everything else, ≈ 1–2 weeks):
- [01](01_data_collection.md) — UniProt-anchored receptor registry; schema-validated retrievers; meta sidecars.
- [02](02_ligand_curation.md) — ChEMBL Structure Pipeline; InChIKey-14 dedup; PU framing for ChEMBL set.
- [07](07_dataset_assembly.md) — remove `-5.0` and `"unknown"` hardcodes; per-family imputation; temporal holdout.

Phase 2 (rebuilds the science, ≈ 2–4 weeks):
- [03](03_receptor_curation.md) + [04](04_binding_site_definition.md) — GPCRdb-curated structures; P2Rank/DeepPocket pocket prediction; pLDDT gating.
- [05](05_docking.md) — re-dock validation; Vina exhaustiveness ≥ 16; pose-ensemble features; protein prep audit.
- [06](06_featurization.md) — interaction fingerprints (ProLIF); collinearity filter; SMILES-only and graph-based baselines.

Phase 3 (rebuilds the ML, ≈ 1–2 weeks):
- [08](08_ml_preprocessing.md) — `sklearn.Pipeline`-ize all preprocessing; scaffold + receptor-grouped splits; temporal holdout enforcement.
- [09](09_feature_selection.md) — stability selection; nested CV.
- [10](10_model_training_eval.md) — single locked model selection rule; bootstrap CIs on macro-F1; per-receptor confusion matrices; SOTA baselines.

Phase 4 (publication readiness, ≈ 1 week):
- [11](11_interpretability.md) — SHAP with mechanistic synthesis; stability of top features.
- [12](12_inference_deployment.md) — train/predict feature parity; calibration; applicability domain; model card.

## Non-goals of this folder

- Manuscript rewriting (writing, citations, figure layout). Reviewer 1 point 9 is real but out of scope here.
- Wet-lab validation suggestions. The reviewers correctly note this is needed; the docs here only address the computational pipeline.
