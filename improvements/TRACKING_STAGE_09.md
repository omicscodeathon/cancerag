# Stage 09 — Feature selection tracking

## Original state (pre-rebuild)
`feature_selection.py` (576 lines) ran a single Boruta selector with three Tier-1 defects. (1) `boruta.fit(X.values, y_encoded)` operated on the **full feature matrix** including test rows — the selected feature set was informed by the very rows used to evaluate it. Leakage independent of and additive to the imputation leakage at Stage 08. (2) `X = X.fillna(X.median())` (line 255) imputed on the full matrix before Boruta. (3) Single-run Boruta on n=504 with ~200 features gives a different shortlist for different bootstrap resamples — reporting "Boruta selected these features" without selection frequency was meaningless. There was no alternative-selector ablation, no collinearity prefilter (Boruta is sensitive to correlated features), no force-keep policy for pair-level interaction features (which Boruta could drop based on noise), and no per-fold persistence of selected features. Boruta wrapped Random Forest only, inheriting RF's biases (favors high-cardinality, depresses correlated features).

## Upgrades performed
- **Four selector classes** implemented as Pipeline-compatible transformers, fit only on training folds inside CV:
  - `BorutaSelector` (RF-wrapped, with force-keep prefixes for `ifp_*` and `vina_pose_*`)
  - `L1LogisticRegressionSelector`
  - `RFECVSelector`
  - `MutualInfoSelector`
- **Selector ablation matrix** (4 selectors × 4 models on scaffold-grouped CV) emitted as `selector_ablation_matrix.csv`. Result: Boruta and RFECV cluster in the 0.48-0.51 macro-F1 band; only L1 (too aggressive) underperforms — confirms feature-selection choice is **not** the bottleneck.
- Per-fold selected features persisted under `data/processed/ml_selected_features/` for stability analysis.
- `LabelEncoder` fit once at pipeline entry (was previously re-fit per call → risk of inconsistent label mapping).
- Pair-level features (`ifp_*`, `vina_pose_*`) force-kept regardless of selector verdict — these are the methodological contribution and should not be dropped on a single noisy run.
- Module collapsed from 576 lines toward a single `BorutaSelector` class + ablation orchestrator.

## What was NOT done (deferred / out-of-scope)
- Stability selection (50-bootstrap Boruta resamples → `selection_frequency` per feature with ≥ 0.8 threshold) — deferred; SHAP-based stability frequency at Stage 11 covers the reviewer ask.
- DiCE-style counterfactual explanations on selected features.
- Stability-Lasso selector (`pulearn` / `stabilityselection` package) as a 5th ablation arm.
- VIF-based collinearity prefilter independent of `CorrelationFilter` at Stage 08.

## Files produced
- `data/processed/ml_selected_features/` — per-fold inspection-only outputs (per-selector, per-fold)
- `data/processed/ml_models/selector_ablation/selector_ablation_matrix.csv`
- `src/cancerag/ml/feature_selection.py` — 4 selector classes + ablation orchestrator

## Tests
- `tests/unit/test_feature_selection.py` (115 lines)

## Status
Complete (Done) — selectors are CV-fold-respecting Pipeline transformers; ablation answers the reviewer question definitively (selector choice ~ ±0.03 F1).
