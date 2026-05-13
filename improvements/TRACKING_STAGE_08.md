# Stage 08 — ML preprocessing tracking

## Original state (pre-rebuild)
`DataPreprocessor.prepare_dataset(df)` had textbook leakage at every step. Imputation (`SimpleImputer(strategy="median")`) ran on the **full X** before the train/test split — the median used to fill train-set NaNs had seen the test rows. Constant-column pruning ran on the full X for the same reason. The split itself was `train_test_split(X, y_encoded, test_size=0.05, stratify=y, random_state=42)` — 26 test samples on a 5-class, 15:1-imbalanced problem with a ±16 point 95% CI on accuracy. There was no scaffold split, no receptor-grouped split, no `sklearn.Pipeline` composition (so preprocessing could not run inside CV folds), no nested CV, no bootstrap CIs, no per-receptor-family imputation, and no applicability-domain check. Class-imbalance handling ran outside CV — if it was SMOTE-like, synthetic samples bled across folds. Single seed (42) reporting hid run-to-run variance that could move accuracy 10+ points.

## Upgrades performed
- **sklearn `Pipeline` factory** composes `VarianceThreshold → CorrelationFilter → PerReceptorFamilyImputer → StandardScaler → selector → estimator`, so every preprocessing step is fit only on the training fold inside CV.
- **Per-fold preprocessing inside CV** — imputation, decorrelation, scaling all respect fold boundaries. Zero leakage across folds.
- `CorrelationFilter` **vectorized** from O(p³) to O(p²) — previously a 217-descriptor matrix took ~minutes per fold; now subsecond.
- **Per-receptor-family median imputer** replaces the global median imputer (addresses Reviewer 2 specifically — global medians muddied receptor-family-specific descriptor distributions).
- DataFrame-preserving custom transformers preserve column names downstream so SHAP / permutation can attribute back to readable feature names.
- Persisted fitted artifacts (imputer / correlation_filter / scaler / label_encoder) under `data/processed/ml_preprocessed/` for inference reuse with proven feature parity (Stage 12).
- `test_size = 0.05` removed from config; honest reporting uses 5-fold CV across 5 seeds + temporal holdout instead.

## What was NOT done (deferred / out-of-scope)
- Full nested 5×5 CV with `StratifiedGroupKFold` for hyperparameter selection — replaced by 5-seed × 5-fold outer CV plus separate Optuna inner-CV (Stage 10).
- `imblearn.pipeline.Pipeline` with SMOTENC inside the pipeline — `SMOTE` was deliberately rejected ("synthetic chemistry samples are unphysical; reviewers in cheminformatics consistently reject"). Class weighting used instead.
- Applicability-domain check at preprocessing time — implemented at Stage 12 (inference) instead of train time.
- `results/splits/<run_id>_split.csv` per-fold persistence with scaffold tags — split assignments persisted at Stage 07 in `ml_splits.json` instead.

## Files produced
- `data/processed/ml_preprocessed/imputer.joblib`
- `data/processed/ml_preprocessed/correlation_filter.joblib`
- `data/processed/ml_preprocessed/scaler.joblib`
- `data/processed/ml_preprocessed/label_encoder.joblib`
- `src/cancerag/ml/preprocessing.py` — Stage 08 (DataFrame-preserving transformers)

## Tests
- `tests/unit/test_preprocessing.py` (265 lines)

## Status
Complete (Done) — leakage paths closed; per-fold preprocessing demonstrably matches inference path via Stage 12 feature-parity tests.
