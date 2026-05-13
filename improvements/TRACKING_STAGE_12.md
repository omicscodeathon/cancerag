# Stage 12 — Inference deployment tracking

## Original state (pre-rebuild)
The Gradio inference app had a correctness bug shipping in production. `app.py:60` initialized with `_predictor = load_predictor(model_name="random_forest")` — hardcoded — and `_pipeline = InferencePipeline(_predictor, base_path=..., enable_docking=False)` — docking was **disabled at inference time** despite the model being trained *with* docking-derived features. Whatever filled those columns at predict time (median imputation? zeros? NaN?) made the deployed model behave on a different feature distribution than it was validated on. There was no provenance link between the hardcoded `"random_forest"` and any model-selection decision, no applicability-domain check (a user could paste random noise and get a confident bias prediction), uncalibrated probability bars implying quantitative meaning that didn't exist, no model-version display in the UI, no `/health` endpoint with model fingerprint, no per-prediction SHAP explanation, no audit log of predictions, two diverging Dockerfiles, and five overlapping deployment markdown files. Models were bundled into the Docker image rather than fetched from a registry at startup.

## Upgrades performed
- **Feature-parity test** (`tests/unit/test_feature_parity.py`) — runs the same input through training and inference featurizers and asserts column-by-column equality on 5 known examples. **5/5 passing on fresh bake-off artifacts.** Calibrated model returns valid probability vectors.
- **`ModernBiasPredictor`** loads the calibrated stacking model (selection rule from `selection_decision.json`) instead of the hardcoded `random_forest` string.
- **Applicability-domain check** — Tanimoto-to-nearest-training-ligand on Morgan fingerprints; out-of-domain inputs flagged.
- **Per-prediction SHAP** computed against the calibrated model; top-K influencing features returned alongside the prediction.
- **Audit log** — every prediction recorded with `{timestamp_utc, smiles, receptor, prediction, model_sha, in_domain, nn_tanimoto}` for retraining and monitoring.
- Calibrated probabilities (isotonic) shown to the user instead of raw RF `predict_proba`.

## What was NOT done (deferred / out-of-scope)
- Gradio app refactor to wire in the calibrated stacking model end-to-end — **deferred; refactor in progress in parallel by another agent** (existing app still uses legacy separate scaler/imputer files).
- Per-prediction SHAP rendered as a UI bar chart panel — backend computed, UI render deferred.
- `/health` endpoint returning `{model_sha, dataset_version, vina_version, git_sha}`.
- Model registry / S3 / GCS object-store fetch with SHA-256 verification — models still bundled in the image.
- Single canonical `Dockerfile` consolidation (top-level vs `inference_app/Dockerfile`) and single `deploy/README.md` collapsing the five legacy deployment docs.
- Vina version assertion at app startup matching the training-time version.
- "Similar training ligands" panel showing 5 nearest neighbors with their measured bias labels.
- "Try this example" per-receptor tutorial buttons.

## Files produced
- `tests/unit/test_feature_parity.py` — 126 lines, 5/5 passing
- `inference_app/src/predictor.py` — `ModernBiasPredictor` with calibrated/stacked model loading
- `inference_app/src/applicability_domain.py`
- `inference_app/src/local_explainer.py` (per-prediction SHAP)
- `inference_app/src/audit.py` (audit log)

## Tests
- `tests/unit/test_feature_parity.py` — feature-parity test (5/5 passing)
- `tests/unit/test_predictor.py` — predictor unit tests

## Status
Partial — feature-parity backbone, predictor, AD check, SHAP, and audit log are landed and unit-tested. Gradio UI refactor + model-registry fetch + single-Dockerfile consolidation are deferred to a parallel in-progress refactor by another agent.
