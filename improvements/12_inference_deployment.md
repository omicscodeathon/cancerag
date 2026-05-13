# 12 — Inference App & Deployment

Modules covered:
- `inference_app/app.py`
- `inference_app/src/inference_pipeline.py`
- `inference_app/src/predictor.py`
- `inference_app/src/docking_extractor.py`
- `inference_app/src/receptor_manager.py`
- `inference_app/Dockerfile` and root `Dockerfile`
- `inference_app/RENDER_DEPLOYMENT.md`, `DOCKER_DEPLOYMENT.md`, `DEPLOYMENT_FIXES_SUMMARY.md`, `DOCKING_FIX.md`

## What the code does today

- Gradio app at `inference_app/app.py`.
- `initialize_app()` (line 60) constructs:
  - `_predictor = load_predictor(model_name="random_forest")` — hardcoded model name.
  - `_pipeline = InferencePipeline(_predictor, base_path=..., enable_docking=False)` — **docking is disabled at inference time.**
- `_receptor_manager`, `_visualizer`, `_result_visualizer`, `_docking_extractor` initialised.
- Recent commits focused on Docker/Vina runtime fixes (`b62ed92`, `2cec367`, `729cc5b`, `4316ce6`) — the deployment has been a moving target.

## Problems

### Tier 1 — correctness

**P12.1 (E7) Train/inference feature parity may be broken.**
`enable_docking=False` at inference (`app.py:76`) means docking-derived features are not computed at predict time. But the model was trained *with* those features. What fills them at inference?
- If they are imputed (e.g., median): the deployed model is effectively descriptor-only at inference, and its predictions do not reflect what was validated.
- If they are zero-filled or NaN-filled: the prediction is garbage relative to the training distribution.
- If they are dynamically computed but the recent commit `b62ed92` "uses pre-converted receptors" — this means docking *does* run for some receptors but with a different pipeline than training.

This is a **correctness bug shipping today**, not a future improvement. It must be audited and fixed before any user trusts the predictions.

**P12.2 Hardcoded `"random_forest"` in `load_predictor`.**
`app.py:74`. There is no provenance link between this string and `results/model_selection_decision.json` ([10_model_training_eval.md](10_model_training_eval.md) F10.1). If the manuscript reports XGBoost as the chosen model and the app loads RF, results don't match.

**P12.3 No applicability-domain check.**
A user can paste *any* SMILES — propranolol, caffeine, or random noise — and the app returns a confident bias prediction. Standard cheminformatics: reject or flag predictions outside the training distribution (Tanimoto < 0.4 to nearest neighbor).

**P12.4 Probability bars shown to user are uncalibrated.**
RF `predict_proba` is uncalibrated; the bar widths in the UI imply quantitative meaning that doesn't exist.

**P12.5 No model-version display in UI.**
A user sees a prediction with no information about which model, which dataset version, which docking config. Required for any user actually relying on the result.

### Tier 2 — engineering / reproducibility

**P12.6 Two Dockerfiles diverging.**
Top-level `Dockerfile` (uncommitted) and `inference_app/Dockerfile`. Drift inevitable.

**P12.7 Models bundled in the image.**
`Dockerfile` (lines copying `random_forest.pkl`, `xgboost.pkl`, etc.) bakes models into the image. Any model retraining requires a full rebuild. Standard practice: pull from a model registry (MLflow, S3, GCS) at startup or via a model-name env var.

**P12.8 Vina installed at image build but not pinned in features.**
The Dockerfile installs Vina 1.2.5; the training pipeline used some version (probably the same — but not asserted). Inference vs train Vina-version drift would change docking-derived features.

**P12.9 Multiple deployment docs without canonical source.**
`DEPLOYMENT_FIXES_SUMMARY.md`, `DOCKER_DEPLOYMENT.md`, `RENDER_DEPLOYMENT.md`, `DOCKING_FIX.md`, `UI_IMPROVEMENT_PLAN.md` — five docs describing what should be one. Docs drift; tests don't.

**P12.10 No health endpoint with model fingerprint.**
A typical `/health` returns model SHA, dataset version, vina version, prep-pipeline version. None present.

**P12.11 No request logging / monitoring.**
Predictions are made without recording: which SMILES, which receptor, which prediction, which version. For a tool that affects research decisions, this audit trail is required (and trivial to add).

**P12.12 Path manipulation via `sys.path.insert`.**
`app.py:26`. Brittle and surprises CI / containerized environments.

**P12.13 Global mutable state for components.**
Lines 51-57. `_predictor`, `_pipeline`, etc. as module-level globals. Hard to test; no isolation between requests.

### Tier 3 — UX / scientific communication

**P12.14 No "this prediction is uncertain" UI affordance.**
Users will treat all predictions as equally confident. The UI should:
- Show calibrated probabilities with explicit "uncertain" badge for low max-class probability.
- Show AD flag prominently when the input is out-of-distribution.
- Show top-K most influential features (SHAP local explanation) per prediction.
- Show similar training ligands and their measured bias labels for comparison.

**P12.15 "Drug-likeness analysis" added but not integrated with the prediction.**
Recent commit `044075c` added drug-likeness; should be a separate panel labelled clearly as different from bias prediction (or the user will conflate them).

**P12.16 No example workflow / tutorial in the UI.**
Inference apps benefit hugely from a "Try this example" button per receptor.

## Standard approach

1. **Train/inference feature parity is non-negotiable.** Same featurizer, same prep pipeline, same Vina version. Audited via a test that runs both and asserts column-by-column equality on a held-out example.
2. **Model selection is provenance-driven**, not hardcoded. App reads the chosen model name from `results/model_selection_decision.json`.
3. **Calibrated probabilities** with applicability-domain flag.
4. **Per-prediction explainability** (SHAP local) shown in UI.
5. **Health / fingerprint endpoint** for ops.
6. **Single deployment doc** consolidating the five.
7. **Request log** for audit and future retraining data.

## Concrete fixes

### F12.1 Audit train/inference feature parity (E7)

```python
# tests/test_inference_parity.py
def test_inference_features_match_training(known_smiles, known_receptor):
    """Run the same input through the training featurizer and the
    inference featurizer; assert equality on every column."""
    train_features = TrainingFeaturizer().featurize(known_smiles, known_receptor)
    infer_features = InferenceFeaturizer().featurize(known_smiles, known_receptor)
    pd.testing.assert_series_equal(train_features, infer_features,
                                   check_names=True)
```

If this test fails, *the deployed model is producing predictions on a different feature distribution than it was trained on*. Block release on test failure.

### F12.2 Read model selection from artifact

```python
# inference_app/src/predictor.py
def load_predictor(decision_path="results/model_selection_decision.json"):
    decision = json.loads(Path(decision_path).read_text())
    model_name = decision["chosen"]
    model = joblib.load(f"results/models/{model_name}.pkl")
    return CalibratedPredictor(model, decision_meta=decision)
```

Remove the `model_name="random_forest"` default in `app.py:74`.

### F12.3 Applicability-domain in the inference path

```python
# inference_app/src/applicability_domain.py
class ApplicabilityChecker:
    def __init__(self, train_fps, threshold=0.4):
        self.train_fps = train_fps
        self.threshold = threshold
    def check(self, smiles) -> dict:
        fp = morgan_fp(Chem.MolFromSmiles(smiles))
        nn_sim = max(DataStructs.TanimotoSimilarity(fp, f) for f in self.train_fps)
        return {"in_domain": nn_sim >= self.threshold,
                "nearest_neighbor_tanimoto": nn_sim}
```

Render an explicit "Out-of-distribution input — prediction unreliable" banner if `in_domain=False`.

### F12.4 Calibrated probabilities + UI affordances

- Use the calibrated model from [10_model_training_eval.md](10_model_training_eval.md) F10.5.
- Show probability with uncertainty: "G-biased: 64% (95% CI: 51–76% from bootstrap calibration)."
- Show a confidence badge: green / yellow / red based on max-class probability and AD flag.

### F12.5 Per-prediction SHAP

```python
# inference_app/src/local_explainer.py
class LocalExplainer:
    def __init__(self, model, background_X):
        self.explainer = shap.TreeExplainer(model, background_X)
    def explain(self, x_row, top_k=5):
        sv = self.explainer.shap_values(x_row)
        # multi-class -> pick the predicted-class shap vector
        ...
        return [{"feature": f, "shap": s} for f, s in top_features]
```

Render as a horizontal bar chart in the result panel: "Top features driving this prediction."

### F12.6 Health endpoint + fingerprint

```python
@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_name": _predictor.name,
        "model_sha256": _predictor.sha,
        "dataset_version": DATASET_VERSION,
        "vina_version": VINA_VERSION,
        "git_sha": GIT_SHA,
        "started_at_utc": STARTED_AT,
    }
```

### F12.7 Single Dockerfile (top-level)

Delete `inference_app/Dockerfile`. Keep root `Dockerfile`. Update CI to build only the root one. Consolidate deployment docs into one `deploy/README.md`.

### F12.8 Model registry instead of bundled models

```dockerfile
# Don't bundle:
# COPY results/models/random_forest.pkl /app/results/models/random_forest.pkl
# Instead at runtime:
ENV MODEL_URI=s3://cancerag-models/v1.2.3/
```

```python
# At app start:
def fetch_model(uri):
    sha_expected = os.environ["MODEL_SHA256"]
    blob = download(uri)
    assert sha256(blob) == sha_expected
    return load(blob)
```

### F12.9 Request logging

```python
# inference_app/src/audit.py
def log_prediction(smiles, receptor, prediction, model_meta, ad_meta):
    record = {"timestamp_utc": now(), "smiles": smiles, "receptor": receptor,
              "prediction": prediction, "model_sha": model_meta["sha"],
              "in_domain": ad_meta["in_domain"], "nn_tanimoto": ad_meta["nn_tanimoto"]}
    AUDIT_LOG.write(json.dumps(record) + "\n")
```

This becomes the data source for future retraining and for "model behaviour over time" monitoring.

### F12.10 Refactor globals into a service object

```python
@dataclass
class InferenceService:
    predictor: Predictor
    pipeline: InferencePipeline
    receptor_manager: ReceptorManager
    visualizer: MolecularVisualizer
    explainer: LocalExplainer
    ad_checker: ApplicabilityChecker

def get_service() -> InferenceService:
    if not hasattr(get_service, "_inst"):
        get_service._inst = _build_service()
    return get_service._inst
```

Removes the 6 module-level globals at `app.py:51-57`.

### F12.11 Pin Vina version to training

```python
# At app startup:
EXPECTED_VINA = "1.2.5"
actual = subprocess.check_output(["vina", "--version"]).decode().strip()
if EXPECTED_VINA not in actual:
    raise RuntimeError(f"Vina version mismatch: expected {EXPECTED_VINA}, got {actual}")
```

### F12.12 UI: similar-training-ligands panel

```python
# Show the 5 nearest training ligands by Tanimoto, with their measured bias labels.
# This gives the user manual sanity check: "the prediction is G-biased and the 5
# most similar training ligands were also G-biased."
```

### F12.13 Single deployment doc

Replace `DEPLOYMENT_FIXES_SUMMARY.md`, `DOCKER_DEPLOYMENT.md`, `RENDER_DEPLOYMENT.md`, `DOCKING_FIX.md`, `UI_IMPROVEMENT_PLAN.md` with one `deploy/README.md` that documents the canonical build / run / debug flow, and deletes the others (or moves them under `deploy/historical/` with a header noting they're superseded).

## Acceptance criteria

- [ ] An automated test asserts feature-vector equality between training and inference paths for ≥ 5 known examples per receptor.
- [ ] `enable_docking` flag is removed from `InferencePipeline`; docking either runs (matching training) or the model was retrained without docking features (one or the other, not both).
- [ ] No hardcoded model name in the app; chosen model is loaded from `results/model_selection_decision.json`.
- [ ] Applicability-domain check runs on every prediction; "out of domain" predictions are flagged in the UI.
- [ ] Predicted probabilities come from a calibrated model.
- [ ] UI shows per-prediction top SHAP features and the 5 nearest training ligands with their labels.
- [ ] `/health` endpoint returns model SHA-256, dataset version, vina version, git SHA.
- [ ] Single canonical `Dockerfile` and single `deploy/README.md`.
- [ ] Models are loaded from a registry / object store with SHA verification — not bundled in the image.
- [ ] Audit log records every prediction with input, output, model fingerprint, AD flag.
- [ ] Vina version at runtime is asserted to match the version used at training time.
