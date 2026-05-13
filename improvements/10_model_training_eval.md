# 10 — Model Training, CV, Evaluation, Metrics

Modules covered:
- `src/cancerag/ml/model_training.py`
- `src/cancerag/ml/model_evaluation.py`
- `src/cancerag/ml/ml_pipeline.py`
- `src/cancerag/ml/generate_final_report.py`

## What the code does today

- `model_training.py` (744 lines) trains LogReg, RF, XGBoost, LightGBM, CatBoost with hardcoded hyperparameters (e.g., RF `min_samples_split=10`, line 105).
- `model_evaluation.py` (615 lines) uses `StratifiedKFold(n_splits=cv, shuffle=True, random_state=...)` for CV (line 78).
- Reported metrics (per the manuscript) include accuracy 76.92% on a 26-sample test set — Reviewer 2 demanded macro-F1.
- The inference app loads `random_forest` by default — implying RF is "the" CancerAg model, but no documented selection rule.
- No comparison with SOTA or even a SMILES-only baseline.

## Problems

### Tier 1 — what's actually being measured

**P10.1 Accuracy as the headline on a 5-class, 15:1-imbalanced problem.**
Reviewer 2's primary complaint. Macro-F1, balanced accuracy, and per-class precision/recall are required.

**P10.2 No documented model-selection rule.**
The repo trains 5+ models, then the inference app picks RF (`inference_app/app.py:74`). There is no documentation of:
- The selection criterion (validation macro-F1? AUC? cost-weighted?).
- Whether the test set was used to pick — if so, that is leakage.
- Why RF over the boosted-tree models that usually outperform it.

This is exactly Reviewer 1 point 6.

**P10.3 No nested CV → reported test metric is inflated.**
With hyperparameter tuning + a single train/test split, the reported test metric is biased optimistically. Standard fix: nested CV (see [08_ml_preprocessing.md](08_ml_preprocessing.md) F8.4).

**P10.4 Hardcoded hyperparameters.**
`model_training.py:105-146` shows RF with `min_samples_split=10`, `min_samples_leaf=...` etc. These were tuned at some point but the tuning protocol isn't recorded. If they were tuned on the test set, results are invalidated.

**P10.5 No baselines.**
- No SMILES-only baseline (descriptors alone, no docking, no receptor features).
- No "majority class" / "stratified random" baselines.
- No SOTA comparison (ChemProp, Uni-Mol, MolBERT) — Reviewer 1 point 7.
- No comparison with a re-implementation of an older biased-agonist predictor (e.g., the descriptors used in Hauser et al. 2017 or similar).

**P10.6 Single-seed reporting.**
With small test sets, run-to-run variance dominates. Standard: run ≥ 5 seeds; report mean ± std.

**P10.7 No per-receptor breakdown.**
A reviewer will ask: does the model work uniformly across receptors, or is performance carried by 1-2 well-represented targets? Per-receptor confusion matrices and per-receptor macro-F1 are mandatory.

**P10.8 No probability calibration.**
RF and gradient-boosted trees produce uncalibrated `predict_proba`. The Gradio app shows probability bars to a user — these numbers are decorative without `CalibratedClassifierCV(method="isotonic", cv=5)`.

**P10.9 No multi-label or multi-task framing.**
A ligand can be biased toward G-protein at receptor X and balanced at receptor Y. Forcing a single `primary_bias_label` per ligand discards the multi-receptor evidence — a reframe to multi-label or multi-task could substantially improve F1 and is a defensible methodological contribution.

### Tier 2 — community-norm gaps

**P10.10 No proper learning-curve analysis.**
The pipeline doesn't report how performance scales with training set size. Standard: train on 25%, 50%, 75%, 100% of train and plot. Tells the reviewer whether more data would help.

**P10.11 No statistical comparison between models.**
Reporting "RF macro-F1 = 0.62, XGBoost macro-F1 = 0.61" without a paired-permutation test or bootstrap CI overlap is meaningless.

**P10.12 No out-of-distribution evaluation.**
Holdout receptors (entire receptor families never seen in training) test whether the model has learned biology vs. memorized receptor identity.

**P10.13 No external holdout reported.**
Reviewer 2 explicitly required this. See [01_data_collection.md](01_data_collection.md) F1.6 and [07_dataset_assembly.md](07_dataset_assembly.md) F7.5 for the data infrastructure; this stage must consume them and report.

### Tier 3 — engineering / reproducibility

**P10.14 744 lines for model training.**
Symptomatic of per-model boilerplate. A `dict[str, estimator_factory]` registry would reduce this to ~150 lines.

**P10.15 No model card.**
For each saved `.pkl`, the standard ML-engineering output is a `model_card.md` (training data hash, hyperparameters, train/val/test metrics, intended use, limitations). Absent.

**P10.16 No MLflow / W&B / DVC tracking.**
Every run produces ad-hoc CSVs in `results/`. Reproducibility is by version-control luck.

## Standard approach

1. **Lock the model selection rule** before training: "select the model with highest validation macro-F1 averaged over outer-CV folds."
2. **Lead with macro-F1**, balanced accuracy, AUC-ROC (one-vs-rest), AUC-PR (per class). Accuracy is supplementary.
3. **Bootstrap CIs and seed-variance** on every reported metric.
4. **Per-receptor confusion matrices** as a supplementary figure.
5. **At least 3 baselines**: majority-class, SMILES-only descriptors, and one GNN (ChemProp).
6. **Calibrated probabilities** via Platt or isotonic.
7. **Model card** per saved checkpoint.
8. **MLflow or DVC** for run tracking.

## Concrete fixes

### F10.1 Single locked model-selection rule

```python
# ml/model_selection.py
SELECTION_RULE = "max_outer_cv_macro_f1_mean"

def select_final_model(nested_cv_results: pd.DataFrame) -> str:
    """Return the model name with highest mean outer-CV macro-F1.
    Ties broken by lower std (more stable)."""
    summary = nested_cv_results.groupby("model").agg(
        mean_f1=("macro_f1", "mean"),
        std_f1=("macro_f1", "std"),
    ).sort_values(["mean_f1", "std_f1"], ascending=[False, True])
    chosen = summary.index[0]
    Path("results/model_selection_decision.json").write_text(json.dumps({
        "rule": SELECTION_RULE,
        "summary": summary.to_dict(),
        "chosen": chosen,
    }, indent=2))
    return chosen
```

The inference app then reads `results/model_selection_decision.json` to know which model to load — no hardcoded `"random_forest"` string.

### F10.2 Reporting suite

```python
# ml/report.py
def report_metrics(y_true, y_pred, y_proba, sample_weight=None,
                   bootstrap_n=1000) -> dict:
    return {
        "macro_f1": _bs_metric(f1_score, y_true, y_pred, average="macro", n=bootstrap_n),
        "balanced_accuracy": _bs_metric(balanced_accuracy_score, y_true, y_pred, n=bootstrap_n),
        "auc_ovr": _bs_metric(roc_auc_score, y_true, y_proba, multi_class="ovr", n=bootstrap_n),
        "per_class_f1": classification_report(y_true, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
```

Report `value [95% CI: lo, hi]` everywhere.

### F10.3 Per-receptor breakdown

```python
def per_receptor_metrics(y_true, y_pred, receptor_ids) -> pd.DataFrame:
    rows = []
    for r in np.unique(receptor_ids):
        mask = receptor_ids == r
        if mask.sum() < 5: continue  # too few to report
        rows.append({
            "receptor": r, "n": mask.sum(),
            "macro_f1": f1_score(y_true[mask], y_pred[mask], average="macro"),
            "balanced_acc": balanced_accuracy_score(y_true[mask], y_pred[mask]),
        })
    return pd.DataFrame(rows)
```

Save as `results/per_receptor_performance.csv`. Plot as a horizontal bar chart in supplementary.

### F10.4 Baselines

```python
# baselines/__init__.py
BASELINES = {
    "majority_class": DummyClassifier(strategy="most_frequent"),
    "stratified": DummyClassifier(strategy="stratified"),
    "smiles_only_rf": Pipeline([
        ("descs", LigandDescriptorExtractor()),  # no receptor or docking features
        ("rf", RandomForestClassifier()),
    ]),
    "morgan_fp_rf": Pipeline([
        ("fp", MorganFingerprintExtractor()),
        ("rf", RandomForestClassifier()),
    ]),
    "chemprop": ChemPropEstimator(),  # wraps chemprop CLI
    "uni_mol": UniMolEstimator(),     # optional, more setup
}
```

Report all baselines in the manuscript Table 2 alongside CancerAg.

### F10.5 Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(base_estimator=final_model,
                                    method="isotonic", cv=5)
calibrated.fit(X_train, y_train)
```

Save the calibrated estimator. The inference app uses it.

### F10.6 Multi-label / multi-task framing (optional methodological contribution)

```python
# ml/multi_task.py
# Each row = (ligand). Targets: a vector indicating bias label per receptor it was tested against.
# Loss: BCE per (ligand, receptor) pair, masked where unobserved.
```

Worth exploring as a stronger methodological story than yet-another descriptor classifier. Could be the actual "novelty" Reviewer 1 was missing.

### F10.7 Model card

```python
# ml/model_card.py
def write_model_card(model, training_meta, val_metrics, test_metrics, path):
    card = f"""# Model Card: {training_meta['model_name']}

## Training Data
- Dataset SHA-256: {training_meta['dataset_sha256']}
- Train rows: {training_meta['n_train']}; held-out test rows: {training_meta['n_test']}
- Split strategy: {training_meta['split_strategy']}

## Hyperparameters
{json.dumps(training_meta['hyperparameters'], indent=2)}

## Performance
- Validation macro-F1: {val_metrics['macro_f1']:.3f} [95% CI: {val_metrics['ci_lo']:.3f}, {val_metrics['ci_hi']:.3f}]
- Test macro-F1: {test_metrics['macro_f1']:.3f} [95% CI: ...]

## Intended Use
Retrospective in-silico hypothesis generation for biased-agonist discovery at GPCRs.
NOT validated for clinical or in-vivo use.

## Limitations
- Single-pose docking features
- Trained on n={training_meta['n_train']}; not benchmarked on receptors outside the BiasDB scope
- ...

## Provenance
- Repo commit: {training_meta['git_sha']}
- Trained at: {training_meta['trained_at_utc']}
- Library versions: {training_meta['lib_versions']}
"""
    path.write_text(card)
```

### F10.8 Run tracking with MLflow

```python
import mlflow
with mlflow.start_run(run_name=f"{model_name}_{seed}"):
    mlflow.log_params(hyperparameters)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("results/per_receptor_performance.csv")
    mlflow.sklearn.log_model(calibrated_model, artifact_path="model")
```

Adopt minimally; even file-based MLflow gives reproducible run history.

### F10.9 Statistical comparison

```python
def paired_permutation_test(scores_a, scores_b, n_perm=10000) -> float:
    """Two-sided paired permutation test on per-fold metrics."""
    ...
```

Used to claim "model A significantly better than model B."

### F10.10 Update headline metric in manuscript

Lead Abstract and Results with: macro-F1 (with 95% CI from bootstrap and seed variance), balanced accuracy, AUC-ROC OvR. Accuracy goes to supplementary.

## Acceptance criteria

- [ ] `results/model_selection_decision.json` exists and is consumed by the inference app (no hardcoded `"random_forest"`).
- [ ] All reported metrics include 95% bootstrap CIs and across-seed std.
- [ ] Macro-F1 leads in every results table; accuracy is supplementary.
- [ ] Per-receptor performance CSV and figure are generated.
- [ ] At least 4 baselines (majority, stratified, SMILES-only, ChemProp) are reported.
- [ ] Calibrated probabilities are produced; the inference app uses them.
- [ ] Each saved model has a `model_card.md` sidecar with the schema in F10.7.
- [ ] MLflow run history exists for every reported result.
- [ ] Hyperparameter tuning was done via inner CV only; no test-set leakage.
- [ ] Statistical comparison (paired permutation or bootstrap CI overlap) backs every "X better than Y" claim.
- [ ] Held-out external set ([01](01_data_collection.md), [07](07_dataset_assembly.md)) is evaluated and reported separately from the scaffold-split test fold.
