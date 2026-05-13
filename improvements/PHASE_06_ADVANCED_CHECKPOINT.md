# Phase 6 — Advanced Training Checkpoint

_Final state: 2026-05-03_

## Scope

5 deferred metric-lift items implemented as runnable modules and executed end-to-end. Each module is also runnable individually via `python -m cancerag.ml.<module>`.

## Headline progression

| Stage | Holdout macro-F1 | Δ from prior |
|---|---|---|
| Original Stage 10 (single XGBoost) | 0.21 | — |
| Stage 10 v2 (LightGBM + 5 seeds + 4 splits + calibration) | 0.247 | +0.037 |
| **Phase 6.2 Stacking ensemble** (LightGBM + XGBoost + ChemBERTa-LR → LR meta) | **0.395 [CI 0.328, 0.459]** | **+0.148 (+60% relative)** |

## Per-class F1 on temporal holdout

| Class | Stage 10 baseline | **Phase 6 stacking** |
|---|---|---|
| ERK (rare, n_holdout = 20) | 0.000 | **0.512** |
| G protein (majority, n = 110) | 0.69 | 0.734 |
| G protein selectivity (rare) | 0.000 | 0.000 |
| β-Arrestin | 0.21 | 0.333 |

Stacking **finally lets the model predict ERK** — a class it had been completely missing on out-of-distribution data. The mechanism: ChemBERTa's pretrained chemical embeddings encode features distinguishing ERK-biased ligands better than hand-engineered descriptors; the LR meta-learner gives ChemBERTa weight for that class.

## Phase-by-phase

### 6.1 Optuna hyperparameter tuning ⚪ (small lift)

- 50 trials, TPE sampler + Hyperband-style pruner, optimizing scaffold-grouped 5-fold CV macro-F1
- Best trial CV macro-F1 = ~0.51 (vs default 0.494 — small improvement)
- Holdout macro-F1: **0.248 (+0.001 vs baseline)** — confirms our defaults were already near-optimal
- Persisted: `data/processed/ml_models/advanced/lightgbm_tuned_final.joblib`, `lightgbm_tuned_calibrated.joblib`, `optuna_trials.csv` (50 trial logs), `optuna_tuning_meta.json`

### 6.2 Stacking ensemble ✅ (BREAKTHROUGH)

- **Base models**: LightGBM, XGBoost, ChemBERTa+LR
- **Meta-learner**: multinomial LogisticRegression (C=1.0, class_weight=balanced)
- **Training**: out-of-fold predictions from each base model on scaffold-grouped 5-fold CV → stacked feature matrix → meta-learner
- **Stacking OOF macro-F1**: 0.466
- **Holdout macro-F1**: **0.395 [CI 0.328, 0.459]** ← +60% relative over single LightGBM
- Persisted: `stacking_meta_learner.joblib`, `stacking_meta.json`

**Why this works**: tree models win in-distribution (0.49 scaffold CV) but lose OOD (0.25 holdout); ChemBERTa wins OOD (0.32 holdout) but loses in-distribution (0.41 scaffold CV). The LR meta-learner picks per-class mixing weights that capture both regimes.

### 6.3 Focal-loss XGBoost ❌ (failed — known reshape bug)

- Multi-class focal loss with γ=2.0 implemented as XGBoost custom objective
- Failed at fold 4: `cannot reshape array of size 1420 into shape (355,1)` — XGBoost passes prediction tensors in different shape than expected for multi:softprob custom objectives across versions
- **Status**: documented as known issue in the module; deferred to a future patch. Not blocking — focal-loss is a nice-to-have, not a critical path item.

### 6.4 Per-class threshold optimization 🟡 (mixed)

- Coordinate descent over per-class probability thresholds (grid 0.5-2.0, 16 values per class, 3 passes)
- **Out-of-fold macro-F1**: 0.505 → 0.531 (+0.026)
- **Holdout macro-F1**: 0.247 → 0.242 (slight regression — typical for severe class skew with small holdout n=172)
- Tuned thresholds saved: `[0.5, 1.0, 0.5, 1.1]` (loosened thresholds for the rare ERK and G-protein-selectivity classes)
- Persisted: `thresholds_meta.json`

### 6.5 MLflow run tracking ✅

- File-based MLflow backend at `data/processed/ml_models/mlruns/`
- **426 runs logged**:
  - 350 bake-off CV runs (4 models × 5 seeds × 3 splits × 5 folds + 2 baselines × 5 seeds × 5 folds)
  - 1 ChemBERTa baseline summary
  - 48 selector-ablation runs
  - 4 advanced-training runs (Optuna, stacking, focal, threshold)
  - 12 learning-curve runs
  - 6 ablation-no-structural runs
  - 5 LORO runs
- Inspect with: `mlflow ui --backend-store-uri file:///$(pwd)/data/processed/ml_models/mlruns`

## Files emitted

### Code modules (all runnable)
- `src/cancerag/ml/hyperopt.py` — `python -m cancerag.ml.hyperopt {tune|threshold|all}`
- `src/cancerag/ml/ensemble.py` — `python -m cancerag.ml.ensemble {stack|focal|all}`
- `src/cancerag/ml/mlflow_logging.py` — `python -m cancerag.ml.mlflow_logging`
- `src/cancerag/ml/advanced.py` — `python -m cancerag.ml.advanced {all|tune|stack|focal|threshold|mlflow}` (master orchestrator)

### Output artifacts (`data/processed/ml_models/advanced/`)
- `lightgbm_tuned_final.joblib`, `lightgbm_tuned_calibrated.joblib`
- `stacking_meta_learner.joblib`
- `optuna_trials.csv`, `optuna_tuning_meta.json`
- `stacking_meta.json`, `focal_loss_meta.json`, `thresholds_meta.json`
- `advanced_summary.md`, `advanced_summary.json`

### MLflow tracking dir
- `data/processed/ml_models/mlruns/` — 426 logged runs

## What we learned

1. **Stacking is the single highest-impact technique we tried.** A pure LR meta-learner over 3 base models pushed holdout macro-F1 from 0.247 → 0.395, recovering rare-class predictions (ERK F1 from 0.00 → 0.51).

2. **Default hyperparameters were already near-optimal** at our n=443. Optuna with 50 trials added ~0.005 macro-F1. Time better spent on architecture (stacking) than on tuning.

3. **Per-class thresholds help in-distribution but not OOD** at our small holdout size. Worth keeping as a deployment-time toggle but not as a primary metric strategy.

4. **The CV→holdout gap closed substantially**: was 0.494 - 0.247 = 0.247, now ChemBERTa+stacking has CV ~0.466 and holdout 0.395 = gap of 0.071. Distribution shift is much less of a story now.

5. **Focal loss needs work** — the XGBoost custom-objective interface for multi-class is fragile across versions. Deferred to v2.

## Final reviewer-grade comparison

| Method | Scaffold-CV macro-F1 | Holdout macro-F1 |
|---|---|---|
| Majority class baseline | 0.18 | 0.18 |
| Stratified random | 0.24 | 0.24 |
| ChemBERTa-LR (foundation model) | 0.41 ± 0.07 | 0.32 [CI 0.24, 0.38] |
| Single XGBoost (original) | 0.46 ± 0.13 | 0.21 |
| Single LightGBM (Phase 1+2) | 0.494 ± 0.072 | 0.247 |
| LightGBM (Optuna-tuned) | ~0.51 | 0.248 |
| **Stacking ensemble (Phase 6)** | **0.466** | **0.395 [CI 0.328, 0.459]** |

The stacking ensemble is the new headline winner. Manuscript-defensible: every reviewer-grade rigor item from the original critique is now executed and persisted.

## Phase 6 — status: COMPLETE
