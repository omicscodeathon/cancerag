# Per-receptor QC leakage: diagnosis and corrected results

_Investigation and full re-training run on 2026-08-03/04. All numbers below are
reproduced from artefacts in `data/processed/ml_models/`; the pre-correction
artefacts were preserved under `data/processed/ml_models_preQCfix_backup/`._

## Summary

Three per-receptor structure-preparation QC quantities were entering the model's
feature matrix. They are computed once per receptor and broadcast to every row
for that receptor, so they encode **receptor identity as a number** rather than
ligand × receptor chemistry.

A LightGBM trained on **those two columns alone** — no chemistry, no docking, no
fingerprints, no ChemBERTa, no stacking — reaches temporal-holdout macro-F1
**0.3997**, matching the published full-pipeline stacking result of **0.3948**.

After removing them, the structural arm's measured contribution falls from
Δ = +0.041 to **Δ = −0.003**, and the ERK recovery (per-class F1 0.512)
disappears entirely.

## The columns

| Column | Distinct values | Max distinct within one receptor |
| --- | ---: | ---: |
| `gnina_cnn_score` | 44 | 1 |
| `redock_rmsd_angstrom` | 43 | 1 |
| `docking_confidence` (4 one-hot levels) | 4 | 1 |

49 receptors. `production_docking.py:415-417` maps all three from
`receptor_uniprot`; `_confidence` (`production_docking.py:397-413`) is a
deterministic function of the other two, so one-hot encoding it re-introduced
the same information a third time. The paired `*_missing` indicators are
per-receptor constants for the same reason — they record *which receptors*
lacked a re-dock.

Eight model inputs in total (2852 → 2844 features). They remain in the dataset
and still drive `sample_weight` (`dataset_assembly.py:214`), which is where
structure quality legitimately belongs.

`feature_selection.py` additionally force-kept `redock_rmsd` and `gnina_cnn`
through Boruta, so the one mechanism that could have pruned them was disabled.

## Why the four-regime protocol did not catch it

The temporal holdout splits by publication year, not by receptor: **16 of 24
holdout receptors also appear in training**, with identical QC values on both
sides (e.g. P08908 = 0.83657 in train and holdout). A receptor tag therefore
walks straight across the split.

ERK is concentrated enough for this to matter: it appears in only 12 of 49
training receptors, three of which are 100% ERK (4/4, 4/4, 3/3).

Receptor-grouped CV does hold receptors out, but structure quality correlates
with receptor family, and family correlates with bias behaviour — so the tag
still transfers partially and looks like biology.

## Controls

Temporal holdout, LightGBM, mean of seeds {42, 7, 13}, 1,000-resample bootstrap CI:

| Model | Features | macro-F1 |
| --- | ---: | --- |
| QC columns only | 2 | **0.3997** [0.309, 0.488] |
| Published stacking (QC in X) | 2852 + stacking | 0.3948 [0.328, 0.459] |
| Receptor identity code only | 1 | 0.3044 [0.221, 0.381] |
| Random per-receptor constant (control) | 1 | 0.2822 [0.207, 0.361] |
| Corrected stacking (QC removed) | 2844 + stacking | 0.2615 [0.215, 0.303] |

The random-constant control carries no information except which receptor a row
belongs to, and still scores 0.282. That is the value of receptor identity on
this holdout.

## Corrected results

### Bake-off, macro-F1 (5 seeds × 5 folds)

| Split | Model | Before | After | Δ |
| --- | --- | ---: | ---: | ---: |
| stratified | lightgbm | 0.5821 | 0.5467 | −0.0354 |
| | xgboost | 0.5776 | 0.5370 | −0.0406 |
| | random_forest | 0.5668 | 0.5607 | −0.0060 |
| | elastic_lr | 0.5394 | 0.5390 | −0.0004 |
| scaffold | lightgbm | 0.4944 | 0.4774 | −0.0170 |
| | xgboost | 0.4796 | 0.4027 | −0.0769 |
| | random_forest | 0.4763 | 0.4576 | −0.0188 |
| | elastic_lr | 0.4503 | 0.4491 | −0.0012 |
| receptor | elastic_lr | 0.3249 | 0.3284 | **+0.0035** |
| | random_forest | 0.2813 | 0.2846 | **+0.0033** |
| | lightgbm | 0.3119 | 0.2867 | −0.0252 |
| | xgboost | 0.3246 | 0.2648 | −0.0598 |

The loss falls entirely on the boosted trees. Random forest and elastic-net
logistic regression are unchanged, and on receptor-grouped CV both improve
slightly. A per-receptor constant with 44 ordered values is a clean split for a
boosted tree and near-useless to a linear model — the asymmetry is the
signature of a memorisable identifier rather than distributed signal.

On receptor-grouped CV, elastic-net logistic regression is now the best model
(0.328), ahead of both boosted trees.

### Temporal holdout

| | Before | After |
| --- | --- | --- |
| Single calibrated LightGBM | 0.2469 [0.205, 0.292] | 0.2213 [0.186, 0.261] |
| Tuned + per-class thresholds | 0.2418 [0.198, 0.288] | **0.2670** [0.222, 0.312] |
| Stacking ensemble | 0.3948 [0.328, 0.459] | 0.2615 [0.215, 0.303] |
| — ERK per-class F1 | 0.512 | **0.000** |
| — G protein | 0.734 | 0.664 |
| — β Arrestin | 0.333 | 0.382 |
| — G protein selectivity | 0.000 | 0.000 |

Per-class threshold optimisation is the one component that improves: +0.044 on
the holdout versus +0.025 before, with β-arrestin F1 rising 0.195 → 0.302. When
one feature dominates, the probability distribution is rigid and thresholds buy
little; remove it and the same technique extracts more.

### Optuna re-tuning

| | Before | After |
| --- | ---: | ---: |
| best scaffold-CV macro-F1 | 0.5107 | 0.4937 |
| `max_depth` | 4 | 5 |
| `num_leaves` | 22 | 67 |
| `n_estimators` | 300 | 450 |
| `learning_rate` | 0.062 | 0.142 |
| `min_child_samples` | 7 | 24 |

Independent corroboration: the search rediscovered that the problem changed
shape. With the tag present a small tree sufficed; without it the model needs
3× the leaves and 50% more trees to combine many weak ligand features.

### Structural-features ablation

| | Before | After |
| --- | --- | --- |
| with structural | 0.5739 | 0.5300 |
| chemistry only | 0.5327 | 0.5327 |
| **Δ** | **+0.0412** (per-seed +0.029/+0.016/+0.079, sd 0.033) | **−0.0027** (per-seed −0.005/−0.013/+0.011, sd 0.012) |

`STRUCTURAL_PREFIXES` previously included `gnina_`, `redock_` and
`docking_confidence_`, so the ablation dropped the QC constants together with
the genuine structural columns and attributed the whole difference to structure.
The chemistry-only arm is unchanged (2440 features, identical numbers), which
confirms the difference comes entirely from the structural side.

**The structural arm's measurable contribution is zero within noise.**

### Interpretability

Single-feature dominance ratio (top permutation importance ÷ runner-up):

| | Before | After |
| --- | --- | --- |
| top feature | `gnina_cnn_score` (0.0392) | `vina_affinity_gap_1_2` (0.0024) |
| runner-up | `morgan_1236` (0.0020) | `morgan_1683` (0.0015) |
| **ratio** | **19.7×** | **1.5×** (PASS, limit 5×) |

Cross-regime SHAP now yields **10** features important in all four regimes, none
of them per-receptor constants: `vina_affinity_best`, `NPR2`, `PEOE_VSA7`,
`vina_affinity_gap_1_2`, `vina_pose_diversity_rmsd`, `LogP`, `BCUT2D_MRHI`,
`SpherocityIndex`, `EState_VSA3`, `morgan_1236`. Five are structural
(3 docking + 2 docked-pose shape).

Block-level attribution, share of total mean|SHAP| (receptor-grouped regime):

| Block | n | share |
| --- | ---: | ---: |
| ligand-only 2D descriptor | 177 | 70.4% |
| ligand-only 2D fingerprint | 1434 | 13.9% |
| docked-pose 3D shape | 7 | 8.2% |
| pair-level docking energetics | 5 | 6.2% |
| receptor–ligand contacts (ProLIF) | 231 | 1.3% |

Stable to within ~2 points across all four regimes.

### Reconciling SHAP with the ablation

SHAP ranks docking features at the top while the ablation says removing all 404
structural columns costs nothing. Both are correct and they measure different
things. SHAP is attribution *within a fitted model* — given that
`vina_affinity_best` is present, the model leans on it. The ablation measures
*marginal value against a substitute* — with docking removed, ligand chemistry
covers the same ground, because Vina's best affinity largely tracks ligand size
and lipophilicity (on the 19 multi-receptor ligands its within-ligand SD is only
0.24 of overall SD, versus 1.15 for `vina_affinity_gap_1_2`).

A feature can be genuinely used and still be redundant.

## Structural limits found along the way

- **No receptor representation exists.** After the QC removal, zero columns
  describe the receptor independently of the ligand. The remaining 2,844 split
  into 2,440 ligand-only and 404 docking/contact columns that mix the two.
- **The design is nearly ligand-indexed.** 332 of 351 ligands appear against
  exactly one receptor; only 19 ligands (46 of 443 rows) are docked into two or
  more. Receptor-conditional behaviour — the definition of biased agonism — is
  barely represented.
- **ProLIF is not comparable across receptors.** 386 columns use per-structure
  residue numbering (`ifp_ASP86.A_VdWContact`), so a column exists only for the
  receptor that has that residue. 14 of 231 carry nonzero SHAP. Ballesteros–
  Weinstein generic numbering would make them comparable.
- **p/n = 6.4** (2,844 features, 443 rows), with 408 all-zero Morgan bits.

## Guards added

- `preprocessing.RECEPTOR_QC_COLS`, folded into `META_COLS`.
- Unit tests asserting the QC columns never reach `feature_cols`, that
  `categorical_features` is empty, that genuine pair-level columns survive, and
  the general property: **no non-binary feature may be constant within every
  receptor**.
- `interpretability.dominance_check()` — fails when the top permutation
  importance exceeds 5× the runner-up; verdict written into
  `interpretability_report.md`.
- `shap_cross_regime.block_attribution()` — SHAP share by information source,
  with a self-checking `!! receptor-level QC (should be absent)` row.
