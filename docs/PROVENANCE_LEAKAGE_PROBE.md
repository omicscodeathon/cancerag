# Provenance leakage probe

Companion to `QC_LEAKAGE_CORRECTION.md`. That note records a leak that four
evaluation regimes failed to catch, and which a two-column model reproduced the
full pipeline's holdout score on. This note records the check that the same
thing has not happened again with the 587-pose set, and the gate that now runs
it automatically.

Measured **2026-10-25** on `cancerag-run`, before any model was trained on the
rebuilt dataset.

## What was at risk

Reaching full docking coverage meant three new per-row properties, none of them
a feature by intent:

| property | values | why it could leak |
| --- | --- | --- |
| `engine_gpu` | 13 pairs on Uni-Dock GPU, 574 on CPU Vina | the 13 are *precisely* the flexible peptides CPU could not finish — not a random sample of rows |
| `exhaustiveness` | 16 for those, 8 for the rest | the same partition under another name |
| `receptor_repaired` | 55 rows over 6 receptors rebuilt from different PDB entries or chains | receptor identity, which predicts much of the label |

The mechanism to fear is the granularity trap: a per-group covariate joined onto
per-pair rows encodes group identity, and group identity carries the label.

## Why the obvious test is not enough

Training on provenance alone and comparing to the majority-class floor — the
test that caught the original leak — is **not interpretable here**.
`receptor_repaired` is a receptor-level property, and receptor identity already
predicts the label. A provenance-only model therefore beats the floor under
stratified CV whether or not provenance adds anything: it is re-reading receptor
identity, a known property of the data that grouped CV already exists to handle.

So the probe runs four comparisons, and the decisive one holds the receptors
out. With receptor identity unavailable, anything provenance still predicts is
signal of its own.

## Result

| test | macro-F1 |
| --- | --- |
| majority-class floor | 0.1894 |
| receptor identity alone, stratified | 0.5197 |
| provenance alone, stratified | 0.1894 |
| **provenance alone, receptor-grouped — the gate** | **0.1894** |

**PASS**, margin `+0.0000` against a `0.02` threshold.

Provenance sits exactly on the floor, and not only in the grouped test — it
scores the floor under *stratified* CV too, meaning the three columns do not
separate the classes even when the model is free to use them to identify the
receptor. Receptor identity being worth 0.5197 over the same rows shows the
comparison has the power to detect an association; provenance simply has none.

That is a stronger pass than expected. It is consistent with the cross-engine
validation: Uni-Dock and CPU Vina agree more closely than CPU Vina agrees with
itself across seeds (mean |Δ| 0.069 against an intrinsic 0.086, n = 15), so the
expected offset was near zero.

## `sample_weight` is unchanged

`sample_weight` derives from `redock_rmsd` and `gnina_cnn_score`, and the
re-dock numbers moved for the repaired receptors, so the weights were expected
to have shifted. Diffed on `pair_key` (unique across all 478 rows;
`inchikey` + `receptor_uniprot` has 77 duplicates and cannot be used):

```
rows: old 478  new 478  matched 478
weights changed: 0 of 478
old weight: mean 0.5440 std 0.0641 min 0.2100 max 0.6000
new weight: mean 0.5440 std 0.0641 min 0.2100 max 0.6000
```

Identical, and the reason is in the timestamps: `docking_features.csv` is dated
22:02 and both the old assembly (22:32) and the rebuild (06:17) read that same
file. The re-dock values moved *before* 22:02, so both datasets already carried
the post-repair numbers. Nothing silently changed between them.

One note on method, because the first attempt got this wrong. Joining on
`canonical_smiles` produced 740 matches from 478 rows against 478 and reported
50 changed weights, all by exactly +0.09 — a many-to-many artefact of duplicate
SMILES, not a real difference. The marginal distributions being identical to
four decimal places was the tell.

## The gate

The probe is `src/cancerag/ml/leakage_probe.py` and runs as the `leakage_probe`
pipeline step, between `ml_preprocessing` and `model_training`. It writes
`data/processed/leakage_probe_provenance.json` and **raises** if the grouped
test clears the floor by more than `MARGIN` (0.02), so no model can be trained
on a provenance mix that has not passed.

If it ever fails, the remedy is to **re-dock the affected rows on a single
engine**, not to filter them out. Excluding rows on an affinity threshold
filters on a value that is itself a model input, and the filter is not
independent of class: selection bias dressed as quality control.

Passing the gate is also not permission to use the columns. A separate
assertion, `assert_provenance_is_not_a_feature`, keeps them out of the feature
matrix — clearing the gate says provenance carries no label signal of its own,
not that it belongs in the model. It describes how a pose was produced, not the
pose.

Tests in `tests/unit/test_leakage_probe.py` cover both directions: provenance
constructed to track the label must FAIL, and provenance that only tracks
receptor identity must PASS. A gate that cannot fire is not a gate; one that
fires on the known receptor→label association would block every run.
