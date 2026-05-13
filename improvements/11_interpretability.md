# 11 — Interpretability (SHAP)

Modules covered:
- `src/cancerag/ml/generate_final_report.py` (referenced)
- Manuscript discussion of feature importance.

## What the code does today

SHAP plots are generated and reported. The manuscript apparently lists "top features" but does not translate them into mechanistic claims.

## Problems

### Tier 1 — methodological

**P11.1 SHAP is not a contribution.**
Reviewer 1 point 8 explicitly noted: SHAP is an external method. Its use does not establish methodological novelty unless it leads to *new and meaningful scientific insight*.

**P11.2 Top features are unstable.**
With ~200 descriptors, n=504, 5 classes, and random splits, the top-K SHAP features differ substantially across seeds and folds. A single SHAP run reports one realization of a noisy distribution.

**P11.3 No mechanistic synthesis.**
Top features appear to be reported as a list (e.g., "TPSA, MolLogP, vina_affinity, ..."). What is missing:
- *Why* should TPSA matter for bias? (E.g., G-protein-biased ligands tend to have higher TPSA because polar contacts with TM7 favor the βArr-incompetent conformation.)
- *Which residues* in the pocket explain the importance?
- *What chemistry* (substructures, scaffolds) does this map to?
- *Are the features actionable* for medicinal chemists?

**P11.4 SHAP on uncalibrated `predict_proba`.**
RF probabilities are uncalibrated. SHAP values computed against them attribute "log-odds-of-uncalibrated-probability" — interpretable only as relative ranking, not as direct evidence weight.

### Tier 2 — community-norm gaps

**P11.5 No alternative interpretability methods.**
- **Permutation importance** as an unbiased complement to SHAP.
- **Counterfactual explanations** (DiCE, Wachter): "this G-biased ligand becomes balanced if its TPSA decreases by 10."
- **Substructure-attribution** for fingerprint features (Morgan bit → atoms in the molecule).
- **PocketAlign / pharmacophore overlay** to translate descriptor importance to 3D pocket regions.

**P11.6 No interaction-fingerprint attribution.**
If [06_featurization.md](06_featurization.md) F6.1 is implemented (ProLIF IFP), then SHAP can directly attribute importance to specific pocket residue contacts (e.g., "H-bond with D3.32 increases probability of G-bias by 0.15"). This is the kind of insight Reviewer 1 was asking for.

**P11.7 No global vs local consistency check.**
For a finding to be biologically meaningful, the global pattern (mean |SHAP| across the dataset) and the local pattern (per-class mean |SHAP|) should agree. A feature important only globally, never per-class, is a confound.

**P11.8 No comparison with literature pharmacophores.**
For each "important feature," compare against published GPCR-bias SAR (e.g., Manglik et al. on µ-opioid bias, Cao et al. on β-arrestin agonism). Agreement = validation; disagreement = a finding worth discussing.

### Tier 3 — engineering

**P11.9 No reproducible figure pipeline.**
SHAP figures should be regenerable from `make report` with the run ID. Current `generate_final_report.py` (483 lines) likely produces them inline.

**P11.10 No SHAP value persistence.**
SHAP values should be saved as `results/interpretability/shap_values.npy` (per-fold, per-class) so reviewers can re-analyze without re-running the model.

## Standard approach

1. **Stability of top features** via SHAP across CV folds and seeds; report selection frequency.
2. **Pair SHAP with permutation importance** — agreement is required for a finding to be reported.
3. **Mechanistic synthesis**: each "top feature" gets ≤ 1 paragraph in the manuscript explaining the proposed mechanism, the literature support, and the testable hypothesis it suggests.
4. **Substructure attribution** for fingerprint bits.
5. **Counterfactuals** for at least 5 example ligands to show actionable chemistry directions.
6. **Calibrated model** as the explanation target.

## Concrete fixes

### F11.1 SHAP across folds with stability frequency

```python
# ml/interpretability.py
import shap

def shap_across_folds(pipeline_factory, X, y, splits, top_k=20) -> pd.DataFrame:
    all_top = []
    for fold_id, (tr, te) in enumerate(splits):
        pipe = pipeline_factory().fit(X.iloc[tr], y[tr])
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        sv = explainer.shap_values(pipe[:-1].transform(X.iloc[te]))
        # multi-class -> list of arrays per class
        global_imp = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
        cols = pipe[:-1].get_feature_names_out()
        top = pd.Series(global_imp, index=cols).nlargest(top_k).index.tolist()
        for f in top:
            all_top.append({"fold": fold_id, "feature": f})
    return (pd.DataFrame(all_top).groupby("feature").size()
            .sort_values(ascending=False) / len(splits))
```

Output: `results/interpretability/shap_stability.csv` — features with frequency ≥ 0.8 across folds are the "stable top features" reported in the manuscript.

### F11.2 Permutation importance complement

```python
from sklearn.inspection import permutation_importance

def perm_importance(model, X, y, n_repeats=30, seed=42) -> pd.DataFrame:
    r = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=seed, n_jobs=-1)
    return pd.DataFrame({
        "feature": X.columns,
        "perm_importance_mean": r.importances_mean,
        "perm_importance_std": r.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
```

Cross-reference top-20 SHAP features with top-20 permutation features. Only features in both lists are "validated."

### F11.3 Mechanistic synthesis template (for the manuscript)

For each stable top feature, the manuscript should report:

> **Feature: `ifp_DRD2_3.32_HBDonor`** (ProLIF: H-bond donor contact with TM3 D3.32 in DRD2 docked pose).
> **Stability**: present in 5/5 outer folds; SHAP rank 2 globally.
> **Mechanism**: D3.32 is the conserved aspartate that recognizes the protonated amine of aminergic ligands. In our cohort, ligands that maintain this H-bond classify predominantly as G-biased; loss of this contact (e.g., in non-aminergic chemotypes) shifts predictions toward β-arrestin bias.
> **Literature support**: Consistent with Hauser et al. (2020), who reported that engagement of D3.32 stabilizes the active-G-protein conformation in DRD2.
> **Testable hypothesis**: Conservative substitutions reducing the basicity of the amine (e.g., morpholine for piperazine) should shift the predicted bias of compound class X.

This is the kind of synthesis that addresses Reviewer 1 point 8.

### F11.4 Counterfactual explanations

```python
# ml/counterfactuals.py
import dice_ml

def counterfactuals_for_examples(model, X, y, example_indices,
                                 desired_class, k=3) -> dict:
    d = dice_ml.Data(...)
    m = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(d, m)
    out = {}
    for i in example_indices:
        cf = explainer.generate_counterfactuals(X.iloc[[i]], total_CFs=k,
                                                desired_class=desired_class)
        out[i] = cf.cf_examples_list[0].final_cfs_df
    return out
```

Pick 5 manuscript-friendly examples (one per bias class) and show their counterfactuals as a supplementary table.

### F11.5 Substructure attribution for fingerprints

```python
# Highlight the atoms in molecule corresponding to the most important Morgan bits:
from rdkit.Chem.Draw import SimilarityMaps

def attribution_image(mol, important_bits, fp_radius=2):
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=fp_radius,
                                               nBits=2048, bitInfo=bit_info)
    weights = [1.0 if i in important_bits else 0.0 for i in range(mol.GetNumAtoms())]
    return SimilarityMaps.GetSimilarityMapFromWeights(mol, weights)
```

### F11.6 Persist SHAP values

```python
np.savez("results/interpretability/shap_values.npz",
         shap_per_class=shap_values, X_eval=X_test.values, y_true=y_test,
         feature_names=X_test.columns.values)
```

## Acceptance criteria

- [ ] SHAP-stability CSV reports per-feature selection frequency across outer folds.
- [ ] Permutation-importance CSV exists; only features in both top-20 lists are reported as "stable."
- [ ] Manuscript Discussion contains ≥ 5 mechanistic synthesis paragraphs in the F11.3 template format.
- [ ] Counterfactual examples are produced for ≥ 5 ligands.
- [ ] Substructure-attribution images exist for top fingerprint bits.
- [ ] SHAP values are saved as `.npz` for downstream re-analysis.
- [ ] Explanations are computed against the calibrated model (see [10_model_training_eval.md](10_model_training_eval.md) F10.5).
- [ ] If interaction-fingerprint features are present, ≥ 3 of them appear in the stable top-20.
