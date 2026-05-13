# 02 — Ligand Curation & Standardization

Module covered: `src/cancerag/preprocessing/ligand_preprocessor.py`

## What the code does today

`LigandPreprocessor.run()`:

1. Loads BiasDB (`_load_biasdb_data`) and ChEMBL (`_load_chembl_data`) CSVs.
2. ChEMBL rows are hardcoded with `bias_category = "Agonist"` (`ligand_preprocessor.py:117`).
3. Concatenates and `drop_duplicates(subset=["canonical_smiles"], keep="first")` (`:267-269`) — on **raw** SMILES strings before standardization.
4. For each SMILES: `Chem.MolFromSmiles` → `_standardize_mol` (Cleanup → FragmentParent → Uncharger → Sanitize). Failures dropped silently.
5. Recomputes `canonical_smiles_standardized` from the standardized molecule (`:293-295`). **No re-deduplication** on the standardized form.
6. Computes MW, LogP, HBD, HBA, TPSA, RotBonds, Lipinski_Violations, PAINS for each molecule.
7. Hard-filters: PAINS=True dropped, Lipinski_Violations > max (1 by default), TPSA > 140, RotBonds > 10.
8. Writes `data/processed/unified_ligands.csv`.

## Problems

### Tier 1 — invalidate downstream steps

**P2.1 (E1) The "unbiased" label is a source-assignment artifact.**
Line 117: `consolidated_df["bias_category"] = "Agonist"`. Every ChEMBL ligand becomes a member of an "Agonist" class regardless of whether bias was ever measured. The model learns BiasDB-vs-ChEMBL provenance, not bias. **This is the deepest scientific flaw in the entire repo and no reviewer caught it.**

**P2.2 Deduplication on raw SMILES.**
Line 267 dedupes on `canonical_smiles` *before* `_standardize_mol` runs. Different atom orderings, salt forms, tautomers, and kekulé-vs-aromatic representations of the same molecule will not collide. After standardization at line 293 the pipeline never re-dedupes — so near-duplicates persist.

**P2.3 BiasDB-first dedup hides label conflicts.**
With `keep="first"` and BiasDB concatenated first, when a SMILES collides the BiasDB row is kept. The conflicting ChEMBL row is silently discarded — including the case where the same molecule has *different* bias labels in the two sources or in two different BiasDB rows (different receptor / different assay).

**P2.4 Ligand–receptor pair is collapsed to ligand-only.**
Line 78 keeps `["smiles", "ligand_name", "receptor_subtype", "bias_category"]`. After `drop_duplicates(subset=["canonical_smiles"])`, a single ligand keeps only one (receptor, bias_category) row even if BiasDB has it measured against three receptors with three labels. Multi-receptor evidence is destroyed.

**P2.5 Bias *quantification* is thrown away at load.**
BiasDB serves continuous bias factors (Δlog(τ/KA)) and ΔΔlog values. `_load_biasdb_data` (line 72-78) keeps only the categorical `bias_category`. The continuous signal is the strongest in the data and would remove the class-imbalance problem; instead the pipeline collapses to a 5-class label.

### Tier 2 — community-norm violations

**P2.6 Standardization recipe is incomplete.**
`_standardize_mol` does Cleanup → FragmentParent → Uncharger → Sanitize. Missing:
- Tautomer canonicalization (`rdMolStandardize.TautomerEnumerator`)
- Stereo normalization (`Chem.AssignStereochemistry(mol, cleanIt=True, force=True)`)
- Isotope stripping
- Curated salt list (FragmentParent silently deletes legitimate cocrystal small molecules)

**P2.7 Hard PAINS filter on curated reference compounds.**
Line 224: `df = df[~df["Has_PAINS"]]`. PAINS was designed for HTS deconvolution, not curation of literature-validated ligands. It flags 5–20% of approved drugs (catechols, isoflavones, tetracyclines). For an adrenergic-receptor study, *every catechol* (i.e., the entire endogenous agonist class) gets dropped.

**P2.8 Hard Lipinski filter on BiasDB.**
BiasDB contains peptides and natural products that violate Lipinski but are biologically real and bias-relevant. Filtering them out shrinks and biases the dataset.

**P2.9 TPSA / RotBonds thresholds are oral-bioavailability rules.**
`tpsa_max=140`, `rotatable_bonds_max=10` (Veber). The downstream task is *bias prediction*, not oral bioavailability. No citation, no sensitivity analysis.

**P2.10 Assay context is dropped.**
BiasDB columns `assay_1`, `assay_2`, `reference_ligand`, `pmid`, `year`, `doi` are loaded into the raw CSV but `_load_biasdb_data` discards them. Without `reference_ligand` the bias-factor concept is mathematically undefined; without `(assay_1, assay_2)` the same ligand-receptor measured in two different assay pairs is collapsed.

### Tier 3 — engineering

**P2.11 Typo: `self.unchoarger`** (line 48).

**P2.12 Silent failures in standardization.**
Lines 278-286: `MolFromSmiles` → `None` and `_standardize_mol` → `None` are dropped without logging which SMILES failed.

**P2.13 Per-molecule property loop is not vectorized.**
Lines 132-159: a Python-loop over 504+ molecules is fine here, but the 60-line indexing kludge at lines 161-217 is a smell — symptomatic of a brittle index-handling bug that should be solved by computing properties as a `pd.DataFrame` indexed identically to `df`.

**P2.14 `logging.basicConfig` at module top** (line 13).

**P2.15 No attrition table.**
A standard curation pipeline emits a per-stage `(stage, n_in, n_out, n_dropped, reasons)` summary. Today the only record is scattered `logger.info` lines.

## Standard approach

1. **Adopt `chembl_structure_pipeline`** (the package ChEMBL itself uses for curation). It implements documented, reproducible standardization rules including all the steps missing in P2.6.
2. **InChIKey-14 deduplication** (the connectivity layer of the InChIKey). Handles tautomers and stereo separately. Compute *after* standardization.
3. **Treat ChEMBL as unlabeled** (PU learning) or drop it entirely — never assign a class label based on database membership.
4. **Preserve all metadata** through curation; let the modeling layer decide what to drop.
5. **Annotate, don't filter** for drug-likeness. PAINS, Lipinski, etc. become columns; downstream code can choose to filter.
6. **Per-stage attrition log** as a structured artifact (`data/processed/attrition.json`).

## Concrete fixes

### F2.1 Drop the "Agonist" hardcode (E1)

Replace the ChEMBL labelling at line 117 with one of:

**Option A — PU framing (recommended if keeping ChEMBL):**
```python
consolidated_df["bias_category"] = pd.NA  # truly unknown
consolidated_df["label_status"] = "unlabeled"
```
The model is then trained as a PU classifier (e.g., `pulearn`, or two-stage spy/biased-SVM).

**Option B — drop ChEMBL entirely:**
Remove `_load_chembl_data` from the unified set. Train a pure multi-class classifier on BiasDB. Reserve ChEMBL purely for inference-time chemical-space comparison.

**Option C — explicit verified-unbiased:**
For each ChEMBL ligand, require evidence in literature that bias has been measured and found absent. Almost no rows will survive — that is the correct outcome.

### F2.2 Replace standardization

```python
# preprocessing/ligand_curation.py
from chembl_structure_pipeline import standardizer, checker

def standardize_smiles(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"status": "parse_error", "mol": None}
    issues = checker.check_molblock(Chem.MolToMolBlock(mol))
    std_mol = standardizer.standardize_mol(mol)
    parent_mol, _ = standardizer.get_parent_mol(std_mol)
    inchikey = Chem.MolToInchiKey(parent_mol)
    return {
        "status": "ok",
        "mol": parent_mol,
        "canonical_smiles": Chem.MolToSmiles(parent_mol),
        "inchikey": inchikey,
        "inchikey14": inchikey[:14],
        "checker_issues": [i[1] for i in issues],
    }
```

### F2.3 InChIKey-14 dedupe with conflict handling

```python
# preprocessing/ligand_curation.py
def dedupe_with_conflict_log(df: pd.DataFrame, key="inchikey14",
                             label_col="bias_category") -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = df.groupby(key)[label_col].nunique()
    conflicts = df[df[key].isin(grouped[grouped > 1].index)]
    deduped = df.drop_duplicates(subset=[key], keep=False)  # drop ALL when conflicting
    return deduped, conflicts
```

Write the conflict frame to `data/processed/dedup_conflicts.csv` for manual triage.

### F2.4 Preserve all metadata; key on (inchikey14, receptor_uniprot, assay_id)

```python
df = pd.read_csv(self.paths["biasdb_input"])
KEEP = ["smiles", "ligand_name", "receptor_subtype", "bias_category",
        "bias_pathway", "reference_ligand", "assay_1", "assay_2",
        "pmid", "year", "doi"]
df = df[KEEP].copy()
df["pair_key"] = df["inchikey14"] + "::" + df["receptor_uniprot"] + "::" + \
                 df["assay_1"].fillna("?") + "::" + df["assay_2"].fillna("?")
```

### F2.5 Annotate, don't filter

```python
# Replace lines 219-240 with:
df["pains_flag"] = df["mol_standardized"].apply(self.pains_filter.HasMatch)
df["lipinski_violations"] = ...  # already computed
# DO NOT drop. Save the flags. Let dataset_assembly decide.
```

The configuration becomes `preprocessing.flag_pains: true`, not `preprocessing.filter_pains: true`.

### F2.6 Attrition log

```python
# preprocessing/ligand_curation.py
class AttritionLogger:
    def __init__(self):
        self.records = []
    def log(self, stage, n_in, n_out, reason=""):
        self.records.append({"stage": stage, "n_in": n_in, "n_out": n_out,
                             "n_dropped": n_in - n_out, "reason": reason})
    def write(self, path):
        pd.DataFrame(self.records).to_json(path, orient="records", indent=2)
```

Call after every transform; write `data/processed/attrition.json`.

### F2.7 Vectorize property calculation

```python
def _props(mol):
    return pd.Series({
        "MW": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol), "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol), "RotBonds": Descriptors.NumRotatableBonds(mol),
    })

props = df["mol_std"].apply(_props)
df = pd.concat([df, props], axis=1)
```

Removes the 60-line indexing kludge at lines 161-217.

### F2.8 Fix the typo

`self.unchoarger` → `self.uncharger` (line 48). Trivial but visible to reviewers reading the supplementary code.

## Acceptance criteria

- [ ] No row in `unified_ligands.csv` has a bias label assigned purely on source membership.
- [ ] Deduplication uses InChIKey-14 on the standardized parent molecule.
- [ ] A `dedup_conflicts.csv` records every label collision and is small enough to triage manually.
- [ ] `unified_ligands.csv` retains `(reference_ligand, assay_1, assay_2, pmid, year, doi)` per row.
- [ ] PAINS, Lipinski violations, TPSA, RotBonds are columns, not filters; the configuration toggles can switch them back to filters explicitly.
- [ ] `attrition.json` exists and reconciles: `n_biasdb + n_chembl − n_dropped == n_unified`.
- [ ] Continuous bias values from BiasDB (where available) are preserved alongside the categorical label.
- [ ] No `logging.basicConfig` in this module; no `unchoarger`; no silent SMILES drops.
