# 06 — Featurization (Ligand + Receptor + Pair)

Modules covered:
- `src/cancerag/features/molecular_descriptors.py`
- `src/cancerag/features/receptor_descriptors.py`
- `src/cancerag/features/receptor_analyzer.py`

## What the code does today

- **Ligand descriptors** (113 lines in `molecular_descriptors.py`): RDKit `Descriptors.descList` (≈200 2D descriptors). README claims Morgan / MACCS fingerprints are also included — needs verification.
- **Receptor pocket descriptors** (503 lines in `receptor_descriptors.py`): pocket volume, surface area, residue composition, hydrophobicity / charge profiles.
- **Pair-level features**: only the **single-pose Vina affinity** from the docking layer.

## Problems

### Tier 1 — invalidates the predictive signal claim

**P6.1 Receptor pocket descriptors are receptor-level, not pair-level.**
Every ligand docked against the same receptor gets the *same* pocket descriptor vector. From the model's perspective, these features are a one-hot receptor identity: they carry zero ligand-specific information. Combined with the source-shortcut leakage in [02_ligand_curation.md](02_ligand_curation.md), the model can learn "if pocket descriptors look like receptor X → assign label Y" without ever using ligand chemistry. Reviewer 1 alluded to this when questioning what CancerAg's "novelty" is beyond combining components.

**P6.2 No interaction-fingerprint features.**
The standard pair-level featurization for protein-ligand modeling is an **interaction fingerprint** (IFP) — a binary or count vector over (pocket residue × interaction type). Open-source tools:
- **ProLIF** (Bouysset & Fiorucci, 2021) — Python, MDAnalysis-backed, supports H-bonds, π-π, π-cation, halogen, hydrophobic, ionic, etc.
- **PLIP** (Adasme et al.) — REST API and Python; widely cited.
- **OddT-IFP** — orchestrates Vina + IFP.

These produce features that *change with the ligand* and directly probe the bias-relevant contacts (e.g., does this ligand contact W6.48? Does it form an ionic bond with D3.32?). Their absence is the largest single missed opportunity in the pipeline.

**P6.3 ~200 RDKit descriptors include severely collinear groups.**
`MolWt`, `HeavyAtomMolWt`, `ExactMolWt` are ~99% correlated. `NumValenceElectrons`, `NumRadicalElectrons`, `MaxPartialCharge`, `MinPartialCharge` and a dozen others form correlated clusters. Without a correlation/VIF prefilter, Boruta and downstream models get confused signal-to-noise. The README claims "200+ features" as a strength — at n=504, this is a curse-of-dimensionality risk.

**P6.4 No SMILES-only baseline.**
A simple sanity check: train the same model on RDKit descriptors *without* receptor or docking features. If macro-F1 is similar, the structure-based half of the pipeline is decorative. This baseline is missing — Reviewer 1 point 7 essentially asks for this when requesting baseline comparisons.

### Tier 2 — community-norm gaps

**P6.5 No graph / 3D representation comparison.**
Reviewer 1 point 3 requested justification for descriptor-based featurization vs. graph or 3D models. Open-source baselines that take days, not weeks:
- **ChemBERTa** / **MolT5** — pre-trained molecular transformers; SMILES → embedding.
- **Uni-Mol** — 3D-aware pre-trained molecular model.
- **ChemProp** (D-MPNN) — graph neural network, gold-standard descriptor-free baseline (Yang et al. 2019, JCIM).
- **MolBERT** — masked-LM pretraining on SMILES.

Two GNN baselines on the same scaffold split would address Reviewer 1 point 3 directly.

**P6.6 No pharmacophore features.**
For GPCR modeling, pharmacophore features (PHA / RDKit pharmacophore fingerprints) are commonly more predictive than 2D descriptors because GPCR pockets recognize specific feature patterns (charged amine 5–7 Å from aromatic ring, etc.).

**P6.7 No conformer-based 3D descriptors.**
RDKit `Descriptors3D` (Asphericity, Eccentricity, NPR1/NPR2, PMI, RadiusOfGyration) require a 3D conformer. If the ligand is already embedded for docking, these are nearly free.

**P6.8 No descriptor stability / scale handling at this layer.**
Some RDKit descriptors range 0-10, others 0-10000. Without per-descriptor scaling at *featurization time* (not just at ML preprocessing), distance-based methods (clustering, neighbor-based AD) are dominated by the largest-scale descriptors.

### Tier 3 — engineering

**P6.9 No descriptor versioning.**
RDKit descriptor definitions occasionally change between versions. The feature matrix has no record of `rdkit_version`, descriptor list, or computation date.

**P6.10 No null/NaN policy at the featurizer.**
Descriptor failures (e.g., `MaxPartialCharge` for radicals) silently produce NaN and are pushed downstream where they get median-imputed across all receptor families ([07_dataset_assembly.md](07_dataset_assembly.md)).

**P6.11 Receptor descriptor module is 503 lines but tightly coupled.**
Without GPCRdb generic-numbering integration, receptor descriptors can't be reported on a per-residue basis comparable across receptors.

## Standard approach

1. **Pair-level features dominate.** 2D ligand descriptors and pocket descriptors are auxiliary. The primary signal should be:
   - Interaction fingerprint (ProLIF) over the docked pose.
   - Per-residue interaction energy (Vina decomposed, or rescored with NNScore / RF-Score / Gnina CNN).
   - Pose-ensemble features ([05_docking.md](05_docking.md)).
2. **Diverse representations** for SOTA comparison: 2D descriptors (current), Morgan FP, Pharmacophore FP, GNN (ChemProp), pre-trained transformer (Uni-Mol).
3. **Feature provenance** with versioned descriptor list.
4. **Collinearity filter** before Boruta.

## Concrete fixes

### F6.1 Add interaction fingerprint module

```python
# features/interaction_fingerprint.py
import prolif as plf

def compute_ifp(receptor_pdb: Path, ligand_pose_sdf: Path,
                pocket_residues: list[str]) -> dict[str, int]:
    receptor = plf.Molecule(read_pdb(receptor_pdb))
    ligand = plf.Molecule(read_sdf(ligand_pose_sdf))
    fp = plf.Fingerprint()
    ifp = fp.run_from_iterable([ligand], receptor)
    df = ifp.to_dataframe()
    # column form: ('LIG', 'D3.32', 'HBDonor') -> 1
    return {f"ifp_{r}_{i}": int(v) for (lig, r, i), v in df.iloc[0].items()}
```

These columns are pair-level: they vary with both ligand and receptor.

### F6.2 Verify or add fingerprint features

```python
# features/molecular_descriptors.py
from rdkit.Chem import AllChem, MACCSkeys

def morgan_fp(mol, n_bits=2048, radius=2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

def maccs_fp(mol):
    return MACCSkeys.GenMACCSKeys(mol)
```

### F6.3 Add SMILES-only baseline

Add a pipeline variant `--features ligand_only` that disables receptor and docking features. Required as a published baseline.

### F6.4 Add ChemProp / Uni-Mol baseline

```python
# baselines/chemprop_baseline.py
# Wrap chemprop CLI: train on the same scaffold split.
# Report macro-F1 alongside CancerAg in the manuscript Table.
```

This single baseline addresses Reviewer 1 point 3 + 7.

### F6.5 Collinearity filter

```python
# features/collinearity.py
def drop_correlated(X: pd.DataFrame, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if (upper[c] > threshold).any()]
    return X.drop(columns=to_drop), to_drop
```

Run **inside the cross-validation fold** (see [08_ml_preprocessing.md](08_ml_preprocessing.md)) — never on the full matrix.

### F6.6 3D descriptors from docked pose

```python
# features/molecular_descriptors_3d.py
from rdkit.Chem import Descriptors3D

def descriptors_3d_from_pose(mol_3d) -> dict:
    return {
        "Asphericity": Descriptors3D.Asphericity(mol_3d),
        "Eccentricity": Descriptors3D.Eccentricity(mol_3d),
        "NPR1": Descriptors3D.NPR1(mol_3d),
        "NPR2": Descriptors3D.NPR2(mol_3d),
        "PMI1": Descriptors3D.PMI1(mol_3d),
        "RadiusOfGyration": Descriptors3D.RadiusOfGyration(mol_3d),
    }
```

### F6.7 Pharmacophore fingerprints

```python
# features/pharmacophore.py
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

def pharm2d_fp(mol):
    return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
```

### F6.8 Featurization provenance

For each feature CSV write `<features>.meta.json`:
```json
{
  "rdkit_version": "2024.09.1",
  "prolif_version": "2.0.3",
  "descriptor_list": ["MolWt","LogP", ...],
  "fingerprint_definitions": {"morgan": {"radius": 2, "n_bits": 2048}},
  "ifp_pocket_residues_per_receptor": {"P14416": ["3.32","6.48", ...]},
  "computed_at_utc": "..."
}
```

### F6.9 Receptor descriptors keyed on GPCRdb numbering

Re-key per-residue receptor descriptors (hydrophobicity, charge, accessibility) from absolute residue numbers to GPCRdb generic numbering. Then "residue 3.32 hydrophobicity" is a feature comparable across receptors instead of "residue 119 in HTR1A vs residue 113 in DRD2."

## Acceptance criteria

- [ ] ProLIF-derived interaction fingerprint columns are present in the feature matrix and vary per (ligand, receptor) pair.
- [ ] Morgan and MACCS fingerprints are confirmed present (or added).
- [ ] A SMILES-only baseline run (no receptor/docking features) is reported in the results.
- [ ] At least one GNN baseline (ChemProp or Uni-Mol) is reported on the same scaffold split.
- [ ] A collinearity filter (≥ 0.95 absolute correlation) is applied inside CV folds.
- [ ] 3D descriptors from the docked pose conformer are computed.
- [ ] Per-residue receptor features use GPCRdb generic numbering as their column suffix (e.g., `pocket_hydrophobicity_3.32`).
- [ ] Featurization metadata sidecar exists with library versions and definitions.
- [ ] No descriptor failure silently propagates as NaN — every NaN has a paired `_missing` indicator.
