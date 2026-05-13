# 05 — Molecular Docking (AutoDock Vina)

Modules covered:
- `src/cancerag/docking/pipeline.py`
- `src/cancerag/docking/preparation.py`
- `src/cancerag/docking/runner.py`
- `src/cancerag/docking/analysis.py`
- `src/cancerag/docking/run_docking.py`
- Config: `configs/config.yaml::docking`

## What the code does today

Per `config.yaml`:
```yaml
docking:
  exhaustiveness: 4   # comment says "Reduced from 8 for faster execution"
  num_modes: 3        # comment says "Reduced from 9 for faster execution"
  num_cpu: 4
  default_binding_site:
    size_x: 25.0
    size_y: 25.0
    size_z: 25.0
```

The docking modules (~660 lines total) wrap Vina via subprocess, prepare ligands by SMILES → 3D embedding via RDKit/OpenBabel, convert to PDBQT, run Vina, and parse the output `.pdbqt` for affinity values. The "best" pose's affinity becomes a feature; per `dataset_assembly.py:371` failures get a hardcoded `-5.0`.

## Problems

### Tier 1 — invalidates the docking-derived features

**P5.1 Vina is run below default parameters.**
`exhaustiveness=4` is half of Vina's default (8). For publication-grade pose prediction the recommendation in the literature is **≥16** (often 32–64 for benchmarking). At 4 the search is effectively a coarse scan and pose RMSD variance is large. The comment "Reduced from 8 for faster execution" admits the trade-off was made for speed, not quality. Reviewer 1 point 4 already raised docking-quality concerns.

**P5.2 `num_modes=3` discards ensemble information.**
Vina's value is in the *distribution* of poses (top-N RMSD spread, energy gap between rank 1 and rank 2). Three modes is too few; nine (the default) is the minimum for any pose-clustering analysis.

**P5.3 No re-docking validation.**
For every receptor with a co-crystal ligand, the standard sanity check is:
1. Extract the bound ligand from the crystal.
2. Re-dock it into its own receptor with the configured box and parameters.
3. Compute RMSD (heavy atoms) of the top-pose vs. the bound pose.
4. RMSD < 2 Å = good; 2–4 Å = marginal; > 4 Å = docking setup is broken for that receptor.

Without this, every docking-derived feature is trusted without evidence. This is the single most damaging gap in the docking layer, and Reviewer 1 explicitly flagged it.

**P5.4 Failed Vina runs are silently filled with `-5.0` kcal/mol.**
`dataset_assembly.py:371`: `df[col] = df[col].fillna(-5.0)`. -5 kcal/mol is a *plausible weak-binder* affinity, not a sentinel. Any model learns "score ≈ -5" as a real binding mode rather than as "Vina failed." This is an existential issue (E6 in the overview).

**P5.5 Single pose used as the feature.**
Only `affinity_best` is consumed downstream. The information-rich features are:
- Mean / std of top-N affinities (ensemble robustness).
- Energy gap between rank 1 and rank N (selectivity of pose).
- Pose-clustering: number of distinct binding modes within 2 Å.
- Per-residue interaction patterns (see [06_featurization.md](06_featurization.md) on ProLIF).

### Tier 2 — preparation gaps

**P5.6 Ligand 3D embedding details are not validated.**
SMILES → 3D requires:
- Choosing a reasonable starting conformer (ETKDG, possibly multiple seeds).
- Energy minimization (MMFF94 or UFF).
- Generating tautomers and protonation states at pH 7.4 (Dimorphite-DL or OpenEye QUACPAC equivalents).
- Stereochemistry preservation: chiral centers and double-bond geometry must survive embedding.

`docking/preparation.py` (126 lines) needs an audit for whether any of this is done.

**P5.7 Receptor PDBQT preparation likely uses generic `obabel`.**
Generic Open Babel `obabel -ipdb -opdbqt` does **not** apply correct AutoDock atom typing for protein residues — it assigns Gasteiger charges but skips the AD4-specific aromatic-carbon assignment, hydrogen merging, and rotatable-bond flagging that `prepare_receptor4.py` (MGLTools) or **Meeko** apply. AutoDock developers have warned against this for years; pose quality suffers materially.

**P5.8 No protonation state at pH 7.4 for ligands.**
Without explicit protonation, RDKit assumes neutral forms. Many GPCR ligands (amines, carboxylates) bind in their charged form at physiological pH. Docking the wrong protomer changes affinity by 1–3 kcal/mol.

**P5.9 No flexible side-chain treatment.**
Vina supports flexible receptor side chains via `--flex`. For GPCRs the toggle-switch residues (W6.48, F6.51) and TM3 D3.32 are known to rearrange on agonist binding. Treating the whole receptor as rigid is a known limitation that Reviewer 1 alluded to.

### Tier 3 — engineering / reproducibility

**P5.10 Vina version is not pinned in features.**
The Dockerfile installs Vina 1.2.5 (top-level `Dockerfile:21`), but no provenance is recorded with each docking result. If a future rebuild uses Vina 1.2.6 the affinities will shift slightly with no audit trail.

**P5.11 No timeout / resource control.**
Vina can hang on pathological inputs. Recent commits (`b62ed92`) mention a docking timeout fix in inference, but it's a workaround (use pre-converted receptors), not a per-job timeout in the docking runner.

**P5.12 `default_binding_site` has size but no center.**
`config.yaml::docking::default_binding_site` defines `size_x/y/z` but no `center_x/y/z`. The fallback path when `binding_sites.json` lacks an entry is unclear — likely the docking either skips the receptor or crashes. Should be documented or removed.

**P5.13 No comparison with a more accurate docking tool.**
Reviewer 1 point 4 explicitly asked for this. Open-source options:
- **Smina** — Vina fork with custom scoring, often better pose quality.
- **Gnina** — CNN-rescored Vina; published as more accurate on pose RMSD (Drugs.AI / Koes lab).
- **DiffDock** — diffusion-model-based, no box required, reports a confidence score.
- **Uni-Mol Docking** — equivariant graph model; high-throughput.
- **AutoDock-GPU / Vina-GPU 2.1** — same scoring, faster.

A comparison on a small subset (re-dock RMSD vs. crystal pose) is sufficient to satisfy the reviewer.

**P5.14 No docking-quality feature in the feature matrix.**
Each row's docking confidence is unknown to the model. A `redock_rmsd` value (per receptor), or a per-row `vina_pose_clustering_score`, would let the model down-weight unreliable rows.

## Standard approach

1. **Re-dock benchmark per receptor** is a non-negotiable QA step.
2. **Prepare receptors with Meeko** (the official AutoDock receptor-prep tool from the Forli lab), not generic obabel.
3. **Prepare ligands with explicit protonation** (Dimorphite-DL at pH 7.4) and 3D embedding with multiple ETKDG conformers.
4. **Run Vina with `exhaustiveness=16, num_modes=9`** as a baseline; use 32 for benchmark/comparison runs.
5. **Extract pose-ensemble features**, not single-pose affinity.
6. **At least one alternative docking tool** for cross-validation (Gnina or Smina).
7. **Pin Vina version in feature provenance.**

## Concrete fixes

### F5.1 Update `config.yaml::docking`

```yaml
docking:
  vina_version: "1.2.5"              # explicit pin
  exhaustiveness: 16                 # raise from 4
  num_modes: 9                       # raise from 3
  num_cpu: 4
  energy_range: 4
  per_job_timeout_seconds: 600       # hard kill after 10 min
  ph_for_protonation: 7.4
  flex_residues:                     # optional flexible side chains per receptor
    HTR1A: ["3.32", "6.48", "7.39"]
  redock_rmsd_threshold_angstrom: 2.5
```

### F5.2 Re-docking benchmark module

```python
# docking/redock_benchmark.py
def redock_all_receptors(registry: pd.DataFrame, vina_cfg: dict) -> pd.DataFrame:
    rows = []
    for _, rec in registry.iterrows():
        if not rec["has_cocrystal_ligand"]: continue
        rmsd = redock_one(rec["pdb_id"], rec["cocrystal_ligand_resname"], vina_cfg)
        rows.append({"uniprot": rec["uniprot"], "pdb_id": rec["pdb_id"],
                     "redock_rmsd": rmsd, "passes": rmsd < vina_cfg["redock_rmsd_threshold_angstrom"]})
    return pd.DataFrame(rows)
```

Output: `results/docking_qa/redock_summary.csv`. Receptors that fail are flagged in the dataset and their docking-derived features are explicitly marked low-confidence (separate column `vina_redock_passed: bool`, NOT median-imputed).

### F5.3 Replace `-5.0` hardcode (E6)

In `dataset_assembly.py`:
```python
# Don't impute. Drop or mark explicitly.
df["vina_affinity"] = df["vina_affinity"]  # leave NaN
df["vina_affinity_missing"] = df["vina_affinity"].isna().astype(int)
```

The model can use `vina_affinity_missing` as a feature and properly handles NaN via boosted-tree native NaN support (LightGBM / XGBoost).

### F5.4 Ligand prep with protonation

```python
# docking/preparation.py
from dimorphite_dl import DimorphiteDL

dimorphite = DimorphiteDL(min_ph=7.0, max_ph=7.4, max_variants=1)

def prepare_ligand(smiles: str, n_conformers: int = 8) -> Path:
    protomers = dimorphite.protonate(smiles)
    smiles_at_ph = protomers[0]
    mol = Chem.MolFromSmiles(smiles_at_ph)
    mol = Chem.AddHs(mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers,
                                      params=AllChem.ETKDGv3())
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    # pick lowest energy conformer
    energies = [AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid).CalcEnergy() for cid in cids]
    best = cids[int(np.argmin(energies))]
    return write_pdbqt_with_meeko(mol, conf_id=best)
```

### F5.5 Receptor prep with Meeko

See [03_receptor_curation.md](03_receptor_curation.md) F3.1.

### F5.6 Pose-ensemble features

```python
# docking/analysis.py
def pose_ensemble_features(vina_output_pdbqt: Path) -> dict:
    poses = parse_vina_output(vina_output_pdbqt)
    affinities = [p.affinity for p in poses]
    rmsds_to_top = [compute_rmsd(p.coords, poses[0].coords) for p in poses[1:]]
    return {
        "vina_affinity_best": affinities[0],
        "vina_affinity_mean_top3": np.mean(affinities[:3]),
        "vina_affinity_gap_1_2": affinities[1] - affinities[0] if len(affinities) > 1 else 0,
        "vina_pose_diversity_rmsd": np.mean(rmsds_to_top) if rmsds_to_top else 0,
        "vina_n_distinct_clusters": cluster_poses(poses, rmsd_threshold=2.0),
    }
```

### F5.7 Comparison with Gnina or Smina

```python
# docking/comparators/gnina.py
def dock_with_gnina(receptor, ligand, box) -> Pose:
    """Gnina = Vina + CNN scoring."""
    ...
```

For a small subset (~50 ligand-receptor pairs from the test set), report per-method affinity correlation, re-dock RMSD distribution, and downstream impact on classifier macro-F1. Sufficient for Reviewer 1 point 4.

### F5.8 Per-job timeout in runner

```python
# docking/runner.py
def run_vina(cmd, timeout_seconds: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, timeout=timeout_seconds, capture_output=True, check=True)
    except subprocess.TimeoutExpired:
        raise DockingTimeoutError(...)
```

### F5.9 Provenance per docking result

Per ligand–receptor result write `<pair_id>.dock.meta.json`:
```json
{
  "vina_version": "1.2.5",
  "exhaustiveness": 16,
  "num_modes": 9,
  "ligand_inchikey": "...",
  "receptor_pdb_id": "6DRY",
  "receptor_uniprot": "P14416",
  "box_center": [...], "box_size": [...],
  "wall_seconds": 87.3,
  "ligand_protomer_at_ph": "...",
  "n_conformers_tried": 8,
  "best_conformer_id": 4,
  "ph": 7.4,
  "vina_seed": 42
}
```

## Acceptance criteria

- [ ] `exhaustiveness ≥ 16` and `num_modes ≥ 9` in default config; the "reduced from 8" comment is gone.
- [ ] `redock_summary.csv` exists for every receptor with a co-crystal ligand; receptors with RMSD > 2.5 Å are flagged in the dataset.
- [ ] No `fillna(-5.0)` anywhere; missing docking values are NaN with a `_missing` indicator column.
- [ ] Receptor PDBQT is generated by Meeko, not generic obabel.
- [ ] Ligand SMILES is protonated at pH 7.4 (Dimorphite-DL) before 3D embedding.
- [ ] Pose-ensemble features (`vina_affinity_mean_top3`, `vina_affinity_gap_1_2`, `vina_pose_diversity_rmsd`, `vina_n_distinct_clusters`) are in the feature matrix.
- [ ] At least one alternative docking tool (Gnina or Smina) is benchmarked on a subset; comparison report is in `results/reports/docking_method_comparison.csv`.
- [ ] Each docking result has a `.dock.meta.json` sidecar with the schema in F5.9.
- [ ] Per-job timeout is enforced; no runaway Vina jobs.
