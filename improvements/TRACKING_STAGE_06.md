# Stage 06 — Featurization tracking

## Original state (pre-rebuild)
Featurization was descriptor-only and overwhelmingly receptor-level. `molecular_descriptors.py` (113 lines) computed ~200 RDKit 2D descriptors with no fingerprint or 3D component verified present. `receptor_descriptors.py` (503 lines) emitted pocket volume, surface area, residue composition, and hydrophobicity/charge profiles — but every ligand docked against the same receptor received the *same* pocket vector, so from the model's perspective these features encoded nothing more than receptor identity. Pair-level signal was a single number: `affinity_best` from Vina. There were no interaction-fingerprint features, no per-residue contact descriptors, no Morgan/MACCS fingerprints with documented bit definitions, no 3D pose descriptors, no collinearity prefilter (RDKit ships `MolWt`/`HeavyAtomMolWt`/`ExactMolWt` at ~99% pairwise correlation, plus a dozen other correlated clusters), and no descriptor-versioning sidecar. NaN failures from descriptors like `MaxPartialCharge` propagated downstream and got median-imputed across all receptor families.

## Upgrades performed
- **217 RDKit 2D descriptors** computed with explicit version pin and per-descriptor NaN policy.
- **2048-bit Morgan fingerprints** (radius=2) and **167-bit MACCS keys** added with bit definitions persisted in featurization metadata.
- **10 RDKit 3D pose descriptors** (Asphericity, Eccentricity, NPR1/NPR2, PMI1-3, RadiusOfGyration, SpherocityIndex, InertialShapeFactor) computed from the docked pose conformer — nearly free given Stage 05 already embedded the ligand.
- **386 ProLIF interaction-fingerprint bits** (pair-level; varies with both ligand and receptor) over the top Vina pose: H-bonds, π-π, π-cation, halogen, hydrophobic, ionic contacts at each pocket residue.
- Pose-loader element-mapping fix recovered 27 pairs whose 3D descriptors had been silently dropped because the PDBQT element column included Vina-specific atom types (e.g., aromatic carbons typed `A` instead of `C`).
- Featurization metadata sidecar (`featurization.meta.json`) records library versions, descriptor list, fingerprint definitions, and per-receptor IFP residue lists.

## What was NOT done (deferred / out-of-scope)
- ChemProp / D-MPNN GNN baseline (Python 3.13 dependency friction; ChemBERTa serves the same Reviewer-1 ask in Stage 10).
- Pharmacophore (Gobbi 2D / RDKit pharmacophore) fingerprints.
- GPCRdb generic-numbering on per-residue receptor descriptors (`pocket_hydrophobicity_3.32`-style column suffixes) — receptor descriptors still keyed on absolute residue numbers.
- DUD-E / property-matched decoy set.
- Substructure-attribution images (Morgan bit → atom highlight) — high effort, low payoff for a methods paper.

## Files produced
- `data/processed/ligand_features.parquet` — 217 RDKit 2D + 2048 Morgan + 167 MACCS
- `data/processed/pose_3d_features.csv` — 10 3D descriptors per docked pose
- `data/processed/interaction_fingerprints.parquet` — 386 ProLIF bits per (ligand, receptor) pair
- `data/processed/featurization.meta.json` — versions + bit definitions

## Tests
- `tests/unit/test_molecular_descriptors.py`

## Status
Complete (Done) — pair-level IFP features now dominate the structural-feature contribution measured at Stage 11 (3 of top 9 validated features come from the docking pipeline).
