# Stage 02 — Ligand curation tracking

## Original state (pre-rebuild)
`LigandPreprocessor.run()` produced a 504-row `unified_ligands.csv` with five existential defects. (1) ChEMBL rows were hardcoded `bias_category="Agonist"` regardless of whether bias had been measured — the model learned a database-of-origin shortcut rather than bias. (2) Deduplication ran on raw `canonical_smiles` *before* standardization, so different atom orderings, salts, and tautomers of the same molecule did not collide; the pipeline never re-deduped on the standardized form. (3) `keep="first"` BiasDB-first dedup silently discarded conflicting ChEMBL rows. (4) `drop_duplicates(subset=["canonical_smiles"])` collapsed multi-receptor evidence: a single ligand kept only one (receptor, bias_category) row even if BiasDB had measurements at three different receptors. (5) PAINS / Lipinski / TPSA / RotBonds were used as hard filters, dropping every catechol (i.e. the entire endogenous adrenergic agonist class) and all peptide ligands. Continuous bias factors and assay context (`reference_ligand`, `assay_1`, `assay_2`, `pmid`) were discarded at load.

## Upgrades performed
- Removed the ChEMBL "Agonist" hardcode; ChEMBL is excluded from the labelled set (decision E1 in `CURATION_JOURNEY.md`).
- Replaced standardization with the Cleanup → FragmentParent → Uncharger → Sanitize → tautomer-canonicalize chain, with a peptide-aware guard skipping `TautomerEnumerator.Canonicalize` for molecules > 50 heavy atoms (preserved real stereo on angiotensin/opioid peptides).
- Switched dedup from raw SMILES to **full InChIKey** (stereo-aware) on the standardized parent molecule, with a six-field composite pair-key: `inchikey :: receptor_uniprot :: bias_pathway :: assay_1 :: assay_2 :: reference_ligand`.
- Reframed PAINS / Lipinski / TPSA / RotBonds as **annotations**, not filters; legacy-filter mode kept behind an opt-in flag.
- Preserved `(reference_ligand, assay_1, assay_2, pmid, year, doi, bias_pathway)` per row for downstream label resolution and continuous-bias work.
- Added per-stage `attrition.json`, `dedup_conflicts.csv`, and a reviewer-facing `dataset_audit.md` with input/output SHA-256 sidecars.
- Verified the 8 BiasDB rows that still merged were chemically identical via 4 independent tests (canonical SMILES, full InChI, molecular formula, heavy-atom count).

## What was NOT done (deferred / out-of-scope)
- Adoption of `chembl_structure_pipeline` (the package ChEMBL itself uses) — current RDKit-only standardization is sufficient for in-scope receptors.
- Per-row continuous bias factor (Δlog(τ/KA)) preservation as a regression target — categorical only; column kept for future regression work.
- PU-learning framing for ChEMBL ligands — ChEMBL was simply dropped from the labelled set instead.

## Files produced
- `data/processed/unified_ligands.csv` (+ `.meta.json`) — 719 rows / 532 unique molecules
- `data/processed/ligands_with_descriptors.csv`
- `data/processed/attrition.json`
- `data/processed/dataset_audit.md`
- `data/processed/unified_ligands.legacy.csv` (legacy 504-row snapshot)

## Tests
- `tests/unit/test_ligand_preprocessor.py` (largest unit test file: 585 lines)
- `tests/unit/test_molecular_descriptors.py`

## Status
Complete (Done) — 98.9% recovery (727 raw → 719 curated), 0 within-key conflicts, `unchoarger` typo fixed, no `logging.basicConfig` left in module.
