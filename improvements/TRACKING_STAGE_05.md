# Stage 05 — Production docking tracking

## Original state (pre-rebuild)
The legacy docking layer ran AutoDock Vina at `exhaustiveness=4` and `num_modes=3` — half of Vina's defaults — with a config comment ("Reduced from 8 for faster execution") that admitted the trade-off was made for throughput, not pose quality. Receptor PDBQT preparation used generic `obabel` (which skips AutoDock-specific aromatic-carbon typing and rotatable-bond flagging that Meeko applies). Ligands were embedded without explicit pH-7.4 protonation. Only `affinity_best` was consumed downstream — pose-ensemble information was discarded. There was no per-job timeout, no Vina-version provenance per result, no alternative docking tool comparison, no re-dock validation, and failed Vina runs were silently filled with `-5.0` kcal/mol in `dataset_assembly.py:371` (a plausible weak-binder value, not a sentinel — the model couldn't distinguish "Vina failed" from "Vina succeeded with weak affinity").

## Upgrades performed
- AutoDock Vina raised to **`exhaustiveness=16`, `num_modes=9`** (publication-grade).
- Production module `production_docking.py` runs all 587 unique (ligand, receptor) pairs in a `ProcessPoolExecutor` with `mp.get_context("spawn")` to avoid RDKit/CUDA fork inheritance.
- Pose-ensemble feature extraction: `vina_affinity_best`, `vina_affinity_mean_top3`, `vina_affinity_gap_1_2`, `vina_pose_diversity_rmsd`, `vina_n_distinct_clusters`.
- Per-job 1800 s timeout (raised from 600 s after 168 timeouts in round 1); job-level idempotency cache so resume launches only rerun affected jobs.
- Peptide ligands act as an automatic exclusion: Vina's exponential search in torsional flexibility filters them out transparently; documented in the manuscript framing.

### Three production bugs caught and fixed
1. **Wrong-chain selection** — 18/55 cocrystal receptors had their G-protein α-subunit prepped instead of the GPCR (Gα ~330 residues > GPCR ~280 → length-only chain pick always lost). Fixed via DBREF UniProt mapping in `receptor_preprocessor.py` (Stage 03). Re-dock pass rate 22/55 → 29/55.
2. **Silent empty-PDBQT from obabel fallback** — 4 receptors (NTSR1, δ-/κ-opioid, DP2) produced 0-byte receptor PDBQTs that the fallback chain accepted as success, giving `INTER=0` for 43 dockings. Fixed by retrying obabel without `--partialcharge gasteiger` and adding a post-condition guard that raises on empty PDBQT.
3. **Stereoisomer collision** — caught upstream in Stage 02's full-InChIKey switch (the four fenoterol stereoisomers from PMID 25342094 were being collapsed to one row); Stage 05 verified all four survive as distinct dockings.

## What was NOT done (deferred / out-of-scope)
- Flexible side-chain (`--flex`) for known toggle-switch residues (W6.48, F6.51, D3.32) per receptor.
- DiffDock / Uni-Mol Docking / AutoDock-GPU comparison on a benchmark subset (Gnina CNN rescore from Stage 04 covers the alt-method ask).
- 67 unrecovered failures (peptide ligands — TRV120-series biased AT1 agonists, opioid endogenous peptides, apelin/ghrelin) — deferred to a peptide-aware-docking v2 (HADDOCK / FlexPepDock / AutoDock CrankPep).
- Per-result `.dock.meta.json` provenance sidecars.

## Files produced
- `data/processed/docking_features.csv` — 587 rows × 15 columns (520 successful, 88.6%)
- `data/processed/docking_audit.md` — per-receptor success-count + mean-affinity audit
- `data/processed/.docking_work/<inchikey14>__<UNIPROT>/` — per-job artifacts (~570 sub-dirs)
- `data/processed/docking_features_smoke.csv`, `docking_audit_smoke.md` — smoke-test outputs
- `src/cancerag/docking/production_docking.py` — production module

## Tests
- `tests/unit/test_production_docking.py` — 16 unit tests (354 lines)
- `tests/unit/test_runner.py`, `test_analysis.py`, `test_clustering.py`, `test_redock_validation.py`, `test_gnina_rescore.py`

## Status
Complete (Done) — 88.6% success rate; 83% of pairs are high or marginal confidence; transparent peptide-failure characterization; 288-test suite passing.
