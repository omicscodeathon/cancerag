# Stage 03 — Receptor preparation tracking

## Original state (pre-rebuild)
`ReceptorPreprocessor.run()` globbed `data/pdb/**/*.pdb` and pushed each through `_clean_pdb_file` → `Bio.PDB.PDBParser` → `PDBIO.save(..., ProteinSelect())`. `ProteinSelect.accept_residue` retained only residues with `id[0] == " "`, indiscriminately stripping every HETATM — including the conserved sodium ion in class A GPCRs (which changes pocket geometry), `MSE` selenomethionines (should remap to MET), and modified residues. There was no protein preparation: no protonation at physiological pH, no altloc resolution, no chain selection, no Meeko/PDBQT conversion (downstream code used generic `obabel`). G-protein chains, nanobodies, and BRIL/T4L fusion inserts were retained because they share the protein HETATM tag space. AlphaFold structures got the same minimal treatment with no pLDDT gating. The "cleaned" file had no metadata sidecar describing what was kept, dropped, or repaired.

## Upgrades performed
- UniProt-anchored RCSB structure selection driven by the canonical receptor registry; preferred PDB per receptor stored at `data/registry/preferred_pdbs.tsv` with manual-override column.
- AlphaFold v6 fallback gated at mean pLDDT ≥ 70 — 27 of 61 receptors fall back; the other 34 use cached PDBs.
- **DBREF UniProt-mapping fix for chain selection** (Bug 1 from STAGE_05_CHECKPOINT). `detect_receptor_chain()` previously preferred chain A by length, which in modern GPCR-Gα cryoEM structures always picked the Gα subunit (~330 residues > GPCR ~280). Now parses `DBREF` records, picks the chain whose UniProt accession matches the target, searches multi-DBREF chains (BRIL-fusion N-terminus + receptor) instead of first-wins, and uses sequence-identity rescue against canonical UniProt FASTA for engineered constructs with only PDB self-references. Re-dock pass rate jumped from 22/55 → 29/55 cocrystal receptors.
- Per-receptor preparation metadata sidecar (`<UNIPROT>.pdb.prep.meta.json`): kept chain, dropped chains, het residues kept/dropped, source SHA-256, AlphaFold/PDB provenance.
- Empty-PDBQT post-condition guard during prep; obabel charge-fallback chain (Bug 2 fix from Stage 05) raises if final PDBQT is 0-bytes or zero-atom-line.
- Reviewer-facing `receptor_audit.md` summarising 61/61 receptors prepared.

## What was NOT done (deferred / out-of-scope)
- PDBFixer-driven loop modeling and explicit PDB2PQR/PROPKA protonation at pH 7.4 — current pipeline relies on Meeko + obabel-fallback charge assignment without an explicit pH stage.
- GPCRdb generic-numbering (Ballesteros-Weinstein) annotation per residue — not integrated; receptor-feature comparability across receptors uses absolute numbering instead.
- Sanity-check tests for 7TM presence, DRY motif, NPxxY motif, and per-receptor pocket residues.
- Glycan / `MSE` → `MET` re-mapping; sodium-ion explicit retention policy.

## Files produced
- `data/processed/receptors/<UNIPROT>.pdb` (61 files)
- `data/processed/receptors/<UNIPROT>.pdb.prep.meta.json` (one per receptor)
- `data/processed/receptors_selected/`
- `data/processed/structure_selection_summary.json`
- `data/processed/receptor_audit.md`
- `data/registry/preferred_pdbs.tsv`

## Tests
- `tests/unit/test_receptor_preprocessor.py` (patched for DBREF chain mapping + sequence-identity rescue)
- `tests/unit/test_pdb_selector.py`
- `tests/unit/test_het_resnames.py`

## Status
Complete (Done) — 61/61 receptors prepared (34 cached PDBs + 27 AlphaFold ≥ pLDDT 70); wrong-chain bug fixed and unit-tested.
