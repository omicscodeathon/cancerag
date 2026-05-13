# Stage 04 — Binding-site definition tracking

## Original state (pre-rebuild)
`ActiveSiteIdentifier.run()` scored every PDB per receptor with a brittle heuristic: `+50` for any non-IGNORE_LIST HETATM, `+30` if `resolution < 3.0`, `+40` if `resolution < 2.5` (an unreachable `elif` — anything < 2.5 is also < 3.0, so best-resolution structures got no advantage), and `+ completeness * 20`. After picking the top PDB it called `extract_binding_site`, which iterated residues and treated **the first HETATM not in a 12-element IGNORE_LIST** as "the ligand." That IGNORE_LIST excluded only water/glycerol/sulfate/PEG and a few ions — it did not filter cholesterol (CLR), oleic acid (OLA), phospholipids (PEE/POV/PCW), detergents (LMT/BOG), glycans (NAG/MAN), or crystallization additives (MES/HEPES/PEG/MPD). Box size was the bound ligand's bounding box + 5 Å — i.e. box volume correlated with the chemotype of whichever ligand happened to be co-crystallized. There was no activation-state filter, no validation of whether the box even covered the canonical orthosteric pocket, no re-dock RMSD check, no pocket-prediction cross-check, and no comparison with an alternative scoring tool.

## Upgrades performed
- Cocrystal-ligand-derived binding boxes consolidated through a single `het_resnames.py` source-of-truth IGNORE_LIST (no more two-file drift).
- Cross-checked every box against P2Rank and fpocket pocket predictions; agreement reported per receptor in the binding-sites audit.
- **Vina re-dock validation** for each cocrystal receptor: extract bound ligand → re-dock into the proposed box at production parameters → require RMSD ≤ 2.5 Å for the box to be flagged "high confidence." Result: 29/55 cocrystal receptors pass after the Stage 03 chain-mapping fix (was 22/55 before).
- **Gnina CNN rescoring** (per-receptor) of the top Vina pose; CNN ≥ 0.7 = additional confidence signal. Joined into Stage 05 docking confidence flag.
- Continuous resolution penalty replaces the broken `elif` step function.
- Structured per-receptor selection report including alternatives considered and rejection reasons.

## What was NOT done (deferred / out-of-scope)
- Activation-state (active / intermediate / inactive) filtering via GPCRdb metadata — selector does not yet prefer active state for agonist-bias prediction.
- Multi-PDB / MD-frame ensemble docking per receptor.
- DeepPocket / Kalasanty / PocketMiner cryptic-pocket comparison (P2Rank + fpocket considered sufficient).
- Per-receptor box size driven by GPCR class via the registry — current boxes still inherit the cocrystal bounding-box dimensions where available.
- GPCRdb-published orthosteric-residue lists used as ground truth for validation.

## Files produced
- `data/processed/binding_sites.json` — 61 binding boxes
- `data/processed/binding_sites_audit.md`
- `data/processed/redock_validation.json` (+ `redock_validation_audit.md`)
- `data/processed/gnina_rescore.json` (+ `gnina_rescore_audit.md`)
- `data/processed/structure_selection_summary.json`
- `src/cancerag/preprocessing/het_resnames.py` (single source of truth)

## Tests
- `tests/unit/test_active_site_identifier.py`
- `tests/unit/test_binding_site.py`
- `tests/unit/test_pocket_predictors.py`
- `tests/unit/test_redock_validation.py`
- `tests/unit/test_gnina_rescore.py`
- `tests/unit/test_het_resnames.py`

## Status
Complete (Done) — boxes are validated by re-dock RMSD + Gnina CNN; per-receptor confidence flags feed Stage 05.
