# 03 — Receptor Curation (PDB Cleaning, Structure Selection)

Modules covered:
- `src/cancerag/preprocessing/receptor_preprocessor.py`
- `src/cancerag/data_collection/receptor_retriever.py` (referenced)
- `src/cancerag/data_collection/alphafold_retriever.py` (referenced)

## What the code does today

`ReceptorPreprocessor.run()`:
1. Globs `data/pdb/**/*.pdb`.
2. For each PDB, runs `_clean_pdb_file` → `Bio.PDB.PDBParser` → `PDBIO.save(..., ProteinSelect())`.
3. `ProteinSelect.accept_residue` keeps only residues whose `id[0] == " "` (i.e., standard residues; drops everything HETATM including water, ions, ligands, lipids, sugars, modified residues, fusion proteins).
4. Output: `data/processed/receptors/<pdb_id>.pdb`.

Structure *selection* (which PDB to use per receptor) lives in `ActiveSiteIdentifier._select_best_structure` — see [04_binding_site_definition.md](04_binding_site_definition.md). This document covers what happens *to* the chosen structure.

## Problems

### Tier 1 — invalidates docking quality

**P3.1 No protein preparation.**
`_clean_pdb_file` strips heteroatoms and re-saves. That is *not* protein preparation. A docking-ready receptor requires:
- Protonation at physiological pH (Reduce / PROPKA / PDB2PQR).
- Removal of alternate locations (altloc 'A' kept, others dropped).
- Decision on structural waters (kept or dropped explicitly, with reasoning).
- Decision on cofactors (Mg²⁺, Na⁺ in GPCR sodium-binding pocket — these *should* be retained for some receptors).
- Capping of chain breaks (usually NME/ACE) or marking them as breaks.
- Assigning Gasteiger or AMBER charges for AutoDock format.
- Conversion to PDBQT (which is what Vina actually consumes).

None of this happens. The "cleaned" PDB is then handed to `docking/preparation.py`, which (per the 126 lines visible) likely runs `obabel` to convert PDB → PDBQT — a one-shot conversion that adds Gasteiger charges but does *not* fix protonation states, doesn't pick altlocs intelligently, and doesn't preserve sodium ions.

**P3.2 `ProteinSelect.accept_residue` is too aggressive.**
By dropping everything with HETATM tag the pipeline removes:
- The conserved sodium ion in the Na⁺-binding pocket of class A GPCRs (residue ID `('H_NA',...)`). This ion stabilizes the inactive state and *changes pocket geometry*. Dropping it without replacing protonation states biases all docking poses toward an artifactual "no-Na" conformation.
- Modified residues (phosphorylated serines, selenomethionine `MSE` substitutions, hydroxyproline, etc.). For GPCRs `MSE` is common and should be re-mapped to `MET`, not deleted.
- N-glycosylation residues (some pocket-adjacent in extracellular domain).
- Fusion-protein residues (T4L, BRIL) — these *should* be removed but only after detecting them by chain ID, not by HETATM status.

**P3.3 No chain selection.**
GPCR PDBs frequently contain multiple receptor chains (asymmetric unit), G-protein chains (Gα, Gβ, Gγ), nanobody, antibody fragments. The current code keeps *all* protein chains, including the G-protein. Vina then sees a 4-chain complex and may dock the ligand into a Gα interface instead of the GPCR pocket.

**P3.4 No alternate-location handling.**
Crystal structures often have altloc A and B for flexible side chains. `Bio.PDB` parses both; `PDBIO.save` writes both unless filtered. Vina parses ambiguous altlocs unpredictably.

### Tier 2 — community-norm violations

**P3.5 No structure quality check before saving.**
A "cleaned" receptor with a 5 Å disordered loop covering the pocket is still saved without warning. Standard practice: report and flag missing residues in the orthosteric pocket region.

**P3.6 No GPCR-specific preprocessing.**
GPCRs need:
- Re-numbering to **GPCRdb generic numbering** (Ballesteros-Weinstein × structure-corrected) so that residue 6.48 in receptor A and receptor B are comparable across the dataset. Without this, "the W6.48 toggle switch is in contact with the ligand" is not computable.
- Loop modeling for missing ECL2 (extracellular loop 2), which lines the pocket entrance.
- Detection of which state (active / intermediate / inactive) the structure represents.

**P3.7 AlphaFold structures get the same minimal treatment.**
`alphafold_retriever.py` (512 lines) downloads AF predictions; they pass through `ReceptorPreprocessor` without:
- pLDDT-based residue masking.
- PAE-based domain trimming.
- Acknowledgement that AF GPCR predictions are usually inactive-like.

A mixed dataset of crystal+cryoEM+AF structures — all preprocessed identically — bakes a structure-source confound into every receptor descriptor.

### Tier 3 — engineering

**P3.8 No idempotency for re-prep.**
Once a cleaned PDB exists at the output path, the cleaning step is skipped (lines 82, 85). If the upstream raw PDB changes (new revision from RCSB), the stale cleaned version is used. No content hashing.

**P3.9 No structured logging of what was removed.**
For each receptor we should know: "removed water (N atoms), kept Na⁺ (M atoms), dropped chain B (L residues, fusion T4L)." Currently the only output is the cleaned PDB.

**P3.10 No tests on biological-sanity of output.**
A test that loads each cleaned receptor and asserts (a) all 7 TM helices are present, (b) the conserved D[E]RY and NPxxY motifs are intact, (c) the orthosteric pocket residues by GPCRdb numbering exist — would catch the worst failures.

## Standard approach

1. **Use a structure preparation toolkit** designed for docking. Open-source options:
   - **PDBFixer** (OpenMM ecosystem) — adds missing atoms, models missing loops, protonates.
   - **PDB2PQR** (with PROPKA) — assigns protonation at user-defined pH.
   - **ADFR / Meeko** — official AutoDock receptor prep (PDB → PDBQT with Gasteiger charges, the *correct* converter, not generic obabel).
   - **Reduce** — adds and optimizes hydrogens.
2. **Use GPCRdb as the canonical GPCR structural source.** Their REST API serves curated PDB lists per receptor with state, fusion, mutation annotations, plus generic-numbering mapping.
3. **Per-residue pLDDT gating for AlphaFold.** Drop residues with pLDDT < 70; if any pocket residue is below threshold, reject the AF model entirely for docking.
4. **Explicit `structure_source` and `activation_state` columns** carried into the feature matrix so downstream models can stratify.

## Concrete fixes

### F3.1 Replace `_clean_pdb_file` with a real prep pipeline

```python
# preprocessing/receptor_curation.py
from pdbfixer import PDBFixer
from openmm.app import PDBFile
from meeko import MoleculePreparation, PDBQTWriterLegacy

def prepare_receptor(input_pdb: Path, output_pdbqt: Path,
                     keep_sodium: bool = True, ph: float = 7.4,
                     gpcr_chain: str = "A") -> dict:
    fixer = PDBFixer(filename=str(input_pdb))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    # Keep only the GPCR chain
    fixer.removeChains([c for c in fixer.topology.chains() if c.id != gpcr_chain])
    # Drop waters but keep ions explicitly
    keep_residues = {"NA"} if keep_sodium else set()
    fixer.removeHeterogens(keepWater=False)  # then re-add Na if removed
    fixer.addMissingHydrogens(pH=ph)

    intermediate = output_pdbqt.with_suffix(".prepped.pdb")
    PDBFile.writeFile(fixer.topology, fixer.positions, open(intermediate, "w"))

    # Convert to PDBQT with Meeko (AutoDock-aware), not obabel
    # ... (Meeko receptor-prep call here)

    return {"residues_kept": ..., "altloc_policy": "A_only",
            "sodium_kept": keep_sodium, "ph": ph}
```

### F3.2 GPCR-specific prep

```python
# preprocessing/receptor_curation.py
def map_gpcrdb_numbering(pdb_path: Path, gpcrdb_id: str) -> dict:
    """Fetch GPCRdb generic numbering and tag each residue."""
    r = requests.get(f"https://gpcrdb.org/services/structure/{gpcrdb_id}/")
    r.raise_for_status()
    return r.json()  # contains residue → 1.50, 2.50, ... mapping
```

Store the mapping as a `<pdb_id>.gpcrdb.json` sidecar.

### F3.3 AlphaFold gating

```python
def gate_alphafold(af_pdb: Path, pocket_residues: list[int],
                   plddt_min: float = 70.0) -> bool:
    """Return True if pocket residues all have pLDDT >= threshold."""
    plddts = parse_plddt(af_pdb)  # b-factor field for AF
    return all(plddts.get(r, 0) >= plddt_min for r in pocket_residues)
```

If the AF model fails the gate, fall back to "no structure available" rather than using a low-confidence pocket.

### F3.4 Structure metadata sidecar

For every prepared receptor write `<pdb_id>.prep.meta.json`:
```json
{
  "source": "pdb_xray|pdb_cryoem|alphafold",
  "pdb_id": "6DRY",
  "uniprot": "P14416",
  "resolution": 2.4,
  "activation_state": "active",
  "kept_chain": "A",
  "fusion_protein_removed": "T4L",
  "sodium_present": true,
  "ph_used_for_protonation": 7.4,
  "missing_pocket_residues": [],
  "altloc_policy": "A_only",
  "preparation_tool_versions": {"pdbfixer": "1.9", "meeko": "0.5.1"}
}
```

### F3.5 Sanity-check tests

```python
# tests/test_receptor_curation.py
def test_prepared_receptor_has_seven_tm_helices(prepared_receptor):
    helix_count = count_helices(prepared_receptor)
    assert helix_count >= 7

def test_dry_motif_intact(prepared_receptor):
    # D(E)RY at TM3.49-51 (Ballesteros-Weinstein)
    assert find_motif_at(prepared_receptor, "DRY", "3.49") or \
           find_motif_at(prepared_receptor, "ERY", "3.49")

def test_pocket_residues_present(prepared_receptor, expected_pocket):
    actual = pocket_residues(prepared_receptor)
    assert set(expected_pocket).issubset(actual)
```

### F3.6 Idempotency by content hash

Replace the existence-only check with:
```python
def needs_reprep(input_pdb: Path, output_pdbqt: Path) -> bool:
    if not output_pdbqt.exists(): return True
    meta = json.loads(output_pdbqt.with_suffix(".prep.meta.json").read_text())
    return meta.get("input_sha256") != sha256(input_pdb)
```

## Acceptance criteria

- [ ] Every receptor in `data/processed/receptors/` has a `.prep.meta.json` with the fields in F3.4.
- [ ] No receptor file contains G-protein chains, nanobodies, or fusion proteins.
- [ ] Sodium ions are retained where appropriate (class A GPCRs in inactive state).
- [ ] Protonation state is set explicitly via PDBFixer + Reduce at pH 7.4.
- [ ] AlphaFold receptors pass the pLDDT gate or are excluded from docking.
- [ ] PDB → PDBQT conversion uses Meeko (AutoDock-official), not generic obabel.
- [ ] Tests verify presence of 7TM, DRY motif, NPxxY motif, and per-receptor pocket residues.
- [ ] Receptor preprocessing is re-run automatically when the input PDB SHA-256 changes.
