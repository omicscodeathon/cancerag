# 04 — Binding-Site / Pocket Definition

Modules covered:
- `src/cancerag/features/active_site_identifier.py`
- `src/cancerag/preprocessing/receptor_preprocessor.py::extract_binding_site`

## What the code does today

`ActiveSiteIdentifier.run()`:

1. Loads `data/pdb/summary.json` → `{receptor: {pdb_id: path}}`.
2. For each receptor, scores every PDB:
   - `+50` if any non-IGNORE_LIST HETATM is present.
   - `+30` if `resolution < 3.0`, `+40` if `resolution < 2.5` (mutually exclusive — see P4.2).
   - `+ completeness * 20` (where completeness = `min(atom_count / 1000, 1.0)`).
3. Picks the top-scoring PDB.
4. Calls `extract_binding_site(pdb_path, ligand_name)`:
   - Parses the PDB; iterates residues; the *first* HETATM residue not in `IGNORE_LIST = {HOH, WAT, SO4, GOL, PO4, EDO, MG, CA, ZN, MN, CL, NA, K}` is treated as "the ligand."
   - Computes bounding box of that residue's atoms; adds 5 Å padding.
   - Returns `{center_x/y/z, size_x/y/z}`.
5. Writes `data/processed/binding_sites.json` and `structure_selection_summary.json`.

If `extract_binding_site` returns `None`, the receptor has no entry in `binding_sites.json`. `config.yaml` defines a `default_binding_site` of 25×25×25 Å but no center — the fallback path is unclear from the code.

## Problems

### Tier 1 — invalidates docking

**P4.1 "First HETATM not in IGNORE_LIST" is not ligand identification.**
GPCR PDBs routinely contain 5–15 distinct HETATM residue types. The IGNORE_LIST in both files (`active_site_identifier.py:79-93` and `receptor_preprocessor.py:131-145`) excludes only water/glycerol/sulfate/PEG-like buffers and a few ions. It does NOT exclude:
- **Lipids/detergents**: `CLR` (cholesterol), `OLA` (oleic acid), `OLC` (1-oleoyl-glycerol), `PEE`/`POV`/`PCW` (phospholipids), `LMT`, `NAG`/`MAN` (glycans), `BOG` (β-octylglucoside), `LFA`/`LMU` (lauryl chains). All are nearly universal in GPCR cryo-EM/X-ray structures.
- **Crystallization additives**: `MES`, `HEPES`, `PEG`/`PG4`, `BU2` (1,4-butanediol), `MPD`.
- **Cofactor / fusion-protein ligands**: nanobody-bound peptides, BRIL-bound co-factors.
- **Allosteric modulators co-bound with orthosteric ligands** (β1AR with cholesterol + isoprenaline, for example).

Because the first iteration order in `Bio.PDB` is chain → residue (sorted by residue number), the "first HETATM" is almost always the wrong molecule. **The orthosteric ligand is rarely first.**

**P4.2 Resolution scoring has a logic bug.**
Lines 112-115:
```python
if metrics["resolution"] < 3.0:
    score += 30.0
elif metrics["resolution"] < 2.5:
    score += 40.0
```
The `elif` branch is **unreachable** — anything `< 2.5` is also `< 3.0`. So a 2.0 Å structure gets +30, identical to a 2.8 Å structure. Best resolution gives no advantage.

**P4.3 No activation-state filter.**
GPCRs crystallize in inactive (antagonist), intermediate, and active (agonist + G-protein) conformations. The orthosteric pocket geometry differs by 3–8 Å between states. Predicting *agonist* bias from a docking into an *inactive* pocket is mechanistically backwards. The selector has no awareness of state.

**P4.4 No filter on engineering modifications.**
GPCR PDBs contain BRIL/T4L fusion insertions, thermostabilizing mutations (e.g., m23 ALA mutations), C-terminal truncations, point mutations to lock conformation. Some of these change pocket geometry; selection by resolution + ligand presence ignores them entirely.

**P4.5 Box size is determined by the bound ligand.**
Lines 174-175: `size = (max_coords - min_coords) + 2*padding`. A small fragment co-crystal yields a small box; a peptide yields a 30+ Å box. This means *box volume correlates with the chemotype of the original co-crystallized ligand*, biasing every subsequent docking against that receptor.

**P4.6 5 Å padding is uniform regardless of ligand size.**
Combined with P4.5, this gives boxes that are too small for drug-like ligands when the cocrystal was a fragment, and too large (allowing exploration of secondary subpockets) when the cocrystal was a peptide.

**P4.7 No verification that the box covers the canonical orthosteric pocket.**
There is no sanity check against known GPCR orthosteric residues (D3.32 in aminergics, conserved W6.48 toggle switch, CWxP/PIF motifs). A trivial Cα-distance check would catch the worst failures.

### Tier 2 — community-norm gaps

**P4.8 Single PDB per receptor.**
Even within "active" states, GPCR pockets fluctuate. A docking ensemble across 3–5 PDBs / MD frames is the standard upgrade.

**P4.9 No pocket-prediction comparison.**
The pipeline relies on co-crystal centroids. There are six categories of pocket prediction (geometric, energy-based, sequence-based, classical-ML, deep-learning, equivariant-DL) — none are used as alternatives or sanity checks. P2Rank, fpocket, DeepPocket, Kalasanty are all open source and can be run in seconds-to-minutes per PDB.

**P4.10 No cryptic-pocket / induced-fit awareness.**
For receptors where bias may involve allosteric or cryptic sites, methods like **PocketMiner** would be relevant. None used.

**P4.11 No GPCRdb pocket residue list.**
GPCRdb publishes pocket residue lists per receptor (orthosteric + allosteric). These should be used as ground truth for box validation and for downstream pair-level pocket descriptors.

### Tier 3 — engineering

**P4.12 Two implementations of `IGNORE_LIST`.**
`active_site_identifier.py:79-93` and `receptor_preprocessor.py:131-145` duplicate the list. They will drift.

**P4.13 `_evaluate_pdb_structure` parses PDB by reading lines.**
String-slicing PDB lines (`line[17:20].strip()`) is fragile to non-canonical PDB files. Use `Bio.PDB` consistently.

**P4.14 Idempotency check is "both files exist."**
Lines 203-208. Doesn't detect upstream changes (new PDB, new IGNORE_LIST). Should hash inputs.

**P4.15 No structured selection report.**
`structure_selection_summary.json` lacks: which alternative PDBs were considered, why each was rejected, the auto-detected ligand identity, residue distance to canonical pocket center. Reviewers will ask.

## Standard approach

1. **Use a curated pocket source** (GPCRdb for GPCRs) as the *primary* pocket definition.
2. **Cross-validate with at least one ML pocket predictor** (P2Rank or DeepPocket) to provide an independent estimate.
3. **Re-dock the crystal ligand** (when present) into the proposed box; require RMSD < 2.5 Å for the box to be considered valid.
4. **Multiple PDBs per receptor**, with state-aware selection (active vs inactive); aggregate docking across the ensemble.
5. **State-aware ligand identification**: filter HETATMs by molecular weight (drug-like 150–800 Da), distance to TM helices, and exclusion of curated buffer/lipid lists.

## Concrete fixes

### F4.1 Replace ad-hoc HETATM picker with curated logic

```python
# features/binding_site/ligand_identifier.py
LIPID_BUFFER_LIST = {
    "HOH","WAT","SO4","GOL","PO4","EDO","MG","CA","ZN","MN","CL","NA","K",
    "CLR","OLA","OLC","PEE","POV","PCW","LMT","NAG","MAN","BOG","LFA","LMU",
    "MES","HEPES","PEG","PG4","BU2","MPD","TRS","ACT","FMT","DMS","BME",
}

def identify_orthosteric_ligand(structure, gpcrdb_pocket_residues,
                                mw_min=150, mw_max=800,
                                max_dist_to_pocket_center=8.0) -> Residue | None:
    """Return the HETATM residue that is most likely the orthosteric ligand."""
    candidates = []
    pocket_center = compute_pocket_center(structure, gpcrdb_pocket_residues)
    for residue in structure.get_residues():
        if residue.id[0] == " ": continue
        resname = residue.get_resname().strip()
        if resname in LIPID_BUFFER_LIST: continue
        mw = approximate_residue_weight(residue)
        if not (mw_min <= mw <= mw_max): continue
        d = distance(residue_centroid(residue), pocket_center)
        if d > max_dist_to_pocket_center: continue
        candidates.append((d, residue))
    if not candidates: return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
```

### F4.2 Fix the resolution-scoring bug

```python
# Replace lines 109-117 with a continuous penalty:
score = 0.0
score += 50.0 if metrics["has_ligand"] else 0.0
score -= 10.0 * max(0.0, metrics["resolution"] - 2.0)  # penalty above 2.0 Å
score += 20.0 * metrics["completeness"]
# state bonus (see F4.3)
score += 30.0 if metrics["activation_state"] == "active" else 0.0
```

### F4.3 State-aware structure selection

```python
def fetch_gpcrdb_state(pdb_id: str) -> str:
    """Returns 'active', 'intermediate', or 'inactive' from GPCRdb."""
    r = requests.get(f"https://gpcrdb.org/services/structure/{pdb_id}/")
    return r.json().get("state", "unknown").lower()
```

### F4.4 Box validation by re-docking

```python
def validate_box_by_redock(receptor_pdbqt, ligand_pdbqt, box, vina_path) -> dict:
    """Redock the crystal ligand into the box; return RMSD to bound pose."""
    pose = vina_dock(receptor_pdbqt, ligand_pdbqt, box, vina_path,
                     exhaustiveness=16, num_modes=9)
    rmsd = compute_rmsd(pose.coords, ligand_pdbqt.bound_coords)
    return {"redock_rmsd": rmsd, "passes": rmsd < 2.5}
```

Receptors whose box fails validation are flagged; their docking-derived features are explicitly marked as low-confidence.

### F4.5 Multi-method pocket comparison

```python
# features/binding_site/pocket_methods.py
def predict_pockets_p2rank(pdb: Path) -> list[Pocket]:
    """Wrap P2Rank JAR; return ranked pockets with center + score."""
    ...

def predict_pockets_fpocket(pdb: Path) -> list[Pocket]: ...
def predict_pockets_deeppocket(pdb: Path) -> list[Pocket]: ...

def consensus_pocket(pockets_per_method: dict[str, list[Pocket]]) -> Pocket:
    """Median centroid across methods; flag if methods disagree by > 5 Å."""
    ...
```

Add a comparison report (`results/reports/pocket_comparison.csv`) per receptor: each method's pocket center, distance to GPCRdb pocket centroid, distance to crystal-ligand centroid.

### F4.6 Standard box size

Don't size the box from the bound ligand. Use a fixed receptor-class-appropriate size (22 Å cube for class A GPCR orthosteric pocket; tunable per receptor in the registry).

```python
# data/registry/receptors.tsv: add column box_size_angstroms
# default 22 for class A, 28 for class B/C, etc.
```

### F4.7 Structured selection report

```python
# Per receptor write structure_selection/<receptor>.json:
{
  "selected_pdb": "6DRY",
  "alternatives": [
    {"pdb": "6DRX", "score": 78.0, "rejected_reason": "inactive state"},
    {"pdb": "7TPN", "score": 82.0, "rejected_reason": "lower resolution"},
  ],
  "selected_ligand_resname": "FB1",
  "ligand_identification_method": "mw_pocket_distance_filter",
  "box_center": [12.3, -4.5, 8.7],
  "box_size_angstroms": 22.0,
  "box_validation": {"redock_rmsd": 1.4, "passes": true},
  "pocket_method_comparison": {
    "p2rank_distance_to_box_center": 1.1,
    "fpocket_distance_to_box_center": 2.3,
    "gpcrdb_distance_to_box_center": 0.8,
  }
}
```

### F4.8 Consolidate `IGNORE_LIST`

Single source of truth: `cancerag/preprocessing/het_resnames.py`.

## Acceptance criteria

- [ ] No receptor uses "first HETATM" logic for ligand identification; all use the MW + pocket-distance filter.
- [ ] Resolution penalty is continuous; the unreachable `elif` is removed.
- [ ] Every selected PDB has an explicit activation-state tag (active / intermediate / inactive) and the selector prefers active for agonist-bias prediction.
- [ ] Every binding box is validated by re-docking the crystal ligand; receptors where re-dock RMSD > 2.5 Å are flagged.
- [ ] At least two independent pocket-prediction methods (e.g., P2Rank + fpocket) are run and compared per receptor; agreement is reported.
- [ ] Box size is set per receptor class via the registry, not by the cocrystal ligand's bounding box.
- [ ] `IGNORE_LIST` exists in exactly one module.
- [ ] `structure_selection/<receptor>.json` exists per receptor with the schema in F4.7.
