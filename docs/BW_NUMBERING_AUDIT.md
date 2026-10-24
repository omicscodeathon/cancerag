# Ballesteros–Weinstein contact remapping: rationale, validation and result

> **Superseded numbers, 2026-10-24.** The reasoning and method in this audit
> stand, but every count here was measured against a 521-pose dataset that is
> now known to have been wrong in two ways: 49 of those poses were ligands
> sitting outside their receptor (five mis-prepared receptors), and the ProLIF
> interaction block itself could only ever emit 3 of its 9 contact types
> because the receptor reached it with 1 bond. Coverage is now 587/587 and the
> featuriser is fixed. **Do not quote the contact counts or the Δ macro-F1
> figures below.** Re-measure against the current artefacts. See
> `DOCKING_REMEDIATION.md`.


_Audit written 2026-09-23. Every number below was re-measured from artefacts in
`data/processed/` at that date; the reproduction script is given at the end.
Where a figure differs from the one quoted in commit `41fd1cc`, the measured
value is used and the discrepancy is recorded in "Numbers that do not
reproduce"._

## Summary

ProLIF names each interaction column after the residue as numbered in the
prepared structure — `ifp_ASP86.A_VdWContact`. That number is a property of the
PDB file, not of biology: prolif receptors are renumbered from 1 and every
construct has a different N-terminal length, so the same functional residue
lands on a different number in every receptor.

The conserved TM3 aspartate that anchors every aminergic ligand was therefore
split across five columns (ASP80/83/84/86/91). Of 384 contact columns, the
median fired in 2 of 443 training rows, and the block carried 1.3 % of model
attribution — not because contacts do not matter, but because each was shattered
into fragments too small to learn from.

Remapping to Ballesteros–Weinstein generic position reduced 384 contact columns
to 122, rebuilt 42 positions from more than one original column, and cut columns
firing in fewer than 5 rows from 295 to 75. Block attribution roughly doubled,
from 1.3 % to 2.8 %.

**It did not change the conclusion.** The receptor-dependent block still
contributes nothing measurable. The remapping matters because it removes
fragmentation as an explanation for that null result.

## The problem, measured

| | before | after |
| --- | ---: | ---: |
| Contact columns | 384 | **122** |
| Columns firing for exactly one receptor | 362 | **80** |
| Columns firing in < 5 rows | 295 | **75** |
| Median rows a column fires in | 2 | 2 |
| SHAP attribution carried by the block | 1.3 % | **2.8 %** |

The single-receptor row is the clearest statement of the problem: before the
remapping **362 of 384 contact columns described one receptor and no other**, so
almost nothing in the block could generalise across receptors by construction.

The median is unchanged because BW numbering merges the *conserved* positions,
which are exactly the ones that were fragmented across receptors. Positions
contacted in a single receptor stay singletons, correctly — they are genuinely
idiosyncratic, not fragmented.

## What the remapping does

BW numbering labels a residue by structural position: helix number, then offset
from that helix's most conserved residue, defined as 50. Position 3.32 is the
same position in every class A GPCR regardless of construct, so a rule learned
at β2-adrenergic can apply at D2 dopamine.

The mapping chain, with GPCRdb as the authority for the generic numbers:

```
prolif residue index  --[sequence alignment]-->  UniProt position
UniProt position      --[GPCRdb residue table]-->  BW generic number
```

Columns are named by **position, not residue identity** (`ifp_BW3.32_HBDonor`,
not `ifp_ASP3.32_HBDonor`). Merging is the point, and the amino acid at a given
position differs between families; keeping the residue name would leave the
fragments separate.

### The trap: read the protonated structure, never the prepared one

ProLIF writes its column names against the `.prolif_H.pdb` file, which is
renumbered from 1. The prepared `.pdb` keeps crystal numbering — κ-opioid runs
57..339. Reading `ifp_ASP82` against the prepared file resolves to a residue
28 Å outside a 22 Å docking box; against the prolif file it is D3.32, the
conserved salt-bridge anchor.

Both lookups succeed silently. The mistake surfaces as implausible pharmacology,
not as an error.

## Validation: anchor agreement, not sequence identity

The obvious gate — require the structure to align to its own canonical UniProt
sequence above some identity threshold — is the wrong one, and rejects precisely
the family the remapping helps most.

Seven aminergic receptors align to their own canonical sequence at only
0.64–0.77, being non-human orthologs or engineered constructs. Their mappings
are nevertheless structurally perfect:

| Accession | Receptor | Identity | Anchors matched | Residues mapped |
| --- | --- | ---: | :---: | ---: |
| P11229 | M1 muscarinic | 0.636 | 5/5 | 271 |
| P14416 | D2 dopamine | 0.704 | 5/5 | 256 |
| P18825 | α-2C adrenergic | 0.735 | 5/5 | 264 |
| P08913 | α-2A adrenergic | 0.751 | 5/5 | 256 |
| P21917 | D4 dopamine | 0.756 | 5/5 | 258 |
| P08172 | M2 muscarinic | 0.758 | 5/5 | 261 |
| P35462 | D3 dopamine | 0.772 | 5/5 | 259 |

An 0.80 identity cut would have discarded all seven.

The gate used instead is **anchor agreement**: the most conserved residue of each
helix (1.50 N, 2.50 D, 3.50 R, 6.50 P, 7.50 P) must land on the expected amino
acid. If an alignment is sound these land correctly; if it has slipped, they do
not. Identity is still computed, but over *aligned pairs* rather than the whole
construct, so a T4-lysozyme or BRIL fusion spliced into ICL3 does not fail on
the fusion partner's length.

Anchor agreement separates cleanly across all 57 receptors:

| Anchors matched | Receptors |
| :---: | ---: |
| 5 of 5 | **54** |
| 4 of 5 | 2 |
| 0 (no class A anchors present) | 1 |

The single receptor matching none is the class C calcium-sensing receptor
(P41180). GPCRdb returns 223 generic positions for it, but none are the class A
anchors — class C does not use class A numbering. It correctly retains its
per-structure column names. That outcome is the gate working, not failing.

## Result

| Position | Original columns merged | Rows fired (of 521 docked pairs) | Largest single fragment replaced |
| --- | ---: | ---: | ---: |
| BW6.51 VdWContact | 25 | 199 | 89 |
| BW3.32 VdWContact | 14 | 125 | 30 |
| BW3.32 HBDonor | 9 | — | — |

Across the block: **91 BW-named columns** rebuilt from 42 multi-source positions,
plus **31 per-structure columns retained** for receptors outside class A
numbering — 122 contact columns in total, down from 384.

## Independent corroboration: κ-opioid

The κ-opioid receptor (P41145, 36 docked pairs) contacts these BW positions in
the shipped features:

> 2.53, 2.60, **3.32**, 3.33, **3.36**, 5.42, **6.48**, **6.55**, **7.35**,
> **7.36**, 7.42, **45.50**

D3.32 is the orthosteric salt-bridge anchor; W6.48 is the toggle-switch
tryptophan; Y7.35/7.36, Y3.33, M3.36, I6.55 and C45.50 line the pocket. This is
textbook orthosteric opioid pharmacology, and it was unreadable before the
remapping — the same contacts existed, spread across dozens of
single-receptor columns.

The same receptor re-docks its co-crystallised ligand to 1.14 Å RMSD, so the
pharmacology is not being read off a bad pose.

## What it changed, and what it did not

Block attribution roughly doubled, from 1.3 % to 2.8 %. It remains small.
Measured across all four evaluation regimes (`shap_block_attribution.csv`):

| Block | Features | SHAP share (scaffold-grouped) |
| --- | ---: | ---: |
| ligand-only 2D descriptor | 181 | 56.9 % |
| ligand-only 2D fingerprint | 1462 | 34.4 % |
| docked-pose 3D shape | 7 | 4.2 % |
| receptor–ligand contacts (BW-merged) | 88 | 2.8 % |
| pair-level docking energetics | 5 | 1.7 % |

Roughly 91 % of attribution is ligand-only. The block-wise ablation
(`data/processed/ml_models/extras/ablation_by_block.csv`) agrees: dropping the
132 receptor-dependent columns changes macro-F1 by −0.002 / +0.000 / +0.004
under receptor-grouped, scaffold-grouped and stratified CV — each inside the
seed spread of its own baseline.

**This is the point of the remapping.** Before it, a reviewer could reasonably
object that the structural arm looked useless only because its features had been
shattered by an arbitrary numbering convention. That objection is now closed:
the contacts are comparable across receptors, they recover canonical
pharmacology, and they still add nothing measurable.

## Defects found while writing this audit

**1. `data/processed/bw_maps.json` is stale and must not be read as describing
the shipped features.** It was written at 10:35 on 2026-08-10; the features were
built at 15:26 the same day, after the anchor-agreement gate replaced an
identity threshold. The file carries **16 empty maps**, including all seven
aminergic receptors above and six receptors that map at identity ≥ 0.96. The
shipped features do not share this defect — D2 alone contacts 18 BW positions
across 96 pairs, which is only possible with the corrected gate. Anyone reading
`bw_maps.json` to learn which receptors were mapped will get the wrong answer.
Regenerate it, or delete it.

**2. The per-receptor validation reports are not persisted at all.** The
`receptor_bw_map()` report dict — identity, anchors matched, residues mapped,
status — is returned and discarded. The 54/2/1 breakdown above had to be
recomputed for this audit; it survives only because the GPCRdb cache
(`data/raw/gpcrdb/`, 114 files) and the `.prolif_H.pdb` structures are both on
disk. Persist the reports alongside the maps.

**3. `bw_column_name()` is dead code and contradicts the shipped convention.**
`src/cancerag/features/bw_numbering.py:290` builds `ifp_ASP3.32_Cationic`, which
embeds the residue identity the remapping exists to discard. It has no callers.
Delete it, or correct it to the `ifp_BW3.32_*` form.

## Numbers that do not reproduce

The following figures from commit `41fd1cc` could not be reproduced against any
artefact on disk, under either a per-column or a per-position reading:

| Claim in commit message | Measured |
| --- | --- |
| 372 → 122 contact columns | **384** → 122 |
| Columns firing < 5 rows: 277 → 75 | **295** → 75 |
| BW3.32 fires in 142 rows, largest fragment 27 | **125** rows, largest fragment **30** |
| BW6.51 merges 25 into 249 rows | 25 into **199** rows |
| 305 of 386 columns fire for one receptor | **362** of **384** |

The column-structure claims (122 columns, 42 multi-source positions, BW3.32
merging 14, BW6.51 merging 25) and the anchor-agreement breakdown (54/2/1) all
reproduce exactly. Only the row-firing counts differ, and they differ in both
directions, which is consistent with the commit having been written against an
intermediate run rather than the artefacts that shipped.

**Use the measured values.** The commit-message figures should not be quoted in
the manuscript.

The 1.3 % pre-remapping attribution figure is also not independently
reproducible — no pre-BW SHAP artefact was retained — and is carried here on the
authority of the commit message alone. If it appears in the manuscript it should
be attributed as such, or re-derived by rebuilding the block from
`interaction_fingerprints.parquet.bak`.

## Reproducing this audit

```bash
# anchor agreement across all 57 receptors (offline; uses data/raw/gpcrdb cache)
python - <<'PY'
import json, sys; sys.path.insert(0, "src")
from pathlib import Path
from cancerag.features.bw_numbering import receptor_bw_map
maps = json.load(open("data/processed/bw_maps.json"))
reports = [receptor_bw_map(a, Path(f"data/processed/receptors/{a}.prolif_H.pdb"))[1]
           for a in sorted(maps)
           if Path(f"data/processed/receptors/{a}.prolif_H.pdb").exists()]
for n in (5, 4, 0):
    print(f"{n} anchors:", sum(1 for r in reports if r["anchors_ok"] == n))
PY

# column-structure statistics
python - <<'PY'
import pandas as pd, json
new = pd.read_parquet("data/processed/interaction_fingerprints.parquet")
old = pd.read_parquet("data/processed/interaction_fingerprints.parquet.bak")
oc = [c for c in old.columns if c.startswith("ifp_")]
nc = [c for c in new.columns if c.startswith("ifp_")]
print(f"contact columns {len(oc)} -> {len(nc)}")
print(f"firing <5 rows  {((old[oc]!=0).sum()<5).sum()} -> {((new[nc]!=0).sum()<5).sum()}")
cm = json.load(open("data/processed/bw_column_map.json"))
print("multi-source positions:", sum(1 for v in cm.values() if len(v) > 1))
PY
```

## See also

- [`QC_LEAKAGE_CORRECTION.md`](QC_LEAKAGE_CORRECTION.md) — the per-receptor
  covariate leakage, diagnosed by the same granularity argument that motivates
  this remapping: ask what a column can actually vary with.
- `src/cancerag/features/bw_numbering.py` — implementation and its docstrings.
- `data/processed/ml_models/interpretability/shap_block_attribution.csv` —
  post-remapping block attribution.
- `data/processed/ml_models/extras/ablation_by_block.csv` — the block-wise
  ablation this remapping was a precondition for.
