# Achieving full docking coverage: defects found and fixed

_Investigation 2026-10-21 to 2026-10-24. Every number below was measured on the
run host (`cancerag-run`, c2d-standard-8, 8 vCPU / 31 GB) or read from artefacts
in `data/processed/`. Where a claim could not be measured it says so._

## Summary

The pipeline reported **554 of 587 pairs docked**. Counting only poses that make
real contact with their receptor, the true figure was **503**. The 51-pose gap
was not visible in any artefact: the affinities were recorded, the quality gate
passed, and nothing downstream complained.

Remediation brought coverage to **587 / 587**, and found that **no ligand in the
dataset is undockable**. Every failure was a resource limit or a mis-prepared
receptor. The sampling-depth exception and peptide-engine substitution that
earlier drafts of this work contemplated are not required and should not be
written into the methods.

Defects are grouped below by what produced them. Each is stated with the
measurement that exposed it, because in every case the wrong number looked
entirely reasonable in isolation.

---

## 1. Receptor preparation

Five of 61 receptors were prepared such that the docking box did not enclose a
pocket of the intended protein. All five produced affinities near zero.

| UniProt | receptor | before | after | fault |
|---|---|---|---|---|
| P41145 | κ-opioid | ≈0 (36 poses) | median −8.46 | prepared chain `R`, which does not exist in 8DZP |
| P41146 | NOP | ≈0 (10 poses) | −6.2 … −9.7 | site from 5DHG, receptor from 4EA3 |
| P41597 | CCR2 | 0.000 | −7.321 | receptor was a 20-atom fragment; box built from an `ACE` cap |
| P51681 | CCR5 | −2.197 | −6.833 | receptor was a 46-atom chain; box built from `TYS` |
| P20309 | M3 | −0.027 | −7.677 | 8EA0 orthosteric pocket too narrow for the ligand |

### 1.1 Root cause for CCR2 and CCR5: capping groups read as ligands

`LIGAND_AUTO_DETECT_IGNORE` (`preprocessing/het_resnames.py`) held 135 entries
and correctly excluded waters, ions, buffers, lipids, glycans and modified
residues such as `MSE`, `SEP`, `TPO` and `PTR`. It did **not** exclude:

```
ACE   N-terminal acetyl cap
NH2   C-terminal amide cap
TYS   sulfotyrosine
```

These are HETATM records attached to the protein, not ligands, and they are
small — so co-crystal auto-detection selected one as "the orthosteric ligand"
and centred a 22 Å box on it. The registry records the consequence plainly:

```
P41597  CCR2   7XA3   ligand_resname = ACE
P51681  CCR5   7O7F   ligand_resname = TYS
```

`TYS` matters more than the others: sulfotyrosine is characteristic of
chemokine-receptor N-termini, so the failure concentrated on that family rather
than scattering randomly.

Fixed by adding a `TERMINAL_CAPS` group (`ACE`, `NH2`, `NME`, `FOR`) and `TYS`
to `MODIFIED_AMINO_ACIDS`. Covered by `tests/unit/test_het_terminal_caps.py`,
which also asserts that genuine ligands (`MRV`, `WEC`, `U99`, `IXO`, `Q6Q`, …)
still pass through, so the list cannot be over-widened.

### 1.2 A prior repair pass fixed the wrong thing

`data/registry/preferred_pdbs.tsv` shows three of these receptors had already
been repaired once:

```
P41145  8DZP  manual_override: "repaired: chain-identity gate; was 7UL2"
P41597  7XA3  manual_override: "repaired: chain-identity gate; was 7P8X"
P51681  7O7F  manual_override: "repaired: chain-identity gate; was 5YY4"
```

The chain-identity gate identified the right receptors. The repair swapped the
PDB. Nobody re-checked what the binding box was built from, which was still a
capping group — so the structure changed and the fault persisted. **A repair
that does not re-validate its own output is not a repair.**

### 1.3 How the receptors were rebuilt

The recipe that worked, applied five times:

1. Choose the structure by **sequence identity to the UniProt canonical**, not
   by size. Rebuilding CCR5 from "largest chain of the largest structure" gave
   7F1T chain A (423 aa, 77.7% identity — a fusion construct) and a **worse**
   result than the broken original (−0.600 vs −2.197). Identity selected 4MBS
   (83.1%), which gave −6.833.
2. Prefer a structure with a **genuine co-crystal ligand**, and take the box
   from that ligand's centroid **in the same file as the receptor chain**. This
   guarantees receptor and box share a coordinate frame.
3. Where no usable co-crystal ligand exists, fall back to P2Rank on the
   *prepared receptor* (same frame by construction). Used for P41146:
   pocket score 23.53, probability 0.890.
4. Assert the box centre lies inside the receptor's coordinate hull **and** that
   ≥50 receptor atoms lie within 12 Å, before docking.

---

## 2. Quality gates that recorded the fault and passed anyway

### 2.1 Re-dock validation has no physical-plausibility check

The κ-opioid record:

```
P41145   rmsd_angstrom: 1.087   passes: true   top_pose_affinity: -0.464
```

A 31-heavy-atom ligand in its own pocket should score roughly −8 to −11.
−0.464 means it is touching almost nothing. The gate passed it because it
checks RMSD and the Gnina CNN score and never asks whether the binding energy
is physically possible.

RMSD compares the docked pose to the crystal ligand. Both were in the same
(wrong) frame, so they agreed to 1.09 Å while neither was near the protein.
**RMSD cannot detect a frame mismatch. Affinity detects it immediately.**

Across all 53 re-dock records only three had affinity weaker than −4, and
P41145 was the only one marked `passes: true`.

Required: an affinity floor in the gate, with the reason recorded on failure.

### 2.2 Chain-identity gate records a number nobody acts on

P51681's prep sidecar carries `chain_identity_to_uniprot: 0.6667` and the
receptor was accepted regardless — a 46-atom chain. The check runs, the value
is stored, nothing rejects on it.

### 2.3 Affinity ≈ 0 has two opposite causes

This matters for any gate built on affinity alone:

| case | contacts within 4 Å | closest approach | meaning |
|---|---|---|---|
| κ-opioid P41145 | ~0 | — | ligand in empty space (box misplaced) |
| M3 P20309 | 156 | **2.20 Å** | ligand clashing (pocket too narrow) |

Both report ≈0 kcal/mol. Only a contact count distinguishes them, and they need
opposite responses — relocate the box versus curate a different conformation.
A gate that sees only the affinity cannot tell them apart.

---

## 3. Featurisation

Fixed under PR #13 and PR #14; recorded here because they affect every
pose-derived feature.

**ProLIF receptor had no bonds.** `mda.Universe(pdb)` derives bonds from
`CONECT` records only, and OpenMM writes those for disulfides alone. A
6672-atom receptor reached ProLIF with **1 bond and 0 aromatic atoms**, so six
of nine interaction types could never fire. The released block held 122 columns
over 521 pairs: `VdWContact` 111, `HBDonor` 9, `Cationic` 2, and nothing else.
Zero hydrophobic contacts across 521 GPCR complexes is not a result.

Reading the same file with RDKit's PDB reader gives 6756 bonds and 327 aromatic
atoms. Measured over 40 pairs: contacts 179 → 737, mean residues contacted
4.4 → 12.7, interaction types present 2 → 5.

**Pose loader guessed ligand chemistry.** When template bond-order assignment
failed, `load_pose_mol` sanitised with `SANITIZE_KEKULIZE` disabled and returned
that — every ring non-aromatic — under a docstring claiming it remained "usable
by ProLIF for … π-stacking perception". It fired on **112 of 455 poses (24.6%)**
and logged at `debug`. Cause: PDBQT drops formal charge, so a protonated
nitrogen reads as valence 5. Replaced with Meeko's own PDBQT round-trip:
522 of 522 poses now load correctly.

**Re-dock torsion counts were wrong.** Co-crystal ligands were parsed from PDB
with no chemistry template, so bond orders came from geometry. Against RCSB
ideal components, **39 of 53 (74%)** had the wrong `TORSDOF`, 37 inflated,
median +2 and up to +8. The κ-opioid showcase ligand `U99` was re-docked with
9 torsions against a true 6.

---

## 4. Scheduling and resources

These produced the majority of the "undockable" pairs, and none of them is a
property of the chemistry.

### 4.1 Concurrency must be bounded by memory, not cores

Peak RSS is **~5–6 GB per job**, set by the grid and conformer space, not by
thread count (confirmed: 5072 MB at both `--cpu 8` and `--cpu 2`). Running 8
jobs on 8 cores demands ~40 GB on a 31 GB host:

```
Out of memory: Killed process (vina) anon-rss:5191808kB
Out of memory: Killed process (vina) anon-rss:5270492kB
Out of memory: Killed process (vina) anon-rss:5758584kB
```

The rule is `max_concurrent = available_memory / peak_RSS_per_job` (≈6 here).
Memory demand rises with torsion count, so sizing by cores destroys **exactly
the largest ligands** — a systematic, class-correlated loss rather than a random
one.

### 4.2 A time budget must be in core-seconds and adaptive

The original cap was a flat 1800 s of wall-clock, applied to single-threaded
jobs. Two separate errors: the unit (wall-clock vs core-seconds, which coincide
only at one thread) and the calibration.

Measured on the run host, `MJQQGUJPWJCFMO__P30556` (27 torsions) needs **4512
core-seconds** and was allowed 4050 — it failed by 11%.

The deeper problem is that **cost is a property of the ligand–receptor pair,
not the ligand**:

| ligand | torsions | receptor A | receptor B |
|---|---|---|---|
| `GZWUQPQBOGLSIM` | 51 | MC1: 10,195 s ✓ | MC3: >10,800 s ✗ |
| `AGTSSZRZBSNTGQ` | 56 | P35372: 9,558 s ✓ | P41143: >10,800 s ✗ |

Identical ligand, identical torsion count, opposite outcome — pocket shape
decides. Any torsion-only formula will therefore clip some pairs arbitrarily.
The budget must detect the timeout and retry with an extension; MC3 succeeded
at 15,691 s once given one.

### 4.3 Observed cost curve

Exhaustiveness 16, 8 threads, uncontended:

| torsions | wall seconds |
|---|---|
| ~6 (drug-like) | seconds |
| 23 | 1,027 |
| 39 | 8,011 |
| 51 | 10,195 – 15,691 |
| 56 | 9,558 – 10,682 |
| 60 | 12,415 |

### 4.4 AutoDock atom types are case-sensitive

Barium at the calcium-sensing receptor (P41180) is a genuine orthosteric
agonist, not a curation artefact. An unsupported-element remap substituted
`Ba → CA`, which Vina rejects:

```
PDBQT parsing error: Atom type CA is not a valid AutoDock type
```

Calcium is `Ca`. With the correct casing the ion docks at −1.624 in under a
second. `Ca`, `Mg` and `Zn` all return the identical score, so the choice of
substitute metal does not influence the result — only its validity as a
parseable type.

---

## 5. Pose validity criterion

A flat affinity threshold is wrong at both extremes of ligand size, because
binding energy scales with the number of heavy atoms.

| pair | affinity | heavy atoms | ligand efficiency |
|---|---|---|---|
| `MFJVHWQDKIEBCM__Q9H3N8` | −3.867 | 20 | 0.193 |
| `XVOCEQLNJQGCQG__Q92847` | −2.980 | 110 | 0.027 |
| `XDFCIPNJCBUZJN__P41180` (barium) | −1.624 | 1 | 1.624 |

A −4 cutoff rejects a healthy fragment and a monatomic ion; a ligand-efficiency
cutoff rejects large peptides that are binding legitimately.

Validity is not a question of binding *strength* but of whether the search
converged and the ligand is in contact with the receptor. The criterion used
here is **`rc == 0` and affinity < −1**, with contact count available to
distinguish the two failure modes in §2.3.

Counts under each rule, for transparency:

| criterion | valid |
|---|---|
| affinity ≤ −4 | 562 |
| **affinity < −1** | **566** |
| ligand efficiency ≥ 0.10 | 554 |

---

## 6. Engine substitution and its limits

Uni-Dock v1.2.0 (CUDA, Vina scoring function) was evaluated on Kaggle T4s to
accelerate the flexible tail. It reproduces CPU Vina closely — and the
reproduction was judged against Vina's own run-to-run spread rather than an
invented tolerance:

```
n = 15     signed mean(cpu − gpu) = +0.011 kcal/mol
|cpu − gpu| : mean 0.069   max 0.259
|gpu − gpu| : mean 0.086   max 0.347     <- intrinsic noise, two seeds
```

Cross-engine disagreement is **smaller than the engine's disagreement with
itself**, so the affinities are interchangeable.

Two limits were measured:

- Uni-Dock **silently rejects** some ligands — `GZWUQPQBOGLSIM` (51 torsions)
  exited in 2–3 s with `rc=0` and no output against all four of its receptors.
  CPU Vina docks it in 10,195 s. Silent rejection is worse than failure here,
  because the batch reports success.
- GPU speed-up is throughput across many ligands, not latency on one hard
  ligand; a single large peptide is not meaningfully faster.

13 poses were merged from Uni-Dock where CPU had produced none, with engine and
exhaustiveness recorded per pair in `docking_provenance.json`.

**Leakage note.** Engine, exhaustiveness and receptor-structure provenance are
per-row properties correlated with ligand class — the GPU set is precisely the
hard peptides. That is the same granularity trap as the per-receptor QC columns
in `QC_LEAKAGE_CORRECTION.md`, which four evaluation regimes failed to catch.
Before any model is trained on this matrix, train on the provenance columns
alone and confirm they do not predict the label.

---

## 7. Outcome

| | pairs |
|---|---|
| reported before remediation | 554 |
| **valid before remediation** | **503** |
| **valid after remediation** | **587** |

ERK, the rarest class (59 measurements) and the one the four-way formulation
exists to capture, was missing 8.5% of its pairs against 1–2% for every other
class — a non-random loss concentrated on the melanocortin receptors and one
ligand, `[D-Trp8]-γ-MSH`, which appears at four of them. All five ERK pairs were
recovered; each needed 10,000–25,500 s, and each had failed at the original
3 h cap.

## 8. What must change in the pipeline

1. Affinity plausibility floor in the re-dock gate, with the reason recorded.
2. Assert the box centre lies inside the receptor hull with ≥50 atoms within
   12 Å, before docking.
3. Reject on `chain_identity_to_uniprot` rather than only recording it.
4. Concurrency bounded by `available_memory / peak_RSS_per_job`.
5. Adaptive time budget: core-seconds, detect timeout, extend and retry.
6. Contact count alongside affinity, to separate "no contacts" from "clashing".
7. Distinguish failure modes in logs. `TIMEOUT` was used for any unparseable
   result, which hid OOM kills for most of a day; vina's stderr was discarded.
   Exit codes carry the answer: `124` timeout, `137` SIGKILL, `0` success.
