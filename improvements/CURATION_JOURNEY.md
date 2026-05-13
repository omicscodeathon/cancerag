# CancerAg ligand-curation journey

This document records the iterative work that brought the CancerAg data
curation pipeline from a broken legacy state to a defensible, reviewer-
ready dataset. It exists so that anyone reading the repo a year from now
— including a manuscript reviewer — can understand exactly what
transformed the raw 727-row BiasDB cache into the final
`unified_ligands.csv`, what decisions were made, and why.

The journey covers Stage 02 (ligand curation) and Stage 03 (receptor
curation). Pipeline stages downstream are not yet wired into a running
pipeline at the time of writing.

---

## 1. Where we started

The legacy pipeline produced a `unified_ligands.csv` with 504 rows, of
which 99 were silently labelled `bias_category="Agonist"` because they
came from ChEMBL — a database where bias was *not* measured. The
classifier reported 76.92% accuracy, but the reviewers (Scientific
Reports, March 2026 rejection) flagged the result as untrustworthy. Our
internal audit confirmed five existential problems at the curation
layer:

| ID | Problem |
|---|---|
| E1 | ChEMBL rows hard-coded to `bias_category="Agonist"` — model learned database-of-origin shortcut, not bias |
| E2 | InChIKey-only deduplication collapsed measurements at different receptors / assays into single rows |
| E3 | Receptor names matched by free-text string ("5HT1A receptor", "5-HT1A receptor", "HTR1A" all treated as different) |
| E4 | PAINS / Lipinski hard filters silently dropped catechols (every adrenaline-class drug) and natural-product peptides |
| E5 | No provenance — no input/output checksums, no per-stage attrition log |

Total raw input: **727 rows from a single BiasDB scrape (Sep 18 2025)**.

---

## 2. Decisions taken

Three policy questions were locked in by the lead author before any code
changes were made:

| Decision | Choice | Why |
|---|---|---|
| Receptor lookup mode | **Strict** — fail loudly when a receptor isn't in the canonical registry | Forces the registry to stay current; eliminates the non-deterministic "first ChEMBL search hit wins" problem |
| Label framing | **BiasDB-only multi-class** — drop ChEMBL "Agonist" rows entirely | Removes E1 source-shortcut; every row in the curated set is a real measured bias |
| Drug-likeness filters | **Annotate, never filter** | PAINS removed catechols and Lipinski removed peptides under the legacy code — both are real biological molecules with bias data |

Two further decisions were made later in the journey:

| Decision | Choice |
|---|---|
| Composite key for the experimental unit | **`inchikey :: receptor_uniprot :: bias_pathway :: assay_1 :: assay_2 :: reference_ligand`** (Tier 1 + Tier 2 — see Section 4) |
| Tautomer canonicalization | **Skip for molecules with > 50 heavy atoms** (peptides) — the enumerator's 1000-tautomer ceiling silently strips stereochemistry |

---

## 3. What we did, pass by pass

The dedup key evolved through five iterations. Each iteration recovered
roughly 80 more rows of measurement signal that the previous one had
collapsed.

### Pass v1 — bare InChIKey-14 (legacy starting point)
- Key: `inchikey14`
- Result: 504 rows including 99 hard-coded "Agonist" labels
- Problem: a single molecule was treated as one row even when measured at multiple receptors / assays

### Pass v2 — composite (receptor + assay)
- Key: `inchikey14 :: receptor_uniprot :: assay_1 :: assay_2`
- Result: **459 BiasDB-only rows, 184 "conflicts"**
- Fixed: E1 (no more source-shortcut labels)
- New finding: 184 conflicts seemed huge

### Pass v3 — added bias_pathway
- Key: `inchikey14 :: receptor_uniprot :: assay_1 :: assay_2 :: bias_pathway`
- Result: **620 rows, 71 conflicts**
- Discovery: most "conflicts" in v2 were the same compound measured under three orthogonal pathway comparisons (Go vs β-Arr, β-Arr vs Gi, Go vs Gi) collapsed by an assay-only key. Adding `bias_pathway` distinguished them.

### Pass v4 — full InChIKey (stereo-aware)
- Key: `inchikey :: receptor_uniprot :: bias_pathway :: assay_1 :: assay_2`
- Result: **697 rows, 0 conflicts**
- Fixed: stereoisomers no longer collapsed. The four fenoterol stereoisomers from PMID 25342094 — which the original authors measured separately because (R,R)-fenoterol is the FDA-approved active species and the others are weaker — survive as four distinct rows.
- Bonus: CCR2 and CCR5 reappeared (they had previously lost every row to the BRET/BRET conflict pattern).

### Pass v5 — added reference_ligand
- Key: `inchikey :: receptor_uniprot :: bias_pathway :: assay_1 :: assay_2 :: reference_ligand`
- Result: **719 rows, 0 conflicts**
- Recovered: buprenorphine at μ-opioid (measured against morphine vs DAMGO in two papers — different references → mathematically different bias factors).

### Pass v6 — peptide-aware tautomer guard
- Standardizer change: skip `TautomerEnumerator.Canonicalize` for molecules with > 50 heavy atoms.
- Result: **719 rows** (no row count change), but the InChIKey of every angiotensin-class peptide changed from blank-stereo `*-UHFFFAOYSA-N` to a real stereo block (e.g. `*-AOEIQLFRSA-N`). This proved that the previous standardizer was silently stripping stereochemistry from peptides.

---

## 4. The composite key, explained

The final dedup key has six components, organised in two tiers:

**Tier 1 — required (the biology):**
- `inchikey` — the molecule, including stereo and protonation
- `receptor_uniprot` — the receptor, anchored to a stable UniProt accession
- `bias_pathway` — which two pathways were compared (e.g. "Go / β-Arr"). Bias is *always* defined relative to a comparison; without this, "G-protein biased" is meaningless.

**Tier 2 — required for experimental defensibility:**
- `assay_1`, `assay_2` — the readout technologies (BRET, cAMP, GTPγS, etc.). The same biology measured in two different technologies can give different bias factors due to system bias.
- `reference_ligand` — bias is computed relative to a reference (Δlog(τ/KA) by Kenakin & Christopoulos). Buprenorphine vs morphine ≠ buprenorphine vs DAMGO.

A row is a duplicate of another row only when **all six components match**.

---

## 5. Verification of the 8 remaining merged rows

After all six dedup-key fields were in place and the peptide guard was
applied, 8 raw BiasDB rows still collapsed into other rows. Each one was
verified to be **chemically identical** to its merge partner using four
independent tests:

| Test | What it confirms |
|---|---|
| Same canonical SMILES | Same atomic graph + same stereochemistry (RDKit canonicalizer is deterministic) |
| Same molecular formula | Necessary but not sufficient |
| Same heavy-atom count | Necessary but not sufficient |
| Same full InChI string | Independent of canonical SMILES; same answer means same molecule |

All 8 rows passed all 4 tests. Per-pair detail:

| Receptor | Names BiasDB stored | Verified identity |
|---|---|---|
| AT1 | `SVdF-[(Sar1,Val5,D-Phe8)-AngII]` ≡ `[Val5]Sarmesin` (3 pair_keys × 2 rows = 6 rows) | C₄₈H₆₉N₁₃O₁₀, 71 heavy atoms, identical canonical SMILES + InChI |
| D1 | `PF-1119` ≡ `4-(4-(3,5-Dimethylpyridazin-4-yl)-3-methylphenoxy)furo[3,2-c]-pyridine` | C₂₀H₁₇N₃O₂, 25 heavy atoms; project code vs IUPAC systematic name for the same compound |
| CCR5 | `J113863` ≡ `UCB35625` | C₃₀H₃₇Cl₂N₂O₂⁺, 36 heavy atoms; two industry codes for the same compound |
| μ-opioid | `5,6-Dichloro-1-(...)-benzo[d]imidazol-2(3H)-one` ≡ `SR-15099` | C₁₉H₁₈BrCl₂N₃O, 26 heavy atoms; IUPAC vs SR project code |
| D2 | `7-(4-(4-(3,4-bis(trimethyl-oxidaneyl)phenyl)piperazin-1-yl)butoxy)-3,4-dihydroquinolin-2(1H)-one` ≡ `7-(4-(4-(3-Ethoxyphenyl)piperazin-1-yl)butoxy)-3,4-dihydroquinolin-2(1H)-one` | C₂₅H₃₃N₃O₃, 31 heavy atoms; one name appears to be a transcription error in BiasDB (trimethyl-oxidaneyl instead of ethoxy) |
| μ-opioid | `SR-15098` ≡ `SR-14969` | C₂₀H₁₉Cl₃FN₃O, 28 heavy atoms; two SR project codes — same molecule |

These are not bugs in the curator. They are BiasDB-side aliasing: the
same chemical structure recorded under multiple names (project codes,
synonyms, IUPAC variants, and in one case a likely transcription error).
The curator correctly unifies them.

---

## 6. Final state of Stage 02 + Stage 03

| Metric | Value |
|---|---|
| Raw BiasDB rows | 727 |
| Curated rows in `unified_ligands.csv` | **719 (98.9% recovery)** |
| Same-context conflicts | **0** |
| Unique molecules (full InChIKey) | 532 |
| Unique molecules (connectivity layer InChIKey-14) | 519 |
| Unique receptors (UniProt-anchored) | **61 / 61** |
| Receptor structures prepared | **61 / 61** |
| ↳ from cached PDB | 34 |
| ↳ from AlphaFold (mean pLDDT ≥ 70) | 27 |
| Test suite | **223 passed, 1 skipped, 0 failed** |

Bias-class distribution after curation:

| bias_category | rows |
|---|---:|
| G protein | (largest, see `data/processed/dataset_audit.md`) |
| β Arrestin | second-largest |
| ERK | smaller |
| G protein selectivity | smallest |

(Exact counts regenerate on every curation run; see the auto-generated
`dataset_audit.md` for the canonical numbers tied to a specific
input/output SHA-256.)

---

## 7. Artifacts the curator emits

Every curation run produces the following set of files, each
provenance-stamped:

| Path | Purpose |
|---|---|
| `data/processed/unified_ligands.csv` | The curated training-eligible table |
| `data/processed/unified_ligands.csv.meta.json` | SHA-256 of input + output, label distribution, attrition trace, curation timestamp, framing notes |
| `data/processed/dedup_conflicts.csv` | Same-context label disagreements (currently empty — no conflicts after v5 key) |
| `data/processed/attrition.json` | Per-stage row counts |
| `data/processed/dataset_audit.md` | Reviewer-facing markdown summary |
| `data/processed/receptors/<UNIPROT>.pdb` | Prepared receptor structures (61 files) |
| `data/processed/receptors/<UNIPROT>.pdb.prep.meta.json` | Per-receptor prep sidecar (kept chain, dropped chains, het residues kept/dropped, source SHA-256) |
| `data/processed/receptor_audit.md` | Reviewer-facing receptor summary |
| `data/registry/preferred_pdbs.tsv` | Per-receptor PDB selection with manual-override column |
| `data/raw/biasdb_data.csv.meta.json` | Provenance for the upstream scraped BiasDB cache |
| `data/raw/alphafold/AF-<UNIPROT>-F1.pdb` (+ `.meta.json`) | Cached AlphaFold models (27 files; 6.6 MB total) |

---

## 8. Existential issues at the curation layer — status

| ID | Problem | Status |
|---|---|---|
| E1 | ChEMBL "Agonist" hard-coded label | **Fixed** — ChEMBL excluded from labelled set entirely |
| E2 | InChIKey-only dedup collapsing measurements | **Fixed** — six-field composite key |
| E3 | Free-text receptor names | **Fixed** — 62-row UniProt-anchored registry; strict mode |
| E4 | PAINS / Lipinski hard filters dropping real compounds | **Fixed** — annotation-only by default, opt-in `legacy_filters` flag |
| E5 | No provenance | **Fixed** — every artifact has SHA-256 sidecar; per-stage attrition log; reviewer-facing audit MD |

---

## 9. What's NOT done yet

The data curation layer is now satisfactory. The remaining stages have
helper functions and tests in place but are not yet wired into the
running pipeline:

- **Stage 04** — Binding-site / docking-box definition (needs Stage 03 receptors as input; will need a pocket predictor like fpocket / P2Rank for the 27 AlphaFold receptors that have no co-crystal ligand).
- **Stage 05** — AutoDock Vina docking against the 61 prepared receptors.
- **Stage 06** — Featurization (Morgan / MACCS fingerprints; 3D descriptors from docked pose; ProLIF interaction fingerprints).
- **Stage 07** — Dataset assembly with `pair_key`, `sample_weight`, temporal holdout split.
- **Stage 08** — ML preprocessing inside an sklearn `Pipeline` (composing the per-receptor-family imputer, correlation filter, scaffold-grouped split, etc.).
- **Stages 09–12** — Feature selection (Boruta inside CV folds), model training with locked selection rule, interpretability (SHAP stability), inference deployment.

Each remaining stage will follow the same pattern Stage 02 and Stage 03
followed: surface the policy questions, lock decisions explicitly, wire
the helpers into the running pipeline, run on real data, emit a
reviewer-facing audit, then move on.

---

## 10. Manuscript-ready summary paragraphs

Suggested wording for the resubmission Methods section (subject to
revision):

> "Raw bias measurements were retrieved from BiasDB (snapshot of
> 2025-09-18; SHA-256 `dae42d09…`), yielding 727 rows. Each row was
> standardized using RDKit (Cleanup → FragmentParent → Uncharger →
> tautomer canonicalization for molecules ≤ 50 heavy atoms) and
> deduplicated on a six-field composite key:
> `(InChIKey, UniProt, bias-pathway-comparison, assay 1, assay 2,
> reference ligand)`. This key represents the unit of measurement of an
> independent bias-factor determination as defined by the operational
> model of agonism. After deduplication, 8 rows from the raw BiasDB
> cache merged into 4 curated rows because their underlying SMILES
> strings encode chemically identical molecules; identity was verified
> by canonical SMILES, full InChI, molecular formula, and heavy-atom
> count agreement. The final curated table contains 719 measurements
> across 532 unique molecules and 61 receptors, with no within-key
> label conflicts."

> "ChEMBL data were excluded from the labelled training set, since
> ChEMBL records do not encode pathway-bias measurements; legacy
> versions of this pipeline that hard-coded ChEMBL rows as `Agonist`
> were found to introduce a database-of-origin shortcut and have been
> removed."

> "All 61 receptor structures were prepared from cached PDBs (n=34) or
> from AlphaFold v6 predictions (n=27, gated at mean pLDDT ≥ 70).
> Receptor preparation removed engineered fusion proteins (T4L, BRIL,
> nanobodies) and G-protein chains, retained the conserved sodium ion
> when present, and filtered alternate locations to a single canonical
> conformation."

---

_Last updated: 2026-04-28. This document is regenerated by hand when
the curation policy changes; the auto-generated `dataset_audit.md` and
`receptor_audit.md` files capture the per-run numerical state._
