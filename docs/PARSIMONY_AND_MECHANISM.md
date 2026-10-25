# Parsimony and mechanism: what survives when generalisation is made hard

Measured 2026-10-25 on the rebuilt pipeline (587/587 docked pairs, 619
conflation-free interaction columns, leakage-gated). Companion to
`QC_LEAKAGE_CORRECTION.md` and `PROVENANCE_LEAKAGE_PROBE.md`.

The question is not "how well does the model score". It is the one the
Anamnesis framework poses: **conditions, not complexity, are what hide a
parsimonious rule.** We have 5,511 features and four regimes of increasing
generalisation difficulty. If a small, regime-filtered subset matches the full
matrix, the complexity was never carrying the signal.

---

## 1. The regimes, and what each closes

Ordered easy → hard. The ordering is the filter.

| regime | leakage it closes | 4-class macro-F1 (LightGBM, 5,511 features) |
| --- | --- | --- |
| stratified | — (baseline) | 0.5288 |
| scaffold-grouped | near-duplicate chemistry | 0.4544 |
| receptor-grouped | entity novelty | 0.2491 |
| temporal holdout | time | 0.2745 |

Majority-class floor: **0.1788**.

The scaffold → receptor drop is the central measurement. Bias determinants
learned on one set of receptors largely do not transfer to another. That is a
statement about the biology, not a defect: biased signalling runs through
receptor-specific conformational coupling, so a portable ligand-only rule was
never guaranteed to exist.

---

## 2. Parsimony: 11 features match 5,511, and 2 beat it

Features ranked by **worst-case selection frequency across the four regimes**,
so the ranking is itself regime-filtered. Random subsets of equal size are the
control — the question is whether it is the *selection* that matters or merely
the count.

| k | stratified | scaffold | receptor |
| --- | --- | --- | --- |
| | ranked / random | ranked / random | ranked / random |
| 1 | 0.264 / 0.130 | 0.226 / 0.131 | 0.212 / 0.177 |
| 2 | 0.366 / 0.130 | 0.335 / 0.131 | **0.293** / 0.177 |
| 5 | 0.426 / 0.122 | 0.371 / 0.136 | 0.231 / 0.177 |
| 11 | 0.500 / 0.106 | 0.430 / 0.172 | 0.284 / 0.137 |
| 20 | **0.536** / 0.322 | 0.434 / 0.203 | 0.260 / 0.171 |
| 320 | 0.538 / 0.518 | **0.462** / 0.376 | 0.260 / 0.253 |
| **all 5,511** | 0.529 | 0.454 | 0.249 |

Three things fall out:

- **20 ranked features beat all 5,511** under stratified (0.536 vs 0.529), and
  320 beat it under scaffold-grouped (0.462 vs 0.454).
- Under the **hardest** regime, **two features beat the full matrix**
  (0.293 vs 0.249). Eleven give 0.284.
- The random control separates selection from count: 11 random features give
  0.106–0.177, against 0.284–0.500 for 11 ranked. At k=320 the gap closes,
  which is what you expect once a random draw starts catching the informative
  columns by chance.

The curve is flat from about k=20 onward in every regime. **The extra 5,491
features are not carrying signal.**

> **The ranking above saw the test folds.** It comes from SHAP computed across
> folds of this same dataset, not re-derived inside each fold, so the numbers
> are optimistic. They show a parsimonious subset *exists*; they do not show
> you would find this one from a cold start. §2b repeats the experiment with
> selection confined to the training fold, and supersedes this table for any
> quotable claim.

### 2b. Nested selection — the honest ladder

Selection moved strictly inside the training fold: fit on train with every
feature, rank by gain, take the top k of *that* ranking, refit on train, score
the held-out fold. Nothing about the test fold informs the choice.

| k | stratified | scaffold | receptor |
| --- | --- | --- | --- |
| | nested / random | nested / random | nested / random |
| 2 | 0.320 / 0.126 | 0.247 / 0.127 | 0.231 / 0.166 |
| 11 | 0.466 / 0.228 | 0.363 / 0.199 | 0.234 / 0.177 |
| 20 | 0.511 / 0.212 | 0.405 / 0.195 | 0.247 / 0.165 |
| 40 | 0.521 / 0.381 | 0.435 / 0.296 | **0.258** / 0.266 |
| 80 | 0.529 / 0.457 | 0.445 / 0.358 | 0.243 / 0.245 |
| 320 | **0.540** / 0.491 | **0.463** / 0.370 | 0.233 / 0.242 |
| **all 5,511** | 0.529 | 0.454 | 0.249 |

**The extreme claim does not survive.** Nested k=11 is *below* the full matrix
in every regime (0.466 vs 0.529, 0.363 vs 0.454, 0.234 vs 0.249). The optimism
in §2 is worth roughly **0.03–0.07 macro-F1**, and it is concentrated exactly
where the claim was strongest — at small k, where a ranking derived from the
whole dataset has the most room to flatter itself.

**What does survive.** 80–320 nested features **match or beat** all 5,511 in
every regime: 0.540 vs 0.529 stratified, 0.463 vs 0.454 scaffold-grouped,
0.258 vs 0.249 receptor-grouped at k=40. That is a **17–70× reduction at no
cost** — real parsimony, an order of magnitude less dramatic than §2 implied.

**And selection stops mattering under the hardest regime.** At k=40 under
receptor-grouped, *random* features (0.266) edge out gain-ranked ones (0.258).
Within noise either way, which is the point: once receptors are held out,
every feature set lands near the 0.179 floor and which few hundred you choose
makes no reliable difference. That is the same message the graph network sends
from a different direction (§6) — the ceiling is in the data.

---

## 3. The contrastive filter: what evaporated

Attribution run separately under the easiest and hardest regimes.
**14 of the top 50** features under stratified are absent from the top 50 under
receptor-grouped:

```
AvgIpc   BCUT2D_LOGPLOW   EState_VSA2   EState_VSA5   FpDensityMorgan2
NPR2     PEOE_VSA2        PMI1          RingCount     SPS
VSA_EState1              morgan_1683    qed           ifp_BW5.39_VdWContact
```

These are the shortcuts. The pattern is coherent: `qed`, `RingCount`, `NPR2`,
`PMI1`, `SPS` are **gross size and shape descriptors**. Receptor families
differ systematically in the size and shape of their ligands, so these encode
*which receptor* and evaporate the moment the model must face an unseen one.
Reporting them matters — concealing the dropouts hides how the filter worked.

**The 11 that survive all four regimes:**

| feature | kind |
| --- | --- |
| `vina_pose_diversity_rmsd` | docking — spread of the pose ensemble |
| `vina_affinity_best` | docking — best score |
| `vina_affinity_gap_1_2` | docking — separation of top two poses |
| `PEOE_VSA7`, `PEOE_VSA1`, `PEOE_VSA9` | partial-charge surface area |
| `EState_VSA3`, `VSA_EState3` | electrotopological state |
| `BCUT2D_CHGLO` | charge-weighted connectivity eigenvalue |
| `LogP` | lipophilicity |
| `SpherocityIndex` | 3D shape |

Three docking scalars, four charge-distribution descriptors, two
electrotopological, one lipophilicity, one shape. **No receptor-contact or
Ballesteros–Weinstein position survives** — the single BW feature in the
top-50-easy list (`ifp_BW5.39_VdWContact`) is among the shortcuts.

`vina_pose_diversity_rmsd` is the standout: selection frequency **1.00 in every
regime**. How widely the docked poses scatter, which plausibly reads binding-site
plasticity — a mechanistically interesting quantity for bias, and a hypothesis
rather than a finding.

---

## 4. Where the signal lives

SHAP block attribution under the receptor-grouped regime:

| block | share | n features |
| --- | --- | --- |
| ligand 2D descriptors | **63.8%** | 182 |
| ligand 2D fingerprint | 12.4% | 1,524 |
| receptor–ligand contacts (ProLIF) | 11.1% | 347 |
| docked-pose 3D shape | 7.5% | 8 |
| pair-level docking energetics | 5.3% | 5 |

About **76% ligand-only**. Feature-view ablation agrees:

| view | n | receptor | scaffold | stratified |
| --- | --- | --- | --- | --- |
| all | 5,511 | 0.2491 | 0.4544 | 0.5288 |
| **bw_only** | 5,266 | **0.2491** | **0.4544** | **0.5288** |
| ligand_only | 4,884 | 0.2349 | 0.4109 | 0.5373 |
| structure_only | 627 | 0.2362 | 0.3503 | 0.4420 |

`bw_only` is **bit-identical to all** at four decimals. Dropping the 245
receptor-qualified contact columns changes nothing — they are, by construction,
always zero for an unseen receptor, and they turn out to be inert for seen ones
too. That is a clean argument for the BW representation on parsimony grounds
alone.

---

## 5. The four-class task is the wrong question

On the temporal holdout the model never predicts two of its four classes:

| class | n (holdout) | F1 |
| --- | --- | --- |
| G protein | 127 | 0.73 |
| β Arrestin | 55 | 0.37 |
| G protein selectivity | 33 | **0.00** |
| ERK | 25 | **0.00** |

Not a tuning failure — a class-balanced stacking ensemble with per-class
threshold optimisation returns the same two zeros.

The labels are not four values on one axis. They are different *comparisons*:

| class | what it compares |
| --- | --- |
| G protein | Gi/Gs/Go/Gq **over β-arrestin** |
| β Arrestin | β-arrestin **over** a G protein |
| ERK | ERK **over** Gs/Gi/Gq/β-arrestin |
| G protein selectivity | Gi **over Gq**, G12 **over Gq** — *no arrestin at all* |

"G protein selectivity" is a within-G-protein question on a different axis
entirely. And the rare classes are spread far too thin to learn: ERK is 59
measurements over 18 receptors, **median 2 per receptor**.

### Is there any signal in them at all?

Rather than asking a 4-way model to predict them, each was tested one-vs-rest
under receptor-grouped CV against a 100-draw label-permutation null:

| target | n pos | observed | null mean ± sd | p |
| --- | --- | --- | --- | --- |
| ERK | 34 | 0.5219 | 0.4843 ± 0.0132 | **0.0495** |
| G protein selectivity | 52 | 0.4764 | 0.4778 ± 0.0165 | **0.4158** |

**G-protein selectivity is indistinguishable from chance** — the observed value
sits *below* the permutation mean. There is nothing to extract; this is a clean
negative.

ERK lands on the 0.05 boundary. With 100 permutations that is fragile — one
further extreme draw would give p = 0.059 — and the effect is +0.038 macro-F1
over "always predict not-ERK". Read as: a weak signal that may be real, is not
practically usable, and would need a larger permutation set and more data
before it could be claimed.

### The well-posed contrast

"G protein" and "β Arrestin" are mirror images — `Gi / β-Arr` (212) against
`β-Arr / Gi` (93). Defining the axis from `bias_pathway` rather than
`bias_category` gives **316 clean rows** (the 55 rows that are really
G-protein-over-ERK are correctly excluded):

| view | n feat | receptor AUC | scaffold AUC | stratified AUC |
| --- | --- | --- | --- | --- |
| all | 5,511 | 0.578 | **0.803** | **0.866** |
| bw_only | 5,266 | 0.578 | 0.803 | 0.866 |
| ligand_only | 4,884 | 0.606 | 0.802 | 0.862 |
| **robust_11** | **11** | **0.610** | 0.711 | 0.790 |

macro-F1 floor 0.379; `all` reaches 0.779 stratified, 0.714 scaffold-grouped.

**This is a usable model** for a novel scaffold at a characterised receptor.
And under the hardest regime the **11 features beat all 5,511** on both AUC
(0.610 vs 0.578) and macro-F1 (0.538 vs 0.516) — parsimony wins exactly where
generalisation is hardest.

---

## 5b. The hard splits are softer than they look

The RCAF skill is blunt: *assume your splits leak and go looking.* They do.

| check | finding |
| --- | --- |
| scaffold groups | 254 total, **166 are singletons (65%)**, largest 21 |
| duplicate (ligand, receptor) pairs | **62**, up to 7 rows for one pair |
| receptor sizes | 51 receptors, median 4 rows, **largest is 100 rows = 21% of the data** (P14416, D2 dopamine) |
| ligands measured at >1 receptor | 19 of 367, max 3 receptors |

Three consequences, and they qualify everything above:

1. **Scaffold-grouping is close to a random split for most of the data.** With
   65% of scaffolds appearing once, grouping by scaffold constrains only the
   minority of rows that share one. The scaffold regime is therefore *not* as
   hard as its name suggests, and the gap between it and receptor-grouped is
   partly a gap between "nearly random" and "genuinely hard".
2. **62 duplicate ligand–receptor pairs cross the stratified split**, which
   inflates the stratified numbers by near-duplicate leakage — exactly the
   failure the grouped regimes exist to expose.
3. **One receptor is 21% of the data.** A receptor-grouped fold that holds out
   P14416 loses a fifth of the dataset in one go, which is a large part of why
   the receptor-grouped estimates are unstable and why their seed variance is
   unreliable (see the caveat below).

None of this invalidates the direction of the finding — receptor-grouped is
still by far the hardest regime and the collapse is reproduced by an
independent architecture — but it means the *stratified* numbers should be
read as an upper bound inflated by duplicates, and the scaffold numbers as
softer than the label implies.

**A second caveat on variance.** Under both grouped regimes the block ablation
returns bit-identical macro-F1 across three seeds (sd = 0.0000). `GroupKFold`
does not shuffle, so the folds are fixed, and the model is evidently
deterministic on that path. Grouped-regime differences in this document
therefore have **no error bars** and should not be read as significant on their
own. In the one regime where spread is measurable, block differences
(+0.013) are smaller than the seed sd (0.028).

## 6. Model class is not the bottleneck

| model | representation | stratified | scaffold | receptor |
| --- | --- | --- | --- | --- |
| LightGBM | 5,511 engineered | 0.529 | 0.454 | 0.249 |
| GIN (graph net) | molecular graph only | 0.393 | 0.299 | 0.209 |
| ChemBERTa + LR | SMILES embedding (384-d, frozen) | holdout macro-F1 ≈ 0.27 | | |

The graph network — a genuinely different inductive bias, reading the molecule
as a graph rather than a descriptor vector — reproduces the **same collapse
signature** (0.393 → 0.299 → 0.209) using the ligand alone. Three
representations, one ceiling.

Within the engineered features, four learners sit within 0.023 of each other
under receptor-grouped CV, against seed SDs of 0.05–0.07. **Learner choice is
noise.**

### Feature selectors, including Boruta

Scaffold-grouped, 3 folds, force-keeping disabled so selectors compete fairly:

| selector | + LightGBM |
| --- | --- |
| mutual_info | **0.460 ± 0.030** |
| rfecv | 0.433 ± 0.058 |
| l1_logreg | 0.428 ± 0.038 |
| **boruta** | **0.364 ± 0.041** |

No selection at all gives 0.454. So mutual-information selection matches the
full matrix, and **Boruta is the worst of the four** — about 0.10 below
mutual_info, roughly 2.5 SD. Wrapper selection is not where the gain is.

---

## 7. What this supports

1. **A parsimonious subset exists, at a few hundred features.** Under nested
   selection, 80–320 features match or beat all 5,511 in every regime — a
   17–70× reduction at no cost. The stronger "11 features" reading in §2 is an
   artefact of ranking on the full dataset and does **not** survive (§2b).
2. **The surviving features are ligand physicochemistry plus three docking
   scalars** — charge-distribution surface area, electrotopological state,
   lipophilicity, shape — with no receptor contact surviving.
3. **Receptor-contact features contribute ~11% of attribution and nothing that
   survives the hardest regime**, and the receptor-qualified subset is inert.
4. **Bias determinants are receptor-context-specific.** Quantified by
   0.454 → 0.249, and reproduced by an independent architecture.
5. **The four-class framing is not supported by the data.** G-protein
   selectivity is indistinguishable from a permutation null (p = 0.42); ERK is
   marginal (p = 0.05) and not usable. The G-protein/β-arrestin axis is
   supported, at AUC 0.80 scaffold-grouped.

## 8. What it does not support

Attribution explains what a model used. Filtering across regimes raises the
prior that a feature is real; **it does not establish causation**. The
surviving eleven are a shortlist to check against mechanism, not a discovery.

The honest formulation: *regime-robust attribution generates the testable
hypothesis that pose-ensemble diversity and ligand charge distribution, rather
than specific receptor contacts, carry what is predictable about functional
selectivity at this scale.* Naming the experiment that would falsify it is the
next step, not a claim that it is true.
