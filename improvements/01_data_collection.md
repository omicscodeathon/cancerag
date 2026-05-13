# 01 — Data Collection (BiasDB, ChEMBL, PDB, AlphaFold)

Modules covered:
- `src/cancerag/data_collection/biasdb_retriever.py`
- `src/cancerag/data_collection/chembl_retriever.py`
- `src/cancerag/data_collection/receptor_retriever.py`
- `src/cancerag/data_collection/alphafold_retriever.py`

## What the code does today

- **BiasDB**: HTTP GET against `https://biasdb.drug-design.de/data_0/query?user_query=default_query`; the JSON list response is wrapped in a DataFrame using a 22-column hardcoded header list (`biasdb_retriever.py:73-96`); saved to `data/raw/biasdb_data.csv`. Idempotent via file existence check.
- **ChEMBL**: per-receptor name → `target_api.search()` → first `Homo sapiens / SINGLE PROTEIN` hit (`chembl_retriever.py:78-92`); then `mechanism.action_type=AGONIST` lookup (`chembl_retriever.py:135`); per-molecule `molecule_api.get(chembl_id)` to fetch `canonical_smiles`. Configured but unused: `chembl_activity_types`, `chembl_min_confidence_score`, `chembl_activity_threshold_nm`.
- **PDB**: per-receptor batch fetch capped by `max_pdb_files_per_receptor: 10`. Files saved under `data/pdb/<receptor>/`.
- **AlphaFold**: per-UniProt model download (separate from `download_alphafold_structures.py` at repo root).

## Problems

### Tier 1 — invalidate downstream steps

**P1.1 Positional column mapping for BiasDB.**
`biasdb_retriever.py:105` does `pd.DataFrame(data_list, columns=headers)` with a hardcoded `headers` list. If BiasDB ever reorders or adds a column the entire label vector misaligns silently. There is no schema check, no field validation, no source-version pin.

**P1.2 ChEMBL target search is non-deterministic across releases.**
`chembl_retriever.py:92` returns `hs_targets[0]` after a name-based search. ChEMBL's search ranking can change between monthly releases. "5-HT1A receptor" can match `CHEMBL214` (HTR1A) or close paralogs. Not pinned to UniProt.

**P1.3 `action_type=AGONIST` is the wrong filter.**
The `mechanism` table is hand-curated and very sparse — it captures only a small fraction of known agonists. The pipeline misses thousands of legitimate ChEMBL agonists and includes some inverse-agonist mis-curations. The `chembl_activity_*` config keys are declared but never read.

**P1.4 No external / temporal holdout reserved at collection time.**
Reviewer 2 explicitly required this. The current pipeline has no notion of "data added after cutoff date X is for held-out evaluation only."

### Tier 2 — reproducibility / FAIR compliance

**P1.5 No source versioning.**
- BiasDB: no fetch timestamp, no payload hash.
- ChEMBL: no release version (`new_client` returns whatever's current).
- PDB: no entry deposition / revision date.
- AlphaFold: no model version (v3 vs v4).

A reviewer cannot reproduce the exact dataset. Required for any FAIR / Nature-tier submission.

**P1.6 Receptor naming is ad-hoc.**
`chembl_retriever.py:62` does `search_name = receptor_name.replace("-", " ")`. There is no canonical receptor table. "5-HT1A" can be `5-ht1a receptor`, `HTR1A`, `5HT1A_HUMAN`, or `P08908` — the same receptor will give different downstream joins depending on what string was passed.

**P1.7 No de-duplication across sources at retrieval time.**
ChEMBL and BiasDB overlap (BiasDB ligands often have ChEMBL IDs). Today this overlap is detected at most at the SMILES-string level in `LigandPreprocessor` after both sets are fetched. Should be detected and resolved at retrieval.

### Tier 3 — engineering hygiene

**P1.8 `sys.exit(1)` inside library code.**
`biasdb_retriever.py:67,70,113`. A retriever should raise, not exit; otherwise it's unusable from notebooks, tests, or the inference app.

**P1.9 Logging configured at module import.**
`chembl_retriever.py:14` calls `logging.basicConfig`. Belongs at the entry point only.

**P1.10 PDB download has no rate-limit handling beyond `NetworkRetrier`.**
`receptor_retriever.py` (575 lines) — needs an audit for RCSB API courtesy (max 10 req/s), retry-after honoring.

## Standard approach

1. **Canonical receptor registry** keyed by **UniProt accession**. Every retriever joins through this registry. Resolves P1.2, P1.6, P1.7.
2. **Schema-validated payloads**. Pydantic models for BiasDB, ChEMBL, PDB metadata. Fail loudly on schema drift. Resolves P1.1.
3. **Provenance sidecars**. Every artifact gets a `*.meta.json` with `{source_url, fetch_timestamp_utc, source_version, query_params, sha256, row_count}`. Resolves P1.5.
4. **Activity-table-driven ChEMBL retrieval**, not mechanism-table-driven. Use `assay_type='F'`, `standard_relation='='`, `standard_type ∈ {EC50, pEC50}`, `standard_value < 1000 nM`, `confidence_score >= 8`. Resolves P1.3.
5. **Temporal split at retrieval time**. Tag every BiasDB row with `publication_year`; route `year >= cutoff` to a separate `data/holdout/` path that the ML pipeline cannot read. Resolves P1.4.

## Concrete fixes

### F1.1 Add `data/registry/receptors.tsv`

Columns: `uniprot_accession, gene_symbol, biasdb_name, chembl_target_id, gpcrdb_id, gpcrdb_class, gpcrdb_family, preferred_pdb_active, preferred_pdb_inactive, alphafold_id, notes`.

Build with `scripts/build_receptor_registry.py` from a curated YAML. Pin to a specific UniProt release date in the file header.

### F1.2 Replace name-based ChEMBL target lookup

```python
# chembl_retriever.py
def _get_target_by_uniprot(self, uniprot: str) -> str | None:
    targets = list(self.target_api.filter(
        target_components__accession=uniprot,
        target_type="SINGLE PROTEIN",
        organism="Homo sapiens",
    ))
    return targets[0]["target_chembl_id"] if targets else None
```

Drop `_get_target_chembl_id` entirely.

### F1.3 Pydantic schemas for retrievers

```python
# data_collection/schemas.py
class BiasDBRow(BaseModel):
    ligand_name: str
    smiles: str
    receptor_subtype: str
    bias_category: str
    reference_ligand: str | None
    assay_1: str | None
    assay_2: str | None
    publication_title: str | None
    doi: str | None
    pmid: str | None
    year: int | None
    # ... numeric props
```

Use `BiasDBRow.model_validate(row_dict)` for every record. Reject batch on first schema error.

### F1.4 Provenance sidecars

```python
# data_collection/provenance.py
def write_meta(payload_path: Path, source_url: str, source_version: str, params: dict):
    sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    meta = {
        "source_url": source_url,
        "fetch_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "source_version": source_version,
        "query_params": params,
        "sha256": sha,
        "row_count": _count_rows(payload_path),
    }
    payload_path.with_suffix(payload_path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2))
```

Call from every retriever after writing the artifact.

### F1.5 Activity-table-based ChEMBL retrieval

```python
# chembl_retriever.py
def _fetch_agonists(self, target_id: str, receptor_name: str) -> pd.DataFrame:
    activities = list(self.activity_api.filter(
        target_chembl_id=target_id,
        assay_type="F",
        standard_type__in=self.config["chembl_activity_types"],
        standard_relation="=",
        standard_value__lte=self.config["chembl_activity_threshold_nm"],
        confidence_score__gte=self.config["chembl_min_confidence_score"],
    ))
    # join to molecule_api for canonical_smiles, dedupe by molecule_chembl_id
```

### F1.6 Temporal holdout at retrieval

```python
# biasdb_retriever.py
HOLDOUT_YEAR_CUTOFF = 2024  # configurable
def split_temporal(df, cutoff):
    return df[df["year"] < cutoff], df[df["year"] >= cutoff]
```

Write `data/raw/biasdb_data.csv` (training-eligible) and `data/holdout/biasdb_holdout.csv` (held out). The ML pipeline must not load the holdout file until final reporting.

### F1.7 Replace `sys.exit(1)` with `raise`

`biasdb_retriever.py:67,70,113`. Define a `RetrievalError` exception type.

## Acceptance criteria

- [ ] `data/registry/receptors.tsv` exists; every other retriever consumes it.
- [ ] All retrievers emit `<artifact>.meta.json` with the provenance fields above.
- [ ] Pydantic schema validation runs on every BiasDB and ChEMBL row; schema drift produces a loud failure.
- [ ] ChEMBL retrieval is activity-table-based and respects all `chembl_*` config keys; the `mechanism`-only path is removed.
- [ ] A reproducibility script `scripts/reproduce_data.sh` re-fetches and verifies the SHA-256s on a clean machine.
- [ ] A `data/holdout/` directory exists with `<source>_holdout.csv` files; an explicit assert in the ML pipeline blocks reading these paths during training.
- [ ] No `logging.basicConfig` calls remain in retriever modules.
