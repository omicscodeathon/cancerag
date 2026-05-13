# Stage 01 — Data collection tracking

## Original state (pre-rebuild)
The legacy data collection layer combined a hardcoded BiasDB scraper with a fragile ChEMBL retriever. `biasdb_retriever.py` issued an HTTP GET against the BiasDB query endpoint and re-wrapped the JSON response into a 22-column DataFrame using a positional `headers` list — any column re-order on the BiasDB side would silently misalign every label. ChEMBL retrieval used name-based target search (`target_api.search()` → first `Homo sapiens / SINGLE PROTEIN` hit) and a `mechanism.action_type=AGONIST` filter that captures only a small fraction of real agonists. PDB and AlphaFold retrievers had no fetch timestamps, no source-version pins, no payload hashes, and called `sys.exit(1)` from inside library code. The configured `chembl_activity_*` keys were declared but never read.

## Upgrades performed
- Replaced positional BiasDB column mapping with Pydantic schema validation; schema drift now raises loudly instead of silently misaligning labels.
- Added a UniProt-anchored canonical receptor registry (`data/registry/receptors.tsv`) so every retriever joins on a stable accession rather than free-text receptor names.
- Switched ChEMBL target lookup from name search to UniProt-keyed `target_components__accession` filter, eliminating the non-deterministic "first ranked hit wins" problem.
- Added provenance sidecars (`*.meta.json`) carrying `{source_url, fetch_timestamp_utc, sha256, row_count, query_params}` for every fetched artifact.
- Replaced `sys.exit(1)` with raised `RetrievalError`; removed module-level `logging.basicConfig` calls.

## What was NOT done (deferred / out-of-scope)
- ChEMBL activity-table-driven retrieval (`assay_type='F'`, `standard_relation='='`, EC50/pEC50 thresholds) — `chembl_activity_*` config keys still unused; ChEMBL was excluded from labelled training entirely (Stage 02 decision) so this never became blocking.
- Temporal split *at retrieval time* with a separate `data/holdout/` directory unreadable by the trainer — temporal holdout enforcement was instead implemented at Stage 07 (assembly time).
- A `scripts/reproduce_data.sh` end-to-end re-fetch + SHA-256 verification harness.

## Files produced
- `data/raw/biasdb_data.csv` (+ `.meta.json`)
- `data/registry/receptors.tsv`, `data/registry/preferred_pdbs.tsv`
- `data/raw/alphafold/AF-<UNIPROT>-F1.pdb` (+ `.meta.json`)
- ChEMBL cache CSVs (excluded from labelled set downstream)

## Tests
- `tests/unit/test_biasdb_retriever.py`
- `tests/unit/test_chembl_retriever.py`
- `tests/unit/test_receptor_retriever.py`
- `tests/unit/test_alphafold_fetcher.py`
- `tests/unit/test_registry.py`
- `tests/unit/test_provenance.py`
- `tests/unit/test_schemas.py`

## Status
Complete (Done) — schema validation + UniProt anchoring + provenance landed; activity-driven ChEMBL retrieval deferred because ChEMBL is no longer in the labelled set.
