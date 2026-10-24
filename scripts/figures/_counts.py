"""Pipeline counts shared by the two schematic figures, computed from artefacts.

The curation funnel replays the row filters of
``cancerag.ml.dataset_assembly.assemble_dataset`` on the stored inputs, and
the build asserts that the replay ends at the row count of the assembled
parquets, so the schematic cannot drift from the data.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from ._data import Artefacts

SOURCES = [
    "processed/attrition.json",
    "processed/unified_ligands.csv",
    "processed/ligand_features.parquet",
    "processed/docking_features.csv",
    "processed/binding_sites.json",
    "processed/redock_validation.json",
    "processed/ml_ready_dataset.parquet",
    "holdout/dataset_holdout.parquet",
    "processed/ml_models/model_meta.json",
    "processed/ml_models/cv_results_long.csv",
]


def pipeline_counts(art: Artefacts) -> dict:
    att = {s["stage"]: s for s in art.json("processed/attrition.json")}
    raw = att["load_biasdb"]["n_out"]
    after_std = att["standardize"]["n_out"]
    after_dedup = att["dedupe"]["n_out"]

    ul = art.csv("processed/unified_ligands.csv")
    lf = art.parquet("processed/ligand_features.parquet", columns=["inchikey"])
    dock = art.csv("processed/docking_features.csv")

    scoped = ul[ul.inchikey.isin(set(lf.inchikey))]
    out_of_scope = ul[~ul.inchikey.isin(set(lf.inchikey))]
    from rdkit import Chem
    backbone = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)[NX3][CX4][CX3](=O)")
    n_peptide = sum(
        bool(m is not None and m.HasSubstructMatch(backbone))
        for m in (Chem.MolFromSmiles(smi) for smi in out_of_scope.canonical_smiles_std)
    )
    merged = scoped.merge(dock[["inchikey", "receptor_uniprot", "success"]],
                          on=["inchikey", "receptor_uniprot"], how="left")
    docked = merged[merged.success == True]  # noqa: E712
    key = (docked.inchikey + "::" + docked.receptor_uniprot + "::"
           + docked.bias_category.fillna("?") + "::" + docked.bias_pathway.fillna("?")
           + "::" + docked.assay_1.fillna("?") + "::" + docked.assay_2.fillna("?"))
    deduped = docked[~key.duplicated()]
    labelled = deduped[deduped.bias_category.notna()]

    tr, ho = art.train, art.holdout
    if len(labelled) != len(tr) + len(ho):
        raise AssertionError(
            f"funnel replay ends at {len(labelled)} rows, parquets hold {len(tr) + len(ho)}")

    sites = art.json("processed/binding_sites.json")["binding_sites"]
    methods = Counter(s["method"] for s in sites)
    redock = art.json("processed/redock_validation.json")["totals"]
    meta = art.model_meta
    cv = art.csv("processed/ml_models/cv_results_long.csv")
    learners = [m for m in cv.model.unique() if m not in ("majority_class", "stratified")]

    n_pairs = len(dock)
    n_ok = int(dock.success.sum())
    return {
        "raw_rows": raw,
        "after_standardise": after_std,
        "after_dedup": after_dedup,
        "dedup_dropped": raw - after_dedup,
        "molecules": int(ul.inchikey.nunique()),
        "receptors_curated": int(ul.receptor_uniprot.nunique()),
        "unique_pairs": int(ul[["inchikey", "receptor_uniprot"]].drop_duplicates().shape[0]),
        "small_molecule_rows": len(scoped),
        "out_of_scope_rows": len(out_of_scope),
        "out_of_scope_peptide_backbone": int(n_peptide),
        "rows_dock_failed": int((merged.success != True).sum()),  # noqa: E712
        "rows_duplicate_key": int(key.duplicated().sum()),
        "modelled_rows": len(labelled),
        "pairs_docked": n_pairs,
        "pairs_ok": n_ok,
        "pairs_failed": n_pairs - n_ok,
        "dock_success_pct": 100 * n_ok / n_pairs,
        "binding_sites": len(sites),
        "site_methods": dict(methods),
        "redock_passed": redock["passed"],
        "redock_attempted": redock["attempted"],
        "train_rows": len(tr),
        "holdout_rows": len(ho),
        "columns": tr.shape[1],
        "n_features": meta["n_features"],
        "receptors_modelled": int(pd.concat([tr, ho]).receptor_uniprot.nunique()),
        "ligands_modelled": int(pd.concat([tr, ho]).inchikey.nunique()),
        "n_learners": len(learners),
        "n_seeds": int(cv.seed.nunique()),
        "n_folds": int(meta["cv_n_folds"]),
        "n_cv_fits": int(len(cv[cv.model.isin(learners)])),
    }
