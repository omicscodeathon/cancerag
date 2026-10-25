"""Can docking provenance alone predict the label?

The trap this guards against is the one recorded in ``docs/QC_LEAKAGE_CORRECTION.md``:
a per-group covariate joined onto per-pair rows encodes group identity, and group
identity carries the label. All four evaluation regimes failed to catch it, and a
two-column model reproduced the full pipeline's holdout score.

The 587-pose set carries three such covariates, none of them a feature by intent:

====================  ==========================================================
``engine``            13 pairs on Uni-Dock GPU, the rest on CPU Vina -- and the
                      13 are *precisely* the flexible peptides CPU could not
                      finish, so the column is not a random sample of rows
``exhaustiveness``    16 for those, the default for the rest
``receptor_repaired`` 6 receptors rebuilt from different PDB entries or chains
====================  ==========================================================

Why four comparisons and not one
--------------------------------

Provenance-only versus the majority-class floor, on its own, is not interpretable
here. ``receptor_repaired`` is a *receptor-level* property and receptor identity
already predicts much of the label, so a provenance-only model beats the floor
under stratified CV whether or not provenance adds anything -- it is re-reading
receptor identity, which is a known property of the data rather than a new leak.

So the decisive test holds the receptors out. With receptor identity unavailable,
anything provenance still predicts is signal of its own:

1. ``floor``                       majority class
2. ``receptor_identity``           one-hot receptor, stratified -- what the known
                                   receptor->label association is worth
3. ``provenance_stratified``       expected between 1 and 2 if it is only
                                   re-reading receptor identity
4. ``provenance_receptor_grouped`` **the gate**

Only test 4 can fail. If it clears the floor by more than ``MARGIN``, the engine
mix is unusable as merged and the affected rows must be re-docked on a single
engine -- not filtered out. Excluding rows on an affinity threshold would filter
on a value that is itself a model input, and the filter is not independent of
class: selection bias dressed as quality control.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)

SEED = 0
N_TREES = 300
N_SPLITS = 5

#: How far above the majority-class floor counts as signal rather than noise.
#: Macro-F1 on four imbalanced classes moves by a few thousandths between seeds;
#: 0.02 is comfortably outside that and well inside the 0.33 that receptor
#: identity is worth.
MARGIN = 0.02

#: Columns the probe is allowed to see. Anything describing *how* a pose was
#: produced belongs here; anything describing the pose itself does not.
PROVENANCE_COLS = ("engine_gpu", "exhaustiveness", "receptor_repaired")

DEFAULT_OUT = "processed/leakage_probe_provenance.json"


def build_provenance_frame(data_dir: Path) -> pd.DataFrame:
    """One row per labelled docked pair, carrying only provenance and the label."""
    data_dir = Path(data_dir)
    dock = pd.read_csv(data_dir / "processed" / "docking_features.csv")
    lig = pd.read_csv(data_dir / "processed" / "unified_ligands.csv")

    labels = (lig[["inchikey", "receptor_uniprot", "bias_category"]]
              .dropna(subset=["bias_category"])
              .drop_duplicates(subset=["inchikey", "receptor_uniprot"]))
    df = dock.merge(labels, on=["inchikey", "receptor_uniprot"], how="inner")

    prov_path = data_dir / "processed" / "docking_provenance.json"
    prov: dict = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    df["engine_gpu"] = df["pair_id"].isin(set(prov)).astype(int)
    df["exhaustiveness"] = df["pair_id"].map(
        lambda p: prov.get(p, {}).get("exhaustiveness", 8)).astype(int)

    repairs = data_dir / "registry" / "receptor_repairs.tsv"
    if repairs.exists():
        # the file carries a `#` provenance preamble before the header row
        repaired = set(pd.read_csv(repairs, sep="\t", comment="#")["uniprot"].astype(str))
    else:
        repaired = set()
    df["receptor_repaired"] = df["receptor_uniprot"].isin(repaired).astype(int)
    return df


def _macro_f1(X, y, cv, groups=None) -> float:
    clf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
    return f1_score(y, cross_val_predict(clf, X, y, cv=cv, groups=groups),
                    average="macro")


def run_probe(df: pd.DataFrame) -> dict:
    """The four comparisons. Returns the results and a PASS/FAIL verdict."""
    y = df["bias_category"].to_numpy()
    groups = df["receptor_uniprot"].to_numpy()
    n_rec = len(np.unique(groups))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    gkf = GroupKFold(n_splits=min(N_SPLITS, n_rec))

    floor = f1_score(
        y,
        cross_val_predict(DummyClassifier(strategy="most_frequent"),
                          np.zeros((len(y), 1)), y, cv=skf),
        average="macro",
    )
    X_prov = df[list(PROVENANCE_COLS)].to_numpy()
    X_rec = pd.get_dummies(df["receptor_uniprot"], dtype=int).to_numpy()

    gated = _macro_f1(X_prov, y, gkf, groups=groups)
    margin = gated - floor
    results = {
        "floor_majority_class": floor,
        "receptor_identity_stratified": _macro_f1(X_rec, y, skf),
        "provenance_stratified": _macro_f1(X_prov, y, skf),
        "provenance_receptor_grouped": gated,
        "margin_over_floor": margin,
        "margin_threshold": MARGIN,
        "verdict": "FAIL" if margin > MARGIN else "PASS",
        "n_rows": int(len(df)),
        "n_receptors": int(n_rec),
        "n_gpu_rows": int(df["engine_gpu"].sum()),
        "n_repaired_rows": int(df["receptor_repaired"].sum()),
        "provenance_columns": list(PROVENANCE_COLS),
        "seed": SEED,
    }
    return results


def assert_provenance_is_not_a_feature(ml_ready: pd.DataFrame) -> None:
    """No provenance column may reach the model's feature matrix.

    The granularity assertion already covers per-receptor QC columns. This
    extends it to provenance: the probe clearing the gate says the columns carry
    no label signal of their own, not that they are safe to train on.
    """
    present = [c for c in PROVENANCE_COLS if c in ml_ready.columns]
    if present:
        raise AssertionError(
            f"provenance columns reached the feature matrix: {present}. "
            "These describe how a pose was produced, not the pose. See "
            "docs/PROVENANCE_LEAKAGE_PROBE.md."
        )


def default_data_dir() -> Path:
    """Same convention as ``cancerag.pipeline``: ``CANCERAG_DATA_DIR``, else ``data``."""
    return Path(os.environ.get("CANCERAG_DATA_DIR", "data")).resolve()


def main(data_dir: Path | str | None = None) -> dict:
    root = Path(data_dir) if data_dir else default_data_dir()
    df = build_provenance_frame(root)
    logger.info("leakage probe: %d labelled pairs, %d receptors, %d GPU rows, "
                "%d rows on repaired receptors",
                len(df), df["receptor_uniprot"].nunique(),
                int(df["engine_gpu"].sum()), int(df["receptor_repaired"].sum()))

    results = run_probe(df)
    for k in ("floor_majority_class", "receptor_identity_stratified",
              "provenance_stratified", "provenance_receptor_grouped"):
        logger.info("  %-34s %.4f", k, results[k])
    logger.info("  margin over floor %+.4f (threshold %.2f) -> %s",
                results["margin_over_floor"], MARGIN, results["verdict"])

    out = root / DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    logger.info("written: %s", out)

    if results["verdict"] == "FAIL":
        raise RuntimeError(
            f"provenance predicts the label with receptors held out: macro-F1 "
            f"{results['provenance_receptor_grouped']:.4f} vs floor "
            f"{results['floor_majority_class']:.4f} "
            f"({results['margin_over_floor']:+.4f} > {MARGIN}). The engine mix "
            "is unusable as merged; re-dock the affected rows on one engine. Do "
            "not filter them out -- an affinity threshold filters on a model "
            "input and is not independent of class."
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    main()
