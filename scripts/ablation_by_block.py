"""Block-wise ablation: separate ligand 3D shape from receptor-derived features.

`reviewer_extras.ablation_drop_structural` drops one combined "structural"
block, but that block mixes three different information sources: descriptors of
the docked *ligand's* 3D shape (which need no receptor), pair-level docking
energetics, and receptor-contact fingerprints. A ligand is a structure too, so
the combined Delta cannot answer "does the receptor help?" — only "does the
whole docking arm help?". This runs each block separately.

Run:
    PYTHONPATH=src python scripts/ablation_by_block.py
"""
from __future__ import annotations

import json as _json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from cancerag.ml.preprocessing import get_X_y_groups, build_full_pipeline
from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight

warnings.filterwarnings("ignore")

POSE_3D = ("Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
           "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex",
           "pose_3d_missing")
# needs the receptor to compute at all
RECEPTOR_DEPENDENT = ("vina_", "ifp_", "n_residues_contacted", "n_total_contacts",
                      "n_poses", "ifp_missing", "ifp_no_contacts")

SEEDS = (42, 7, 13)
FOLDS = 3


def cols_matching(X, prefixes):
    return [c for c in X.columns
            if any(c.startswith(p) or c == p for p in prefixes)]


def main() -> None:
    df = pd.read_parquet("data/processed/ml_ready_dataset.parquet")
    le = joblib.load("data/processed/ml_preprocessed/label_encoder.joblib")
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)
    winner = _json.loads(
        Path("data/processed/ml_models/selection_decision.json").read_text())["chosen"]
    factory = MODEL_FACTORIES[winner]

    pose = cols_matching(X, POSE_3D)
    recep = cols_matching(X, RECEPTOR_DEPENDENT)
    print(f"model={winner}  total={X.shape[1]}  "
          f"ligand-3D-shape={len(pose)}  receptor-dependent={len(recep)}")

    # The published ablation ran StratifiedKFold while the manuscript described
    # it as scaffold-grouped. Stratified is the optimistic regime this paper
    # argues against, so the headline Delta must be reported on the grouped
    # splits as well -- especially receptor-grouped, which is the only split
    # that asks whether receptor features help on an unseen receptor.
    variants = {
        "full": X,
        "drop_receptor_dependent": X.drop(columns=recep),   # keeps ligand 3D shape
        "drop_ligand_3d_shape": X.drop(columns=pose),       # keeps receptor features
        "drop_both": X.drop(columns=recep + pose),
    }

    scaf = pd.factorize(df["murcko_scaffold"])[0] if "murcko_scaffold" in df.columns \
        else pd.factorize(df["scaffold"])[0]
    recep = pd.factorize(df["receptor_uniprot"])[0]
    SPLITS = {
        "stratified": lambda Xv, seed: StratifiedKFold(
            n_splits=FOLDS, shuffle=True, random_state=seed).split(Xv, y),
        "scaffold_grouped": lambda Xv, seed: GroupKFold(n_splits=FOLDS).split(Xv, y, scaf),
        "receptor_grouped": lambda Xv, seed: GroupKFold(n_splits=FOLDS).split(Xv, y, recep),
    }

    rows = []
    for split_name, splitter in SPLITS.items():
      for name, Xv in variants.items():
        for seed in SEEDS:
            f1s = []
            for tr, te in splitter(Xv, seed):
                model = factory(n_classes=n_classes, random_state=seed)
                pipe = build_full_pipeline(model)
                last = pipe.steps[-1][0]
                pipe.fit(Xv.iloc[tr], y[tr],
                         **{f"{last}__sample_weight": _combined_weight(y[tr], sw[tr])})
                f1s.append(float(f1_score(y[te], pipe.predict(Xv.iloc[te]),
                                          average="macro", zero_division=0)))
            rows.append({"split": split_name, "variant": name, "seed": seed,
                         "n_features": int(Xv.shape[1]),
                         "macro_f1": float(np.mean(f1s))})
      print(f"  [{split_name}] done")

    out = pd.DataFrame(rows)
    outdir = Path("data/processed/ml_models/extras"); outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "ablation_by_block.csv", index=False)

    print("\n=== mean macro-F1 over 3 seeds x 3 folds ===")
    for split_name in SPLITS:
        sub = out[out.split == split_name]
        base = sub[sub.variant == "full"].macro_f1
        print(f"\n--- {split_name} ---")
        for name in variants:
            m = sub[sub.variant == name].macro_f1
            d = m.mean() - base.mean()
            # seed spread of the baseline, for judging whether Delta is signal
            print(f"  {name:<26} {m.mean():.4f} +/- {m.std():.4f}"
                  f"   (Delta vs full: {d:+.4f})")
        print(f"  [baseline seed spread: +/-{base.std():.4f}]")


if __name__ == "__main__":
    main()
