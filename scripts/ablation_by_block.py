"""Block-wise ablation: separate ligand 3D shape from receptor-derived features.

`reviewer_extras.ablation_drop_structural` drops one combined "structural"
block, but that block mixes three different information sources: descriptors of
the docked *ligand's* 3D shape (which need no receptor), pair-level docking
energetics, and receptor-contact fingerprints. A ligand is a structure too, so
the combined Delta cannot answer "does the receptor help?" -- only "does the
whole docking arm help?". This runs each block separately.

Two things about the design are deliberate, and both exist because the earlier
version of this script could not answer the question it was written to ask.

**The folds are shuffled, and they have to be.** This script used to pass a
`seed` to `GroupKFold`, which ignores it: the three "seeds" re-fit on one fixed
partition, so the spread printed as the yardstick for signal was zero by
construction, and a Delta compared against zero has not been compared at all.
It now asks `make_grouped_cv` for a seeded group permutation (`shuffle=True`),
so each seed is a different partition of receptors or scaffolds and the spread
is real fold-assignment variance. That option is opt-in precisely so this
script can have it without moving any audited number elsewhere (HAL-6168).

**The comparison is paired.** "Does this block contribute?" is a question about
the same data with and without it, so within every (regime, seed, fold) all
four variants are fitted on identical row indices with an identical model
`random_state`. The per-seed Delta is then a paired difference, and its spread
over seeds is the noise the Delta actually has to clear. Comparing two
independently-noisy means instead -- the old shape -- throws away that pairing
and is far less able to resolve a small effect.

Run:
    PYTHONPATH=src python scripts/ablation_by_block.py
    PYTHONPATH=src python scripts/ablation_by_block.py \
        --dataset data/processed/regen_20261123/ml_ready_dataset.parquet \
        --label-encoder data/processed/regen_20261123/ml_preprocessed/label_encoder.joblib
"""
from __future__ import annotations

import argparse
import hashlib
import json as _json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from cancerag.ml.preprocessing import (
    build_full_pipeline,
    get_X_y_groups,
    make_grouped_cv,
)
from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight

warnings.filterwarnings("ignore")

POSE_3D = ("Asphericity", "Eccentricity", "InertialShapeFactor", "NPR1", "NPR2",
           "PMI1", "PMI2", "PMI3", "RadiusOfGyration", "SpherocityIndex",
           "pose_3d_missing")
# needs the receptor to compute at all
RECEPTOR_DEPENDENT = ("vina_", "ifp_", "n_residues_contacted", "n_total_contacts",
                      "n_poses", "ifp_missing", "ifp_no_contacts")

# Ten seeds, not three. Three gives a standard deviation on two degrees of
# freedom, which is too thin to put a confidence interval on; ten is still
# cheap here and makes the interval worth printing.
SEEDS = (42, 7, 13, 101, 2029, 3, 57, 811, 1234, 99)
FOLDS = 3

DEFAULT_DATASET = "data/processed/ml_ready_dataset.parquet"
DEFAULT_LABEL_ENCODER = "data/processed/ml_preprocessed/label_encoder.joblib"
DEFAULT_SELECTION = "data/processed/ml_models/selection_decision.json"


def cols_matching(X, prefixes):
    return [c for c in X.columns
            if any(c.startswith(p) or c == p for p in prefixes)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ci95(vals: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% t interval on the mean of `vals`."""
    from scipy import stats
    n = len(vals)
    if n < 2:
        return (float("nan"), float("nan"))
    sem = float(np.std(vals, ddof=1)) / np.sqrt(n)
    half = float(stats.t.ppf(0.975, n - 1)) * sem
    m = float(np.mean(vals))
    return (m - half, m + half)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--label-encoder", default=DEFAULT_LABEL_ENCODER)
    ap.add_argument("--selection", default=DEFAULT_SELECTION)
    ap.add_argument("--output-dir", default="data/processed/ml_models/extras")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--folds", type=int, default=FOLDS)
    ap.add_argument(
        "--reuse-folds", metavar="CSV",
        help="Skip fitting and rebuild every table from an existing "
             "ablation_by_block_folds.csv. Each row of that file is one fit, "
             "so the per-seed and summary tables are pure functions of it; "
             "this recomputes them without repeating 360 model fits.")
    args = ap.parse_args(argv)

    ds = Path(args.dataset)
    df = pd.read_parquet(ds)
    le = joblib.load(args.label_encoder)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)
    winner = _json.loads(Path(args.selection).read_text())["chosen"]
    factory = MODEL_FACTORIES[winner]

    pose_cols = cols_matching(X, POSE_3D)
    recep_cols = cols_matching(X, RECEPTOR_DEPENDENT)
    print(f"dataset={ds}  rows={len(df)}  model={winner}")
    print(f"total={X.shape[1]}  ligand-3D-shape={len(pose_cols)}  "
          f"receptor-dependent={len(recep_cols)}")

    variants = {
        "full": X,
        "drop_receptor_dependent": X.drop(columns=recep_cols),  # keeps ligand 3D shape
        "drop_ligand_3d_shape": X.drop(columns=pose_cols),      # keeps receptor features
        "drop_both": X.drop(columns=recep_cols + pose_cols),
    }

    scaf_col = "murcko_scaffold" if "murcko_scaffold" in df.columns else "scaffold"

    # Each entry returns a list of (train_idx, test_idx) for one seed. The
    # folds are built ONCE per (regime, seed) and reused across all four
    # variants, which is what makes the Delta paired.
    def splits_for(regime: str, seed: int):
        if regime == "stratified":
            return list(StratifiedKFold(
                n_splits=args.folds, shuffle=True, random_state=seed).split(X, y))
        if regime == "scaffold_grouped":
            return make_grouped_cv(df, group_col=scaf_col, n_splits=args.folds,
                                   seed=seed, shuffle=True)
        if regime == "receptor_grouped":
            return make_grouped_cv(df, group_col="receptor_uniprot",
                                   n_splits=args.folds, seed=seed, shuffle=True)
        raise ValueError(regime)

    REGIMES = ("stratified", "scaffold_grouped", "receptor_grouped")

    if args.reuse_folds:
        long = pd.read_csv(args.reuse_folds)
        print(f"reusing {len(long)} recorded fits from {args.reuse_folds}")
        seeds_used = sorted(long.seed.unique().tolist())
        folds_used = int(long.fold.nunique())
    else:
        long = None
        seeds_used, folds_used = list(args.seeds), args.folds

    rows = []
    for regime in (() if args.reuse_folds else REGIMES):
        for seed in args.seeds:
            splits = splits_for(regime, seed)
            for fold, (tr, te) in enumerate(splits):
                for name, Xv in variants.items():
                    model = factory(n_classes=n_classes, random_state=seed)
                    pipe = build_full_pipeline(model)
                    last = pipe.steps[-1][0]
                    pipe.fit(Xv.iloc[tr], y[tr],
                             **{f"{last}__sample_weight":
                                _combined_weight(y[tr], sw[tr])})
                    f1 = float(f1_score(y[te], pipe.predict(Xv.iloc[te]),
                                        average="macro", zero_division=0))
                    rows.append({
                        "split": regime, "variant": name, "seed": seed,
                        "fold": fold, "n_features": int(Xv.shape[1]),
                        "n_train": int(len(tr)), "n_test": int(len(te)),
                        "macro_f1": f1,
                    })
        print(f"  [{regime}] done ({len(args.seeds)} seeds x {args.folds} folds)")

    if long is None:
        long = pd.DataFrame(rows)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    long.to_csv(outdir / "ablation_by_block_folds.csv", index=False)

    # ---- per-seed means: one complete K-fold CV estimate per seed. This is
    # the unit of replication; the folds inside a seed are not independent of
    # each other, so an SD over folds would understate the real spread.
    #
    # This table keeps the file name and the column names the earlier version
    # wrote, because `scripts/figures/fig08_ablation.py` reads it and derives
    # the plotted band from the per-seed spread. That band used to come out at
    # zero; now that the grouped folds are shuffled it is a real quantity, and
    # the figure picks that up without changing how it reads the file.
    per_seed = (long.groupby(["split", "variant", "seed"], as_index=False)
                    .agg(n_features=("n_features", "first"),
                         macro_f1=("macro_f1", "mean")))
    per_seed.to_csv(outdir / "ablation_by_block.csv", index=False)

    summary_rows = []
    for regime in REGIMES:
        base = per_seed[(per_seed.split == regime) &
                        (per_seed.variant == "full")].set_index("seed").macro_f1
        for name in variants:
            v = per_seed[(per_seed.split == regime) &
                         (per_seed.variant == name)].set_index("seed").macro_f1
            # paired on seed: same fold partition, same model random_state
            delta = (v - base).reindex(base.index)
            lo, hi = ci95(delta.values) if name != "full" else (0.0, 0.0)
            fold_sd = float(long[(long.split == regime) &
                                 (long.variant == name)].macro_f1.std(ddof=1))
            summary_rows.append({
                "split": regime, "variant": name,
                "n_features": int(variants[name].shape[1]),
                "n_seeds": len(base), "n_folds": folds_used,
                "macro_f1_mean": float(v.mean()),
                "macro_f1_sd_over_seeds": float(v.std(ddof=1)),
                "macro_f1_sd_over_folds": fold_sd,
                "delta_vs_full_mean": float(delta.mean()),
                "delta_vs_full_sd": float(delta.std(ddof=1)),
                "delta_ci95_lo": lo, "delta_ci95_hi": hi,
                "delta_excludes_zero": bool(name != "full" and (lo > 0 or hi < 0)),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "ablation_by_block_summary.csv", index=False)

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(ds), "dataset_sha256": sha256(ds),
        "label_encoder": str(args.label_encoder),
        "selection_decision": str(args.selection), "model": winner,
        "seeds": list(seeds_used), "folds": folds_used,
        "grouped_cv_shuffle": True,
        "n_rows": int(len(df)), "n_features_full": int(X.shape[1]),
        "n_ligand_3d_shape": len(pose_cols),
        "n_receptor_dependent": len(recep_cols),
        "pairing": "all variants share fold indices and model random_state "
                   "within each (regime, seed, fold)",
        "replication_unit": "seed (one complete K-fold CV estimate each)",
    }
    (outdir / "ablation_by_block.meta.json").write_text(
        _json.dumps(meta, indent=2) + "\n")

    print(f"\n=== macro-F1, mean +/- SD over {len(seeds_used)} seeds "
          f"({folds_used}-fold CV each; grouped folds shuffled per seed) ===")
    for regime in REGIMES:
        sub = summary[summary.split == regime]
        print(f"\n--- {regime} ---")
        for _, r in sub.iterrows():
            if r.variant == "full":
                print(f"  {r.variant:<26} {r.macro_f1_mean:.4f} "
                      f"+/- {r.macro_f1_sd_over_seeds:.4f}   (baseline)")
            else:
                verdict = "RESOLVED" if r.delta_excludes_zero else "within noise"
                print(f"  {r.variant:<26} {r.macro_f1_mean:.4f} "
                      f"+/- {r.macro_f1_sd_over_seeds:.4f}   "
                      f"Delta {r.delta_vs_full_mean:+.4f} "
                      f"+/- {r.delta_vs_full_sd:.4f}  "
                      f"95% CI [{r.delta_ci95_lo:+.4f}, {r.delta_ci95_hi:+.4f}]  "
                      f"{verdict}")
    print(f"\nwrote {outdir/'ablation_by_block.csv'}, "
          f"{outdir/'ablation_by_block_summary.csv'}, "
          f"{outdir/'ablation_by_block_folds.csv'}, "
          f"{outdir/'ablation_by_block.meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
