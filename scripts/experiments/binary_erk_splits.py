"""Three things the four-class view cannot answer.

1. THE WELL-POSED CONTRAST.  "G protein" and "β Arrestin" are mirror images of
   each other -- Gi/β-Arr (212) against β-Arr/Gi (93) -- so together they form
   one clean axis with 575 measurements. The other two classes are different
   *kinds* of comparison: "G protein selectivity" is Gi-over-Gq, with no
   arrestin in it at all. Define the contrast from `bias_pathway`, the actual
   comparison, rather than `bias_category`, which is a lossy summary: 55 rows
   labelled "G protein" are really G-protein-over-ERK and do not belong on
   this axis.

2. IS THERE ANY ERK SIGNAL AT ALL?  At n=59 over 18 receptors, "predict ERK"
   is not a well-posed request -- every method tried returns F1 = 0.00. The
   answerable question is whether structural features carry *any* ERK
   information above chance. One-vs-rest against a label-permutation null.
   A clean negative is a result; a model that cannot be beaten by its own
   shuffled labels is not.

3. DO THE HARD SPLITS ACTUALLY LEAK?  The RCAF skill is blunt about this:
   "assume yours leaks and go looking." Scaffold grouping can miss a shared
   pharmacophore; an entity split can put two receptors sharing a source
   structure on opposite sides.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

from cancerag.ml.model_training import MODEL_FACTORIES  # noqa: E402
from cancerag.ml.preprocessing import (  # noqa: E402
    build_full_pipeline, get_X_y_groups, make_grouped_cv,
)

SEEDS = [42, 7, 13]
N_PERM = 100   # resolves p to 0.01; finer than any claim this supports
OUT = Path("data/processed/ml_models/binary_erk_splits.json")
results: dict = {}

tr = pd.read_parquet("data/processed/ml_ready_dataset.parquet")
X_all, y4, sw, _, _ = get_X_y_groups(tr)

G_SUBTYPES = ("Gi", "Gs", "Gq", "Go", "G12", "Gz", "G protein")


def axis_from_pathway(p: str):
    """1 = beta-arrestin preferred, 0 = a G protein preferred, None = other axis."""
    if not isinstance(p, str) or "/" not in p:
        return None
    pref, against = (s.strip() for s in p.split("/", 1))
    barr = {"β-Arr", "b-Arr", "βArr", "beta-Arr"}
    if pref in barr and against in G_SUBTYPES:
        return 1
    if pref in G_SUBTYPES and against in barr:
        return 0
    return None


tr["_axis"] = tr["bias_pathway"].map(axis_from_pathway)
mask = tr["_axis"].notna().to_numpy()
print(f"rows on the G-protein/beta-arrestin axis: {mask.sum()} of {len(tr)}")
print(f"  excluded (other axes, e.g. Gi/Gq or ERK/Gs): {(~mask).sum()}")

# ===================================================== 1. the binary contrast
print("\n" + "=" * 70)
print("1. BINARY  G protein vs beta-arrestin   (axis from bias_pathway)")
print("=" * 70)
sub = tr[mask].reset_index(drop=True)
Xb = X_all[mask].reset_index(drop=True)
yb = sub["_axis"].astype(int).to_numpy()
swb = sw[mask]
print(f"n={len(yb)}  beta-arrestin={yb.sum()}  G protein={len(yb)-yb.sum()}")
floor_b = f1_score(yb, np.full_like(yb, np.bincount(yb).argmax()),
                   average="macro", zero_division=0)
print(f"majority-class macro-F1 floor: {floor_b:.4f}")

REGIMES = {
    "stratified": lambda s: list(
        StratifiedKFold(5, shuffle=True, random_state=s).split(Xb, yb)),
    "scaffold_grouped": lambda s: make_grouped_cv(
        sub, group_col="scaffold", n_splits=5, seed=s),
    "receptor_grouped": lambda s: make_grouped_cv(
        sub, group_col="receptor_uniprot", n_splits=5, seed=s),
}


def score(cols, regime, seed, Xsrc, ysrc, wsrc, meta):
    splits = REGIMES[regime](seed) if meta is sub else None
    if splits is None:
        return None
    Xc = Xsrc[cols]
    f1s, mccs, aucs = [], [], []
    for tri, tei in splits:
        p = build_full_pipeline(
            MODEL_FACTORIES["lightgbm"](n_classes=2, random_state=seed),
            variance_threshold=0.0, correlation_threshold=1.01)
        last = p.steps[-1][0]
        try:
            p.fit(Xc.iloc[tri], ysrc[tri], **{f"{last}__sample_weight": wsrc[tri]})
        except TypeError:
            p.fit(Xc.iloc[tri], ysrc[tri])
        pr = p.predict(Xc.iloc[tei])
        f1s.append(f1_score(ysrc[tei], pr, average="macro", zero_division=0))
        mccs.append(matthews_corrcoef(ysrc[tei], pr))
        try:
            aucs.append(roc_auc_score(ysrc[tei], p.predict_proba(Xc.iloc[tei])[:, 1]))
        except Exception:
            pass
    return (float(np.mean(f1s)), float(np.mean(mccs)),
            float(np.mean(aucs)) if aucs else None)


ifp = [c for c in Xb.columns if c.startswith("ifp_")]
bw = [c for c in ifp if c.startswith("ifp_BW")]
structure = [c for c in Xb.columns
             if c.startswith(("ifp_", "vina_", "pose_", "sasa", "rmsd"))]
robust11 = json.loads(
    (Path("data/processed/ml_models") / "rcaf_parsimony.json").read_text()
)["robust_features"] if (Path("data/processed/ml_models")
                         / "rcaf_parsimony.json").exists() else []
robust11 = [c for c in robust11 if c in Xb.columns]

VIEWS = {"all": list(Xb.columns),
         "bw_only": [c for c in Xb.columns if c not in ifp or c in bw],
         "ligand_only": [c for c in Xb.columns if c not in structure]}
if robust11:
    VIEWS["robust_11"] = robust11

brows = []
for v, cols in VIEWS.items():
    for regime in REGIMES:
        vals = [score(cols, regime, s, Xb, yb, swb, sub) for s in SEEDS]
        vals = [x for x in vals if x]
        if vals:
            brows.append({"view": v, "n": len(cols), "regime": regime,
                          "macro_f1": float(np.mean([a for a, _, _ in vals])),
                          "sd": float(np.std([a for a, _, _ in vals])),
                          "mcc": float(np.mean([b for _, b, _ in vals])),
                          "auc": float(np.mean([c for _, _, c in vals
                                                if c is not None]) or 0)})
    print(f"  view {v} ({len(cols)}) done", flush=True)
bd = pd.DataFrame(brows)
print()
print(bd.pivot_table(index=["view", "n"], columns="regime",
                     values=["macro_f1", "auc"]).round(4).to_string())
results["binary"] = {"rows": brows, "floor": float(floor_b),
                     "n": int(len(yb)), "n_pos": int(yb.sum())}

# ============================================== 2. is there ANY ERK signal?
print("\n" + "=" * 70)
print("2. ERK one-vs-rest against a label-permutation null")
print("=" * 70)
classes = sorted(tr["bias_category"].unique())
for target in ("ERK", "G protein selectivity"):
    yt = (tr["bias_category"] == target).astype(int).to_numpy()
    print(f"\n{target}: {yt.sum()} positives of {len(yt)}")

    def one(ys, seed):
        splits = make_grouped_cv(tr, group_col="receptor_uniprot",
                                 n_splits=5, seed=seed)
        if not splits:
            return None
        sc = []
        for tri, tei in splits:
            if len(np.unique(ys[tri])) < 2:
                continue
            p = build_full_pipeline(
                MODEL_FACTORIES["lightgbm"](n_classes=2, random_state=seed),
                variance_threshold=0.0, correlation_threshold=1.01)
            p.fit(X_all.iloc[tri], ys[tri])
            sc.append(f1_score(ys[tei], p.predict(X_all.iloc[tei]),
                               average="macro", zero_division=0))
        return float(np.mean(sc)) if sc else None

    obs = np.mean([v for v in (one(yt, s) for s in SEEDS) if v is not None])
    rng = np.random.default_rng(0)
    null = []
    for i in range(N_PERM):
        v = one(rng.permutation(yt), 42)
        if v is not None:
            null.append(v)
        if (i + 1) % 50 == 0:
            print(f"    perm {i+1}/{N_PERM}", flush=True)
    null = np.array(null)
    pval = float(((null >= obs).sum() + 1) / (len(null) + 1))
    print(f"  observed macro-F1 (receptor-grouped): {obs:.4f}")
    print(f"  permutation null: mean {null.mean():.4f}  sd {null.std():.4f}  "
          f"95th pct {np.percentile(null, 95):.4f}")
    print(f"  p = {pval:.4f}  -> "
          f"{'signal above chance' if pval < 0.05 else 'NOT distinguishable from chance'}")
    results[f"onevsrest_{target}"] = {
        "observed": float(obs), "null_mean": float(null.mean()),
        "null_sd": float(null.std()), "p_value": pval, "n_perm": len(null),
        "n_positive": int(yt.sum())}

# ================================================= 3. do the splits leak?
print("\n" + "=" * 70)
print("3. SPLIT INTEGRITY  (assume they leak, go looking)")
print("=" * 70)
sc_sizes = tr.groupby("scaffold").size()
print(f"scaffolds: {len(sc_sizes)}  singletons: {(sc_sizes == 1).sum()}  "
      f"largest: {sc_sizes.max()}")
print(f"  -> {(sc_sizes == 1).sum() / len(sc_sizes):.0%} of scaffold groups hold one row, "
      "so scaffold-grouping is close to random for most of the data")

dup = tr.groupby(["inchikey14", "receptor_uniprot"]).size()
print(f"same ligand+receptor appearing more than once: {(dup > 1).sum()} "
      f"(max {dup.max()} rows)")

rec_sizes = tr.groupby("receptor_uniprot").size()
print(f"receptors: {len(rec_sizes)}  median rows {rec_sizes.median():.0f}  "
      f"largest {rec_sizes.max()} ({rec_sizes.idxmax()})")
print(f"  top receptor is {rec_sizes.max()/len(tr):.0%} of the data -- a "
      "receptor-grouped fold holding it out loses that share at once")

lig_across = tr.groupby("inchikey14")["receptor_uniprot"].nunique()
print(f"ligands measured at >1 receptor: {(lig_across > 1).sum()} of "
      f"{len(lig_across)}  (max {lig_across.max()} receptors)")
print("  -> these cross receptor-grouped folds by construction: the same "
       "molecule appears on both sides, only the receptor differs")

results["split_integrity"] = {
    "n_scaffolds": int(len(sc_sizes)),
    "scaffold_singletons": int((sc_sizes == 1).sum()),
    "largest_scaffold": int(sc_sizes.max()),
    "n_receptors": int(len(rec_sizes)),
    "largest_receptor_share": float(rec_sizes.max() / len(tr)),
    "ligands_at_multiple_receptors": int((lig_across > 1).sum()),
    "max_receptors_per_ligand": int(lig_across.max()),
}

OUT.write_text(json.dumps(results, indent=2, default=float))
print(f"\nwritten: {OUT}")
