"""Cross-regime SHAP robustness analysis.

Runs SHAP on the winning model under each of the four evaluation regimes
(stratified, scaffold-grouped, receptor-grouped CV, and the temporal-holdout
final model), records the top features per regime, then reports which features
stand out *across* regimes. A feature that survives the hardest regimes
(receptor-grouped, temporal) is a far stronger mechanistic candidate than one
that is only "important" under the friendly stratified split.

Outputs (data/processed/ml_models/interpretability/):
  - shap_per_regime.csv       : (feature, regime, selection_frequency, mean_abs_shap)
  - shap_cross_regime.csv     : (feature, n_regimes_top, regimes, mean_selection_freq)
"""
from __future__ import annotations
import warnings, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

from cancerag.ml.preprocessing import get_X_y_groups, make_grouped_cv, build_full_pipeline
from cancerag.ml.model_training import MODEL_FACTORIES, _combined_weight
from cancerag.ml.interpretability import _shap_for_pipeline

DATA = "data/processed/ml_ready_dataset.parquet"
HOLD = "data/holdout/dataset_holdout.parquet"
LE = "data/processed/ml_preprocessed/label_encoder.joblib"
DEC = "data/processed/ml_models/selection_decision.json"
OUT = Path("data/processed/ml_models/interpretability")
SEEDS = (42, 7, 13)
N_FOLDS = 5
TOP_N = 20

REGIMES = {
    "stratified": None,            # StratifiedKFold
    "scaffold":   "scaffold",      # Bemis-Murcko grouped
    "receptor":   "receptor_uniprot",  # UniProt grouped (hardest CV)
}


def top_features_for_regime(df, X, y, sw, factory, n_classes, group_col):
    """Selection frequency of each feature across seeds x folds for one regime."""
    freq, total, n_runs = {}, {}, 0
    for seed in SEEDS:
        if group_col is None:
            splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                          random_state=seed).split(X, y))
        else:
            splits = make_grouped_cv(df, group_col=group_col, n_splits=N_FOLDS, seed=seed)
            if not splits:
                continue
        for tr, te in splits:
            model = factory(n_classes=n_classes, random_state=seed)
            pipe = build_full_pipeline(model, impute=True, variance_threshold=1e-4,
                                       correlation_threshold=0.97, scale=True)
            cw = _combined_weight(y[tr], sw[tr])
            last = pipe.steps[-1][0]
            try:
                pipe.fit(X.iloc[tr], y[tr], **{f"{last}__sample_weight": cw})
                shap_values, feat_names, _ = _shap_for_pipeline(pipe, X.iloc[te])
                if isinstance(shap_values, list):
                    abs_shap = np.mean([np.abs(s).mean(axis=0) for s in shap_values], axis=0)
                elif getattr(shap_values, "ndim", 0) == 3:
                    abs_shap = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    abs_shap = np.abs(shap_values).mean(axis=0)
                top_idx = np.argsort(abs_shap)[-TOP_N:]
                for f, v in zip(feat_names, abs_shap):
                    total[f] = total.get(f, 0.0) + float(v)
                for i in top_idx:
                    f = feat_names[i]; freq[f] = freq.get(f, 0) + 1
                n_runs += 1
            except Exception as exc:
                print(f"  [{group_col}] fold failed: {exc}")
    rows = [{"feature": f, "selection_frequency": freq.get(f, 0) / max(1, n_runs),
             "mean_abs_shap": total.get(f, 0.0) / max(1, n_runs)}
            for f in set(total) | set(freq)]
    return pd.DataFrame(rows), n_runs


def _per_row_abs_shap(shap_values):
    """Return per-(row, feature) |SHAP|, averaged over classes -> (n_rows, n_feat)."""
    if isinstance(shap_values, list):                       # list[class] of (n,p)
        return np.mean([np.abs(s) for s in shap_values], axis=0)
    if getattr(shap_values, "ndim", 0) == 3:               # (n, p, class)
        return np.abs(shap_values).mean(axis=2)
    return np.abs(shap_values)                              # (n, p)


def holdout_top(df, X, y, sw, factory, n_classes, n_boot=5):
    """Bootstrapped SHAP on the temporal holdout, on equal footing with the CV
    regimes: for each of the SEEDS a final model is fit and explained on the
    holdout, then holdout rows are bootstrap-resampled n_boot times; a feature's
    selection frequency is the fraction of the SEEDS x n_boot resamples in which
    it lands in the top-N by mean|SHAP|."""
    ho = pd.read_parquet(HOLD)
    le = joblib.load(LE)
    ho_X, ho_y, _, _, _ = get_X_y_groups(ho, label_encoder=le)
    for c in X.columns:
        if c not in ho_X.columns:
            ho_X[c] = 0.0
    ho_X = ho_X[X.columns]
    cw = _combined_weight(y, sw)
    freq, total, n_runs = {}, {}, 0
    for seed in SEEDS:
        model = factory(n_classes=n_classes, random_state=seed)
        pipe = build_full_pipeline(model, impute=True, variance_threshold=1e-4,
                                   correlation_threshold=0.97, scale=True)
        last = pipe.steps[-1][0]
        pipe.fit(X, y, **{f"{last}__sample_weight": cw})
        shap_values, feat_names, X_used = _shap_for_pipeline(pipe, ho_X, max_samples=172)
        per_row = _per_row_abs_shap(shap_values)            # (n_rows, n_feat)
        n_rows = per_row.shape[0]
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = rng.integers(0, n_rows, n_rows)
            abs_shap = per_row[idx].mean(axis=0)
            top_idx = np.argsort(abs_shap)[-TOP_N:]
            for f, v in zip(feat_names, abs_shap):
                total[f] = total.get(f, 0.0) + float(v)
            for i in top_idx:
                f = feat_names[i]; freq[f] = freq.get(f, 0) + 1
            n_runs += 1
    rows = [{"feature": f, "selection_frequency": freq.get(f, 0) / max(1, n_runs),
             "mean_abs_shap": total.get(f, 0.0) / max(1, n_runs)}
            for f in set(total) | set(freq)]
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    le = joblib.load(LE)
    X, y, sw, le, _ = get_X_y_groups(df, label_encoder=le)
    n_classes = len(le.classes_)
    winner = json.loads(Path(DEC).read_text())["chosen"]
    factory = MODEL_FACTORIES[winner.split("_")[0] if "_" in winner else winner]
    print(f"winner={winner} | classes={n_classes}")

    per = {}
    for name, gc in REGIMES.items():
        print(f"=== regime: {name} (group_col={gc}) ===")
        tab, n_runs = top_features_for_regime(df, X, y, sw, factory, n_classes, gc)
        tab["regime"] = name
        per[name] = tab
        print(f"    {n_runs} runs; top-5: "
              f"{tab.sort_values('selection_frequency', ascending=False)['feature'].head(5).tolist()}")

    print("=== regime: temporal (holdout final model) ===")
    tab = holdout_top(df, X, y, sw, factory, n_classes)
    tab["regime"] = "temporal"
    per["temporal"] = tab
    print(f"    top-5: {tab.sort_values('mean_abs_shap', ascending=False)['feature'].head(5).tolist()}")

    allrows = pd.concat(per.values(), ignore_index=True)
    allrows.to_csv(OUT / "shap_per_regime.csv", index=False)

    # cross-regime: a feature is "top" in a regime if selection_frequency >= 0.5
    # (for CV regimes) or it is in the temporal top-20 (freq == 1.0).
    regimes = list(per.keys())
    feats = set(allrows["feature"])
    summary = []
    for f in feats:
        hits, freqs = [], []
        for rg in regimes:
            row = per[rg][per[rg]["feature"] == f]
            sf = float(row["selection_frequency"].iloc[0]) if len(row) else 0.0
            freqs.append(sf)
            if sf >= 0.5:
                hits.append(rg)
        summary.append({"feature": f, "n_regimes_top": len(hits),
                        "regimes": ";".join(hits),
                        "mean_selection_freq": round(float(np.mean(freqs)), 3),
                        **{f"sf_{rg}": round(freqs[i], 3) for i, rg in enumerate(regimes)}})
    summ = pd.DataFrame(summary).sort_values(
        ["n_regimes_top", "mean_selection_freq"], ascending=False).reset_index(drop=True)
    summ.to_csv(OUT / "shap_cross_regime.csv", index=False)

    print("\n=== FEATURES BY CROSS-REGIME ROBUSTNESS ===")
    print(summ.head(20).to_string(index=False))
    survive_all = summ[summ["n_regimes_top"] == len(regimes)]["feature"].tolist()
    print(f"\nSurvive ALL {len(regimes)} regimes (sf>=0.5 each): {survive_all}")


if __name__ == "__main__":
    main()
