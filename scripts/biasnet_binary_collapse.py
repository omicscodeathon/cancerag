"""Reproduce the BiasNet (Sanchez et al., 2021) binary-collapse comparison row of
Table 4.

Collapses the four-class CancerAg holdout predictions to the same binary
label space BiasNet evaluates: G-protein-bias = (G protein ∪ G-protein-
selectivity) vs β-arrestin-bias, dropping ERK rows. Evaluates the released
calibrated single-LightGBM model on this collapse, with 1 000-resample
bootstrap 95 % CIs (seed 42).

Run:
    PYTHONPATH=src python scripts/biasnet_binary_collapse.py
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from cancerag.ml.preprocessing import get_X_y_groups

warnings.filterwarnings("ignore")

CAL_PATH = "data/processed/ml_models/lightgbm_final_calibrated.joblib"
LE_PATH = "data/processed/ml_preprocessed/label_encoder.joblib"
TRAIN_PATH = "data/processed/ml_ready_dataset.parquet"
HOLDOUT_PATH = "data/holdout/dataset_holdout.parquet"
SEED = 42
N_BOOT = 1000


def main() -> None:
    cal = joblib.load(CAL_PATH)
    le = joblib.load(LE_PATH)
    X_train, _, _, _, _ = get_X_y_groups(pd.read_parquet(TRAIN_PATH), label_encoder=le)
    ho_X, ho_y, _, _, _ = get_X_y_groups(pd.read_parquet(HOLDOUT_PATH), label_encoder=le)
    for col in X_train.columns:
        if col not in ho_X.columns:
            ho_X[col] = 0.0
    ho_X = ho_X[X_train.columns]

    pred = cal.predict(ho_X)
    classes = list(le.classes_)
    gp_idx = [i for i, c in enumerate(classes) if c in ("G protein", "G protein selectivity")]
    erk_idx = [i for i, c in enumerate(classes) if c == "ERK"]
    keep = ~np.isin(ho_y, erk_idx)
    y_bin = np.where(np.isin(ho_y[keep], gp_idx), 0, 1)
    p_bin = np.where(np.isin(pred[keep], gp_idx), 0, 1)

    acc = float(accuracy_score(y_bin, p_bin))
    mf1 = float(f1_score(y_bin, p_bin, average="macro"))
    rng = np.random.default_rng(SEED)
    n = len(y_bin)
    accs, f1s = [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        accs.append(accuracy_score(y_bin[idx], p_bin[idx]))
        f1s.append(f1_score(y_bin[idx], p_bin[idx], average="macro"))

    print(f"n (ERK rows dropped): {n}")
    print(f"binary accuracy: {acc:.4f}  [{np.percentile(accs, 2.5):.3f}, "
          f"{np.percentile(accs, 97.5):.3f}]")
    print(f"binary macro-F1: {mf1:.4f}  [{np.percentile(f1s, 2.5):.3f}, "
          f"{np.percentile(f1s, 97.5):.3f}]")


if __name__ == "__main__":
    main()
