"""Temporal-holdout macro-F1 across modelling generations (Table 2 as a chart)."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from . import _style as S
from ._data import Artefacts

NAME = "fig03_holdout_progression"
TITLE = "Holdout macro-F1 across modelling generations"
DESCRIPTION = """
Temporal-holdout (year ≥ 2018, n = 172) macro-F1 with 1,000-resample bootstrap
95 % CIs for each modelling generation: the calibrated bake-off winner
(random forest), the Optuna-tuned model (the search tuned LightGBM — model
names are read from the artefacts), that tuned model with per-class decision
thresholds, the
ChemBERTa-77M-MTR + logistic-regression baseline and the stacking ensemble.
The dashed line is the majority-class macro-F1 from cross-validation. The v1
single-XGBoost value (0.210) is a historical single-seed run with no artefact
on disk; it is drawn hatched, without a CI, and labelled as such.
"""
SOURCES = [
    "processed/ml_models/model_meta.json",
    "processed/ml_models/advanced/optuna_tuning_meta.json",
    "processed/ml_models/advanced/thresholds_meta.json",
    "processed/ml_models/baselines/chemberta/chemberta_summary.json",
    "processed/ml_models/advanced/stacking_meta.json",
    "processed/ml_models/cv_results_summary.csv",
]

# Historical v1 value (manuscript Table 2). No artefact exists for this run.
V1_XGBOOST_HOLDOUT = 0.210


def _ci(d: dict) -> tuple[float, float, float]:
    return d["point_estimate"], d["ci_lo"], d["ci_hi"]


def build(art: Artefacts):
    meta = art.model_meta
    opt = art.json(SOURCES[1])
    thr = art.json(SOURCES[2])
    cb = art.json(SOURCES[3])
    st = art.json(SOURCES[4])
    summ = art.csv(SOURCES[5])
    floor = float(summ[summ.model == "majority_class"].macro_f1_mean.iloc[0])

    # Model names come from the artefacts: the Optuna search tuned LightGBM
    # even though the bake-off winner is random forest (label corrected in
    # optuna_tuning_meta.json on 2026-10-18).
    tuned_label = S.MODEL_LABEL.get(opt["model"], opt["model"])
    thr_est = thr.get("threshold_model", {}).get("estimator", opt["model"])
    thr_label = S.MODEL_LABEL.get(thr_est, thr_est) + " (tuned)"
    winner = S.MODEL_LABEL[meta["selection_decision"]["chosen"]]
    rows = [
        ("Gen 1  Single XGBoost (v1)\nhistorical run, no artefact",
         (V1_XGBOOST_HOLDOUT, None, None), S.GREY_LIGHT, True),
        (f"Gen 2  {winner}\ncalibrated, bake-off winner",
         _ci(meta["holdout_metrics"]["macro_f1"]), S.BLUE, False),
        (f"Gen 2′  {tuned_label}\nOptuna, {opt['n_trials']} trials",
         _ci(opt["holdout_metrics"]["macro_f1"]), S.BLUE, False),
        (f"Gen 2″  {thr_label}\n+ per-class thresholds",
         _ci(thr["holdout_metrics_with_tuned_thresholds"]["macro_f1_tuned_thresholds"]),
         S.BLUE, False),
        ("Gen 3  ChemBERTa-77M-MTR + LR\nligand-only embedding",
         _ci(cb["holdout"]["macro_f1"]), S.AQUA, False),
        ("Gen 4  Stacking ensemble\nLightGBM + XGBoost + ChemBERTa-LR",
         _ci(st["holdout_metrics"]["macro_f1"]), S.ORANGE, False),
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    facts = {}
    n = len(rows)
    for i, (label, (pt, lo, hi), color, hist) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, pt, height=0.56, color=color if not hist else S.SURFACE,
                edgecolor=S.GREY_DARK if hist else S.SURFACE,
                hatch="////" if hist else None, linewidth=0.8 if hist else 0,
                zorder=2)
        if lo is not None:
            ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], color=S.TEXT_PRIMARY,
                        lw=0.8, capsize=2.5, zorder=3)
            txt = f"{pt:.3f}  [{lo:.3f}, {hi:.3f}]"
            ax.text(hi + 0.008, y, txt, va="center", fontsize=S.FS_SMALL,
                    color=S.TEXT_PRIMARY)
        else:
            ax.text(pt + 0.008, y, f"{pt:.3f}  (no CI)", va="center",
                    fontsize=S.FS_SMALL, color=S.TEXT_SECONDARY)
        facts[label.split("\n")[0].strip()] = (
            f"{pt:.3f}" + (f" [{lo:.3f}, {hi:.3f}]" if lo is not None else " (historical, no artefact)")
        )
    ax.axvline(floor, color=S.TEXT_SECONDARY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(floor, n - 0.45, f" majority-class floor {floor:.3f}", fontsize=6.8,
            color=S.TEXT_SECONDARY, va="bottom", ha="left")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=S.FS_SMALL)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.52)
    ax.set_ylim(-0.6, n - 0.1)
    ax.set_xlabel("Temporal-holdout macro-F1 (bootstrap 95 % CI, n = 172)")
    S.xgrid(ax)
    facts["majority-class floor (CV)"] = f"{floor:.3f}"
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
