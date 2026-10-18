"""Stacking-ensemble data flow, as implemented in ``ensemble.run_stacking``."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from . import _diagram as D
from . import _style as S
from ._data import Artefacts

NAME = "fig13_stacking_architecture"
TITLE = "Stacking-ensemble data flow"
DESCRIPTION = """
The stacking ensemble as `cancerag.ml.ensemble.run_stacking` builds it.
LightGBM and XGBoost use the engineered feature matrix; the third base
learner is a logistic-regression head on frozen ChemBERTa-77M-MTR SMILES
embeddings. Each produces out-of-fold class probabilities under
scaffold-grouped 5-fold CV, concatenated into a 12-column matrix for a
multinomial logistic-regression meta-learner. The code applies no
post-hoc calibration to the stacked output. Holdout numbers come from
`stacking_meta.json`; the meta-learner's macro-F1 on its own training
matrix is in-sample and is labelled as such.
"""
SOURCES = [
    "processed/ml_models/advanced/stacking_meta.json",
    "processed/ml_models/model_meta.json",
    "processed/ml_models/baselines/chemberta/chemberta_summary.json",
]

BASE_LABEL = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "chemberta_lr": "ChemBERTa-LR"}


def build(art: Artefacts):
    st = art.json(SOURCES[0])
    meta = art.model_meta
    cb = art.json(SOURCES[2])
    n, p, k = meta["n_train_eligible"], meta["n_features"], meta["n_classes"]
    bases = st["base_models"]
    hm = st["holdout_metrics"]["macro_f1"]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    D.blank(ax, (0, 100), (0, 100))
    h = 11

    inp = D.box(ax, 22, 85, 56, h, "Train-eligible set",
                f"n = {n} · {p} engineered features · SMILES", S.STAGE_COLOR["source"])
    xs = {"lightgbm": 2, "xgboost": 35, "chemberta_lr": 68}
    subs = {
        "lightgbm": f"gradient-boosted trees\non {p} features",
        "xgboost": f"gradient-boosted trees\non {p} features",
        "chemberta_lr": f"frozen {cb['embedding_dim']}-d SMILES embedding\n+ logistic regression",
    }
    oofs = []
    for b in sorted(bases, key=lambda m: xs.get(m, 0)):
        x = xs.get(b, 35)
        bx = D.box(ax, x, 61, 30, 14, BASE_LABEL.get(b, b), subs.get(b, ""),
                   S.STAGE_COLOR["ml"], title_size=7.8)
        D.arrow(ax, inp["bottom"], bx["top"], rad=0.0)
        ob = D.note(ax, x + 5, 43, 20, 9, f"out-of-fold probabilities\n{n} × {k}")
        D.arrow(ax, bx["bottom"], ob["top"])
        oofs.append(ob)
    stack = D.box(ax, 18, 25, 64, h, "Stacked matrix",
                  f"{n} × {k * len(bases)} = {k} classes × {len(bases)} base learners",
                  S.STAGE_COLOR["features"])
    for ob in oofs:
        D.arrow(ax, ob["bottom"], stack["top"])
    ml = D.box(ax, 18, 5, 64, h, "Meta-learner", st["meta_learner"],
               S.STAGE_COLOR["deploy"])
    D.arrow(ax, stack["bottom"], ml["top"])
    ax.text(84, 10.5,
            f"holdout macro-F1\n{hm['point_estimate']:.3f} "
            f"[{hm['ci_lo']:.3f}, {hm['ci_hi']:.3f}]\n"
            f"in-sample fit {st['stacking_oof_macro_f1']:.3f}",
            fontsize=6.4, color=S.TEXT_SECONDARY, va="center")
    ax.text(51.5, 57.5, "scaffold-grouped 5-fold CV", fontsize=6.4,
            color=S.TEXT_MUTED, ha="left", va="center")

    facts = {
        "base learners": [BASE_LABEL.get(b, b) for b in bases],
        "train rows / features / classes": f"{n} / {p} / {k}",
        "stacked columns": k * len(bases),
        "meta-learner": st["meta_learner"],
        "holdout macro-F1": f"{hm['point_estimate']:.3f} [{hm['ci_lo']:.3f}, {hm['ci_hi']:.3f}]",
        "meta-learner in-sample macro-F1": round(st["stacking_oof_macro_f1"], 3),
        "post-hoc calibration of stacked output": "none in ensemble.run_stacking",
    }
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
