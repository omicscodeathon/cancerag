"""End-to-end CancerAg pipeline schematic, with counts read from the artefacts."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from . import _diagram as D
from . import _style as S
from ._counts import SOURCES as COUNT_SOURCES
from ._counts import pipeline_counts
from ._data import Artefacts

NAME = "fig00_pipeline_schematic"
TITLE = "CancerAg methods and data flow"
DESCRIPTION = """
Data sources, structure preparation, docking, featurisation and modelling,
with every count read from the artefacts at build time: curated measurements,
molecules and receptors (`unified_ligands.csv`, `attrition.json`),
binding-site methods counted per record in `binding_sites.json` (the file's
own `totals` block is stale), re-dock validation (`redock_validation.json`),
docking success over unique pairs (`docking_features.csv`), the assembled
matrix and temporal split (parquets), and the CV design (`cv_results_long.csv`).
The dashed arrow marks the per-receptor quality flags, which enter training
only as sample weights, never as features. The bottom row is the study's
central design axis: the same matrix and the same learners evaluated under
three splits that differ only in what the model is asked to generalise to.
"""
SOURCES = COUNT_SOURCES


def build(art: Artefacts):
    c = pipeline_counts(art)
    sm = c["site_methods"]
    col = S.STAGE_COLOR

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    D.blank(ax, (0, 100), (0, 100))
    h, ts, ss = 11.5, 7.6, 6.2

    # Row 1: sources
    s1 = D.box(ax, 2, 86, 30, h, "BiasDB", f"{c['raw_rows']} raw bias measurements",
               col["source"], sub_size=ss)
    s2 = D.box(ax, 35, 86, 30, h, "RCSB PDB", f"{c['binding_sites']} GPCR structures",
               col["source"], sub_size=ss)
    s3 = D.box(ax, 68, 86, 30, h, "UniProt", "canonical receptor sequences",
               col["source"], sub_size=ss)

    # Row 2: curation & structure preparation
    y2, w2 = 67, 22
    c1 = D.box(ax, 2, y2, w2, h, "Ligand curation",
               f"{c['after_dedup']} measurements\n{c['molecules']} molecules",
               col["curation"], title_size=ts, sub_size=ss)
    c2 = D.box(ax, 25.5, y2, w2, h, "Binding sites",
               f"{sm.get('cocrystal_ligand', 0)} cocrystal, {sm.get('consensus', 0)} consensus,\n"
               f"{sm.get('registry_repair', 0)} registry repair, {sm.get('primary_only', 0)} predicted",
               col["curation"], title_size=ts, sub_size=ss)
    c3 = D.box(ax, 49, y2, 26, h, "Receptor preparation",
               "DBREF chain, identity gate,\nPDBFixer, Meeko",
               col["curation"], title_size=ts, sub_size=ss)
    c4 = D.box(ax, 77, y2, 21, h, "Re-dock QC",
               f"{c['redock_passed']}/{c['redock_attempted']} pass RMSD ≤ 2.5 Å\n+ Gnina CNN score",
               col["curation"], title_size=ts, sub_size=ss)
    D.arrow(ax, s1["bottom"], c1["top"])
    D.arrow(ax, s2["bottom"], c2["top"])
    D.arrow(ax, s2["bottom"], c3["top"])
    D.arrow(ax, s3["bottom"], c3["top"])
    D.arrow(ax, c3["right"], c4["left"])

    # Row 3: docking & featurisation
    y3 = 47
    d1 = D.box(ax, 2, y3, 45, h, "Production docking (AutoDock Vina)",
               f"exhaustiveness 16, 9 modes\n{c['pairs_ok']}/{c['pairs_docked']} unique pairs "
               f"docked ({c['dock_success_pct']:.1f} %)", col["docking"], sub_size=ss)
    f1 = D.box(ax, 51, y3, 39, h, "Featurisation",
               f"{c['modelled_rows']} × {c['columns']} matrix, {c['n_features']} features\n"
               "2D, Morgan, MACCS, pose 3D, Vina, ProLIF (BW)",
               col["features"], sub_size=ss)
    D.arrow(ax, c1["bottom"], (14, y3 + h))
    D.arrow(ax, c2["bottom"], (30, y3 + h))
    D.arrow(ax, c3["bottom"], (42, y3 + h))
    D.arrow(ax, d1["right"], f1["left"])

    # Row 4: machine learning
    y4, w4 = 26, 22.5
    m1 = D.box(ax, 2, y4, w4, h, "Temporal split",
               f"{c['train_rows']} train-eligible (< 2018)\n{c['holdout_rows']} holdout (≥ 2018)",
               col["ml"], title_size=ts, sub_size=ss)
    m2 = D.box(ax, 26.5, y4, w4, h, "Cross-validation",
               f"{c['n_learners']} learners × {c['n_seeds']} seeds × {c['n_folds']} folds\n"
               f"× 3 splits = {c['n_cv_fits']} fits",
               col["ml"], title_size=ts, sub_size=ss)
    m3 = D.box(ax, 51, y4, w4, h, "Stacking ensemble",
               "LightGBM + XGBoost +\nChemBERTa-LR → LR",
               col["ml"], title_size=ts, sub_size=ss)
    m4 = D.box(ax, 75.5, y4, 22.5, h, "Interpretation",
               "cross-regime SHAP,\nblock attribution",
               col["ml"], title_size=ts, sub_size=ss)
    D.elbow(ax, [(87, y3), (87, 40.5), (13.25, 40.5), (13.25, y4 + h)], dashed=False)
    for a_, b_ in ((m1, m2), (m2, m3), (m3, m4)):
        D.arrow(ax, a_["right"], b_["left"])
    # QC flags reach training only as sample weights.
    D.elbow(ax, [(94, y2), (94, 44.0), (37.75, 44.0), (37.75, y4 + h)])
    ax.text(62, 44.8, "per-receptor QC flags → sample weights only (never features)",
            fontsize=5.9, color=S.TEXT_SECONDARY, ha="center", va="bottom")

    # Row 5: the three evaluation regimes -- the study's design axis.
    y5, h5 = 3, 10.0
    r1 = D.box(ax, 2, y5, 30.5, h5, "Stratified",
               "random split\nligand and receptor both seen", col["deploy"],
               title_size=ts, sub_size=ss)
    r2 = D.box(ax, 34.75, y5, 30.5, h5, "Scaffold-grouped",
               "unseen Bemis–Murcko scaffold\nnovel chemistry", col["deploy"],
               title_size=ts, sub_size=ss)
    r3 = D.box(ax, 67.5, y5, 30.5, h5, "Receptor-grouped",
               "unseen receptor\nnovel protein target", col["deploy"],
               title_size=ts, sub_size=ss)
    for r in (r1, r2, r3):
        D.elbow(ax, [(37.75, y4), (37.75, y5 + h5 + 4.5),
                     (r["top"][0], y5 + h5 + 4.5), r["top"]], dashed=False)
    ax.text(70, y5 + h5 + 5.2, "the same matrix and the same learners, asked to "
            "generalise to three different things", fontsize=5.9,
            color=S.TEXT_SECONDARY, ha="center", va="bottom")

    facts = {k: c[k] for k in (
        "raw_rows", "after_dedup", "molecules", "binding_sites", "site_methods",
        "redock_passed", "redock_attempted", "pairs_docked", "pairs_ok",
        "modelled_rows", "columns", "n_features", "train_rows", "holdout_rows",
        "n_learners", "n_seeds", "n_folds", "n_cv_fits")}
    facts["dock_success_pct"] = round(c["dock_success_pct"], 1)
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
