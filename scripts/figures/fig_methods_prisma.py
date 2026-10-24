"""PRISMA-style curation flow from raw BiasDB rows to the modelled dataset."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from . import _diagram as D
from . import _style as S
from ._counts import SOURCES as COUNT_SOURCES
from ._counts import pipeline_counts
from ._data import Artefacts

NAME = "fig_methods_prisma"
TITLE = "Curation flow: raw BiasDB rows to the modelled dataset"
DESCRIPTION = """
Row counts at every curation step, with the rows each step excludes. The
counts are recomputed from the artefacts by replaying the row filters of
`dataset_assembly.assemble_dataset` (scope filter on ligands with computed
features, successful-docking filter, composite-key de-duplication, label
filter), and the build fails if the replay does not end at the row count of
the assembled parquets. Docking runs on unique (ligand, receptor) pairs, so its
counts are pairs, given inside the docking step; the peptide ligands that dominate
docking failures are mostly removed earlier, by the small-molecule scope
filter.
"""
SOURCES = COUNT_SOURCES


def build(art: Artefacts):
    c = pipeline_counts(art)
    after_dock = c["small_molecule_rows"] - c["rows_dock_failed"]

    fig, ax = plt.subplots(figsize=(6.6, 7.4))
    D.blank(ax, (0, 100), (0, 110))

    for lab, y, h, col in (
        ("Identification", 96, 12, S.STAGE_COLOR["source"]),
        ("Curation", 54, 40, S.STAGE_COLOR["curation"]),
        ("Docking", 40, 12, S.STAGE_COLOR["docking"]),
        ("Modelling", 1, 37, S.STAGE_COLOR["ml"]),
    ):
        ax.add_patch(plt.Rectangle((0, y), 1.0, h, color=col, zorder=1))
        ax.text(3.2, y + h / 2, lab, rotation=90, ha="center", va="center",
                fontsize=7, color=S.TEXT_SECONDARY)

    X, W, H = 8, 52, 9.0
    side_x, side_w = 66, 32
    col_c, col_d, col_m = (S.STAGE_COLOR[k] for k in ("curation", "docking", "ml"))

    b1 = D.box(ax, X, 98, W, H, "BiasDB bias-factor records",
               f"{c['raw_rows']} raw rows", S.STAGE_COLOR["source"])
    b2 = D.box(ax, X, 84, W, H, "RDKit standardisation",
               f"{c['after_standardise']} rows", col_c)
    b3 = D.box(ax, X, 70, W, H, "Six-field composite-key de-duplication",
               f"{c['after_dedup']} measurements · {c['molecules']} molecules · "
               f"{c['receptors_curated']} receptors", col_c)
    b4 = D.box(ax, X, 56, W, H, "Small-molecule scope",
               f"{c['small_molecule_rows']} rows", col_c)
    b5 = D.box(ax, X, 41, W, 11, "Pair docked successfully",
               f"{after_dock} rows\nVina on {c['pairs_docked']} unique pairs: {c['pairs_ok']} "
               f"succeeded ({c['dock_success_pct']:.1f} %), {c['pairs_failed']} failed",
               col_d)
    b6 = D.box(ax, X, 28, W, H, "Assembled feature matrix",
               f"{c['modelled_rows']} rows × {c['columns']} columns "
               f"({c['n_features']} features)", col_m)
    tr = D.box(ax, X, 4, 24, 13, "Train-eligible",
               f"year < 2018\nn = {c['train_rows']}", col_m)
    ho = D.box(ax, X + 28, 4, 24, 13, "Temporal holdout",
               f"year ≥ 2018\nn = {c['holdout_rows']}", col_m)
    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)):
        D.arrow(ax, a["bottom"], b["top"])
    D.arrow(ax, (X + 12, 28), tr["top"])
    D.arrow(ax, (X + 40, 28), ho["top"])

    def excl(y, text, src):
        n = D.note(ax, side_x, y, side_w, 8.0, text)
        D.arrow(ax, (src["right"][0], src["bottom"][1] - 2.5), n["left"], dashed=True)

    excl(84.5, f"{c['raw_rows'] - c['after_standardise']} rows failed parsing", b1)
    excl(70.5, f"{c['dedup_dropped']} duplicate rows removed\n(0 within-key conflicts)", b2)
    excl(56.5, f"{c['out_of_scope_rows']} rows outside small-molecule scope\n"
               f"({c['out_of_scope_peptide_backbone']} with a peptide backbone)", b3)
    excl(42.5, f"{c['rows_dock_failed']} row whose pair failed docking", b4)
    excl(28.5, f"{c['rows_duplicate_key']} row with a duplicate\nassembly key", b5)

    facts = {k: c[k] for k in (
        "raw_rows", "after_standardise", "after_dedup", "dedup_dropped",
        "molecules", "receptors_curated", "small_molecule_rows",
        "out_of_scope_rows", "out_of_scope_peptide_backbone", "rows_dock_failed", "rows_duplicate_key",
        "modelled_rows", "columns", "n_features", "train_rows", "holdout_rows",
        "pairs_docked", "pairs_ok", "pairs_failed")}
    facts["dock_success_pct"] = round(c["dock_success_pct"], 1)
    facts["rows after docking filter"] = after_dock
    return fig, facts


if __name__ == "__main__":
    from ._cli import run_one
    sys.exit(run_one(sys.modules[__name__]))
