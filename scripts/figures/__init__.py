"""Manuscript figures, regenerated from the run artefacts.

``PYTHONPATH=src uv run python -m scripts.figures`` builds every figure into
``manuscript/figures/`` (override with ``--out``) and writes ``FIGURES.md``
there; ``python -m scripts.figures.<name>`` builds one.

Every figure module exposes ``NAME`` (output basename), ``TITLE``,
``DESCRIPTION``, ``SOURCES`` (artefact paths relative to ``data/``) and
``build(art) -> (figure, facts)``, where ``facts`` are the key numbers the
figure shows, read from the artefacts at build time.
"""

from __future__ import annotations

import importlib

# Build order = filename order. Filenames are kept from the 2026-08-25 set so
# manuscript cross-references stay stable; fig06 changed content (the old
# "validated features" figure plotted leaked QC columns) and therefore name.
MODULE_NAMES = [
    "fig00_pipeline_schematic",
    "fig_methods_prisma",
    "fig01_dataset_overview",
    "fig02_bake_off_results",
    "fig03_holdout_progression",
    "fig04_per_class_breakthrough",
    "fig05_calibration",
    "fig06_shap_block_attribution",
    "fig07_learning_curve",
    "fig08_ablation",
    "fig09_loro",
    "fig10_selector_ablation",
    "fig11_temporal_shift",
    "fig12_confusion_matrix",
    "fig13_stacking_architecture",
    "fig14_unsupervised_clustering",
    "fig15_shap_cross_regime",
]

# Figures from the old set whose content was wrong and that no module rebuilds.
RETIRED = {
    "fig06_validated_features": (
        "plotted the per-receptor QC columns that leaked receptor identity; "
        "the corrected validated-feature set is empty — replaced by "
        "fig06_shap_block_attribution"
    ),
}


def load(name: str):
    return importlib.import_module(f"{__name__}.{name}")


def all_modules():
    return [load(n) for n in MODULE_NAMES]
