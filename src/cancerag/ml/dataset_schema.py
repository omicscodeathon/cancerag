"""Pandera schema for the Stage 07 ML-ready dataset.

Validates required columns, dtypes, allowed-value constraints on labels,
sample-weight bounds, and pair_key uniqueness. Drift in any upstream stage
will surface as a SchemaError instead of a silent downstream failure.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


# Allowed bias_category values from BiasDB
_BIAS_CATEGORIES = ["G protein", "β Arrestin", "G protein selectivity", "ERK"]

# Required columns + dtype + constraints. Optional feature blocks (Morgan,
# MACCS, IFP bits) are not enumerated because there are 2000+ of them — we
# instead validate the count via a separate `validate_feature_block_counts`.
DATASET_SCHEMA = DataFrameSchema(
    {
        # ------------- meta / identity
        "pair_key": Column(str, unique=True, nullable=False),
        "inchikey": Column(str, nullable=False),
        "receptor_uniprot": Column(str, nullable=False),
        "scaffold": Column(str, nullable=True),
        # ------------- target
        "bias_category": Column(
            str,
            checks=Check.isin(_BIAS_CATEGORIES),
            nullable=False,
        ),
        # ------------- vina pose-ensemble (always present after success filter)
        "vina_affinity_best": Column(float, nullable=True),
        "vina_affinity_mean_top3": Column(float, nullable=True),
        "vina_affinity_gap_1_2": Column(float, nullable=True),
        "vina_pose_diversity_rmsd": Column(float, nullable=True),
        "vina_n_distinct_clusters": Column(
            "Int64", nullable=True,  # nullable int from merge
        ),
        # ------------- confidence joinage
        "docking_confidence": Column(str, nullable=True),
        "redock_rmsd_angstrom": Column(float, nullable=True),
        "gnina_cnn_score": Column(float, nullable=True),
        # ------------- pose-3D missingness
        "pose_3d_missing": Column(int, checks=Check.isin([0, 1])),
        "ifp_missing": Column(int, checks=Check.isin([0, 1])),
        "ifp_no_contacts": Column(int, checks=Check.isin([0, 1])),
        # ------------- weights
        "sample_weight": Column(
            float,
            checks=[Check.greater_than(0), Check.less_than_or_equal_to(1.0)],
            nullable=False,
        ),
        # ------------- bookkeeping
        "year": Column(float, nullable=True),  # int with NaN -> float
        "source": Column(str, nullable=False),
    },
    strict=False,  # extra cols (morgan_*, maccs_*, ifp_*) are allowed
    coerce=True,
)


def validate_feature_block_counts(
    df, *, expected_min_morgan: int = 2048, expected_min_maccs: int = 167,
    expected_min_ifp: int = 50,
) -> dict:
    """Sanity-check the number of fingerprint / IFP columns. Returns counts."""
    morgan = sum(1 for c in df.columns if c.startswith("morgan_"))
    maccs = sum(1 for c in df.columns if c.startswith("maccs_"))
    ifp = sum(1 for c in df.columns if c.startswith("ifp_"))
    if morgan < expected_min_morgan:
        raise ValueError(
            f"Morgan bit count too low: {morgan} < {expected_min_morgan}"
        )
    if maccs < expected_min_maccs:
        raise ValueError(
            f"MACCS bit count too low: {maccs} < {expected_min_maccs}"
        )
    if ifp < expected_min_ifp:
        raise ValueError(
            f"IFP bit count suspiciously low: {ifp} < {expected_min_ifp}"
        )
    return {"morgan": morgan, "maccs": maccs, "ifp": ifp}


def validate(df) -> dict:
    """Validate the assembled dataset against DATASET_SCHEMA.

    Returns a dict with the validated dataframe shape + feature counts.
    Raises pandera.errors.SchemaError if validation fails.
    """
    DATASET_SCHEMA.validate(df, lazy=True)
    counts = validate_feature_block_counts(df)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "feature_counts": counts,
    }
