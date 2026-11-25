"""`make_grouped_cv`'s seeded shuffling, and the fact that it stays opt-in.

The block ablation used to report a seed-to-seed spread of exactly zero on both
grouped regimes, and then used that zero as the yardstick for whether a feature
block's contribution was signal. The cause was `GroupKFold`: it takes no
`shuffle` and no `random_state`, so the seed a caller passed was discarded and
every seed re-fitted on one fixed partition.

The fix adds `shuffle=True`. These tests pin both halves of it: that shuffling
genuinely re-partitions and is reproducible, and that the default path is
byte-for-byte what it was, because every audited scaffold- and receptor-grouped
number in the manuscript comes out of that default (HAL-6168).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from cancerag.ml.preprocessing import make_grouped_cv


def frame(n_groups: int = 12, per_group: int = 5) -> pd.DataFrame:
    """Rows carrying a group label, deliberately uneven in group size."""
    rows = []
    for g in range(n_groups):
        # sizes 1..per_group, cycling, so greedy balancing has work to do
        for i in range((g % per_group) + 1):
            rows.append({"grp": f"G{g:02d}", "val": g * 100 + i})
    return pd.DataFrame(rows)


def as_sets(splits):
    """Comparable, order-insensitive representation of a fold assignment."""
    return sorted(tuple(sorted(te.tolist())) for _, te in splits)


# --------------------------------------------------------------- default path

def test_default_is_unchanged_plain_groupkfold():
    """No `shuffle` must reproduce plain GroupKFold exactly.

    This is the guard on every audited number: the bake-off, nested parsimony,
    SHAP attribution and the binary ERK contrast all call this function with
    its defaults.
    """
    df = frame()
    groups = df["grp"].astype(str).values
    expected = [
        (np.asarray(tr), np.asarray(te))
        for tr, te in GroupKFold(n_splits=3).split(df, groups=groups)
    ]
    got = make_grouped_cv(df, group_col="grp", n_splits=3)

    assert len(got) == len(expected)
    for (gtr, gte), (etr, ete) in zip(got, expected):
        np.testing.assert_array_equal(gtr, etr)
        np.testing.assert_array_equal(gte, ete)


@pytest.mark.parametrize("seed", [42, 7, 13, 2029])
def test_default_ignores_seed(seed):
    """The old behaviour, made explicit rather than left as a surprise."""
    df = frame()
    base = as_sets(make_grouped_cv(df, group_col="grp", n_splits=3, seed=42))
    assert as_sets(make_grouped_cv(df, group_col="grp", n_splits=3, seed=seed)) == base


# -------------------------------------------------------------- shuffled path

def test_shuffle_changes_partition_across_seeds():
    """Different seeds must give different fold assignments.

    Without this the ablation's "3 seeds" are one measurement repeated.
    """
    df = frame(n_groups=24)
    seen = {
        seed: as_sets(make_grouped_cv(df, group_col="grp", n_splits=3,
                                      seed=seed, shuffle=True))
        for seed in (42, 7, 13, 101, 2029)
    }
    distinct = {repr(v) for v in seen.values()}
    assert len(distinct) == len(seen), (
        f"{len(seen)} seeds collapsed to {len(distinct)} distinct partitions"
    )


@pytest.mark.parametrize("seed", [42, 7, 2029])
def test_shuffle_is_reproducible_for_one_seed(seed):
    df = frame(n_groups=24)
    a = make_grouped_cv(df, group_col="grp", n_splits=3, seed=seed, shuffle=True)
    b = make_grouped_cv(df, group_col="grp", n_splits=3, seed=seed, shuffle=True)
    assert len(a) == len(b)
    for (atr, ate), (btr, bte) in zip(a, b):
        np.testing.assert_array_equal(atr, btr)
        np.testing.assert_array_equal(ate, bte)


@pytest.mark.parametrize("seed", [42, 7, 13, 101, 2029])
def test_shuffle_never_straddles_a_group(seed):
    """The whole point of grouped CV: no group may appear on both sides."""
    df = frame(n_groups=24)
    groups = df["grp"].astype(str).values
    for tr, te in make_grouped_cv(df, group_col="grp", n_splits=3,
                                  seed=seed, shuffle=True):
        assert not (set(groups[tr]) & set(groups[te]))


@pytest.mark.parametrize("seed", [42, 7, 2029])
def test_shuffle_folds_partition_every_row_exactly_once(seed):
    df = frame(n_groups=24)
    splits = make_grouped_cv(df, group_col="grp", n_splits=3,
                             seed=seed, shuffle=True)
    test_idx = np.concatenate([te for _, te in splits])
    np.testing.assert_array_equal(np.sort(test_idx), np.arange(len(df)))
    for tr, te in splits:
        np.testing.assert_array_equal(np.sort(np.concatenate([tr, te])),
                                      np.arange(len(df)))


@pytest.mark.parametrize("seed", [42, 7, 13, 101, 2029])
def test_shuffle_keeps_folds_roughly_balanced(seed):
    """Greedy assignment to the lightest fold, so no fold may run away.

    A naive round-robin over permuted groups can leave one fold several times
    the size of another when group sizes are uneven, which would make the
    per-fold scores incomparable.
    """
    df = frame(n_groups=24)
    sizes = [len(te) for _, te in
             make_grouped_cv(df, group_col="grp", n_splits=3,
                             seed=seed, shuffle=True)]
    assert min(sizes) > 0
    assert max(sizes) <= 2 * min(sizes), sizes


@pytest.mark.parametrize("seed", [42, 2029])
def test_shuffle_is_invariant_to_row_order(seed):
    """Same rows in a different order must give the same partition of groups.

    np.unique sorts the group labels before permuting them, so the fold a
    group lands in cannot depend on where its rows happen to sit.
    """
    df = frame(n_groups=24)
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    def group_to_fold(frame_):
        groups = frame_["grp"].astype(str).values
        out = {}
        for k, (_, te) in enumerate(make_grouped_cv(
                frame_, group_col="grp", n_splits=3, seed=seed, shuffle=True)):
            for g in set(groups[te]):
                out[g] = k
        return out

    assert group_to_fold(df) == group_to_fold(shuffled)


def test_too_few_groups_returns_empty_under_both_paths():
    df = pd.DataFrame({"grp": ["only"] * 6, "val": range(6)})
    assert make_grouped_cv(df, group_col="grp", n_splits=3) == []
    assert make_grouped_cv(df, group_col="grp", n_splits=3, shuffle=True) == []


def test_folds_capped_at_group_count_under_shuffle():
    df = frame(n_groups=2)
    splits = make_grouped_cv(df, group_col="grp", n_splits=5,
                             seed=42, shuffle=True)
    assert len(splits) == 2
