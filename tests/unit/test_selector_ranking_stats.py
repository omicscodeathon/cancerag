"""The ranking arithmetic, pinned.

The defect this guards against is the one HAL-6168 fixed upstream: a spread
quoted over the wrong replication unit, and an ordering reported from means
whose difference the design cannot resolve. These tests fix both halves --
what the script computes, and what it refuses to compute.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from scripts.selector_ranking_stats import ci95, contrasts, main, ranking


def per_seed(rows):
    return pd.DataFrame(rows, columns=["scope", "selector", "seed", "macro_f1"])


# ---- ci95 -----------------------------------------------------------------

def test_ci95_is_nan_for_one_value():
    lo, hi = ci95(pd.Series([0.5]))
    assert math.isnan(lo) and math.isnan(hi)


def test_ci95_of_a_constant_series_has_zero_width():
    lo, hi = ci95(pd.Series([0.4, 0.4, 0.4, 0.4]))
    assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4)


def test_ci95_brackets_the_mean_and_widens_with_spread():
    tight = ci95(pd.Series([0.40, 0.41, 0.39, 0.40]))
    loose = ci95(pd.Series([0.10, 0.70, 0.20, 0.60]))
    assert tight[0] < 0.40 < tight[1]
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])


def test_ci95_ignores_nan():
    lo, hi = ci95(pd.Series([0.4, float("nan"), 0.4, 0.4]))
    assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4)


# ---- ranking --------------------------------------------------------------

def test_ranking_orders_by_mean_within_each_scope():
    df = per_seed([
        ("pooled", "a", 1, 0.50), ("pooled", "a", 2, 0.52),
        ("pooled", "b", 1, 0.40), ("pooled", "b", 2, 0.42),
        ("lgbm", "a", 1, 0.10), ("lgbm", "a", 2, 0.12),
        ("lgbm", "b", 1, 0.90), ("lgbm", "b", 2, 0.92),
    ])
    out = ranking(df).set_index(["scope", "selector"])
    assert out.loc[("pooled", "a"), "rank"] == 1
    assert out.loc[("pooled", "b"), "rank"] == 2
    # Scopes are ranked independently: the pooled winner may lose per learner.
    assert out.loc[("lgbm", "b"), "rank"] == 1


def test_ranking_counts_seeds_not_rows():
    df = per_seed([("pooled", "a", s, 0.5) for s in (1, 2, 3)])
    assert int(ranking(df).n_seeds.iloc[0]) == 3


def test_ranking_sd_is_over_seeds():
    df = per_seed([("pooled", "a", 1, 0.30), ("pooled", "a", 2, 0.50)])
    row = ranking(df).iloc[0]
    assert row["mean"] == pytest.approx(0.40)
    assert row["sd"] == pytest.approx(pd.Series([0.30, 0.50]).std(ddof=1))


# ---- contrasts ------------------------------------------------------------

def test_a_constant_lead_is_resolved_from_zero():
    df = per_seed([("pooled", "a", s, 0.50 + 0.001 * s) for s in range(1, 6)]
                  + [("pooled", "b", s, 0.45 + 0.001 * s) for s in range(1, 6)])
    c = contrasts(df).set_index(["scope", "selector_a", "selector_b"])
    row = c.loc[("pooled", "a", "b")]
    assert row.delta == pytest.approx(0.05)
    assert row.delta_excludes_zero


def test_a_lead_that_swings_sign_across_seeds_is_not_resolved():
    # Same mean difference as above on average, but the per-seed difference
    # changes sign: two means far apart that the design cannot order.
    df = per_seed([("pooled", "a", 1, 0.90), ("pooled", "a", 2, 0.10),
                   ("pooled", "a", 3, 0.90), ("pooled", "a", 4, 0.10),
                   ("pooled", "b", 1, 0.10), ("pooled", "b", 2, 0.90),
                   ("pooled", "b", 3, 0.10), ("pooled", "b", 4, 0.90)])
    c = contrasts(df).set_index(["scope", "selector_a", "selector_b"])
    assert not c.loc[("pooled", "a", "b")].delta_excludes_zero


def test_the_contrast_is_paired_not_a_difference_of_means():
    # a beats b by exactly 0.02 in every seed while both wander over 0.3.
    # Unpaired, that spread swamps the lead; paired, the lead is exact.
    base = {1: 0.20, 2: 0.50, 3: 0.35, 4: 0.48, 5: 0.22}
    df = per_seed([("pooled", "a", s, v + 0.02) for s, v in base.items()]
                  + [("pooled", "b", s, v) for s, v in base.items()])
    row = contrasts(df).set_index(["scope", "selector_a", "selector_b"]).loc[
        ("pooled", "a", "b")]
    assert row.delta == pytest.approx(0.02)
    assert row.delta_sd == pytest.approx(0.0, abs=1e-12)
    assert row.delta_excludes_zero


def test_contrasts_are_antisymmetric():
    df = per_seed([("pooled", "a", s, 0.5 + 0.01 * s) for s in range(1, 4)]
                  + [("pooled", "b", s, 0.4 + 0.02 * s) for s in range(1, 4)])
    c = contrasts(df).set_index(["scope", "selector_a", "selector_b"])
    ab = c.loc[("pooled", "a", "b")]
    ba = c.loc[("pooled", "b", "a")]
    assert ab.delta == pytest.approx(-ba.delta)
    assert ab.delta_ci95_lo == pytest.approx(-ba.delta_ci95_hi)
    assert bool(ab.delta_excludes_zero) is bool(ba.delta_excludes_zero)


def test_contrasts_cover_every_ordered_pair_per_scope():
    df = per_seed([(sc, s, seed, 0.5)
                   for sc in ("pooled", "lgbm")
                   for s in ("a", "b", "c")
                   for seed in (1, 2)])
    c = contrasts(df)
    assert len(c) == 2 * 3 * 2  # 2 scopes x 3x2 ordered pairs


def test_one_seed_gives_no_interval_and_no_verdict():
    df = per_seed([("pooled", "a", 1, 0.5), ("pooled", "b", 1, 0.4)])
    row = contrasts(df).iloc[0]
    assert math.isnan(row.delta_ci95_lo)
    assert bool(row.delta_excludes_zero) is False


# ---- the refusals ---------------------------------------------------------

def write_pair(tmp_path, seeds_rows, long_rows=None):
    s = tmp_path / "selector_ablation_seeds.csv"
    pd.DataFrame(seeds_rows, columns=["selector", "model", "seed", "macro_f1"]).to_csv(
        s, index=False)
    if long_rows is None:
        return s, None
    lg = tmp_path / "selector_ablation_long.csv"
    pd.DataFrame(long_rows, columns=[
        "selector", "model", "seed", "fold", "macro_f1", "n_features_selected",
        "fallback_reason"]).to_csv(lg, index=False)
    return s, lg


def test_main_refuses_a_single_seed(tmp_path, capsys):
    s, _ = write_pair(tmp_path, [("a", "lgbm", 42, 0.5), ("b", "lgbm", 42, 0.4)])
    assert main(["--seeds-csv", str(s), "--output-dir", str(tmp_path / "o")]) == 1
    assert "one seed" in capsys.readouterr().err


def test_main_refuses_a_fallback_cell(tmp_path, capsys):
    s, lg = write_pair(
        tmp_path,
        [("a", "lgbm", seed, 0.5) for seed in (1, 2)],
        [("a", "lgbm", 1, 0, 0.5, 10, None),
         ("a", "lgbm", 2, 0, 0.5, 10, "ImportError: boruta")],
    )
    code = main(["--seeds-csv", str(s), "--long-csv", str(lg),
                 "--output-dir", str(tmp_path / "o")])
    assert code == 1
    assert "fell back" in capsys.readouterr().err


def test_main_refuses_a_nan_fit(tmp_path, capsys):
    s, lg = write_pair(
        tmp_path,
        [("a", "lgbm", seed, 0.5) for seed in (1, 2)],
        [("a", "lgbm", 1, 0, 0.5, 10, None),
         ("a", "lgbm", 2, 0, float("nan"), 10, None)],
    )
    assert main(["--seeds-csv", str(s), "--long-csv", str(lg),
                 "--output-dir", str(tmp_path / "o")]) == 1
    assert "NaN" in capsys.readouterr().err


def test_main_refuses_when_two_selectors_produced_the_identical_fit(tmp_path, capsys):
    s, lg = write_pair(
        tmp_path,
        [(sel, "lgbm", seed, 0.5) for sel in ("a", "b") for seed in (1, 2)],
        [("a", "lgbm", 1, 0, 0.5, 1667, None),
         ("b", "lgbm", 1, 0, 0.5, 1667, None)],
    )
    # The collision column is what the ablation writes; nothing in
    # fallback_reason records it, which is the case this guards.
    df = pd.read_csv(lg)
    df["collides_with"] = ["b", "a"]
    df.to_csv(lg, index=False)
    assert main(["--seeds-csv", str(s), "--long-csv", str(lg),
                 "--output-dir", str(tmp_path / "o")]) == 1
    assert "identical" in capsys.readouterr().err


def test_main_writes_both_artefacts_and_pools_over_learners(tmp_path):
    rows = [("a", m, seed, v) for m, v in (("lgbm", 0.60), ("rf", 0.40))
            for seed in (1, 2, 3)]
    rows += [("b", m, seed, v) for m, v in (("lgbm", 0.30), ("rf", 0.10))
             for seed in (1, 2, 3)]
    s, _ = write_pair(tmp_path, rows)
    out = tmp_path / "o"
    assert main(["--seeds-csv", str(s), "--output-dir", str(out)]) == 0
    rank = pd.read_csv(out / "selector_ablation_ranking.csv")
    cont = pd.read_csv(out / "selector_ablation_contrasts.csv")
    pooled = rank[rank.scope == "pooled"].set_index("selector")
    assert pooled.loc["a", "mean"] == pytest.approx(0.50)  # (0.60 + 0.40) / 2
    assert pooled.loc["b", "mean"] == pytest.approx(0.20)
    assert set(rank.scope) == {"pooled", "lgbm", "rf"}
    assert set(cont.scope) == {"pooled", "lgbm", "rf"}
