#!/usr/bin/env python3
"""Turn a seeded selector ablation into a testable ranking.

``run_selector_ablation`` writes per-fit scores. Reading a ranking off those
means is what HAL-6168 was about: the folds inside one seed are not
independent, so a spread taken over folds is not the spread a ranking has to
survive. This script does the arithmetic once, into an artefact, so figures and
prose quote it instead of each recomputing it.

Two things come out:

``selector_ablation_ranking.csv``
    One complete estimate per (scope, selector, seed) collapsed to a mean, an
    SD and a two-sided 95% t interval **over seeds**. ``scope`` is either a
    learner name or ``pooled`` (the mean across learners within that seed).

``selector_ablation_contrasts.csv``
    Every ordered pair of selectors as a *paired* per-seed difference. Within a
    seed every selector saw the same folds and the same model seed, so the
    difference is a difference on the same data, not a gap between two
    independently noisy means. ``delta_excludes_zero`` is the only column that
    licenses a sentence about one selector beating another.
"""

from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

import pandas as pd
from scipy import stats


def ci95(values: pd.Series) -> tuple[float, float]:
    """Two-sided 95% t interval for the mean. NaN when n < 2."""
    v = values.dropna()
    n = len(v)
    if n < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / (n**0.5)
    return float(v.mean() - half), float(v.mean() + half)


def ranking(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scope, sel), g in per_seed.groupby(["scope", "selector"]):
        lo, hi = ci95(g.macro_f1)
        rows.append({
            "scope": scope, "selector": sel, "n_seeds": int(g.macro_f1.notna().sum()),
            "mean": float(g.macro_f1.mean()), "sd": float(g.macro_f1.std(ddof=1)),
            "ci95_lo": lo, "ci95_hi": hi,
        })
    out = pd.DataFrame(rows).sort_values(["scope", "mean"], ascending=[True, False])
    out["rank"] = out.groupby("scope")["mean"].rank(ascending=False, method="min").astype(int)
    return out.reset_index(drop=True)


CONTRAST_COLUMNS = [
    "scope", "selector_a", "selector_b", "n_seeds", "delta", "delta_sd",
    "delta_ci95_lo", "delta_ci95_hi", "delta_excludes_zero",
]


def contrasts(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, g in per_seed.groupby("scope"):
        wide = g.pivot(index="seed", columns="selector", values="macro_f1")
        for a, b in permutations(sorted(wide.columns), 2):
            d = (wide[a] - wide[b]).dropna()
            lo, hi = ci95(d)
            rows.append({
                "scope": scope, "selector_a": a, "selector_b": b,
                "n_seeds": int(len(d)), "delta": float(d.mean()),
                "delta_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                "delta_ci95_lo": lo, "delta_ci95_hi": hi,
                # A pair whose paired interval straddles zero is a pair this
                # design cannot order, however far apart the two means look.
                "delta_excludes_zero": bool(lo > 0 or hi < 0) if len(d) > 1 else False,
            })
    # Columns are declared, not inferred: a single selector yields no ordered
    # pairs, and a frame with no columns breaks every downstream `.scope`.
    return pd.DataFrame(rows, columns=CONTRAST_COLUMNS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds-csv", type=Path, required=True,
                    help="selector_ablation_seeds.csv from run_selector_ablation")
    ap.add_argument("--long-csv", type=Path, default=None,
                    help="selector_ablation_long.csv, audited for fallbacks")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    seeds = pd.read_csv(args.seeds_csv)
    if seeds.seed.nunique() < 2:
        print(f"[error] {args.seeds_csv} has {seeds.seed.nunique()} seed(s); a "
              "ranking cannot be tested on one seed. Re-run with --seeds and "
              "--shuffle-folds.", file=sys.stderr)
        return 1

    # A cell that fell back is not a measurement of the selector it is filed
    # under, so it must not reach a ranking.
    if args.long_csv is not None:
        long = pd.read_csv(args.long_csv)
        bad = long[long.fallback_reason.notna()]
        if len(bad):
            print(f"[error] {len(bad)} of {len(long)} fits fell back "
                  f"({sorted(set(bad.selector))}); refusing to rank.", file=sys.stderr)
            return 1
        if long.macro_f1.isna().any():
            print(f"[error] {int(long.macro_f1.isna().sum())} fits are NaN; "
                  "refusing to rank.", file=sys.stderr)
            return 1
        if "collides_with" in long.columns:
            # An all-empty column round-trips through CSV as float64 NaN, so
            # neither fillna("") nor astype(bool) is safe on it.
            col = long["collides_with"]
            coll = long[col.notna() & col.astype(str).str.strip().ne("")]
            if len(coll):
                print(f"[error] {len(coll)} of {len(long)} fits are identical "
                      f"to another selector's fit on the same (model, seed, "
                      f"fold) ({sorted(set(coll.selector))}); at least one of "
                      "each pair is not a measurement of its own selector. "
                      "Refusing to rank.", file=sys.stderr)
                return 1

    per_model = seeds.rename(columns={"model": "scope"})[
        ["scope", "selector", "seed", "macro_f1"]]
    pooled = (seeds.groupby(["selector", "seed"], as_index=False).macro_f1.mean()
                   .assign(scope="pooled"))
    per_seed = pd.concat([per_model, pooled], ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = ranking(per_seed)
    cont = contrasts(per_seed)
    rank.to_csv(args.output_dir / "selector_ablation_ranking.csv", index=False)
    cont.to_csv(args.output_dir / "selector_ablation_contrasts.csv", index=False)

    pool = rank[rank.scope == "pooled"]
    print(f"pooled across {int(pool.n_seeds.iloc[0])} seeds "
          f"(mean over learners within each seed):")
    for r in pool.itertuples():
        print(f"  {r.rank}. {r.selector:<12} {r.mean:.4f} +/- {r.sd:.4f}  "
              f"95% CI [{r.ci95_lo:.4f}, {r.ci95_hi:.4f}]")
    order = list(pool.selector)
    print("\nadjacent pairs in that order (paired per-seed difference):")
    cp = cont[cont.scope == "pooled"].set_index(["selector_a", "selector_b"])
    for a, b in zip(order, order[1:]):
        r = cp.loc[(a, b)]
        print(f"  {a} - {b} = {r.delta:+.4f}  95% CI "
              f"[{r.delta_ci95_lo:+.4f}, {r.delta_ci95_hi:+.4f}]  "
              f"{'RESOLVED' if r.delta_excludes_zero else 'within noise'}")
    n_res = int(cont[cont.scope == "pooled"].delta_excludes_zero.sum()) // 2
    print(f"\n{n_res} of {len(order) * (len(order) - 1) // 2} selector pairs "
          f"are resolved from zero at the pooled scope.")
    print(f"\nWritten to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
