"""A selector must never score under its own name without having selected.

Every selector here absorbs a failure by keeping all the columns it was
handed. That is a defensible recovery for one bad fold, and an indefensible
one for a broken call: the pipeline still fits, still scores, and the ablation
files the number under the selector's name. On 2026-11-25 that is exactly what
happened -- `multi_class="multinomial"` was removed in scikit-learn 1.7, so
`L1LogRegSelector` raised `TypeError` on construction, kept all ~1,700 columns
that survived preprocessing, and came top of the selector ranking. Only
`BorutaSelector` reported a reason, so the run's own fallback count read zero.

Two rules follow, and these tests pin both:

1. A signature fault (`TypeError`) propagates. It is a fault in this repo, not
   a property of the data, and no fold is going to fix it.
2. A genuine failure is recorded on the estimator as ``fallback_reason_``, so a
   caller can refuse to publish the number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cancerag.ml.feature_selection import (
    L1LogRegSelector,
    MISelector,
    RFECVSelector,
    flag_no_selection_collisions,
)

SRC = Path(__file__).resolve().parents[2] / "src"


def toy(n: int = 60, p: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)),
                     columns=[f"f{i}" for i in range(p)])
    # Two informative columns, so a working selector drops something.
    y = ((X.f0 + X.f1 + 0.1 * rng.normal(size=n)) > 0).astype(int).to_numpy()
    return X, y


# ---- rule 1: a signature fault is not absorbed ----------------------------

def test_mi_does_not_absorb_a_signature_fault(monkeypatch):
    import sklearn.feature_selection as fs

    def boom(*a, **k):
        raise TypeError("mutual_info_classif() got an unexpected keyword")

    monkeypatch.setattr(fs, "mutual_info_classif", boom)
    X, y = toy()
    with pytest.raises(TypeError):
        MISelector(force_keep_prefixes=()).fit(X, y)


def test_l1_does_not_absorb_a_signature_fault(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def boom(self, *a, **k):
        raise TypeError("LogisticRegression.fit() got an unexpected keyword")

    monkeypatch.setattr(LogisticRegression, "fit", boom)
    X, y = toy()
    with pytest.raises(TypeError):
        L1LogRegSelector(force_keep_prefixes=()).fit(X, y)


def test_rfecv_does_not_absorb_a_signature_fault(monkeypatch):
    from sklearn.feature_selection import RFECV

    def boom(self, *a, **k):
        raise TypeError("RFECV.fit() got an unexpected keyword")

    monkeypatch.setattr(RFECV, "fit", boom)
    X, y = toy()
    with pytest.raises(TypeError):
        RFECVSelector(force_keep_prefixes=()).fit(X, y)


# ---- rule 2: a genuine failure is recorded --------------------------------

def test_mi_records_a_genuine_failure_and_keeps_every_column(monkeypatch):
    import sklearn.feature_selection as fs

    def boom(*a, **k):
        raise ValueError("could not estimate mutual information")

    monkeypatch.setattr(fs, "mutual_info_classif", boom)
    X, y = toy()
    sel = MISelector(force_keep_prefixes=()).fit(X, y)
    assert sel.fallback_reason_ is not None
    assert "ValueError" in sel.fallback_reason_
    assert sel.selected_ == sorted(X.columns)


def test_l1_records_a_genuine_failure_and_keeps_every_column(monkeypatch):
    from sklearn.linear_model import LogisticRegression

    def boom(self, *a, **k):
        raise ValueError("saga failed to converge")

    monkeypatch.setattr(LogisticRegression, "fit", boom)
    X, y = toy()
    sel = L1LogRegSelector(force_keep_prefixes=()).fit(X, y)
    assert sel.fallback_reason_ is not None
    assert sel.selected_ == sorted(X.columns)


def test_rfecv_records_a_genuine_failure_and_keeps_every_column(monkeypatch):
    from sklearn.feature_selection import RFECV

    def boom(self, *a, **k):
        raise ValueError("ran out of memory ranking features")

    monkeypatch.setattr(RFECV, "fit", boom)
    X, y = toy()
    sel = RFECVSelector(force_keep_prefixes=()).fit(X, y)
    assert sel.fallback_reason_ is not None
    assert sel.selected_ == sorted(X.columns)


# ---- the live call actually works on the installed scikit-learn -----------

@pytest.mark.parametrize("factory", [
    pytest.param(lambda: L1LogRegSelector(force_keep_prefixes=(), C=1.0),
                 id="l1_logreg"),
    pytest.param(lambda: MISelector(force_keep_prefixes=(), k=4), id="mutual_info"),
])
def test_selector_selects_on_the_installed_sklearn(factory):
    """The regression test for the real defect.

    This fails the moment a selector's own call stops working against the
    pinned scikit-learn, instead of quietly returning every column.
    """
    X, y = toy()
    sel = factory().fit(X, y)
    assert sel.fallback_reason_ is None, sel.fallback_reason_
    # Keeping every column is what the silent fallback looked like. Keeping
    # none is a different (and visible) problem, so both bounds are asserted.
    assert 0 < len(sel.selected_) < len(X.columns), (
        f"kept {len(sel.selected_)} of {len(X.columns)} columns"
    )


def test_no_module_passes_multi_class_to_logistic_regression():
    """`multi_class` was removed in scikit-learn 1.7 and this repo pins >=1.7.2.

    Removing it from one call site and leaving three behind is how the defect
    survived: it was dropped from `make_elastic_lr` on 2026-11-25 while
    `L1LogRegSelector` and both ChemBERTa heads kept passing it.
    """
    offenders = []
    for f in sorted(SRC.rglob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            if "multi_class=" in code:
                offenders.append(f"{f.relative_to(SRC)}:{i}")
    assert not offenders, f"multi_class= still passed in: {offenders}"


# ---- the detector that does not rely on a selector noticing ---------------

def long_frame(rows):
    return pd.DataFrame(rows, columns=[
        "selector", "model", "seed", "fold", "macro_f1", "n_features_selected"])


def test_two_selectors_with_an_identical_fit_are_flagged_as_each_other():
    out = flag_no_selection_collisions(long_frame([
        ("l1_logreg", "lgbm", 42, 2, 0.4306958549409392, 1667),
        ("rfecv", "lgbm", 42, 2, 0.4306958549409392, 1667),
        ("mutual_info", "lgbm", 42, 2, 0.4512000000000000, 200),
    ]))
    by_sel = out.set_index("selector").collides_with
    assert by_sel["l1_logreg"] == "rfecv"
    assert by_sel["rfecv"] == "l1_logreg"
    assert by_sel["mutual_info"] == ""


def test_a_three_way_collision_names_the_other_two():
    out = flag_no_selection_collisions(long_frame([
        (s, "lgbm", 42, 0, 0.5, 900) for s in ("l1_logreg", "rfecv", "mutual_info")
    ]))
    by_sel = out.set_index("selector").collides_with
    assert by_sel["l1_logreg"] == "mutual_info, rfecv"
    assert by_sel["rfecv"] == "l1_logreg, mutual_info"


def test_the_same_score_on_a_different_feature_count_is_not_a_collision():
    out = flag_no_selection_collisions(long_frame([
        ("l1_logreg", "lgbm", 42, 0, 0.5, 1667),
        ("rfecv", "lgbm", 42, 0, 0.5, 342),
    ]))
    assert (out.collides_with == "").all()


def test_the_same_fit_in_a_different_fold_is_not_a_collision():
    out = flag_no_selection_collisions(long_frame([
        ("l1_logreg", "lgbm", 42, 0, 0.5, 1667),
        ("rfecv", "lgbm", 42, 1, 0.5, 1667),
    ]))
    assert (out.collides_with == "").all()


def test_the_same_fit_under_a_different_seed_is_not_a_collision():
    out = flag_no_selection_collisions(long_frame([
        ("l1_logreg", "lgbm", 42, 0, 0.5, 1667),
        ("rfecv", "lgbm", 7, 0, 0.5, 1667),
    ]))
    assert (out.collides_with == "").all()


def test_one_selector_repeated_is_not_a_collision_with_itself():
    out = flag_no_selection_collisions(long_frame([
        ("rfecv", "lgbm", 42, 0, 0.5, 1667),
        ("rfecv", "xgboost", 42, 0, 0.5, 1667),
    ]))
    assert (out.collides_with == "").all()


def test_failed_fits_are_not_collisions_with_each_other():
    out = flag_no_selection_collisions(long_frame([
        ("l1_logreg", "lgbm", 42, 0, float("nan"), 0),
        ("rfecv", "lgbm", 42, 0, float("nan"), 0),
    ]))
    assert (out.collides_with == "").all()


def test_the_detector_preserves_the_frame_it_was_given():
    df = long_frame([("rfecv", "lgbm", 42, 0, 0.5, 1667)])
    before = df.copy()
    out = flag_no_selection_collisions(df)
    pd.testing.assert_frame_equal(df, before)
    assert list(out.columns)[:-1] == list(before.columns)
    assert out.collides_with.tolist() == [""]
