"""The provenance-leakage gate, and the assertion that backs it up.

HAL-4071. The failure this guards against is the one in
`docs/QC_LEAKAGE_CORRECTION.md`: a per-group covariate joined onto per-pair rows
encodes group identity, group identity carries the label, and all four
evaluation regimes miss it.

The tests that matter here are the two that could be got wrong in opposite
directions:

- a gate that never fires is worthless, so one test constructs provenance that
  *does* predict the label with receptors held out and asserts FAIL;
- a gate that fires on the known receptor->label association would block every
  run forever, so another gives provenance that only tracks receptor identity
  and asserts PASS.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cancerag.ml import leakage_probe as lp

CLASSES = ["G protein", "β Arrestin", "G protein selectivity", "ERK"]


def _frame(n_receptors: int = 10, per_receptor: int = 20, *,
           provenance_predicts: bool = False, seed: int = 0) -> pd.DataFrame:
    """Synthetic pairs. Each receptor gets one dominant class, as in the real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_receptors):
        acc = f"R{r:03d}"
        dominant = CLASSES[r % len(CLASSES)]
        for i in range(per_receptor):
            label = dominant if rng.random() < 0.8 else rng.choice(CLASSES)
            rows.append({"pair_id": f"L{r}_{i}__{acc}", "receptor_uniprot": acc,
                         "bias_category": label})
    df = pd.DataFrame(rows)

    if provenance_predicts:
        # engine tracks the LABEL directly, cutting across receptors -- the
        # situation the gate exists to catch.
        df["engine_gpu"] = (df["bias_category"] == "ERK").astype(int)
        df["exhaustiveness"] = np.where(df["engine_gpu"] == 1, 16, 8)
        df["receptor_repaired"] = (df["bias_category"] == "β Arrestin").astype(int)
    else:
        # provenance is a property of the receptor, which is the real shape:
        # it can proxy receptor identity, but carries nothing once receptors
        # are held out.
        gpu = {f"R{r:03d}" for r in range(2)}
        rep = {f"R{r:03d}" for r in range(3, 6)}
        df["engine_gpu"] = df["receptor_uniprot"].isin(gpu).astype(int)
        df["exhaustiveness"] = np.where(df["engine_gpu"] == 1, 16, 8)
        df["receptor_repaired"] = df["receptor_uniprot"].isin(rep).astype(int)
    return df


@pytest.mark.unit
class TestGate:
    def test_receptor_level_provenance_passes(self):
        """Provenance that only tracks receptor identity must not fail the gate.

        If it did, the gate would block every run on a property of the data we
        already know about and have chosen grouped CV to handle.
        """
        res = lp.run_probe(_frame(provenance_predicts=False))
        assert res["verdict"] == "PASS"
        assert res["margin_over_floor"] <= lp.MARGIN

    def test_provenance_that_tracks_the_label_fails(self):
        """A gate that cannot fail is not a gate."""
        res = lp.run_probe(_frame(provenance_predicts=True))
        assert res["verdict"] == "FAIL"
        assert res["provenance_receptor_grouped"] > res["floor_majority_class"]

    def test_receptor_identity_beats_the_floor(self):
        """Sanity on the comparison that makes test 4 interpretable.

        Receptor identity predicting the label is the known property; if this
        did not hold, the synthetic frame would not resemble the real data and
        the PASS above would mean nothing.
        """
        res = lp.run_probe(_frame(provenance_predicts=False))
        assert res["receptor_identity_stratified"] > res["floor_majority_class"]

    def test_main_raises_and_still_writes_the_result_on_fail(self, tmp_path, monkeypatch):
        df = _frame(provenance_predicts=True)
        monkeypatch.setattr(lp, "build_provenance_frame", lambda _: df)
        with pytest.raises(RuntimeError, match="provenance predicts the label"):
            lp.main(tmp_path)
        written = json.loads((tmp_path / lp.DEFAULT_OUT).read_text())
        assert written["verdict"] == "FAIL", (
            "the result must be on disk even when the gate fails, or the run "
            "report cannot say why the pipeline stopped"
        )


@pytest.mark.unit
class TestGranularityAssertion:
    def test_provenance_in_the_feature_matrix_is_refused(self):
        X = pd.DataFrame({"vina_affinity_best": [1.0], "engine_gpu": [0]})
        with pytest.raises(AssertionError, match="engine_gpu"):
            lp.assert_provenance_is_not_a_feature(X)

    def test_a_clean_feature_matrix_passes(self):
        X = pd.DataFrame({"vina_affinity_best": [1.0], "ifp_BW3.32_Cationic": [1]})
        lp.assert_provenance_is_not_a_feature(X)

    def test_passing_the_gate_does_not_make_the_columns_features(self):
        """The two checks answer different questions.

        Clearing the gate says provenance carries no label signal of its own.
        It does not say the columns belong in the model -- they describe how a
        pose was produced, not the pose.
        """
        df = _frame(provenance_predicts=False)
        assert lp.run_probe(df)["verdict"] == "PASS"
        with pytest.raises(AssertionError):
            lp.assert_provenance_is_not_a_feature(df)


@pytest.mark.unit
def test_provenance_columns_are_declared_once():
    """The gate and the assertion must read the same list, or they drift."""
    assert set(lp.PROVENANCE_COLS) == {"engine_gpu", "exhaustiveness",
                                       "receptor_repaired"}


@pytest.mark.unit
def test_repairs_tsv_comment_preamble_is_skipped(tmp_path):
    """`receptor_repairs.tsv` opens with `#` provenance lines before the header.

    Reading it without `comment="#"` raises a tokenising error, which is how
    this surfaced the first time.
    """
    (tmp_path / "processed").mkdir(parents=True)
    (tmp_path / "registry").mkdir()
    (tmp_path / "registry" / "receptor_repairs.tsv").write_text(
        "# a note about why these rows exist\n"
        "# spanning several lines, with\ttabs in them\n"
        "uniprot\treceptor\tpdb\nP41145\tkappa-opioid\t8DZP\n")
    pd.DataFrame({"pair_id": ["a__P41145"], "inchikey": ["a"],
                  "receptor_uniprot": ["P41145"]}).to_csv(
        tmp_path / "processed" / "docking_features.csv", index=False)
    pd.DataFrame({"inchikey": ["a"], "receptor_uniprot": ["P41145"],
                  "bias_category": ["ERK"]}).to_csv(
        tmp_path / "processed" / "unified_ligands.csv", index=False)

    out = lp.build_provenance_frame(tmp_path)
    assert out.loc[0, "receptor_repaired"] == 1
    assert out.loc[0, "engine_gpu"] == 0, "no provenance file means CPU"
    assert out.loc[0, "exhaustiveness"] == 8
