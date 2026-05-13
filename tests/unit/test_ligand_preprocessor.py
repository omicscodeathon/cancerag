"""
Tests for cancerag.preprocessing.ligand_preprocessor.

Covers standardize_smiles (parse/empty/error paths), standardize_dataframe
(drops failed rows + logs attrition), dedupe_with_conflict_log (separates
unanimous from conflicting groups), annotate_drug_likeness (annotation, not
filtering), mark_label_status (no source-shortcut "Agonist" label), and the
typo fix in the legacy LigandPreprocessor.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cancerag.preprocessing.ligand_preprocessor import (
    AttritionLogger,
    annotate_drug_likeness,
    dedupe_with_conflict_log,
    mark_label_status,
    merge_sources,
    standardize_dataframe,
    standardize_smiles,
)


@pytest.mark.unit
class TestStandardize:
    def test_empty_input(self):
        assert standardize_smiles("")["status"] == "empty"
        assert standardize_smiles(None)["status"] == "empty"  # type: ignore[arg-type]

    def test_garbage_returns_parse_error(self):
        assert standardize_smiles("not_smiles$$$")["status"] == "parse_error"

    def test_neutralizes_charged_amine(self):
        # Protonated amine -> neutral amine in canonical form
        result = standardize_smiles("CC[NH3+]")
        assert result["status"] == "ok"
        assert "+" not in result["canonical_smiles"]

    def test_keeps_largest_fragment(self):
        # Salt: ethylamine HCl. The Cl- counterion should be dropped.
        result = standardize_smiles("CCN.[Cl-]")
        assert result["status"] == "ok"
        assert "Cl" not in result["canonical_smiles"]

    def test_inchikey14_present_and_short(self):
        result = standardize_smiles("CCO")
        assert len(result["inchikey14"]) == 14
        assert result["inchikey14"].isalpha()

    def test_tautomer_collision_via_inchikey14(self):
        # Acetaldehyde and vinyl alcohol — same connectivity, different
        # tautomer. Their InChIKey-14 prefix should match after tautomer
        # canonicalization.
        a = standardize_smiles("CC=O")
        b = standardize_smiles("C=CO")
        assert a["status"] == b["status"] == "ok"
        assert a["inchikey14"] == b["inchikey14"]

    def test_large_peptide_skips_tautomer_canonicalization(self):
        """Two genuinely different peptides (different sequences) must NOT
        end up with the same blank-stereo InChIKey because of the tautomer
        canonicalizer giving up at its 1000-tautomer ceiling. The fix is
        to skip the canonicalizer for molecules above ~50 heavy atoms.

        Both inputs here are real angiotensin II analogs from BiasDB
        PMID 30514808 — `SVdF-AngII` (D-Phe at position 8) and Sarmesin
        (Val at position 5). They differ in residue composition.
        """
        svdf = standardize_smiles(
            "OC1=CC=C(C[C@@H](C(N[C@@H](C(C)C)C(N[C@@H](CC2=CNC=N2)"
            "C(N3[C@@H](CCC3)C(N[C@@H](CC4=CC=CC=C4)C(O)=O)=O)=O)=O)=O)"
            "NC([C@@H](NC([C@H](CCCNC(N)=N)NC(CN)=O)=O)C(C)C)=O)C=C1"
        )
        sarmesin = standardize_smiles(
            "O=C(N[C@@H](CCCNC(N)=N)C(N[C@@H](C(C)C)C(N[C@@H](CC1=CC=C(C=C1)O)"
            "C(N[C@@H](C(C)C)C(N[C@@H](CC2=CNC=N2)C(N3[C@@H](CCC3)"
            "C(N[C@@H](CC4=CC=CC=C4)C(O)=O)=O)=O)=O)=O)=O)CN)C"
        )
        assert svdf["status"] == sarmesin["status"] == "ok"
        # The two peptides have different sequences -> different InChIKeys.
        assert svdf["inchikey"] != sarmesin["inchikey"], (
            "tautomer canonicalization is collapsing distinct peptides; "
            "the heavy-atom threshold guard is not effective"
        )


@pytest.mark.unit
class TestStandardizeDataframe:
    def test_drops_failures_and_logs_attrition(self):
        df = pd.DataFrame(
            {"canonical_smiles": ["CCO", "garbage$$$", "", "CCN"]}
        )
        attr = AttritionLogger()
        out = standardize_dataframe(df, attrition=attr)
        assert len(out) == 2
        assert set(out["canonical_smiles_std"]).issubset({"CCO", "CCN"})
        assert attr.records[0]["stage"] == "standardize"
        assert attr.records[0]["n_in"] == 4
        assert attr.records[0]["n_out"] == 2

    def test_keeps_extra_columns(self):
        df = pd.DataFrame(
            {"canonical_smiles": ["CCO"], "ligand_name": ["ethanol"], "src": ["x"]}
        )
        out = standardize_dataframe(df)
        assert "ligand_name" in out.columns
        assert out.iloc[0]["ligand_name"] == "ethanol"


@pytest.mark.unit
class TestDedupe:
    def test_unanimous_kept_once(self):
        df = pd.DataFrame(
            {
                "inchikey14": ["AAAA", "AAAA", "BBBB"],
                "bias_category": ["G", "G", "Arr"],
            }
        )
        deduped, conflicts = dedupe_with_conflict_log(df)
        assert len(deduped) == 2
        assert len(conflicts) == 0

    def test_conflicts_extracted_and_dropped(self):
        df = pd.DataFrame(
            {
                "inchikey14": ["AAAA", "AAAA", "BBBB"],
                "bias_category": ["G", "Arr", "Balanced"],
            }
        )
        deduped, conflicts = dedupe_with_conflict_log(df)
        assert len(deduped) == 1
        assert deduped.iloc[0]["inchikey14"] == "BBBB"
        assert len(conflicts) == 2
        assert set(conflicts["bias_category"]) == {"G", "Arr"}

    def test_attrition_logged(self):
        df = pd.DataFrame(
            {"inchikey14": ["A", "A", "B"], "bias_category": ["x", "y", "z"]}
        )
        attr = AttritionLogger()
        dedupe_with_conflict_log(df, attrition=attr)
        assert attr.records[0]["stage"] == "dedupe"

    def test_missing_key_column_raises(self):
        df = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(KeyError):
            dedupe_with_conflict_log(df)


@pytest.mark.unit
class TestAnnotateDrugLikeness:
    def test_appends_columns_without_filtering(self):
        df = pd.DataFrame({"canonical_smiles_std": ["CCO", "CCN"]})
        out = annotate_drug_likeness(df)
        assert len(out) == 2  # nothing dropped
        for col in (
            "MW", "LogP", "HBD", "HBA", "TPSA",
            "RotBonds", "Lipinski_Violations", "has_pains",
        ):
            assert col in out.columns
        assert out["MW"].iloc[0] > 0

    def test_handles_invalid_smiles_gracefully(self):
        df = pd.DataFrame({"canonical_smiles_std": ["", "not-a-smiles"]})
        out = annotate_drug_likeness(df)
        # Columns exist; values are NaN where the molecule could not be parsed
        assert out["MW"].isna().all()


@pytest.mark.unit
class TestMarkLabelStatus:
    def test_distinguishes_labeled_from_unlabeled(self):
        df = pd.DataFrame(
            {"bias_category": ["G-biased", None, "", " Arr ", float("nan")]}
        )
        out = mark_label_status(df)
        assert list(out["label_status"]) == [
            "labeled", "unlabeled", "unlabeled", "labeled", "unlabeled",
        ]

    def test_does_not_invent_labels(self):
        # Whatever the source, this function never assigns a bias_category
        # value — that was the source-shortcut bug (E1).
        df = pd.DataFrame({"bias_category": [None, None]})
        out = mark_label_status(df)
        assert out["bias_category"].isna().all()


@pytest.mark.unit
class TestMergeSources:
    def test_concat_with_required_columns(self):
        a = pd.DataFrame(
            {"inchikey14": ["A"], "source": ["BiasDB"], "label_status": ["labeled"]}
        )
        b = pd.DataFrame(
            {"inchikey14": ["B"], "source": ["ChEMBL"], "label_status": ["unlabeled"]}
        )
        merged = merge_sources([a, b])
        assert len(merged) == 2
        assert set(merged["source"]) == {"BiasDB", "ChEMBL"}

    def test_rejects_missing_required_columns(self):
        a = pd.DataFrame({"inchikey14": ["A"]})
        with pytest.raises(KeyError):
            merge_sources([a])


@pytest.mark.unit
class TestLigandPreprocessorRun:
    """End-to-end integration test for the rewritten LigandPreprocessor.run().

    Verifies the Stage-02 decisions are actually in effect when the pipeline
    is invoked the way ``main.py`` invokes it:

    - BiasDB-only (no ChEMBL hardcoded labels) — E1 fix.
    - InChIKey-14 dedup, with conflicts surfaced to a separate CSV.
    - Drug-likeness columns present, but every BiasDB row survives by default.
    - Attrition log + provenance sidecar emitted.
    """

    def _write_synthetic_biasdb(self, path) -> None:
        """A tiny but realistic BiasDB CSV. Receptor names match the
        canonical registry so strict-mode receptor lookup succeeds.

        Includes:
        - Same molecule, same receptor, same assay-pair -> dedup on pair_key.
        - Same molecule, same receptor, *different* assay-pair -> kept as
          two rows under the composite key.
        - Same molecule at *different* receptors -> kept as two rows.
        - Garbage SMILES dropped at standardization.
        - Genuine same-context label conflict surfaced to dedup_conflicts.csv.
        """
        import pandas as _pd

        rows = [
            # 1. Ethanol at D2, BRET/cAMP, Go / β-Arr -> G protein
            {"smiles": "CCO", "ligand_name": "ethanol_a",
             "receptor_subtype": "D2 receptor",
             "bias_category": "G protein", "bias_pathway": "Go / β-Arr",
             "reference_ligand": "DA", "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "1", "year": 2018, "doi": "10.1/a"},
            # 2. SAME everything (true duplicate) -> dedup
            {"smiles": "CCO", "ligand_name": "ethanol_b",
             "receptor_subtype": "D2 receptor",
             "bias_category": "G protein", "bias_pathway": "Go / β-Arr",
             "reference_ligand": "DA", "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "2", "year": 2019, "doi": "10.1/b"},
            # 3. SAME pair_key but contradictory label -> real conflict.
            {"smiles": "CCO", "ligand_name": "ethanol_c",
             "receptor_subtype": "D2 receptor",
             "bias_category": "Arrestin", "bias_pathway": "Go / β-Arr",
             "reference_ligand": "DA", "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "3", "year": 2020, "doi": "10.1/c"},
            # 4. Same molecule, same receptor, DIFFERENT assay pair ->
            # distinct pair_key, kept.
            {"smiles": "CCO", "ligand_name": "ethanol_d",
             "receptor_subtype": "D2 receptor",
             "bias_category": "Arrestin", "bias_pathway": "β-Arr / Gi",
             "reference_ligand": "DA",
             "assay_1": "Tango", "assay_2": "PathHunter",
             "pmid": "4", "year": 2020, "doi": "10.1/d"},
            # 5. Same molecule, DIFFERENT receptor -> distinct pair_key, kept.
            {"smiles": "CCO", "ligand_name": "ethanol_e",
             "receptor_subtype": "μ receptor",
             "bias_category": "G protein", "bias_pathway": "Go / β-Arr",
             "reference_ligand": "DAMGO",
             "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "5", "year": 2021, "doi": "10.1/e"},
            # 6. Garbage SMILES -> dropped during standardization.
            {"smiles": "definitely-not-smiles$$$", "ligand_name": "junk",
             "receptor_subtype": "D2 receptor",
             "bias_category": "G protein", "bias_pathway": "Go / β-Arr",
             "reference_ligand": "DA", "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "6", "year": 2020, "doi": "10.1/f"},
            # 7. Methanol at OPRK1 -> unique molecule, unique row
            {"smiles": "CO", "ligand_name": "methanol",
             "receptor_subtype": "κ receptor",
             "bias_category": "β Arrestin", "bias_pathway": "β-Arr / Gi",
             "reference_ligand": "U69593",
             "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "7", "year": 2021, "doi": "10.1/g"},
            # 8. Same molecule at same receptor with same assay-pair but
            # DIFFERENT bias_pathway -> stays as its own measurement.
            {"smiles": "CCO", "ligand_name": "ethanol_pathway_alt",
             "receptor_subtype": "D2 receptor",
             "bias_category": "G protein selectivity",
             "bias_pathway": "Go / Gi",
             "reference_ligand": "DA", "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "8", "year": 2020, "doi": "10.1/h"},
            # 9. Stereoisomer test: a chiral molecule with a defined
            # stereocenter. Same receptor, same assay-pair, same pathway
            # as the otherwise-identical row 10 below — but a different
            # stereoisomer. Under inchikey14 they'd collapse; under the
            # full InChIKey they stay distinct.
            {"smiles": "C[C@H](N)Cc1ccccc1", "ligand_name": "(R)-amphetamine",
             "receptor_subtype": "D2 receptor",
             "bias_category": "G protein", "bias_pathway": "Go / β-Arr",
             "reference_ligand": "DA", "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "9", "year": 2022, "doi": "10.1/i"},
            # 10. The other enantiomer.
            {"smiles": "C[C@@H](N)Cc1ccccc1", "ligand_name": "(S)-amphetamine",
             "receptor_subtype": "D2 receptor",
             "bias_category": "Arrestin", "bias_pathway": "Go / β-Arr",
             "reference_ligand": "DA", "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "10", "year": 2022, "doi": "10.1/j"},
            # 11. Same molecule, same receptor, same assay-pair, same
            # pathway-comparison as row 7 — but a DIFFERENT reference
            # ligand (the bias factor is computed relative to the
            # reference, so this is a genuinely different measurement).
            # This is the buprenorphine-vs-morphine vs buprenorphine-vs-
            # DAMGO case from the real data.
            {"smiles": "CO", "ligand_name": "methanol_alt_ref",
             "receptor_subtype": "κ receptor",
             "bias_category": "G protein", "bias_pathway": "β-Arr / Gi",
             "reference_ligand": "U50488",  # different from row 7
             "assay_1": "BRET", "assay_2": "cAMP",
             "pmid": "11", "year": 2022, "doi": "10.1/k"},
        ]
        _pd.DataFrame(rows).to_csv(path, index=False)

    def _config(self, tmp_path) -> dict:
        biasdb_path = tmp_path / "biasdb_data.csv"
        self._write_synthetic_biasdb(biasdb_path)
        return {
            "paths": {
                "biasdb_input": str(biasdb_path),
                "processed_data": str(tmp_path / "processed"),
                "chembl_raw": str(tmp_path / "chembl"),
            },
            "preprocessing": {"legacy_filters": False},
        }

    def test_runs_end_to_end_no_source_shortcut(self, tmp_path):
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        proc = LigandPreprocessor(cfg)
        proc.run()

        out_csv = Path(cfg["paths"]["processed_data"]) / "unified_ligands.csv"
        assert out_csv.exists()

        df = pd.read_csv(out_csv)
        # E1 fix: no row carries the legacy hardcoded "Agonist" label.
        assert (df["bias_category"] != "Agonist").all()
        legitimate_labels = {
            "G protein", "Arrestin", "β Arrestin", "Balanced",
            "G protein selectivity",
        }
        assert set(df["bias_category"].dropna()).issubset(legitimate_labels)
        assert set(df["source"].unique()) == {"BiasDB"}
        # Every row must have a UniProt anchor (registry-strict).
        assert df["receptor_uniprot"].notna().all()
        assert (df["receptor_uniprot"].str.len() > 0).all()

    def test_composite_key_separates_distinct_measurements(self, tmp_path):
        """The same molecule (CCO) appears at four distinct
        (receptor, assay-pair, bias_pathway) combinations:
          A) D2 + BRET/cAMP + Go/β-Arr  -> rows 1,2 (G protein) and row 3
                                           (Arrestin) share this pair_key
                                           -> real conflict -> dedup_conflicts.
          B) D2 + Tango/PathHunter + β-Arr/Gi -> row 4 -> kept.
          C) μ-opioid + BRET/cAMP + Go/β-Arr  -> row 5 -> kept.
          D) D2 + BRET/cAMP + Go/Gi (different pathway comparison from A!)
              -> row 8 -> kept. THIS is the fix: under an assay-only key
              this would have collapsed into A and been treated as a fake
              conflict. Pathway-aware key preserves it.
        Surviving ethanol rows in the curated output: 3 (B, C, D).
        """
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        df = pd.read_csv(
            Path(cfg["paths"]["processed_data"]) / "unified_ligands.csv"
        )
        cco_rows = df[df["canonical_smiles_std"] == "CCO"]
        assert len(cco_rows) == 3, (
            f"pathway-aware composite key must keep 3 ethanol rows "
            f"(D2 Tango β-Arr/Gi, μ BRET Go/β-Arr, D2 BRET Go/Gi). "
            f"Got {len(cco_rows)}."
        )
        # All three surviving rows must have distinct pair_keys.
        assert cco_rows["pair_key"].nunique() == 3

    def test_pair_key_column_present_and_unique_in_output(self, tmp_path):
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        df = pd.read_csv(
            Path(cfg["paths"]["processed_data"]) / "unified_ligands.csv"
        )
        assert "pair_key" in df.columns
        # pair_key uniquely identifies a (molecule, receptor, assay-pair)
        # row in the deduped output.
        assert df["pair_key"].is_unique

    def test_only_real_same_context_conflicts_surface(self, tmp_path):
        """Of the synthetic data, only the (CCO, D2, BRET/cAMP) trio with
        contradicting labels (G protein vs Arrestin) is a *real*
        same-context conflict. The cross-receptor and cross-assay rows
        must NOT appear in conflicts.csv."""
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        conflicts_path = (
            Path(cfg["paths"]["processed_data"]) / "dedup_conflicts.csv"
        )
        assert conflicts_path.exists()
        conflicts = pd.read_csv(conflicts_path)
        # All conflicts must share the same pair_key (a real disagreement).
        assert conflicts["pair_key"].nunique() == 1

    def test_stereoisomers_kept_as_separate_rows(self, tmp_path):
        """The two amphetamine enantiomers (rows 9 and 10) share
        connectivity (same inchikey14) but differ in stereochemistry.
        With the full-InChIKey composite key, they must survive as two
        distinct rows. (R,R)-fenoterol vs (S,R)-fenoterol etc. is the
        real-world version of this — the original BiasDB authors
        reported each stereoisomer separately because their bias
        differs."""
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        df = pd.read_csv(
            Path(cfg["paths"]["processed_data"]) / "unified_ligands.csv"
        )
        amph = df[df["ligand_name"].str.contains("amphetamine", na=False)]
        assert len(amph) == 2, (
            f"both amphetamine enantiomers must survive; got {len(amph)}"
        )
        # Distinct full InChIKeys (the stereo layer differs even though
        # the inchikey14 prefix is identical).
        assert amph["inchikey"].nunique() == 2
        assert amph["inchikey14"].nunique() == 1
        assert amph["pair_key"].nunique() == 2

    def test_different_reference_ligand_kept_as_separate_rows(self, tmp_path):
        """Methanol at κ-opioid appears twice in the synthetic data:
        - row 7: vs reference U69593
        - row 11: vs reference U50488
        Same molecule, same receptor, same assay-pair, same pathway
        comparison — but different reference ligand. Bias is calculated
        Δlog(τ/KA) relative to the reference, so these are mathematically
        distinct measurements and must stay as two rows."""
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        df = pd.read_csv(
            Path(cfg["paths"]["processed_data"]) / "unified_ligands.csv"
        )
        meoh = df[df["canonical_smiles_std"] == "CO"]
        assert len(meoh) == 2, (
            f"both methanol-vs-different-reference rows must survive; "
            f"got {len(meoh)}"
        )
        assert set(meoh["reference_ligand"]) == {"U69593", "U50488"}
        assert meoh["pair_key"].nunique() == 2

    def test_dataset_audit_md_emitted(self, tmp_path):
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        audit_path = (
            Path(cfg["paths"]["processed_data"]) / "dataset_audit.md"
        )
        assert audit_path.exists()
        content = audit_path.read_text()
        for section in (
            "# CancerAg dataset audit",
            "## Top-line numbers",
            "## Bias-class distribution",
            "## Receptors by GPCRdb family",
            "## Top 15 receptors",
            "## Top 15 assay-pair combinations",
            "## Per-stage attrition",
            "## Reading the assay columns",
            "## Recommended manuscript framing",
        ):
            assert section in content, f"missing section: {section}"
        # Curated rows count appears in the top-line block
        assert "curated rows" in content

    def test_unknown_receptor_raises_in_strict_mode(self, tmp_path):
        """If a BiasDB row references a receptor that isn't in the
        registry, the curator must raise rather than silently producing
        a UniProt-less row. This is the registry-strict decision in code."""
        import pandas as _pd

        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        # Replace the synthetic CSV with one that names a fake receptor.
        rogue = _pd.DataFrame([{
            "smiles": "CCO", "ligand_name": "x",
            "receptor_subtype": "totally-fake-receptor",
            "bias_category": "G protein",
            "reference_ligand": "?", "assay_1": "X", "assay_2": "Y",
            "pmid": "1", "year": 2020, "doi": "?",
        }])
        rogue.to_csv(cfg["paths"]["biasdb_input"], index=False)
        with pytest.raises(KeyError, match="not in registry"):
            LigandPreprocessor(cfg).run()

    def test_attrition_log_emitted(self, tmp_path):
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        attrition_path = (
            Path(cfg["paths"]["processed_data"]) / "attrition.json"
        )
        assert attrition_path.exists()
        records = pd.read_json(attrition_path, orient="records")
        # We expect at minimum: load_biasdb, standardize, dedupe.
        assert {"load_biasdb", "standardize", "dedupe"}.issubset(
            set(records["stage"])
        )

    def test_drug_likeness_annotated_not_filtered(self, tmp_path):
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        df = pd.read_csv(
            Path(cfg["paths"]["processed_data"]) / "unified_ligands.csv"
        )
        for col in (
            "MW", "LogP", "HBD", "HBA", "TPSA",
            "RotBonds", "Lipinski_Violations", "has_pains",
        ):
            assert col in df.columns

    def test_provenance_sidecar(self, tmp_path):
        import json as _json

        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        cfg = self._config(tmp_path)
        LigandPreprocessor(cfg).run()
        meta_path = (
            Path(cfg["paths"]["processed_data"]) / "unified_ligands.csv.meta.json"
        )
        assert meta_path.exists()
        meta = _json.loads(meta_path.read_text())
        assert meta["source"].startswith("BiasDB-only")
        assert meta["row_count"] >= 2
        assert meta["input_sha256"] is not None
        assert meta["output_sha256"] is not None
        assert isinstance(meta["attrition"], list)


@pytest.mark.unit
class TestLegacyTypoFixed:
    def test_uncharger_attribute_name(self):
        from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

        # `unchoarger` was a misspelling that survived for years.
        # The fix renames it to the correct `uncharger`.
        config = {
            "paths": {
                "chembl_raw": "tests/data/chembl",
                "biasdb_input": "tests/data/biasdb_data.csv",
                "processed_data": "tests/data/processed",
            },
            "preprocessing": {"legacy_filters": False},
        }
        proc = LigandPreprocessor(config)
        # `uncharger` is no longer a class attribute (the standalone
        # `standardize_smiles` owns the uncharger now); the test asserts the
        # legacy typo `unchoarger` is gone from the source.
        import inspect

        src = inspect.getsource(LigandPreprocessor)
        assert "unchoarger" not in src
