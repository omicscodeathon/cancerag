"""Tests for cancerag.data_collection.schemas."""

from __future__ import annotations

import pytest

from cancerag.data_collection.schemas import (
    BIASDB_HEADER_ORDER,
    BiasDBRow,
    SchemaDriftError,
    parse_biasdb_payload,
)


@pytest.mark.unit
class TestBiasDBSchema:
    def test_positional_row_round_trip(self):
        positional = [
            "Lorcaserin",
            "Clc1ccc2[nH]cc(c2c1)-c1ccc(Br)cc1",
            "",
            "5-HT receptor",
            "5-HT2C",
            "5HT2C receptor",
            "G protein-biased",
            "Gαq",
            "5-HT",
            "BRET",
            "cAMP",
            "Some title",
            "Author",
            "10.1/abc",
            "12345",
            "2018",
            "260.1",
            "3.2",
            "2",
            "1",
            "3",
            "44.1",
        ]
        row = BiasDBRow.from_positional(positional, BIASDB_HEADER_ORDER)
        assert row.ligand_name == "Lorcaserin"
        assert row.year == 2018
        assert row.molecular_weight == pytest.approx(260.1)
        assert row.bias_category == "G protein-biased"

    def test_year_coercion_handles_garbage(self):
        row = BiasDBRow(year="not-a-year")
        assert row.year is None
        assert BiasDBRow(year="").year is None
        assert BiasDBRow(year=None).year is None

    def test_float_coercion(self):
        row = BiasDBRow(molecular_weight="", logp="3.5", tpsa=None)
        assert row.molecular_weight is None
        assert row.logp == 3.5

    def test_arity_mismatch_raises(self):
        with pytest.raises(ValueError, match="arity mismatch"):
            BiasDBRow.from_positional([1, 2, 3], BIASDB_HEADER_ORDER)

    def test_parse_payload_dict_form(self):
        payload = [
            {"ligand_name": "X", "smiles": "CCO", "year": "2020"},
            {"ligand_name": "Y", "smiles": "CCC", "year": ""},
        ]
        rows = parse_biasdb_payload(payload)
        assert len(rows) == 2
        assert rows[0].year == 2020
        assert rows[1].year is None

    def test_parse_payload_positional(self):
        payload = [
            ["A"] + [""] * (len(BIASDB_HEADER_ORDER) - 1),
        ]
        rows = parse_biasdb_payload(payload)
        assert rows[0].ligand_name == "A"

    def test_parse_payload_rejects_non_list(self):
        with pytest.raises(SchemaDriftError, match="must be a list"):
            parse_biasdb_payload({"oops": 1})  # type: ignore[arg-type]

    def test_parse_payload_rejects_unexpected_field(self):
        with pytest.raises(SchemaDriftError, match="row 0 failed"):
            parse_biasdb_payload([{"unknown_column": "x"}])

    def test_parse_payload_rejects_bad_arity(self):
        with pytest.raises(SchemaDriftError, match="arity mismatch"):
            parse_biasdb_payload([[1, 2, 3]])
