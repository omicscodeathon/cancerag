"""Tests for cancerag.data_collection.biasdb_retriever."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from cancerag.data_collection import biasdb_retriever
from cancerag.data_collection.provenance import sha256_of


@pytest.mark.unit
class TestBiasDBRetriever:
    def test_idempotent_when_csv_exists(self, tmp_path: Path):
        out = tmp_path / "biasdb.csv"
        out.write_text("ligand_name,smiles\nFoo,CCO\n")
        df = biasdb_retriever.download_biasdb_data(str(out))
        assert len(df) == 1
        assert df.iloc[0]["ligand_name"] == "Foo"

    def test_network_failure_raises_typed_error(self, tmp_path: Path):
        out = tmp_path / "biasdb.csv"

        class _ExplodingSession:
            def get(self, *_a, **_k):
                raise requests.exceptions.ConnectionError("boom")

        with patch.object(
            biasdb_retriever, "create_retry_session", return_value=_ExplodingSession()
        ):
            with pytest.raises(biasdb_retriever.BiasDBRetrievalError):
                biasdb_retriever.download_biasdb_data(
                    str(out), network_config={"max_retries": 0}
                )

    def test_invalid_json_raises_typed_error(self, tmp_path: Path):
        out = tmp_path / "biasdb.csv"

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("not json")

        class _FakeSession:
            def get(self, *_a, **_k):
                return _FakeResponse()

        with patch.object(
            biasdb_retriever, "create_retry_session", return_value=_FakeSession()
        ):
            with pytest.raises(biasdb_retriever.BiasDBRetrievalError):
                biasdb_retriever.download_biasdb_data(
                    str(out), network_config={"max_retries": 0}
                )

    def test_schema_drift_raises_typed_error(self, tmp_path: Path):
        out = tmp_path / "biasdb.csv"

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return [{"unexpected_column": "value"}]

        class _FakeSession:
            def get(self, *_a, **_k):
                return _FakeResponse()

        with patch.object(
            biasdb_retriever, "create_retry_session", return_value=_FakeSession()
        ):
            with pytest.raises(
                biasdb_retriever.BiasDBRetrievalError, match="schema drift"
            ):
                biasdb_retriever.download_biasdb_data(
                    str(out), network_config={"max_retries": 0}
                )

    def test_writes_provenance_sidecar(self, tmp_path: Path):
        out = tmp_path / "biasdb.csv"

        class _FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return [
                    {
                        "ligand_name": "Lorcaserin",
                        "smiles": "CCO",
                        "year": "2018",
                    }
                ]

        class _FakeSession:
            def get(self, *_a, **_k):
                return _FakeResponse()

        with patch.object(
            biasdb_retriever, "create_retry_session", return_value=_FakeSession()
        ):
            df = biasdb_retriever.download_biasdb_data(
                str(out), network_config={"max_retries": 0}
            )
        assert len(df) == 1
        meta = json.loads((tmp_path / "biasdb.csv.meta.json").read_text())
        assert meta["row_count"] == 1
        assert meta["source_url"] == biasdb_retriever.BIASDB_URL
        assert meta["sha256"] == sha256_of(out)

    def test_logging_basicConfig_only_inside_main_guard(self):
        """basicConfig must not run at import — only when invoked as
        ``python -m cancerag.data_collection.biasdb_retriever``."""
        import inspect
        import re

        src_lines = inspect.getsource(biasdb_retriever).splitlines()
        main_guard_idx = next(
            (
                i for i, ln in enumerate(src_lines)
                if 'if __name__ == "__main__"' in ln
            ),
            None,
        )
        assert main_guard_idx is not None, "module must declare __main__ guard"
        call_re = re.compile(r"^\s*logging\.basicConfig\s*\(")
        bc_indices = [i for i, ln in enumerate(src_lines) if call_re.match(ln)]
        for i in bc_indices:
            assert i > main_guard_idx, (
                f"logging.basicConfig at line {i + 1} is outside __main__ guard"
            )
