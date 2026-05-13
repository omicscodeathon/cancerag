"""Tests for cancerag.preprocessing.alphafold_fetcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from cancerag.preprocessing import alphafold_fetcher
from cancerag.preprocessing.alphafold_fetcher import (
    AlphaFoldFetchError,
    _mean_plddt_from_pdb,
    fetch_alphafold_pdb,
)


def _af_pdb_text(plddts: list[float]) -> str:
    """Build a tiny AlphaFold-shaped PDB where each ATOM CA carries the
    pLDDT in the B-factor column (cols 61-66)."""
    lines = ["HEADER    SYNTHETIC AF MODEL"]
    for i, p in enumerate(plddts, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A{i:>4}    "
            f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00{p:6.2f}           C"
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


@pytest.mark.unit
class TestMeanPlddt:
    def test_simple_average(self):
        text = _af_pdb_text([90.0, 80.0, 70.0])
        mean, n = _mean_plddt_from_pdb(text)
        assert n == 3
        assert mean == pytest.approx(80.0)

    def test_empty_pdb_returns_zero(self):
        mean, n = _mean_plddt_from_pdb("HEADER\nEND\n")
        assert n == 0
        assert mean == 0.0

    def test_only_ca_atoms_count(self):
        text = "HEADER\n"
        text += (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\n"
            "ATOM      2  CB  ALA A   1       0.000   0.000   0.000  1.00 50.00           C\n"
        )
        text += "END\n"
        mean, n = _mean_plddt_from_pdb(text)
        assert n == 1
        assert mean == pytest.approx(90.0)


@pytest.mark.unit
class TestFetchAlphaFold:
    def _mock_response(self, text: str, status: int = 200):
        class _Resp:
            status_code = status
            def __init__(self): self.text = text
            def raise_for_status(self):
                if self.status_code >= 400:
                    err = requests.exceptions.HTTPError(f"{self.status_code}")
                    err.response = self
                    raise err
        return _Resp()

    def test_passes_gate_writes_artifact_and_meta(self, tmp_path: Path):
        text = _af_pdb_text([90.0] * 200)
        with patch.object(
            alphafold_fetcher.requests, "get",
            return_value=self._mock_response(text),
        ):
            out = fetch_alphafold_pdb("P14416", tmp_path, plddt_threshold=70.0)
        assert out["passes_gate"] is True
        assert out["mean_plddt"] == pytest.approx(90.0)
        assert Path(out["output_pdb"]).exists()
        meta_path = Path(out["output_pdb"]).with_suffix(".pdb.meta.json")
        assert meta_path.exists()

    def test_below_threshold_does_not_pass(self, tmp_path: Path):
        text = _af_pdb_text([60.0] * 200)
        with patch.object(
            alphafold_fetcher.requests, "get",
            return_value=self._mock_response(text),
        ):
            out = fetch_alphafold_pdb("P14416", tmp_path, plddt_threshold=70.0)
        assert out["passes_gate"] is False
        assert out["mean_plddt"] == pytest.approx(60.0)

    def test_404_raises_typed_error(self, tmp_path: Path):
        with patch.object(
            alphafold_fetcher.requests, "get",
            return_value=self._mock_response("", status=404),
        ):
            with pytest.raises(AlphaFoldFetchError, match="No AlphaFold model"):
                fetch_alphafold_pdb("FAKE0", tmp_path)

    def test_garbage_body_raises(self, tmp_path: Path):
        with patch.object(
            alphafold_fetcher.requests, "get",
            return_value=self._mock_response("not a pdb at all"),
        ):
            with pytest.raises(AlphaFoldFetchError, match="does not look like a PDB"):
                fetch_alphafold_pdb("P14416", tmp_path)

    def test_idempotent_on_cached(self, tmp_path: Path):
        text = _af_pdb_text([85.0] * 200)
        cached = tmp_path / "AF-P14416-F1.pdb"
        cached.write_text(text)
        # No mock — should NOT call requests because the file exists
        out = fetch_alphafold_pdb("P14416", tmp_path, plddt_threshold=70.0)
        assert out["from_cache"] is True
        assert out["passes_gate"] is True
