"""Tests for cancerag.data_collection.provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from cancerag.data_collection.provenance import (
    read_meta,
    sha256_of,
    verify_meta,
    write_meta,
)


@pytest.mark.unit
class TestProvenance:
    def test_write_and_read_roundtrip(self, tmp_path: Path):
        artifact = tmp_path / "data.csv"
        artifact.write_text("a,b\n1,2\n")
        meta_path = write_meta(
            artifact,
            source_url="http://example.com",
            source_version="v1",
            query_params={"q": "default"},
            row_count=1,
        )
        assert meta_path.exists()
        meta = read_meta(artifact)
        assert meta["source_url"] == "http://example.com"
        assert meta["source_version"] == "v1"
        assert meta["row_count"] == 1
        assert meta["sha256"] == sha256_of(artifact)
        assert "fetch_timestamp_utc" in meta

    def test_verify_detects_tamper(self, tmp_path: Path):
        artifact = tmp_path / "data.csv"
        artifact.write_text("a,b\n1,2\n")
        write_meta(artifact, source_url="x")
        assert verify_meta(artifact) is True
        artifact.write_text("a,b\n1,3\n")
        assert verify_meta(artifact) is False

    def test_extra_fields(self, tmp_path: Path):
        artifact = tmp_path / "x.json"
        artifact.write_text("{}")
        write_meta(artifact, source_url="x", extra={"custom_tag": "stage_01"})
        assert read_meta(artifact)["custom_tag"] == "stage_01"

    def test_missing_artifact_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            write_meta(tmp_path / "ghost.csv", source_url="x")
