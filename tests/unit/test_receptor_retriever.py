"""Tests for cancerag.data_collection.receptor_retriever (rewritten,
UniProt-anchored)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cancerag.data_collection import receptor_retriever
from cancerag.data_collection.receptor_retriever import (
    ReceptorRetrievalError,
    ReceptorRetriever,
)
from cancerag.data_collection.registry import ReceptorRegistry


def _registry() -> ReceptorRegistry:
    return ReceptorRegistry.load("data/registry/receptors.tsv")


def _retriever(tmp_path: Path, **kw) -> ReceptorRetriever:
    return ReceptorRetriever(
        output_dir=tmp_path / "pdb",
        registry=_registry(),
        network_config={"max_retries": 0},
        **kw,
    )


def _uniprot_record(pdb_blocks: list[dict]) -> dict:
    """Build a UniProt-shaped record with PDB cross-references."""
    return {
        "accession": "TEST",
        "dbReferences": [
            {
                "type": "PDB",
                "id": b["pdb_id"],
                "properties": {
                    "method": b.get("method", "X-ray"),
                    "resolution": b.get("resolution", "2.50 A"),
                    "chains": b.get("chains", "A=1-300"),
                },
            }
            for b in pdb_blocks
        ],
    }


@pytest.mark.unit
class TestParsePdbRefs:
    def test_extracts_only_pdb_type(self):
        rec = {
            "dbReferences": [
                {"type": "PDB", "id": "6CM4",
                 "properties": {"method": "X-ray", "resolution": "2.87 A",
                                 "chains": "A=35-443"}},
                {"type": "GO", "id": "GO:0001234"},
                {"type": "PDB", "id": "7DFP",
                 "properties": {"method": "X-ray", "resolution": "3.10 A",
                                 "chains": "A"}},
            ]
        }
        out = ReceptorRetriever.parse_pdb_refs(rec)
        ids = [r["pdb_id"] for r in out]
        assert ids == ["6CM4", "7DFP"]

    def test_handles_missing_resolution(self):
        rec = _uniprot_record([{"pdb_id": "X", "method": "EM", "resolution": "-"}])
        out = ReceptorRetriever.parse_pdb_refs(rec)
        assert out[0]["resolution"] == float("inf")

    def test_handles_empty_dbreferences(self):
        assert ReceptorRetriever.parse_pdb_refs({}) == []
        assert ReceptorRetriever.parse_pdb_refs({"dbReferences": None}) == []


@pytest.mark.unit
class TestFilterAndRank:
    def test_drops_high_resolution(self, tmp_path: Path):
        r = _retriever(tmp_path, resolution_cutoff=3.0, max_pdbs_per_receptor=10)
        refs = [
            {"pdb_id": "A", "method": "X-ray", "resolution": 2.0, "chains": ""},
            {"pdb_id": "B", "method": "X-ray", "resolution": 2.5, "chains": ""},
            {"pdb_id": "C", "method": "X-ray", "resolution": 4.5, "chains": ""},
        ]
        out = r.filter_and_rank(refs)
        assert [x["pdb_id"] for x in out] == ["A", "B"]

    def test_sorted_by_resolution_best_first(self, tmp_path: Path):
        r = _retriever(tmp_path, resolution_cutoff=10.0)
        refs = [
            {"pdb_id": "MID", "method": "X-ray", "resolution": 2.5, "chains": ""},
            {"pdb_id": "BEST", "method": "X-ray", "resolution": 1.5, "chains": ""},
            {"pdb_id": "WORST", "method": "X-ray", "resolution": 4.0, "chains": ""},
        ]
        out = r.filter_and_rank(refs)
        assert [x["pdb_id"] for x in out] == ["BEST", "MID", "WORST"]

    def test_caps_at_max_pdbs(self, tmp_path: Path):
        r = _retriever(tmp_path, resolution_cutoff=10.0, max_pdbs_per_receptor=2)
        refs = [
            {"pdb_id": str(i), "method": "X-ray", "resolution": float(i),
             "chains": ""} for i in range(5)
        ]
        out = r.filter_and_rank(refs)
        assert len(out) == 2

    def test_drops_disallowed_method(self, tmp_path: Path):
        r = _retriever(tmp_path, allowed_methods=("X-ray",))
        refs = [
            {"pdb_id": "X", "method": "X-ray", "resolution": 2.0, "chains": ""},
            {"pdb_id": "E", "method": "EM", "resolution": 2.0, "chains": ""},
        ]
        assert [x["pdb_id"] for x in r.filter_and_rank(refs)] == ["X"]


@pytest.mark.unit
class TestRunWritesArtifacts:
    def test_run_downloads_and_emits_meta(self, tmp_path: Path):
        r = _retriever(tmp_path, max_pdbs_per_receptor=2)
        rec = _uniprot_record([
            {"pdb_id": "6CM4", "resolution": "2.87 A"},
            {"pdb_id": "7DFP", "resolution": "3.10 A"},
        ])

        def _fake_uniprot_fetch(uniprot):
            return rec

        def _fake_download(pdb_id, dest):
            dest.write_text(f"HEADER {pdb_id}\nEND\n")
            return True

        with patch.object(r, "_fetch_uniprot_record",
                          side_effect=_fake_uniprot_fetch), \
             patch.object(r, "_download_pdb", side_effect=_fake_download):
            summary = r.run(restrict_to={"P14416"})

        assert summary["totals"]["uniprots_processed"] == 1
        assert summary["totals"]["pdbs_downloaded"] == 2
        out_dir = tmp_path / "pdb" / "P14416"
        assert (out_dir / "6CM4.pdb").exists()
        assert (out_dir / "7DFP.pdb").exists()
        assert (out_dir / "6CM4.pdb.meta.json").exists()
        per_uniprot = json.loads((out_dir / "_uniprot_summary.json").read_text())
        assert per_uniprot["uniprot"] == "P14416"
        assert per_uniprot["kept_after_filter"] == ["6CM4", "7DFP"]

    def test_uniprot_fetch_failure_recorded(self, tmp_path: Path):
        r = _retriever(tmp_path)
        with patch.object(
            r, "_fetch_uniprot_record",
            side_effect=ReceptorRetrievalError("boom"),
        ):
            summary = r.run(restrict_to={"P14416"})
        assert summary["totals"]["uniprot_fetch_failures"] == 1
        assert summary["receptors"][0]["status"] == "uniprot_fetch_failed"

    def test_no_pdbs_after_filter(self, tmp_path: Path):
        r = _retriever(tmp_path, resolution_cutoff=0.5)  # nothing passes
        with patch.object(
            r, "_fetch_uniprot_record",
            return_value=_uniprot_record([
                {"pdb_id": "X", "resolution": "3.0 A"}
            ]),
        ):
            summary = r.run(restrict_to={"P14416"})
        assert summary["totals"]["uniprots_with_zero_pdbs"] == 1


@pytest.mark.unit
class Test404Handling:
    """A 404 from RCSB is permanent — the retriever must skip the PDB
    immediately, not loop on it via the NetworkRetrier (whose default
    config is `max_retries: null` = retry indefinitely)."""

    def test_404_returns_false_without_retry(self, tmp_path: Path):
        r = _retriever(tmp_path)

        class _Resp:
            status_code = 404
            text = ""

            def raise_for_status(self):
                err = receptor_retriever.requests.exceptions.HTTPError(
                    "404 Client Error: Not Found"
                )
                err.response = self
                raise err

        call_count = {"n": 0}

        def _fake_get(*_a, **_k):
            call_count["n"] += 1
            return _Resp()

        with patch.object(r.session, "get", side_effect=_fake_get):
            ok = r._download_pdb("9DYF", tmp_path / "9DYF.pdb")

        assert ok is False
        assert call_count["n"] == 1, (
            f"404 should be a single attempt, got {call_count['n']}"
        )
        assert not (tmp_path / "9DYF.pdb").exists()


@pytest.mark.unit
class TestFromConfig:
    def test_wires_config_keys(self, tmp_path: Path):
        cfg = {
            "data_collection": {
                "max_pdb_files_per_receptor": 5,
                "max_resolution_angstrom": 3.5,
            },
            "network": {"max_retries": 0},
        }
        r = ReceptorRetriever.from_config(
            output_dir=tmp_path / "pdb", config=cfg, registry=_registry()
        )
        assert r.max_pdbs_per_receptor == 5
        assert r.resolution_cutoff == 3.5
