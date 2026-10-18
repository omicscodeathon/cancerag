"""Tests for cancerag.preprocessing.chain_verification (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cancerag.preprocessing import chain_verification as cv

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
CANONICAL = ("MGQPGNGSAFLLAPNGSHAPDHDVTQERDEVWVVGMGIVMSLIVLAIVFGNVLVITAIAK"
             "FERLQTVTNYFITSLACADLVMGLAVVPFGAAHILMKMWTFGNFWCEFWTSIDVLCVTAS")


def _write_chain(path: Path, seq: str, chain: str = "R") -> None:
    lines = []
    for i, aa in enumerate(seq, start=1):
        lines.append(
            f"ATOM  {i:>5}  CA  {ONE_TO_THREE[aa]:>3} {chain}{i:>4}    "
            f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _fasta(acc: str, seq: str) -> str:
    return f">sp|{acc}|TEST_HUMAN Test OS=Homo sapiens SV=3\n{seq[:60]}\n{seq[60:]}\n"


@pytest.fixture
def receptor_dir(tmp_path):
    d = tmp_path / "receptors"
    d.mkdir()
    # P00001: exact construct, with a sidecar
    _write_chain(d / "P00001.pdb", CANONICAL[10:110])
    (d / "P00001.pdb.prep.meta.json").write_text(json.dumps(
        {"kept_chain": "R", "output_sha256": "stale", "waters_dropped": 3}))
    _write_chain(d / "P00001.prolif_H.pdb", CANONICAL[10:110])  # derived file
    # P00002: the wrong molecule (stands in for a Galpha chain), no sidecar
    _write_chain(d / "P00002.pdb", "W" * 80)
    return d


def _fake_fetch(calls):
    def fetch(acc):
        calls.append(acc)
        return _fasta(acc, CANONICAL)
    return fetch


@pytest.mark.unit
class TestFasta:
    def test_parse_first_record(self):
        header, seq = cv.parse_fasta(_fasta("P1", CANONICAL) + ">second\nAAAA\n")
        assert header.startswith("sp|P1|") and seq == CANONICAL

    def test_cache_is_written_then_reused(self, tmp_path):
        calls = []
        h1, s1 = cv.canonical_sequence("P1", tmp_path, fetch=_fake_fetch(calls))
        _h2, s2 = cv.canonical_sequence("P1", tmp_path, fetch=_fake_fetch(calls))
        assert calls == ["P1"]
        assert (tmp_path / "P1.fasta").exists()
        assert s1 == s2 == CANONICAL and cv._sequence_version(h1) == 3

    def test_empty_fasta_is_an_error_and_not_cached(self, tmp_path):
        with pytest.raises(ValueError):
            cv.canonical_sequence("P1", tmp_path, fetch=lambda _: ">x\n")
        assert not (tmp_path / "P1.fasta").exists()


@pytest.mark.unit
class TestRun:
    def test_derived_files_are_not_receptors(self, receptor_dir):
        names = [p.name for p in cv.prepared_receptors(receptor_dir)]
        assert names == ["P00001.pdb", "P00002.pdb"]

    def test_summary_and_sidecars(self, receptor_dir, tmp_path):
        summary_path = tmp_path / "summary.json"
        s = cv.run(receptor_dir=receptor_dir, cache_dir=tmp_path / "cache",
                   summary_path=summary_path, repair_report=tmp_path / "none.json",
                   backup_suffix=".bak-test", fetch=_fake_fetch([]))
        assert s == json.loads(summary_path.read_text())
        assert s["n_prepared_receptors"] == 2 and s["n_verified"] == 2
        by = {r["uniprot"]: r for r in s["receptors"]}

        good = by["P00001"]
        assert good["chain_identity_to_uniprot"] == pytest.approx(1.0)
        assert good["chain_coverage"] == pytest.approx(1.0)
        assert good["passes_gate"] and good["verified_chain"] == "R"
        assert good["sidecar_output_sha_matches_file"] is False

        bad = by["P00002"]
        assert not bad["passes_gate"]
        assert bad["sidecar"] is None and "no prep sidecar" in bad["sidecar_missing_reason"]
        assert s["below_gate"] == ["P00002"] and s["n_below_gate"] == 1
        assert s["identity"]["min"] == bad["chain_identity_to_uniprot"]

        meta = json.loads((receptor_dir / "P00001.pdb.prep.meta.json").read_text())
        assert meta["chain_identity_to_uniprot"] == pytest.approx(1.0)
        assert meta["chain_coverage"] == pytest.approx(1.0)
        assert meta["waters_dropped"] == 3  # other fields untouched
        assert meta["chain_verification"]["passes_gate"] is True
        backup = receptor_dir / "P00001.pdb.prep.meta.json.bak-test"
        assert json.loads(backup.read_text()) == {
            "kept_chain": "R", "output_sha256": "stale", "waters_dropped": 3}

    def test_backup_is_not_overwritten_on_rerun(self, receptor_dir, tmp_path):
        kw = {"receptor_dir": receptor_dir, "cache_dir": tmp_path / "cache",
              "summary_path": tmp_path / "s.json", "backup_suffix": ".bak-test",
              "repair_report": tmp_path / "none.json", "fetch": _fake_fetch([])}
        cv.run(**kw)
        cv.run(**kw)
        backup = json.loads(
            (receptor_dir / "P00001.pdb.prep.meta.json.bak-test").read_text())
        assert "chain_identity_to_uniprot" not in backup

    def test_no_write_leaves_sidecars_alone(self, receptor_dir, tmp_path):
        before = (receptor_dir / "P00001.pdb.prep.meta.json").read_text()
        cv.run(receptor_dir=receptor_dir, cache_dir=tmp_path / "cache",
               summary_path=tmp_path / "s.json", write_sidecars=False,
               repair_report=tmp_path / "none.json", fetch=_fake_fetch([]))
        assert (receptor_dir / "P00001.pdb.prep.meta.json").read_text() == before

    def test_repair_report_explains_missing_sidecar(self, receptor_dir, tmp_path):
        rr = tmp_path / "repair.json"
        rr.write_text(json.dumps([{"acc": "P00002", "chosen_pdb": "7XYZ",
                                   "chain": "R", "status": "repaired"}]))
        s = cv.run(receptor_dir=receptor_dir, cache_dir=tmp_path / "cache",
                   summary_path=tmp_path / "s.json", repair_report=rr,
                   fetch=_fake_fetch([]))
        rec = next(r for r in s["receptors"] if r["uniprot"] == "P00002")
        assert "7XYZ" in rec["sidecar_missing_reason"]

    def test_fetch_failure_is_reported_not_fatal(self, receptor_dir, tmp_path):
        def fetch(acc):
            if acc == "P00002":
                raise OSError("offline")
            return _fasta(acc, CANONICAL)
        s = cv.run(receptor_dir=receptor_dir, cache_dir=tmp_path / "cache",
                   summary_path=tmp_path / "s.json", write_sidecars=False,
                   repair_report=tmp_path / "none.json", fetch=fetch)
        assert s["n_verified"] == 1 and s["failures"][0]["uniprot"] == "P00002"
        assert s["n_prepared_receptors"] == 2
