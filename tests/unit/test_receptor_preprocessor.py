"""
Tests for cancerag.preprocessing.receptor_preprocessor.

Covers chain detection, multi-chain dropping, water removal, conserved Na+
retention, altloc filtering, and the content-hash idempotency of
``prepare_receptor`` (re-running on unchanged input is a cheap cache hit).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from Bio.PDB import PDBParser

from cancerag.preprocessing.receptor_preprocessor import (
    RETAIN_HET_RESNAMES_DEFAULT,
    detect_receptor_chain,
    prepare_receptor,
)


def _write_synthetic_pdb(path: Path, *, n_chain_a: int = 220, n_chain_b: int = 5) -> None:
    """Emit a tiny synthetic PDB with two chains, a water, the conserved Na+,
    and one alternate-location atom on chain A residue 1.

    PDB ATOM record column layout (1-indexed):
      1-6   record (ATOM/HETATM)
      7-11  serial number
      13-16 atom name
      17    altloc
      18-20 resname
      22    chain
      23-26 resnum
      31-38 x  (8.3f)
      39-46 y  (8.3f)
      47-54 z  (8.3f)
      55-60 occupancy
      61-66 b-factor
      77-78 element
    """
    lines: list[str] = []
    serial = 1

    def atom(record: str, name: str, altloc: str, resname: str, chain: str,
             resnum: int, x: float, y: float, z: float, element: str) -> str:
        nonlocal serial
        ln = (
            f"{record:<6}{serial:>5} {name:<4}{altloc:1}{resname:>3} "
            f"{chain:1}{resnum:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"
        )
        serial += 1
        return ln

    # Chain A — n_chain_a ALA residues, all with a single CA atom.
    for i in range(1, n_chain_a + 1):
        lines.append(atom("ATOM", "CA", " ", "ALA", "A", i, float(i), 0.0, 0.0, "C"))
    # Add an alternate location ('B') for residue 1 to verify altloc filtering.
    lines.append(atom("ATOM", "CB", "B", "ALA", "A", 1, 1.5, 0.5, 0.0, "C"))

    # Chain B — n_chain_b residues (a pretend small partner chain).
    for i in range(1, n_chain_b + 1):
        lines.append(atom("ATOM", "CA", " ", "GLY", "B", i, 100.0 + i, 0.0, 0.0, "C"))

    # Conserved Na+ ion (HETATM, resname NA) in chain A.
    lines.append(atom("HETATM", "NA", " ", " NA", "A", 901, 5.0, 5.0, 5.0, "Na"))
    # Cholesterol HETATM (CLR) — should be dropped.
    lines.append(atom("HETATM", "C1", " ", "CLR", "A", 902, 8.0, 8.0, 8.0, "C"))
    # A water — should be dropped.
    lines.append(atom("HETATM", "O", " ", "HOH", "A", 950, 9.0, 9.0, 9.0, "O"))

    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def _residue_inventory(pdb_path: Path) -> tuple[set[str], set[str], int, int]:
    """Return (chains, het_resnames, n_water, n_atoms_with_altloc_B)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", str(pdb_path))
    chains: set[str] = set()
    het: set[str] = set()
    n_water = 0
    n_altloc_b = 0
    for model in structure:
        for chain in model:
            chains.add(chain.id)
            for res in chain:
                if res.id[0].startswith("H_"):
                    het.add(res.get_resname().strip().upper())
                if res.id[0] == "W" or res.get_resname().strip().upper() == "HOH":
                    n_water += 1
                for atom in res:
                    if atom.altloc == "B":
                        n_altloc_b += 1
        break
    return chains, het, n_water, n_altloc_b


@pytest.mark.unit
class TestDetectReceptorChain:
    def test_picks_long_chain(self, tmp_path: Path):
        pdb = tmp_path / "rec.pdb"
        _write_synthetic_pdb(pdb, n_chain_a=300, n_chain_b=10)
        assert detect_receptor_chain(pdb) == "A"

    def test_returns_none_when_no_chain_qualifies(self, tmp_path: Path):
        pdb = tmp_path / "tiny.pdb"
        _write_synthetic_pdb(pdb, n_chain_a=10, n_chain_b=5)
        assert detect_receptor_chain(pdb) is None

    def test_target_uniprot_picks_dbref_chain_over_longer_one(
        self, tmp_path: Path
    ):
        """GPCR-Gα complex pattern: chain A is the longer Gα subunit, chain R
        is the shorter receptor that maps to the target UniProt. Without
        target_uniprot, we'd pick A (length heuristic). With it, we must
        prefer R because DBREF says so."""
        pdb = tmp_path / "complex.pdb"
        _write_synthetic_pdb(pdb, n_chain_a=350, n_chain_b=0)
        # Append a third chain "R" of 280 residues and the DBREF records.
        with pdb.open("a") as f:
            for i in range(1, 281):
                f.write(
                    f"ATOM  {i:5d}  CA  ALA R{i:4d}    "
                    f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
                )
            f.write("TER\n")
            f.write("END\n")
        # Prepend DBREFs at the top by rewriting the file.
        body = pdb.read_text()
        dbref = (
            "DBREF  TEST A    1   354  UNP    P63096   GNAI1_HUMAN      1    354\n"
            "DBREF  TEST R    1   280  UNP    P08908   5HT1A_HUMAN      1    280\n"
        )
        pdb.write_text(dbref + body)
        # Without target_uniprot we'd pick A (350 > 280, plus prefer="A").
        assert detect_receptor_chain(pdb) == "A"
        # With target_uniprot, DBREF wins.
        assert detect_receptor_chain(pdb, target_uniprot="P08908") == "R"
        # Unrelated UniProt → fall back to length heuristic.
        assert detect_receptor_chain(pdb, target_uniprot="Q99999") == "A"

    def test_dbref_pdb_self_reference_is_ignored(self, tmp_path: Path):
        """A `DBREF ... PDB` line points to the PDB itself (engineered
        construct) and must not be treated as a UniProt mapping."""
        pdb = tmp_path / "engineered.pdb"
        _write_synthetic_pdb(pdb, n_chain_a=300, n_chain_b=0)
        body = pdb.read_text()
        pdb.write_text(
            "DBREF  TEST A    1   300  PDB    TEST     TEST             1    300\n"
            + body
        )
        # No UNP DBREF -> length heuristic chooses A regardless of target.
        assert detect_receptor_chain(pdb, target_uniprot="P08908") == "A"


@pytest.mark.unit
class TestPrepareReceptor:
    def test_drops_extra_chain_water_and_lipid_keeps_sodium(self, tmp_path: Path):
        in_pdb = tmp_path / "raw.pdb"
        out_pdb = tmp_path / "clean.pdb"
        _write_synthetic_pdb(in_pdb, n_chain_a=220, n_chain_b=5)

        meta = prepare_receptor(in_pdb, out_pdb)

        chains, het, n_water, n_altloc_b = _residue_inventory(out_pdb)
        assert chains == {"A"}, f"expected only chain A, got {chains}"
        assert "NA" in het, "conserved Na+ should be retained"
        assert "CLR" not in het, "cholesterol should be dropped"
        assert "HOH" not in het, "water should be dropped"
        assert n_water == 0
        assert n_altloc_b == 0, "altloc B atoms should be filtered out"

        assert meta["kept_chain"] == "A"
        assert meta["dropped_chains"] == ["B"]
        assert "NA" in meta["het_residues_kept"]
        assert "CLR" in meta["het_residues_dropped"]
        assert meta["waters_dropped"] >= 1
        assert meta["altloc_atoms_dropped"] >= 1

    def test_writes_meta_sidecar(self, tmp_path: Path):
        in_pdb = tmp_path / "raw.pdb"
        out_pdb = tmp_path / "clean.pdb"
        _write_synthetic_pdb(in_pdb)

        prepare_receptor(in_pdb, out_pdb)
        meta_path = out_pdb.with_suffix(out_pdb.suffix + ".prep.meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["input_sha256"]
        assert meta["output_sha256"]
        assert "prepared_at_utc" in meta

    def test_idempotent_on_unchanged_input(self, tmp_path: Path):
        in_pdb = tmp_path / "raw.pdb"
        out_pdb = tmp_path / "clean.pdb"
        _write_synthetic_pdb(in_pdb)
        first = prepare_receptor(in_pdb, out_pdb)
        # Second call with unchanged input must return cached meta verbatim.
        second = prepare_receptor(in_pdb, out_pdb)
        assert first == second

    def test_reprep_when_input_changes(self, tmp_path: Path):
        in_pdb = tmp_path / "raw.pdb"
        out_pdb = tmp_path / "clean.pdb"
        _write_synthetic_pdb(in_pdb, n_chain_a=220)
        first = prepare_receptor(in_pdb, out_pdb)
        # Mutate the input file and re-prep — the sha must change.
        _write_synthetic_pdb(in_pdb, n_chain_a=250)
        second = prepare_receptor(in_pdb, out_pdb)
        assert first["input_sha256"] != second["input_sha256"]

    def test_explicit_chain_override(self, tmp_path: Path):
        in_pdb = tmp_path / "raw.pdb"
        out_pdb = tmp_path / "clean.pdb"
        _write_synthetic_pdb(in_pdb, n_chain_a=220)
        meta = prepare_receptor(in_pdb, out_pdb, keep_chain="A")
        assert meta["kept_chain"] == "A"

    def test_raises_when_no_chain_qualifies(self, tmp_path: Path):
        in_pdb = tmp_path / "raw.pdb"
        out_pdb = tmp_path / "clean.pdb"
        _write_synthetic_pdb(in_pdb, n_chain_a=10, n_chain_b=5)
        with pytest.raises(ValueError, match="no chain"):
            prepare_receptor(in_pdb, out_pdb)


@pytest.mark.unit
class TestRetainHetDefaults:
    def test_sodium_in_default_retain_set(self):
        assert "NA" in RETAIN_HET_RESNAMES_DEFAULT
