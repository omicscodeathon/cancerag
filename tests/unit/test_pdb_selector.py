"""Tests for cancerag.preprocessing.pdb_selector."""

from __future__ import annotations

from pathlib import Path

import pytest

from cancerag.preprocessing.pdb_selector import (
    PDBCandidate,
    score_candidate,
    select_best_pdb,
)


def _write_minimal_pdb(
    path: Path,
    *,
    resolution: float | None = None,
    het_resnames: tuple[str, ...] = (),
    chains: tuple[str, ...] = ("A",),
    g_alpha_chain: bool = False,
    fusion: bool = False,
    residues_per_chain: int = 220,
) -> None:
    """Emit a synthetic PDB. Default chain length (220 residues) is above
    the GPCR-receptor threshold so existing tests don't trip the
    fragment-rejection guard. Pass a smaller value to test fragment-
    rejection behaviour."""
    lines: list[str] = []
    if resolution is not None:
        lines.append(
            f"REMARK   2 RESOLUTION.    {resolution:.2f} ANGSTROMS."
        )
    if g_alpha_chain:
        lines.append("COMPND    MOL_ID: 2; MOLECULE: GUANINE NUCLEOTIDE-BINDING "
                     "PROTEIN G(S) SUBUNIT ALPHA (GNAS);")
    if fusion:
        lines.append("COMPND    MOL_ID: 3; MOLECULE: ENDOLYSIN (T4L);")
    serial = 1
    for ch in chains:
        for i in range(1, residues_per_chain + 1):
            lines.append(
                f"ATOM  {serial:>5}  CA  ALA {ch}{i:>4}    "
                f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
            )
            serial += 1
    for i, h in enumerate(het_resnames, start=1):
        lines.append(
            f"HETATM{serial:>5}  C   {h:>3} A{900 + i:>4}    "
            f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.unit
class TestScoring:
    def test_higher_resolution_wins(self):
        a = PDBCandidate("A", Path("/x"), resolution=2.0, has_orthosteric_ligand=True,
                         detected_state="active")
        b = PDBCandidate("B", Path("/x"), resolution=2.8, has_orthosteric_ligand=True,
                         detected_state="active")
        c = PDBCandidate("C", Path("/x"), resolution=3.5, has_orthosteric_ligand=True,
                         detected_state="active")
        assert score_candidate(a) > score_candidate(b) > score_candidate(c)

    def test_active_state_bonus(self):
        active = PDBCandidate("A", Path("/x"), resolution=2.0,
                              has_orthosteric_ligand=True, detected_state="active")
        inactive = PDBCandidate("B", Path("/x"), resolution=2.0,
                                has_orthosteric_ligand=True,
                                detected_state="inactive_likely")
        assert score_candidate(active) - score_candidate(inactive) >= 30.0

    def test_fusion_protein_penalty(self):
        no_fusion = PDBCandidate("A", Path("/x"), resolution=2.0,
                                  has_orthosteric_ligand=True,
                                  detected_state="active")
        with_fusion = PDBCandidate("B", Path("/x"), resolution=2.0,
                                    has_orthosteric_ligand=True,
                                    detected_state="active",
                                    has_fusion_protein=True)
        assert score_candidate(no_fusion) - score_candidate(with_fusion) == 10.0

    def test_extra_chains_penalty(self):
        single = PDBCandidate("A", Path("/x"), resolution=2.0,
                              has_orthosteric_ligand=True, n_chains=1)
        multi = PDBCandidate("B", Path("/x"), resolution=2.0,
                             has_orthosteric_ligand=True, n_chains=4)
        assert score_candidate(single) > score_candidate(multi)


@pytest.mark.unit
class TestFragmentRejection:
    """Real-data finding from CXCR2 (P25025) cache: PDB 4Q3H is a tiny
    90-residue N-terminal extracellular domain of CXCR2 at 1.44 Å, while
    full-length CXCR2 cryo-EM structures (~300 residues) sit at ~3 Å.
    The picker was choosing 4Q3H because it had the best resolution,
    even though 90 residues can't host the orthosteric pocket. The
    scorer must reject any candidate whose longest chain is below the
    GPCR full-receptor threshold (200 residues)."""

    def test_full_length_beats_high_resolution_fragment(self, tmp_path):
        from cancerag.preprocessing.pdb_selector import select_best_pdb

        # A) 1.44 Å fragment, 90 residues (the 4Q3H situation).
        frag = tmp_path / "FRAG.pdb"
        _write_minimal_pdb(frag, resolution=1.44, het_resnames=("8NU",),
                           residues_per_chain=90)
        # B) 3.0 Å full-length, 250 residues (the cryo-EM situation).
        full = tmp_path / "FULL.pdb"
        _write_minimal_pdb(full, resolution=3.0, het_resnames=("8NU",),
                           residues_per_chain=250)

        best = select_best_pdb(tmp_path)
        assert best is not None
        assert best.pdb_id == "FULL", (
            f"full-length 250-residue chain must beat the 90-residue "
            f"high-resolution fragment; got {best.pdb_id}"
        )

    def test_all_fragments_returns_none_or_score_neg_inf(self, tmp_path):
        """If every candidate is a fragment, every candidate is scored
        -inf so the receptor will fall back to AlphaFold downstream."""
        from cancerag.preprocessing.pdb_selector import select_best_pdb

        a = tmp_path / "A.pdb"
        _write_minimal_pdb(a, resolution=1.5, het_resnames=("8NU",),
                           residues_per_chain=80)
        b = tmp_path / "B.pdb"
        _write_minimal_pdb(b, resolution=2.0, het_resnames=("8NU",),
                           residues_per_chain=120)

        best = select_best_pdb(tmp_path)
        assert best is not None
        assert best.score == float("-inf"), (
            "all candidates are fragments — the top score must be -inf so "
            "the receptor cleanly falls back to AlphaFold"
        )


@pytest.mark.unit
class TestSelectBestPdb:
    def test_picks_active_state_over_inactive(self, tmp_path):
        active = tmp_path / "ACTV.pdb"
        inactive = tmp_path / "INAC.pdb"
        _write_minimal_pdb(active, resolution=2.5,
                           het_resnames=("8NU",), g_alpha_chain=True)
        _write_minimal_pdb(inactive, resolution=2.0,
                           het_resnames=("8NU",), fusion=True)
        best = select_best_pdb(tmp_path)
        assert best is not None
        assert best.pdb_id == "ACTV"
        assert best.detected_state == "active"

    def test_picks_higher_resolution_when_state_ties(self, tmp_path):
        a = tmp_path / "GOOD.pdb"
        b = tmp_path / "BAD.pdb"
        _write_minimal_pdb(a, resolution=2.0, het_resnames=("8NU",))
        _write_minimal_pdb(b, resolution=3.5, het_resnames=("8NU",))
        best = select_best_pdb(tmp_path)
        assert best.pdb_id == "GOOD"

    def test_ignores_lipid_hetatms_for_ligand_detection(self, tmp_path):
        chol_only = tmp_path / "CHOL.pdb"
        with_drug = tmp_path / "DRUG.pdb"
        _write_minimal_pdb(chol_only, resolution=2.0,
                           het_resnames=("CLR",))
        _write_minimal_pdb(with_drug, resolution=2.0,
                           het_resnames=("8NU",))
        best = select_best_pdb(tmp_path)
        # Ligand-bearing structure must outrank the cholesterol-only one
        assert best.pdb_id == "DRUG"
        assert best.has_orthosteric_ligand is True

    def test_returns_none_for_empty_dir(self, tmp_path):
        assert select_best_pdb(tmp_path / "ghost") is None

    def test_returns_none_for_dir_with_no_pdbs(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a pdb")
        assert select_best_pdb(tmp_path) is None
