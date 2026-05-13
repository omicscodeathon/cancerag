"""Tests for cancerag.features.active_site_identifier (and the
``extract_binding_site`` helper that lives in ``receptor_preprocessor`` but
is exercised here because its picker logic is the same concern)."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

from cancerag.features import active_site_identifier
from cancerag.features.active_site_identifier import (
    StructureMetrics,
    score_structure,
)
from cancerag.preprocessing import receptor_preprocessor


def _write_pdb_with_het(path: Path, het_resname: str) -> None:
    """Tiny PDB with one tagged HETATM near the origin so atom coords parse."""
    lines: list[str] = []
    serial = 1
    for i in range(1, 221):
        lines.append(
            f"ATOM  {serial:>5}  CA  ALA A{i:>4}    "
            f"{float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1
    coords = [(10.0, 0.0, 0.0), (10.5, 0.5, 0.0), (10.0, 0.5, 0.5)]
    for i, (x, y, z) in enumerate(coords):
        lines.append(
            f"HETATM{serial:>5}  C{i:<2} {het_resname:>3} A{900 + i:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.unit
class TestStructureScore:
    def test_higher_resolution_wins(self):
        good = score_structure(StructureMetrics(resolution=2.0, has_ligand=True))
        ok = score_structure(StructureMetrics(resolution=2.8, has_ligand=True))
        worse = score_structure(StructureMetrics(resolution=3.5, has_ligand=True))
        assert good > ok > worse

    def test_ligand_bonus(self):
        with_lig = score_structure(StructureMetrics(resolution=2.0, has_ligand=True))
        no_lig = score_structure(StructureMetrics(resolution=2.0, has_ligand=False))
        assert with_lig - no_lig == pytest.approx(50.0)

    def test_completeness_bonus_clamped(self):
        s = score_structure(StructureMetrics(resolution=2.0, completeness=2.0))
        assert s == pytest.approx(20.0)

    def test_no_negative_for_ideal_structure(self):
        s = score_structure(
            StructureMetrics(resolution=1.5, has_ligand=True, completeness=1.0)
        )
        assert s == pytest.approx(50.0 + 0.0 + 20.0)

    def test_resolution_2_0_breakeven(self):
        s_at = score_structure(StructureMetrics(resolution=2.0, has_ligand=True))
        s_below = score_structure(StructureMetrics(resolution=1.5, has_ligand=True))
        assert s_at == s_below

    def test_inf_resolution_handled(self):
        s = score_structure(StructureMetrics())
        assert s == -math.inf or s < -100


@pytest.mark.unit
class TestActiveSiteIdentifierImports:
    def test_no_local_ignore_list_constant(self):
        """The legacy duplicate IGNORE_LIST literal must be gone."""
        src = inspect.getsource(active_site_identifier)
        assert '"HOH"' not in src or "LIGAND_AUTO_DETECT_IGNORE" in src
        assert "LIGAND_AUTO_DETECT_IGNORE" in src

    def test_no_module_level_basicConfig(self):
        import re

        src_lines = inspect.getsource(active_site_identifier).splitlines()
        guard_idx = next(
            (
                i for i, ln in enumerate(src_lines)
                if 'if __name__ == "__main__"' in ln
            ),
            len(src_lines),
        )
        call_re = re.compile(r"^\s*logging\.basicConfig\s*\(")
        bc_indices = [i for i, ln in enumerate(src_lines) if call_re.match(ln)]
        for i in bc_indices:
            assert i > guard_idx, (
                f"basicConfig at line {i + 1} runs at import"
            )


@pytest.mark.unit
class TestExtractBindingSiteIgnoresLipids:
    def test_cholesterol_skipped_in_auto_detect(self, tmp_path: Path):
        pdb = tmp_path / "with_chol.pdb"
        _write_pdb_with_het(pdb, "CLR")
        bs = receptor_preprocessor.extract_binding_site(str(pdb))
        assert bs is None, "cholesterol should not be picked as a ligand"

    def test_real_ligand_picked(self, tmp_path: Path):
        pdb = tmp_path / "with_drug.pdb"
        _write_pdb_with_het(pdb, "8NU")
        bs = receptor_preprocessor.extract_binding_site(str(pdb))
        assert bs is not None
        assert "center_x" in bs
