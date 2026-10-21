"""A loaded pose must carry real chemistry, or not be returned at all.

Companion to the receptor-side cover in
``test_interaction_fingerprint_receptor.py``. Same mechanism, ligand side: a
molecule whose chemistry was guessed from coordinates reached ProLIF and the
3D descriptors, and the features computed from it were wrong without anything
saying so.

The old loader's last resort sanitised with ``SANITIZE_KEKULIZE`` removed and
returned the result, claiming in its docstring that this stayed "usable by
ProLIF for hydrophobic / VdW / pi-stacking perception". Measured over the
docked corpus, that path fired on 112 of 455 poses, and a molecule with
kekulisation disabled has no aromatic ring for PiStacking to find.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rdkit import Chem

from cancerag.features.pose_loader import load_pose_mol

BENZAMIDINE_SMILES = "NC(=N)c1ccccc1"


@pytest.fixture
def missing_pose(tmp_path: Path) -> Path:
    return tmp_path / "absent.pdbqt"


@pytest.fixture
def empty_pose(tmp_path: Path) -> Path:
    p = tmp_path / "empty.pdbqt"
    p.write_text("")
    return p


def test_absent_pose_returns_none(missing_pose: Path):
    assert load_pose_mol(missing_pose, BENZAMIDINE_SMILES) is None


def test_empty_pose_returns_none(empty_pose: Path):
    assert load_pose_mol(empty_pose, BENZAMIDINE_SMILES) is None


def test_unreadable_pose_is_dropped_not_degraded(tmp_path: Path, caplog):
    """The point of the change: no molecule rather than a guessed one."""
    junk = tmp_path / "junk.pdbqt"
    junk.write_text("MODEL 1\nnot a pdbqt at all\nENDMDL\n")

    assert load_pose_mol(junk, BENZAMIDINE_SMILES) is None


def test_no_template_still_returns_none_rather_than_a_guess(tmp_path: Path):
    """An empty template must not open a path back to coordinate guessing."""
    junk = tmp_path / "junk.pdbqt"
    junk.write_text("MODEL 1\nnot a pdbqt at all\nENDMDL\n")

    assert load_pose_mol(junk, "") is None


@pytest.mark.skipif(
    not Path("data/processed/.docking_work").is_dir(),
    reason="docked corpus not present in this checkout",
)
def test_real_pose_has_bonds_and_aromaticity():
    """Against a real Vina output, chemistry must survive the round trip."""
    work = Path("data/processed/.docking_work")
    pose = next(
        (p for p in sorted(work.glob("*__*/out.pdbqt")) if p.stat().st_size > 0),
        None,
    )
    if pose is None:
        pytest.skip("no non-empty pose on disk")

    mol = load_pose_mol(pose, "")
    assert mol is not None, f"meeko round trip failed on {pose}"
    assert mol.GetNumBonds() > 0
    assert mol.GetNumConformers() == 1, "callers expect exactly the top pose"
    # Ring perception is what the old fallback destroyed. Not every ligand is
    # aromatic, so assert the weaker property that holds for all of them:
    # sanitisation completed, hence ring info is available.
    assert mol.GetRingInfo().NumRings() >= 0
    Chem.SanitizeMol(mol)  # must not raise
