"""The repair registry must be read, not merely kept.

HAL-4155. `data/registry/receptor_repairs.tsv` records six hand-verified
docking boxes. They are what took docking from 503 genuinely valid poses to
587: auto-detection had centred each box outside any pocket -- on a chain
absent from the chosen entry, on an `ACE` cap, on a `TYS` sulfotyrosine -- and
every pose scored about 0 kcal/mol.

Until this change nothing in `src/` or `scripts/` read the file, so a fresh
clone running the full pipeline regenerated the broken boxes. The registry
looked like configuration and behaved like documentation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cancerag.preprocessing.binding_site import (
    REPAIRS_TSV,
    BindingSite,
    ReceptorRepair,
    apply_repair,
    apply_repairs_to_file,
    load_receptor_repairs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _site(**kw) -> BindingSite:
    base = dict(
        uniprot="P41145", biasdb_name="kappa-opioid", pdb_id="8DZP",
        structure_source="pdb", method="cocrystal_ligand",
        center_x=0.0, center_y=0.0, center_z=0.0,
        size_x=22.0, size_y=22.0, size_z=22.0,
    )
    base.update(kw)
    return BindingSite(**base)


@pytest.mark.unit
class TestRegistryParsing:
    def test_the_real_registry_parses(self):
        """The shipped file, not a fixture -- a format change must fail here."""
        repairs = load_receptor_repairs(REPO_ROOT / REPAIRS_TSV)
        assert set(repairs) == {"P41145", "P41146", "P41597",
                                "P51681", "P20309", "P55085"}
        k = repairs["P41145"]
        assert k.pdb == "8DZP"
        assert k.chain == "A", "chain R is the fault being repaired"
        assert k.center == pytest.approx((129.8, 120.1, 93.8))
        assert k.confidence == "high"

    def test_par2_stays_provisional(self):
        """P55085's box was chosen by docking outcome, not from the literature.

        If this ever reads 'high' without HAL-4072 having been done, the flag
        has been laundered.
        """
        repairs = load_receptor_repairs(REPO_ROOT / REPAIRS_TSV)
        assert repairs["P55085"].confidence == "PROVISIONAL"

    def test_comment_preamble_is_skipped(self, tmp_path):
        f = tmp_path / "r.tsv"
        f.write_text(
            "# a note\n# another, with\ttabs\n"
            "uniprot\treceptor\tpdb\tchain\tbox_x\tbox_y\tbox_z\t"
            "box_source\twas\tfault\tconfidence\n"
            "P00001\tx\t1ABC\tA\t1.0\t2.0\t3.0\tcocrystal:LIG\t-\tbroken\thigh\n")
        r = load_receptor_repairs(f)["P00001"]
        assert r.center == (1.0, 2.0, 3.0)
        assert r.box_source == "cocrystal:LIG"

    def test_a_missing_column_is_refused(self, tmp_path):
        f = tmp_path / "r.tsv"
        f.write_text("uniprot\tpdb\tchain\tbox_x\tbox_y\n P1\t1ABC\tA\t1\t2\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_receptor_repairs(f)

    def test_absent_registry_is_not_fatal(self, tmp_path):
        assert load_receptor_repairs(tmp_path / "nope.tsv") == {}


@pytest.mark.unit
class TestApplyRepair:
    def test_the_registry_centre_replaces_the_detected_one(self):
        repair = ReceptorRepair(
            "P41145", "kappa-opioid", "8DZP", "A", (129.8, 120.1, 93.8),
            "cocrystal:U99", "prepared chain R, which does not exist", "high")
        out = apply_repair(_site(center_x=1.0, center_y=2.0, center_z=3.0),
                           repair, 22.0)
        assert (out.center_x, out.center_y, out.center_z) == (129.8, 120.1, 93.8)
        assert out.method == "registry_repair"
        assert out.repair_applied is True
        assert out.repair_box_source == "cocrystal:U99"
        assert "does not exist" in out.repair_fault

    def test_the_repairs_own_confidence_is_carried_through(self):
        repair = ReceptorRepair(
            "P55085", "PAR2", "8ZMD", "R", (136.458, 138.64, 140.33),
            "p2rank:pocket2", "clashing pose", "PROVISIONAL")
        out = apply_repair(_site(uniprot="P55085", confidence="ok"), repair, 22.0)
        assert out.confidence == "PROVISIONAL", (
            "a provisional box must not be normalised to ok on its way into "
            "the artefact"
        )

    def test_box_size_is_untouched(self):
        """The registry records a centre, not a size."""
        repair = ReceptorRepair("P1", "r", "1ABC", "A", (0, 0, 0), "s", "f", "high")
        out = apply_repair(_site(size_x=30.0, size_y=30.0, size_z=30.0),
                           repair, 22.0)
        assert (out.size_x, out.size_y, out.size_z) == (22.0, 22.0, 22.0)

    def test_the_registry_box_differs_from_what_detection_gave(self):
        """The regression that matters.

        If the override were silently dropped, P41145 would fall back to the
        auto-detected centre -- the one that put the box on a chain absent
        from 8DZP. These coordinates must not coincide.
        """
        repairs = load_receptor_repairs(REPO_ROOT / REPAIRS_TSV)
        detected = _site(center_x=0.0, center_y=0.0, center_z=0.0)
        out = apply_repair(detected, repairs["P41145"], 22.0)
        assert (out.center_x, out.center_y, out.center_z) != (
            detected.center_x, detected.center_y, detected.center_z)
        assert out.center_x == pytest.approx(129.8)


@pytest.mark.unit
def test_every_repaired_receptor_is_in_the_preferred_pdbs_registry():
    """A repair points at a PDB entry; the two registries must agree it exists.

    A repair naming an entry the selector no longer prefers would apply a box
    in one structure's coordinate frame to a receptor prepared from another --
    which is the exact fault recorded for P41146.
    """
    repairs = load_receptor_repairs(REPO_ROOT / REPAIRS_TSV)
    preferred_path = REPO_ROOT / "data/registry/preferred_pdbs.tsv"
    if not preferred_path.exists():
        pytest.skip("preferred_pdbs.tsv not available")
    accessions = {ln.split("\t")[0].strip()
                  for ln in preferred_path.read_text().splitlines()[1:]
                  if ln.strip()}
    unknown = sorted(set(repairs) - accessions)
    assert not unknown, f"repairs for receptors absent from preferred_pdbs: {unknown}"


def _row(uniprot, x, y, z, method="cocrystal_ligand", confidence="ok") -> dict:
    return {
        "uniprot": uniprot, "biasdb_name": uniprot, "pdb_id": "1ABC",
        "structure_source": "pdb", "method": method,
        "center_x": x, "center_y": y, "center_z": z,
        "size_x": 22.0, "size_y": 22.0, "size_z": 22.0,
        "confidence": confidence, "notes": "",
    }


def _artefact(tmp_path) -> Path:
    import json

    p = tmp_path / "binding_sites.json"
    p.write_text(json.dumps({
        "generated_at_utc": "2026-10-24T22:00:00+00:00",
        "config": {"box_size_angstrom": 22.0,
                   "agreement_threshold_angstroms": 5.0},
        "binding_sites": [_row("P41145", 1.0, 2.0, 3.0),
                          _row("P99999", 7.0, 8.0, 9.0)],
    }))
    return p


def _registry(tmp_path, body: str) -> Path:
    f = tmp_path / "r.tsv"
    f.write_text(
        "uniprot\treceptor\tpdb\tchain\tbox_x\tbox_y\tbox_z\t"
        "box_source\twas\tfault\tconfidence\n" + body)
    return f


@pytest.mark.unit
class TestApplyRepairsToFile:
    """Writing the registry into an existing artefact, without re-deriving it.

    Measured 2026-10-25: `binding_sites.json` still held the auto-detected
    centres -- P41597 and P51681 at (0,0,0) with `no_pocket_found` -- while
    the poses for those receptors score -7.321 and -6.833. The repairs were
    applied at docking time and never written back, so the file described a
    run that did not happen.
    """

    def test_only_the_named_receptor_moves(self, tmp_path):
        import json

        path = _artefact(tmp_path)
        repairs = _registry(
            tmp_path,
            "P41145\tkappa\t8DZP\tA\t129.8\t120.1\t93.8\tcocrystal:U99\t-\t"
            "box on a chain absent from 8DZP\thigh\n")

        apply_repairs_to_file(path, repairs, audit_path=None)
        got = {s["uniprot"]: s
               for s in json.loads(path.read_text())["binding_sites"]}

        assert got["P41145"]["center_x"] == pytest.approx(129.8)
        assert got["P41145"]["repair_applied"] is True
        assert got["P41145"]["method"] == "registry_repair"
        # a receptor the registry does not name keeps the box docking used --
        # re-deriving it could move it, and the poses are already on disk
        assert got["P99999"]["center_x"] == pytest.approx(7.0)
        assert got["P99999"].get("repair_applied") in (False, None)

    def test_provisional_confidence_survives_the_write(self, tmp_path):
        import json

        path = _artefact(tmp_path)
        repairs = _registry(
            tmp_path,
            "P41145\tkappa\t8DZP\tA\t1\t2\t3\tp2rank:pocket2\t-\t"
            "clashing pose\tPROVISIONAL\n")
        apply_repairs_to_file(path, repairs, audit_path=None)
        got = {s["uniprot"]: s
               for s in json.loads(path.read_text())["binding_sites"]}
        assert got["P41145"]["confidence"] == "PROVISIONAL"

    def test_a_repair_for_an_unknown_receptor_is_refused(self, tmp_path):
        path = _artefact(tmp_path)
        repairs = _registry(
            tmp_path, "P00000\tghost\t1ABC\tA\t1\t2\t3\ts\t-\tf\thigh\n")
        with pytest.raises(ValueError, match="absent from"):
            apply_repairs_to_file(path, repairs, audit_path=None)

    def test_which_repairs_were_applied_is_recorded(self, tmp_path):
        import json

        path = _artefact(tmp_path)
        repairs = _registry(
            tmp_path,
            "P41145\tkappa\t8DZP\tA\t1\t2\t3\tcocrystal:U99\t-\tf\thigh\n")
        apply_repairs_to_file(path, repairs, audit_path=None)
        cfg = json.loads(path.read_text())["config"]
        assert cfg["repairs_applied"] == ["P41145"]
        assert cfg["repairs_tsv"] == str(repairs)
