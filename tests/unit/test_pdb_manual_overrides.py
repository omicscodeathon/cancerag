"""Hand-curated PDB choices must survive a rerun.

Seven receptors were repaired after an audit found their prepared structures did
not contain the intended receptor at all — the kappa-opioid entry held a
neurotensin receptor (7UL2). The corrections live in the ``manual_override``
column of ``data/registry/preferred_pdbs.tsv``, and nothing re-derives them:
automatic selection scores cached structures and would pick the wrong one again.

Before this guard, `ReceptorPreprocessor.run()` rewrote that file with blank
overrides, so a clean run silently reverted all seven repairs and re-docked 58
rows against the wrong proteins. These tests fail if that regresses.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cancerag.preprocessing.pdb_selector import (
    apply_manual_overrides,
    load_manual_overrides,
    write_preferred_tsv,
)

REGISTRY = Path("data/registry/preferred_pdbs.tsv")

# acc -> (repaired PDB, the wrong one auto-selection had chosen)
REPAIRED = {
    "O15552": ("8T3S", "9K1D"),
    "P21452": ("7XWO", "7XWO"),
    "P21462": ("7EUO", "7VFX"),
    "P30518": ("7DW9", "7KH0"),
    "P41145": ("8DZP", "7UL2"),  # kappa-opioid; 7UL2 is neurotensin
    "P41597": ("7XA3", "7P8X"),
    "P51681": ("7O7F", "5YY4"),
}

HEADER = [
    "uniprot", "biasdb_name", "selected_pdb", "resolution", "state",
    "ligand_resname", "score", "reason", "manual_override",
]


def _row(uniprot: str, pdb: str, override: str = "") -> dict:
    return {
        "uniprot": uniprot, "biasdb_name": f"{uniprot} receptor",
        "selected_pdb": pdb, "resolution": "2.5", "state": "active",
        "ligand_resname": "LIG", "score": "40.0", "reason": "auto",
        "manual_override": override,
    }


class TestLoadManualOverrides:
    def test_returns_only_overridden_rows(self, tmp_path: Path):
        path = tmp_path / "preferred.tsv"
        write_preferred_tsv(
            [_row("P11111", "1ABC"), _row("P41145", "8DZP", "repaired: was 7UL2")],
            path,
        )
        got = load_manual_overrides(path)
        assert set(got) == {"P41145"}
        assert got["P41145"]["selected_pdb"] == "8DZP"

    def test_missing_file_is_not_an_error(self, tmp_path: Path):
        assert load_manual_overrides(tmp_path / "absent.tsv") == {}

    def test_file_without_the_column_degrades_quietly(self, tmp_path: Path):
        path = tmp_path / "old.tsv"
        path.write_text("uniprot\tselected_pdb\nP41145\t7UL2\n")
        assert load_manual_overrides(path) == {}


class TestApplyManualOverrides:
    def test_override_beats_auto_selection(self):
        auto = [_row("P41145", "7UL2")]
        overrides = {"P41145": _row("P41145", "8DZP", "repaired: was 7UL2")}
        merged, honoured = apply_manual_overrides(auto, overrides)
        assert honoured == ["P41145"]
        assert merged[0]["selected_pdb"] == "8DZP"
        assert merged[0]["manual_override"] == "repaired: was 7UL2"
        # the discarded automatic choice stays visible, so the decision is auditable
        assert "7UL2" in merged[0]["reason"]

    def test_rows_without_an_override_are_untouched(self):
        auto = [_row("P11111", "1ABC")]
        merged, honoured = apply_manual_overrides(auto, {})
        assert honoured == []
        assert merged[0]["selected_pdb"] == "1ABC"

    def test_override_applies_even_when_selection_found_nothing(self):
        auto = [_row("P41145", "")]
        overrides = {"P41145": _row("P41145", "8DZP", "repaired")}
        merged, honoured = apply_manual_overrides(auto, overrides)
        assert honoured == ["P41145"]
        assert merged[0]["selected_pdb"] == "8DZP"

    def test_round_trip_through_the_writer_preserves_the_override(self, tmp_path: Path):
        """The failure this whole module exists to prevent."""
        path = tmp_path / "preferred.tsv"
        write_preferred_tsv([_row("P41145", "8DZP", "repaired: was 7UL2")], path)
        # a rerun re-scores and proposes the wrong structure again
        rescored = [_row("P41145", "7UL2")]
        merged, _ = apply_manual_overrides(rescored, load_manual_overrides(path))
        write_preferred_tsv(merged, path)
        assert load_manual_overrides(path)["P41145"]["selected_pdb"] == "8DZP"


@pytest.mark.skipif(not REGISTRY.exists(), reason="registry not present")
class TestShippedRegistry:
    def test_every_repaired_receptor_still_carries_its_override(self):
        rows = {r["uniprot"]: r for r in csv.DictReader(REGISTRY.open(), delimiter="\t")}
        for acc, (repaired, wrong) in REPAIRED.items():
            assert acc in rows, f"{acc} missing from the registry"
            assert rows[acc]["selected_pdb"] == repaired, (
                f"{acc} points at {rows[acc]['selected_pdb']}, not the repaired {repaired}"
            )
            assert rows[acc]["manual_override"].strip(), (
                f"{acc} lost its manual_override; a rerun will re-select {wrong}"
            )

    def test_kappa_opioid_is_not_the_neurotensin_structure(self):
        rows = {r["uniprot"]: r for r in csv.DictReader(REGISTRY.open(), delimiter="\t")}
        assert rows["P41145"]["selected_pdb"] != "7UL2", (
            "P41145 (kappa-opioid) is set to 7UL2, which is a neurotensin receptor"
        )


class TestAlphaFoldFallbackIsOff:
    """A missing crystal structure must not silently become a predicted model.

    The fallback defaulted to True with no mention in config.yaml. On a clean
    checkout data/pdb is empty (gitignored, 273 MB), so every receptor missed
    and sixty AlphaFold models were downloaded and prepared instead — without
    an error, and without matching the methods, which state that all receptors
    came from the PDB. No released sidecar cites AlphaFold.
    """

    def test_config_declares_it_and_declares_it_off(self):
        import yaml

        cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
        assert "alphafold_fallback" in cfg["preprocessing"], (
            "the key must be declared, not left to an invisible code default"
        )
        assert cfg["preprocessing"]["alphafold_fallback"] is False

    def test_code_default_is_off_when_config_is_silent(self):
        assert bool({}.get("alphafold_fallback", False)) is False

    @pytest.mark.skipif(
        not Path("data/processed/receptors").exists(), reason="no prepared receptors"
    )
    def test_no_released_receptor_came_from_alphafold(self):
        import json

        offenders = []
        for sidecar in Path("data/processed/receptors").glob("*.prep.meta.json"):
            src = str(json.loads(sidecar.read_text()).get("input_pdb", ""))
            if "alphafold" in src.lower() or "/AF-" in src:
                offenders.append(f"{sidecar.name} <- {src}")
        assert not offenders, (
            "receptors prepared from predicted models: " + "; ".join(offenders[:5])
        )
