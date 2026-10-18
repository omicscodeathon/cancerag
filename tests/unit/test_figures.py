"""Manuscript figure package: module contract, style helpers, and real builds.

The build tests need the gitignored run artefacts; they skip when
``data/processed/ml_models`` is absent (set ``CANCERAG_DATA_DIR`` to point at
a checkout that has them).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import figures
from scripts.figures import _style
from scripts.figures._data import Artefacts, MissingArtefact

PKG_DIR = Path(figures.__file__).parent


def test_registry_matches_modules_on_disk():
    on_disk = {p.stem for p in PKG_DIR.glob("fig*.py")}
    assert set(figures.MODULE_NAMES) == on_disk
    assert len(figures.MODULE_NAMES) == len(set(figures.MODULE_NAMES))


@pytest.mark.parametrize("name", figures.MODULE_NAMES)
def test_module_contract(name):
    mod = figures.load(name)
    assert mod.NAME == name
    assert mod.TITLE and mod.DESCRIPTION.strip()
    assert mod.SOURCES and all(isinstance(s, str) for s in mod.SOURCES)
    assert not any("preQCfix" in s for s in mod.SOURCES)
    assert callable(mod.build)


def test_superseded_run_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Artefacts(tmp_path).path("processed/ml_models_preQCfix_backup/model_meta.json")


def test_missing_artefact_is_reported(tmp_path):
    with pytest.raises(MissingArtefact):
        Artefacts(tmp_path).path("processed/ml_models/model_meta.json")


@pytest.mark.parametrize("fill,expected", [
    ("#ffffff", _style.TEXT_PRIMARY),
    ("#0d366b", "#ffffff"),
    ("#f4f8fe", _style.TEXT_PRIMARY),
])
def test_ink_on_picks_contrasting_text(fill, expected):
    assert _style.ink_on(fill) == expected


def test_class_palette_covers_every_class():
    assert set(_style.CLASS_COLOR) == set(_style.CLASS_ORDER)
    assert len(set(_style.CLASS_COLOR.values())) == len(_style.CLASS_ORDER)


_ART = Artefacts()
_HAVE_DATA = (_ART.models / "model_meta.json").exists()


@pytest.mark.slow
@pytest.mark.skipif(not _HAVE_DATA, reason="run artefacts not available")
@pytest.mark.parametrize("name", figures.MODULE_NAMES)
def test_figure_builds_from_artefacts(name, tmp_path):
    from scripts.figures._cli import build_module

    rec = build_module(figures.load(name), _ART, tmp_path)
    png, pdf = tmp_path / f"{name}.png", tmp_path / f"{name}.pdf"
    assert png.stat().st_size > 10_000
    assert pdf.read_bytes()[:4] == b"%PDF"
    assert rec["facts"], "a figure must report the key numbers it shows"
