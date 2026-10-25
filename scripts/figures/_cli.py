"""Command-line plumbing shared by ``python -m scripts.figures`` and each module."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

from . import _style
from ._data import DEFAULT_OUT, Artefacts, MissingArtefact, default_data_dir


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"output directory (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--data", type=Path, default=None,
        help=f"data root holding processed/ and holdout/ (default: {default_data_dir()})",
    )
    return p


def build_module(mod: ModuleType, art: Artefacts, out: Path) -> dict:
    """Build one figure module; returns its record (paths + key numbers)."""
    _style.apply()
    fig, facts = mod.build(art)
    paths = _style.save(fig, mod.NAME, out)
    # A module whose inputs depend on a decision the run made -- which learner
    # won, say -- cannot name them in a constant. Those define `sources(art)`;
    # everything else keeps the plain `SOURCES` list.
    sources = list(mod.sources(art) if hasattr(mod, "sources") else mod.SOURCES)
    dates = {
        s: datetime.fromtimestamp(art.path(s).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        for s in sources
    }
    return {
        "name": mod.NAME,
        "title": mod.TITLE,
        "description": mod.DESCRIPTION.strip(),
        "sources": sources,
        "source_dates": dates,
        "facts": facts,
        "files": [p.name for p in paths],
    }


def run_one(mod: ModuleType) -> int:
    args = parser(f"Build {mod.NAME}").parse_args()
    art = Artefacts(args.data) if args.data else Artefacts()
    try:
        rec = build_module(mod, art, args.out)
    except MissingArtefact as exc:
        print(f"[skip] {mod.NAME}: {exc}", file=sys.stderr)
        return 2
    print(f"[ok]   {mod.NAME} -> {args.out}")
    print(json.dumps(rec["facts"], indent=2, ensure_ascii=False, default=str))
    return 0
