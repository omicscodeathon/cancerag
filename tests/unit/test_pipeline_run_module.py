"""A step body must not see the pipeline's command line.

HAL-4194. `_run_module` calls each step's entrypoint in-process. Those are
written as `main(argv=None)` and fall through to `parse_args(argv)`; with
`argv=None` argparse reads `sys.argv[1:]`, which belongs to the pipeline.

Running `--from dataset_assembly --force --no-zenml` therefore killed the
clustering step with

    pipeline.py: error: unrecognized arguments: --from dataset_assembly --force

one step from the end of a four-hour chain. It had never shown up because the
systemd unit invoked the pipeline with no arguments at all, so `sys.argv[1:]`
was empty and argparse was satisfied -- every historical run took the single
path that works.
"""

from __future__ import annotations

import argparse
import sys
import types

import pytest

from cancerag import pipeline as pl


def _module_that_parses_args(name: str) -> types.ModuleType:
    """A stand-in for the real step entrypoints: `main(argv=None)` + argparse."""
    mod = types.ModuleType(name)

    def main(argv=None):
        ap = argparse.ArgumentParser(description=name)
        ap.add_argument("--dataset", default="d")
        ap.add_argument("--no-umap", action="store_true")
        mod.parsed = ap.parse_args(argv)
        mod.argv_seen = list(sys.argv)
        return mod.parsed

    mod.main = main
    sys.modules[name] = mod
    return mod


@pytest.fixture
def pipeline_argv(monkeypatch):
    """sys.argv as it looks mid-run with flags, which is what broke it."""
    monkeypatch.setattr(
        sys, "argv",
        ["pipeline.py", "--from", "dataset_assembly", "--force", "--no-zenml"],
    )


@pytest.mark.unit
def test_step_with_argparse_survives_pipeline_flags(pipeline_argv):
    name = "_hal4194_ok"
    mod = _module_that_parses_args(name)
    try:
        pl._run_module(name)           # raised SystemExit(2) before the fix
    finally:
        sys.modules.pop(name, None)
    assert mod.parsed.dataset == "d", "the step should get its own defaults"
    assert mod.argv_seen == [name], (
        "the step must not be shown the pipeline's command line"
    )


@pytest.mark.unit
def test_argv_is_restored_afterwards(pipeline_argv):
    name = "_hal4194_restore"
    _module_that_parses_args(name)
    before = list(sys.argv)
    try:
        pl._run_module(name)
    finally:
        sys.modules.pop(name, None)
    assert sys.argv == before


@pytest.mark.unit
def test_argv_is_restored_when_the_step_raises(pipeline_argv):
    """A failing step must not leave argv clobbered for the steps after it."""
    name = "_hal4194_boom"
    mod = types.ModuleType(name)

    def main(argv=None):
        raise RuntimeError("step failed")

    mod.main = main
    sys.modules[name] = mod
    before = list(sys.argv)
    try:
        with pytest.raises(RuntimeError, match="step failed"):
            pl._run_module(name)
    finally:
        sys.modules.pop(name, None)
    assert sys.argv == before


@pytest.mark.unit
def test_kwargs_still_reach_the_entrypoint(pipeline_argv):
    """Steps that take real keyword arguments must keep working."""
    name = "_hal4194_kwargs"
    mod = types.ModuleType(name)
    seen = {}

    def run_all(*, workers=None):
        seen["workers"] = workers

    mod.run_all = run_all
    sys.modules[name] = mod
    try:
        pl._run_module(name, "run_all", workers=8)
    finally:
        sys.modules.pop(name, None)
    assert seen == {"workers": 8}


@pytest.mark.unit
@pytest.mark.parametrize("module", [
    "cancerag.ml.unsupervised_clustering",
    "cancerag.preprocessing.chain_verification",
    "cancerag.preprocessing.binding_site",
])
def test_the_exposed_modules_still_parse_args(module):
    """These three call parse_args; the guard is what makes them safe.

    If one stops doing so the guard is merely redundant, not wrong -- but the
    list in the issue should then be updated, so flag it here.
    """
    import importlib
    from pathlib import Path

    src = Path(importlib.import_module(module).__file__).read_text()
    assert "parse_args" in src, (
        f"{module} no longer parses args; update HAL-4194's exposure list"
    )
