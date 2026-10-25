"""Resume must distinguish present from fresh.

HAL-4141. `StepSpec.complete()` checked only that the outputs existed, which is
deliberate -- the artefacts on disk are the source of truth, not a framework
cache. But presence is not freshness.

On 2026-10-25 `ml_ready_dataset.parquet` was written at 22:32 from interaction
fingerprints that were replaced at 01:18 and corrected again at 05:52. `--list`
kept reporting `dataset_assembly  yes`, so every later run skipped the rebuild
and trained on a feature matrix where 215 columns no longer existed and 281
current ones were absent -- about 40% wrong. Nothing in any run said so.
"""

from __future__ import annotations

import os

import pytest

from cancerag import pipeline as pl


def _touch(p, when: float):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    os.utime(p, (when, when))
    return p


def _spec(name, outputs, inputs=(), body=None):
    s = pl.StepSpec(name, name, list(outputs), body or (lambda: None))
    s.inputs = list(inputs)
    return s


@pytest.mark.unit
class TestStaleness:
    def test_the_dataset_assembly_case(self, tmp_path):
        """The exact shape of the incident, in miniature."""
        fingerprints = _touch(tmp_path / "interaction_fingerprints.parquet", 5_000)
        dataset = _touch(tmp_path / "ml_ready_dataset.parquet", 1_000)

        step = _spec("dataset_assembly", [dataset], [fingerprints])
        assert step.exists(), "the output is on disk, as it was then"
        assert step.stale(), "and it predates the fingerprints it was built from"
        assert not step.complete(), "so resume must not skip it"
        assert step.state() == "stale"

    def test_fresh_output_is_complete(self, tmp_path):
        src = _touch(tmp_path / "in.parquet", 1_000)
        out = _touch(tmp_path / "out.parquet", 5_000)
        step = _spec("s", [out], [src])
        assert step.complete()
        assert step.state() == "yes"

    def test_missing_output_is_not_stale(self, tmp_path):
        src = _touch(tmp_path / "in.parquet", 1_000)
        step = _spec("s", [tmp_path / "absent.parquet"], [src])
        assert not step.exists()
        assert not step.stale(), "absent is 'no', not 'stale'"
        assert step.state() == "no"

    def test_no_declared_inputs_is_never_stale(self, tmp_path):
        """The first step in the chain has nothing upstream to be older than."""
        out = _touch(tmp_path / "out.csv", 1_000)
        step = _spec("curation", [out], [])
        assert step.complete()
        assert step.state() == "yes"

    def test_the_oldest_output_decides(self, tmp_path):
        """One fresh output must not mask a stale sibling.

        A step writing several artefacts can rewrite one and leave another, so
        the comparison has to be against the oldest.
        """
        src = _touch(tmp_path / "in.parquet", 3_000)
        fresh = _touch(tmp_path / "fresh.json", 9_000)
        old = _touch(tmp_path / "old.csv", 1_000)
        assert _spec("s", [fresh, old], [src]).stale()
        assert not _spec("s", [fresh], [src]).stale()

    def test_the_newest_input_decides(self, tmp_path):
        src_old = _touch(tmp_path / "a.parquet", 1_000)
        src_new = _touch(tmp_path / "b.parquet", 9_000)
        out = _touch(tmp_path / "out.parquet", 5_000)
        assert _spec("s", [out], [src_old, src_new]).stale()
        assert not _spec("s", [out], [src_old]).stale()


@pytest.mark.unit
class TestDirectoryArtefacts:
    def test_a_file_deep_in_a_directory_counts(self, tmp_path):
        """A directory's own mtime does not move when a nested file is rewritten.

        `receptors` and `pdb` are directory outputs, so staleness has to look
        inside them or it silently never fires for those steps.
        """
        d = tmp_path / "receptors"
        _touch(d / "a" / "b" / "P41145.pdb", 9_000)
        os.utime(d, (1_000, 1_000))
        out = _touch(tmp_path / "binding_sites.json", 5_000)
        assert pl.newest_mtime([d]) == 9_000
        assert _spec("binding_sites", [out], [d]).stale()

    def test_newest_mtime_ignores_absent_paths(self, tmp_path):
        present = _touch(tmp_path / "there.csv", 4_000)
        assert pl.newest_mtime([tmp_path / "nope.csv", present]) == 4_000
        assert pl.newest_mtime([tmp_path / "nope.csv"]) is None

    def test_empty_directory_falls_back_to_its_own_mtime(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        os.utime(d, (7_000, 7_000))
        assert pl.newest_mtime([d]) == 7_000


@pytest.mark.unit
class TestChainWiring:
    def test_every_step_takes_the_previous_step_s_outputs(self):
        specs = pl.build_specs()
        assert specs[0].inputs == [], "nothing precedes the first step"
        for prev, nxt in zip(specs, specs[1:]):
            assert nxt.inputs == list(prev.outputs), (
                f"{nxt.name} should read {prev.name}'s outputs"
            )

    def test_staleness_propagates_down_the_chain(self, tmp_path):
        """Re-running an early step must invalidate everything after it.

        This is what makes wiring inputs from the chain sufficient: the
        fingerprints moving has to reach model_training, several steps later.
        """
        a_out = _touch(tmp_path / "a.csv", 9_000)   # just rebuilt
        b_out = _touch(tmp_path / "b.csv", 2_000)   # built before that
        c_out = _touch(tmp_path / "c.csv", 3_000)   # and so was this

        b = _spec("b", [b_out], [a_out])
        c = _spec("c", [c_out], [b_out])
        assert b.stale(), "b was built before a was rebuilt"
        # c is not yet stale against b's current file -- it becomes stale once
        # b reruns and rewrites it, which is exactly the propagation.
        assert not c.stale()
        os.utime(b_out, (10_000, 10_000))
        assert c.stale(), "rebuilding b must invalidate c"

    def test_list_state_is_one_of_three(self):
        for spec in pl.build_specs():
            assert spec.state() in {"yes", "no", "stale"}
