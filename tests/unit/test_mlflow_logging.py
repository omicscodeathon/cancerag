"""`log_all_runs` must work inside the pipeline's own MLflow run.

HAL-4159. `cancerag.pipeline` opens a run for the pipeline as a whole, then
the `advanced` step calls `log_all_runs`, which opened its own *top-level*
run per CV fold. MLflow refuses a second active top-level run:

    Exception: Run with UUID ... is already active. To start a new run, first
    end the current run with mlflow.end_run(). To start a nested run, call
    start_run with nested=True

So `advanced` died on its final phase and `advanced_summary.json` was never
written -- while `optuna_trials.csv` and the tuned models sat on disk, which
made the step look like it had simply not run.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cancerag.ml import mlflow_logging as ml


@pytest.fixture
def results(tmp_path, monkeypatch) -> Path:
    """A minimal ml_models tree with one CV row and one extras row."""
    models = tmp_path / "ml_models"
    (models / "extras").mkdir(parents=True)
    pd.DataFrame([{
        "model": "lightgbm", "split": "scaffold_kfold", "fold": 0, "seed": 42,
        "macro_f1": 0.5, "macro_f1_ci_lo": 0.4, "macro_f1_ci_hi": 0.6,
        "balanced_accuracy": 0.5, "accuracy": 0.5, "n_train": 400, "n_test": 78,
    }]).to_csv(models / "cv_results_long.csv", index=False)
    pd.DataFrame([{"n_train": 100, "macro_f1": 0.3}]).to_csv(
        models / "extras" / "learning_curve.csv", index=False)
    monkeypatch.setattr(ml, "ML_MODELS", models)
    return tmp_path


@pytest.mark.unit
def test_logs_runs_standalone(results):
    """Run directly, nothing active: the runs stay top-level, as before."""
    import mlflow

    assert mlflow.active_run() is None
    out = ml.log_all_runs(mlruns_dir=results / "mlruns")
    assert out["n_runs_logged"] == 2
    assert mlflow.active_run() is None, "must not leave a run open"


@pytest.mark.unit
def test_logs_runs_inside_an_active_run(results):
    """The case that killed the advanced step.

    Before the fix this raised "Run with UUID ... is already active" on the
    first CV row.
    """
    import mlflow

    mlflow.set_tracking_uri(f"file://{(results / 'mlruns').resolve()}")
    mlflow.set_experiment("cancerag-pipeline")
    with mlflow.start_run(run_name="pipeline"):
        parent = mlflow.active_run().info.run_id
        out = ml.log_all_runs(mlruns_dir=results / "mlruns")
        assert out["n_runs_logged"] == 2
        still = mlflow.active_run()
        assert still is not None, "the outer run must survive"
        assert still.info.run_id == parent, (
            "log_all_runs must not close or replace the pipeline's run"
        )
    assert mlflow.active_run() is None


@pytest.mark.unit
def test_nested_flag_follows_whether_a_run_is_active(monkeypatch):
    """`_record` decides nesting from the live state, not a constant."""
    seen = {}

    class FakeMlflow:
        def __init__(self, active):
            self._active = active

        def active_run(self):
            return object() if self._active else None

        def start_run(self, **kw):
            seen.update(kw)
            return kw

    ml._record(FakeMlflow(active=False), run_name="x")
    assert seen["nested"] is False
    ml._record(FakeMlflow(active=True), run_name="x")
    assert seen["nested"] is True


@pytest.mark.unit
def test_every_call_site_goes_through_the_helper():
    """A sixth `mlflow.start_run` added later would reintroduce the bug."""
    src = Path(ml.__file__).read_text()
    assert "with mlflow.start_run(" not in src, (
        "open runs through _record so they nest under the pipeline's run"
    )
