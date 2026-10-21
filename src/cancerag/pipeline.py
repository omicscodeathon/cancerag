"""The CancerAg pipeline as one connected, resumable chain.

Before this module the pipeline existed as eighteen commands run by hand in an
order recorded nowhere, with `cancerag.main` covering the first few and crashing
on the rest. This module declares the whole chain once: what each step runs,
what it must produce, and what to measure about it afterwards.

Three properties it is built for.

**Durable.** Every step declares its output artefacts. A step whose outputs are
all present is skipped, so a run that dies at hour nine resumes at hour nine.
Resume is decided by artefacts on disk, not by a status flag that can outlive
the file it describes.

**Observable.** Every step reports how long it took, what it wrote, and a few
facts specific to it — rows curated, pairs docked, columns after the BW merge,
macro-F1 by split. The probe runs after the step and reads the artefact, so the
number is measured, never asserted. Results stream to a JSONL and a status JSON
as they happen, so a twelve-hour run is legible while it is still running.

**Two ways to drive it.** The step bodies are plain functions. ZenML wraps them
for the DAG and the dashboard; `--no-zenml` runs the identical sequence through
a bare loop. If the orchestrator fails mid-run the fallback picks up from the
completed artefacts, which is the whole point of putting resume in the steps
rather than in the framework.

    python -m cancerag.pipeline --list
    python -m cancerag.pipeline --no-zenml --workers 32
    python -m cancerag.pipeline --only docking --force
    python -m cancerag.pipeline --from featurisation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Paths are resolved against the data root so the same code runs on a laptop and
# on a VM with the data on an attached disk.
DATA = Path(os.environ.get("CANCERAG_DATA_DIR", "data")).resolve()
PROC = DATA / "processed"
HOLDOUT = DATA / "holdout"
MODELS = PROC / "ml_models"
RUN_DIR = Path(os.environ.get("CANCERAG_RUN_DIR", PROC / "pipeline_runs")).resolve()

# How many docking jobs run at once, and how many threads each gets. Vina scales
# sub-linearly past a few threads, so throughput on independent jobs favours more
# jobs over fatter ones; the default is one thread each across every core.
N_WORKERS = int(os.environ.get("CANCERAG_WORKERS", os.cpu_count() or 4))
THREADS_PER_JOB = int(os.environ.get("CANCERAG_THREADS_PER_JOB", 1))


# --------------------------------------------------------------------- results


@dataclass
class StepResult:
    """What one step did. Written to the run log and surfaced in the dashboard."""

    name: str
    status: str  # ran | skipped | failed
    seconds: float = 0.0
    outputs: dict[str, int] = field(default_factory=dict)  # artefact -> bytes
    metrics: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    started_at: str = ""
    finished_at: str = ""

    def line(self) -> str:
        mark = {"ran": "\033[1;32m✓\033[0m", "skipped": "\033[1;33m•\033[0m",
                "failed": "\033[1;31m✗\033[0m"}.get(self.status, "?")
        bits = " ".join(f"{k}={v}" for k, v in list(self.metrics.items())[:4])
        dur = f"{self.seconds / 60:.1f}m" if self.seconds >= 60 else f"{self.seconds:.1f}s"
        return f"{mark} {self.name:<24} {self.status:<8} {dur:>7}  {bits}"


# ----------------------------------------------------------------------- probes
#
# A probe reads a step's artefact after it runs and returns a few facts worth
# seeing. Probes never raise: a broken probe must not fail a good step, so each
# one is wrapped. They are the difference between "stage 7 finished" and "stage 7
# docked 521 of 587 pairs, 66 failures, median affinity -8.5".


def _safe(fn: Callable[[], dict]) -> dict:
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - telemetry must not break the run
        return {"probe_error": f"{type(exc).__name__}: {exc}"[:120]}


def _csv_rows(path: Path) -> int:
    with path.open() as fh:
        return max(0, sum(1 for _ in fh) - 1)


def probe_curation() -> dict:
    import pandas as pd

    df = pd.read_csv(PROC / "unified_ligands.csv")
    return {
        "measurements": len(df),
        "molecules": df["inchikey"].nunique(),
        "receptors": df["receptor_uniprot"].nunique(),
    }


def probe_receptors() -> dict:
    pdbs = list((PROC / "receptors").glob("*.pdb"))
    return {"prepared": len({p.stem.split(".")[0] for p in pdbs})}


def probe_binding_sites() -> dict:
    d = json.loads((PROC / "binding_sites.json").read_text())
    recs = d.get("binding_sites", [])
    methods: dict[str, int] = {}
    for r in recs:
        methods[r.get("method", "?")] = methods.get(r.get("method", "?"), 0) + 1
    return {"sites": len(recs), **{f"method_{k}": v for k, v in methods.items()}}


def probe_redock() -> dict:
    d = json.loads((PROC / "redock_validation.json").read_text())
    t = d.get("totals", {})
    return {"attempted": t.get("attempted"), "passed": t.get("passed")}


def probe_docking() -> dict:
    import pandas as pd

    df = pd.read_csv(PROC / "docking_features.csv")
    ok = df["vina_affinity_best"].notna()
    return {
        "pairs": len(df),
        "succeeded": int(ok.sum()),
        "failed": int((~ok).sum()),
        "median_affinity": round(float(df.loc[ok, "vina_affinity_best"].median()), 3),
    }


def probe_featurisation() -> dict:
    import pandas as pd

    ifp = pd.read_parquet(PROC / "interaction_fingerprints.parquet")
    contact = [c for c in ifp.columns if c.startswith("ifp_")]
    bw = [c for c in contact if c.startswith("ifp_BW")]
    return {
        "contact_columns": len(contact),
        "bw_mapped": len(bw),
        "raw_retained": len(contact) - len(bw),
    }


def probe_dataset() -> dict:
    import pandas as pd

    tr = pd.read_parquet(PROC / "ml_ready_dataset.parquet")
    ho = pd.read_parquet(HOLDOUT / "dataset_holdout.parquet")
    return {"train_rows": len(tr), "holdout_rows": len(ho), "columns": tr.shape[1]}


def probe_training() -> dict:
    d = json.loads((MODELS / "model_meta.json").read_text())
    mf = d.get("holdout_metrics", {}).get("macro_f1", {})
    sel = json.loads((MODELS / "selection_decision.json").read_text())
    return {
        "holdout_macro_f1": round(float(mf.get("point_estimate", float("nan"))), 4),
        "holdout_ci_lo": round(float(mf.get("ci_lo", float("nan"))), 4),
        "holdout_ci_hi": round(float(mf.get("ci_hi", float("nan"))), 4),
        "n_features": d.get("n_features"),
        "chosen": sel.get("chosen"),
    }


def probe_shap() -> dict:
    import pandas as pd

    df = pd.read_csv(MODELS / "interpretability" / "shap_cross_regime.csv")
    cols = [c for c in df.columns if c.startswith("sf_")]
    robust = df[(df[cols] >= 0.5).all(axis=1)]
    return {"features_scored": len(df), "robust_all_regimes": len(robust)}


def probe_selector() -> dict:
    import pandas as pd

    df = pd.read_csv(MODELS / "selector_ablation" / "selector_ablation_summary.csv")
    best = df.groupby("selector")["mean"].mean().idxmax()
    return {"cells": len(df), "best_selector": best}


def probe_clustering() -> dict:
    d = json.loads((MODELS / "clustering" / "clustering_results.json").read_text())
    return {"k_star": d.get("k_star"), "n_features": d.get("n_features_used")}


def probe_figures() -> dict:
    out = Path(os.environ.get("CANCERAG_FIG_DIR", "manuscript/figures"))
    return {"png": len(list(out.glob("*.png"))), "pdf": len(list(out.glob("*.pdf")))}


# ------------------------------------------------------------------ step bodies
#
# Each body calls the existing entrypoint. Nothing here reimplements science; the
# chain's job is ordering, resume and telemetry.


def _run_module(module: str, func: str = "main", **kwargs) -> None:
    mod = __import__(module, fromlist=[func])
    getattr(mod, func)(**kwargs)


def body_curation() -> None:
    import yaml
    from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    LigandPreprocessor(cfg).run()


def body_fetch_structures() -> None:
    """Download the crystal structures the receptor step prepares from.

    Without this, `data/pdb` is empty on a fresh clone, `select_best_pdb` finds
    nothing, and receptor preparation falls through to its AlphaFold fallback
    for every receptor — silently, because the fallback is enabled by default.
    The run completes and the structures are plausible, but they are predicted
    models rather than the crystal structures the published pipeline used.
    """
    import pandas as pd
    import yaml
    from cancerag.data_collection import receptor_retriever

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    unified = PROC / "unified_ligands.csv"
    restrict = None
    if unified.exists():
        restrict = set(pd.read_csv(unified)["receptor_uniprot"].dropna().unique())
    # from_config is the supported constructor; the direct __init__ needs a
    # registry, and main.py:237 passes arguments the signature does not have.
    retriever = receptor_retriever.ReceptorRetriever.from_config(
        output_dir=DATA / "pdb", config=cfg,
    )
    retriever.run(restrict_to=restrict)


def probe_structures() -> dict:
    pdb_dir = DATA / "pdb"
    files = list(pdb_dir.rglob("*.pdb")) + list(pdb_dir.rglob("*.cif"))
    return {"receptors_with_structures": len([d for d in pdb_dir.iterdir() if d.is_dir()]),
            "structure_files": len(files)}


def probe_receptor_sources() -> dict:
    """Crystal structure or predicted model? The distinction the paper rests on."""
    import json as _json

    sidecars = list((PROC / "receptors").glob("*.prep.meta.json"))
    sources = {"pdb": 0, "alphafold": 0, "unknown": 0}
    for sc in sidecars:
        try:
            src = str(_json.loads(sc.read_text()).get("input_pdb", ""))
        except Exception:  # noqa: BLE001
            sources["unknown"] += 1
            continue
        if "alphafold" in src.lower() or "/AF-" in src:
            sources["alphafold"] += 1
        elif src:
            sources["pdb"] += 1
        else:
            sources["unknown"] += 1
    return {"prepared": len(sidecars), **{f"from_{k}": v for k, v in sources.items()}}


def body_receptors() -> None:
    import yaml
    from cancerag.preprocessing import receptor_preprocessor

    cfg = yaml.safe_load(Path("configs/config.yaml").read_text())
    receptor_preprocessor.ReceptorPreprocessor(cfg).run()


def body_docking() -> None:
    from cancerag.docking.production_docking import main as dock_main

    dock_main(workers=N_WORKERS, cpu_threads_per_job=THREADS_PER_JOB)


# ------------------------------------------------------------------- step specs


@dataclass
class StepSpec:
    name: str
    description: str
    outputs: list[Path]
    body: Callable[[], None]
    probe: Callable[[], dict] | None = None
    optional: bool = False  # a failure here does not stop the chain

    def complete(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)


def build_specs() -> list[StepSpec]:
    """The chain, in dependency order."""
    return [
        StepSpec(
            "curation", "BiasDB → curated measurements under the six-field key",
            [PROC / "unified_ligands.csv", PROC / "attrition.json"],
            body_curation, probe_curation,
        ),
        StepSpec(
            "fetch_structures", "download crystal structures from the RCSB PDB",
            [DATA / "pdb"], body_fetch_structures, probe_structures,
        ),
        StepSpec(
            "receptors", "PDB → prepared, chain-gated receptor structures",
            [PROC / "receptors"], body_receptors, probe_receptor_sources,
        ),
        StepSpec(
            "binding_sites", "cocrystal centroid, else P2Rank/fpocket consensus",
            [PROC / "binding_sites.json"],
            lambda: _run_module("cancerag.preprocessing.binding_site"),
            probe_binding_sites,
        ),
        StepSpec(
            "redock_validation", "re-dock the cocrystal ligand, RMSD vs crystal pose",
            [PROC / "redock_validation.json"],
            lambda: _run_module("cancerag.preprocessing.redock_validation", "redock_all"),
            probe_redock,
        ),
        StepSpec(
            "gnina_rescore", "CNN pose score for each re-dock",
            [PROC / "gnina_rescore.json"],
            lambda: _run_module("cancerag.preprocessing.gnina_rescore",
                                "rescore_existing_redock_runs"),
            optional=True,
        ),
        StepSpec(
            "chain_verification", "every prepared chain aligned to its canonical UniProt",
            [PROC / "receptor_chain_verification.json"],
            lambda: _run_module("cancerag.preprocessing.chain_verification"),
        ),
        StepSpec(
            "docking", "AutoDock Vina over every unique (ligand, receptor) pair",
            [PROC / "docking_features.csv"], body_docking, probe_docking,
        ),
        StepSpec(
            "featurisation", "descriptors, fingerprints, pose shape, ProLIF → BW positions",
            [PROC / "interaction_fingerprints.parquet", PROC / "ligand_features.parquet",
             PROC / "bw_column_map.json"],
            lambda: _run_module("cancerag.features.featurization"),
            probe_featurisation,
        ),
        StepSpec(
            "dataset_assembly", "join the blocks, label, weight, split by year",
            [PROC / "ml_ready_dataset.parquet", HOLDOUT / "dataset_holdout.parquet",
             PROC / "ml_splits.json"],
            lambda: _run_module("cancerag.ml.dataset_assembly"),
            probe_dataset,
        ),
        StepSpec(
            "ml_preprocessing", "fold-internal impute, variance, correlation, scale",
            [PROC / "ml_preprocessed" / "preprocessing_meta.json"],
            lambda: _run_module("cancerag.ml.preprocessing", "run_preprocessing"),
        ),
        StepSpec(
            "model_training", "four learners × three splits × five seeds, then the holdout",
            [MODELS / "model_meta.json", MODELS / "selection_decision.json",
             MODELS / "cv_results_long.csv"],
            lambda: _run_module("cancerag.ml.model_training", "run_model_training"),
            probe_training,
        ),
        StepSpec(
            "chemberta", "frozen ChemBERTa embeddings + logistic head",
            [MODELS / "baselines" / "chemberta" / "chemberta_summary.json"],
            lambda: _run_module("cancerag.ml.baselines", "run_chemberta_baseline"),
            optional=True,
        ),
        StepSpec(
            "advanced", "Optuna, per-class thresholds, stacking ensemble, MLflow",
            [MODELS / "advanced" / "advanced_summary.json"],
            lambda: _run_module("cancerag.ml.advanced", "run_all"),
            optional=True,
        ),
        StepSpec(
            "interpretability", "SHAP stability and permutation importance",
            [MODELS / "interpretability" / "interpretability_report.md"],
            lambda: _run_module("cancerag.ml.interpretability"),
        ),
        StepSpec(
            "shap_cross_regime", "SHAP under all four regimes + block attribution",
            [MODELS / "interpretability" / "shap_cross_regime.csv",
             MODELS / "interpretability" / "shap_block_attribution.csv"],
            lambda: _script("scripts/shap_cross_regime.py"), probe_shap,
        ),
        StepSpec(
            "reviewer_extras", "learning curve, LORO, temporal shift, permutation test",
            [MODELS / "extras" / "learning_curve.csv", MODELS / "extras" / "loro_results.csv"],
            lambda: _run_module("cancerag.ml.reviewer_extras"),
        ),
        StepSpec(
            "block_ablation", "drop the receptor-dependent and pose blocks, three regimes",
            [MODELS / "extras" / "ablation_by_block.csv"],
            lambda: _script("scripts/ablation_by_block.py"),
        ),
        StepSpec(
            "selector_ablation", "four selectors × four learners, force-keeping off",
            [MODELS / "selector_ablation" / "selector_ablation_summary.csv"],
            lambda: _script("scripts/run_selector_ablation.py"), probe_selector,
        ),
        StepSpec(
            "clustering", "unsupervised check that the classes are not already separable",
            [MODELS / "clustering" / "clustering_results.json"],
            lambda: _run_module("cancerag.ml.unsupervised_clustering"),
            probe_clustering,
        ),
        StepSpec(
            "figures", "every manuscript figure, regenerated from the artefacts",
            [], lambda: _module_cli("scripts.figures"), probe_figures,
        ),
    ]


def _script(path: str) -> None:
    subprocess.run([sys.executable, path], check=True, env=_child_env())


def _module_cli(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], check=True, env=_child_env())


def _child_env() -> dict:
    """Child processes inherit thread pinning, or they oversubscribe the box.

    A step that fans out across cores while every BLAS call inside it also grabs
    every core produces threads² of contention and runs slower than serial. The
    pipeline decides the parallelism; the libraries get one thread each.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", "src")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    return env


# -------------------------------------------------------------------- execution


class MlflowTelemetry:
    """Every step as a nested MLflow run, not just the advanced ones.

    MLflow was already in this project, but wired only into `ml.advanced`, so the
    audit trail covered Optuna and stacking and nothing before them. Curation,
    docking and featurisation — the stages where most of the wall clock and most
    of the risk live — left no trace. Here each step opens a nested run under one
    parent per pipeline execution, so `mlflow ui` shows the whole chain: what ran,
    how long it took, and what the probe measured about its output.

    Telemetry must never be the thing that fails a twelve-hour run, so every call
    is guarded and a broken tracker degrades to no tracking.
    """

    def __init__(self, run_id: str, *, enabled: bool = True) -> None:
        self.active = False
        self._mlflow = None
        if not enabled:
            return
        try:
            import mlflow

            uri = os.environ.get("MLFLOW_TRACKING_URI", f"file://{MODELS / 'mlruns'}")
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("cancerag-pipeline")
            mlflow.start_run(run_name=run_id)
            mlflow.set_tags({
                "host": socket.gethostname(),
                "workers": str(N_WORKERS),
                "data_dir": str(DATA),
                "git_sha": _git_sha(),
            })
            self._mlflow = mlflow
            self.active = True
            logger.info("mlflow tracking at %s", uri)
        except Exception as exc:  # noqa: BLE001 - tracking is never load-bearing
            logger.warning("mlflow disabled: %s", exc)

    def step(self, result: StepResult, description: str) -> None:
        if not self.active:
            return
        try:
            with self._mlflow.start_run(run_name=result.name, nested=True):
                self._mlflow.set_tags({"status": result.status, "step": result.name,
                                       "description": description})
                self._mlflow.log_metric("seconds", result.seconds)
                for key, value in result.metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self._mlflow.log_metric(key, float(value))
                    else:
                        self._mlflow.log_param(key, str(value)[:250])
                for name, size in result.outputs.items():
                    self._mlflow.log_param(f"out_{name}"[:250], size)
                if result.error:
                    self._mlflow.log_text(result.error, f"{result.name}_traceback.txt")
        except Exception as exc:  # noqa: BLE001
            logger.warning("mlflow step log failed for %s: %s", result.name, exc)

    def close(self, results: list[StepResult]) -> None:
        if not self.active:
            return
        try:
            ran = [r for r in results if r.status == "ran"]
            self._mlflow.log_metric("steps_total", len(results))
            self._mlflow.log_metric("steps_ran", len(ran))
            self._mlflow.log_metric("steps_skipped",
                                    len([r for r in results if r.status == "skipped"]))
            self._mlflow.log_metric("wall_hours", sum(r.seconds for r in results) / 3600)
            self._mlflow.end_run()
        except Exception as exc:  # noqa: BLE001
            logger.warning("mlflow close failed: %s", exc)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class RunLog:
    """Streams step results to disk as they happen, for live monitoring."""

    def __init__(self, run_id: str) -> None:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.jsonl = RUN_DIR / f"{run_id}.jsonl"
        self.status = RUN_DIR / "status.json"
        self.results: list[StepResult] = []
        self.started = time.time()

    def record(self, result: StepResult) -> None:
        self.results.append(result)
        with self.jsonl.open("a") as fh:
            fh.write(json.dumps(asdict(result)) + "\n")
        self._write_status(current=None)

    def _write_status(self, current: str | None) -> None:
        payload = {
            "run_id": self.run_id,
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "workers": N_WORKERS,
            "started_at": datetime.fromtimestamp(self.started, timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - self.started, 1),
            "current_step": current,
            "steps": [asdict(r) for r in self.results],
        }
        tmp = self.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(self.status)  # atomic, so a reader never sees half a file

    def mark_running(self, name: str) -> None:
        self._write_status(current=name)


def execute(specs: list[StepSpec], *, force: bool = False,
            run_log: RunLog | None = None,
            telemetry: "MlflowTelemetry | None" = None) -> list[StepResult]:
    results: list[StepResult] = []
    for spec in specs:
        started = datetime.now(timezone.utc)
        if run_log:
            run_log.mark_running(spec.name)

        if spec.complete() and not force:
            res = StepResult(spec.name, "skipped",
                             started_at=started.isoformat(),
                             finished_at=started.isoformat(),
                             metrics=_safe(spec.probe) if spec.probe else {})
            results.append(res)
            print(res.line(), flush=True)
            if run_log:
                run_log.record(res)
            if telemetry:
                telemetry.step(res, spec.description)
            continue

        t0 = time.perf_counter()
        try:
            spec.body()
            status, error = "ran", None
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised below
            status = "failed"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
        seconds = time.perf_counter() - t0

        res = StepResult(
            spec.name, status, round(seconds, 2),
            outputs={p.name: (p.stat().st_size if p.is_file() else -1)
                     for p in spec.outputs if p.exists()},
            metrics=_safe(spec.probe) if (spec.probe and status == "ran") else {},
            error=error,
            started_at=started.isoformat(),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        results.append(res)
        print(res.line(), flush=True)
        if run_log:
            run_log.record(res)
        if telemetry:
            telemetry.step(res, spec.description)

        if status == "failed":
            if spec.optional:
                print(f"  \033[1;33m{spec.name} is optional — continuing\033[0m", flush=True)
                continue
            raise RuntimeError(f"step {spec.name} failed:\n{error}")
    return results


# ------------------------------------------------------------------------ zenml


def run_with_zenml(specs: list[StepSpec], force: bool,
                   telemetry: "MlflowTelemetry | None" = None) -> None:
    # Without this the client writes to its own SQLite store and the run never
    # appears in the dashboard the server is serving.
    os.environ.setdefault("ZENML_STORE_URL",
                          os.environ.get("ZENML_SERVER_URL", "http://127.0.0.1:8237"))
    """Same chain, wrapped for the DAG and the dashboard.

    Resume stays in the step bodies rather than in ZenML's cache: the artefacts
    on disk are the source of truth, so a step is skipped because its output
    exists, not because a framework believes nothing changed.
    """
    import cancerag.pipeline as selfmod
    from zenml import pipeline as zen_pipeline, step as zen_step
    from zenml import log_metadata

    def make(spec: StepSpec):
        def _fn() -> None:
            res = execute([spec], force=force, telemetry=telemetry)[0]
            log_metadata(metadata={
                "status": res.status,
                "seconds": res.seconds,
                **{f"metric_{k}": v for k, v in res.metrics.items()},
            })

        _fn.__name__ = spec.name
        _fn.__doc__ = spec.description
        # ZenML resolves a step by importing its module and looking the function
        # up by name. A closure is not reachable that way, and under
        # `python -m cancerag.pipeline` the running module is __main__, which
        # ZenML cannot import — so it raised
        # "module '__main__' has no attribute 'curation'". Publish each
        # generated step on the real, importable module and point it there.
        _fn.__module__ = selfmod.__name__
        _fn.__qualname__ = spec.name
        zstep = zen_step(name=spec.name, enable_cache=False)(_fn)
        setattr(selfmod, spec.name, zstep)
        return zstep

    zen_steps = [make(s) for s in specs]

    @zen_pipeline(name="cancerag", enable_cache=False)
    def cancerag_pipeline() -> None:
        previous = None
        for zs in zen_steps:
            # after= makes the DAG a chain rather than a fan-out; the stages are
            # genuinely sequential and running them in parallel would corrupt
            # shared artefacts.
            previous = zs(after=previous) if previous is not None else zs()

    cancerag_pipeline.__module__ = selfmod.__name__
    setattr(selfmod, "cancerag_pipeline", cancerag_pipeline)
    cancerag_pipeline()


# -------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="show the chain and what is done")
    p.add_argument("--no-zenml", action="store_true", help="bare runner, no orchestrator")
    p.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    p.add_argument("--only", nargs="+", metavar="STEP", help="run just these steps")
    p.add_argument("--from", dest="from_step", metavar="STEP", help="start here")
    p.add_argument("--skip", nargs="+", metavar="STEP", default=[], help="skip these")
    p.add_argument("--workers", type=int, help="docking workers (default: all cores)")
    p.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    if args.workers:
        global N_WORKERS
        N_WORKERS = args.workers

    specs = build_specs()

    if args.list:
        print(f"{'step':<24} {'done':<6} outputs")
        for s in specs:
            print(f"{s.name:<24} {'yes' if s.complete() else 'no':<6} "
                  f"{', '.join(p.name for p in s.outputs) or '—'}")
            print(f"{'':24} {s.description}")
        return 0

    if args.only:
        specs = [s for s in specs if s.name in args.only]
    if args.from_step:
        names = [s.name for s in specs]
        if args.from_step not in names:
            p.error(f"unknown step {args.from_step!r}; see --list")
        specs = specs[names.index(args.from_step):]
    specs = [s for s in specs if s.name not in args.skip]
    if not specs:
        p.error("no steps selected")

    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    print(f"\ncancerag pipeline · {run_id} · {len(specs)} steps · "
          f"{N_WORKERS} workers · data {DATA}\n")

    if args.no_zenml:
        log = RunLog(run_id)
        telemetry = MlflowTelemetry(run_id, enabled=not args.no_mlflow)
        try:
            execute(specs, force=args.force, run_log=log, telemetry=telemetry)
        except RuntimeError as exc:
            telemetry.close(log.results)
            print(f"\n\033[1;31m{exc}\033[0m", file=sys.stderr)
            return 1
        telemetry.close(log.results)
        total = sum(r.seconds for r in log.results)
        print(f"\ncomplete in {total / 3600:.2f} h · log {log.jsonl}")
        if telemetry.active:
            print(f"mlflow ui --backend-store-uri file://{MODELS / 'mlruns'}")
        return 0

    telemetry = MlflowTelemetry(run_id, enabled=not args.no_mlflow)
    try:
        run_with_zenml(specs, force=args.force, telemetry=telemetry)
    finally:
        telemetry.close([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
