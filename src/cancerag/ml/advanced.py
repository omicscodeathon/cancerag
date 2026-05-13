"""
Phase 6 — Advanced training orchestrator.

Runs the 5 deferred metric-lift items in sequence:
  1. Optuna hyperparameter tuning of the bake-off winner
  2. Stacking ensemble (LightGBM + XGBoost + ChemBERTa-LR → LR meta)
  3. Focal-loss XGBoost (rare-class lift)
  4. Per-class threshold optimization (cheap macro-F1 lift)
  5. MLflow run tracking (audit trail)

Run:
    python -m cancerag.ml.advanced            # all 5
    python -m cancerag.ml.advanced tune       # just Optuna
    python -m cancerag.ml.advanced stack      # just stacking
    python -m cancerag.ml.advanced focal      # just focal loss
    python -m cancerag.ml.advanced threshold  # just threshold opt
    python -m cancerag.ml.advanced mlflow     # just MLflow
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed/ml_models/advanced")


def run_all(*, optuna_trials: int = 50) -> dict:
    """Run all 5 phases in dependency order."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phases": {},
    }

    # 1. Optuna tuning (independent — could go first or after stacking)
    logger.info("=" * 60)
    logger.info("Phase 6.1 — Optuna hyperparameter tuning")
    logger.info("=" * 60)
    from cancerag.ml.hyperopt import run_optuna
    summary["phases"]["optuna"] = run_optuna(n_trials=optuna_trials)

    # 2. Stacking ensemble (uses base models + ChemBERTa)
    logger.info("=" * 60)
    logger.info("Phase 6.2 — Stacking ensemble")
    logger.info("=" * 60)
    from cancerag.ml.ensemble import run_stacking, run_focal_loss
    summary["phases"]["stacking"] = run_stacking()

    # 3. Focal-loss XGBoost
    logger.info("=" * 60)
    logger.info("Phase 6.3 — Focal-loss XGBoost")
    logger.info("=" * 60)
    summary["phases"]["focal"] = run_focal_loss(gamma=2.0)

    # 4. Per-class threshold optimization (uses tuned model if available)
    logger.info("=" * 60)
    logger.info("Phase 6.4 — Per-class threshold optimization")
    logger.info("=" * 60)
    from cancerag.ml.hyperopt import optimize_thresholds
    summary["phases"]["threshold"] = optimize_thresholds()

    # 5. MLflow tracking (final — captures all results)
    logger.info("=" * 60)
    logger.info("Phase 6.5 — MLflow run tracking")
    logger.info("=" * 60)
    from cancerag.ml.mlflow_logging import log_all_runs
    summary["phases"]["mlflow"] = log_all_runs()

    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (OUTPUT_DIR / "advanced_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )

    # Aggregated headline metric table
    md = ["# Phase 6 — Advanced training summary", "",
          f"_Generated: {summary['finished_at_utc']}_  ", ""]

    md.append("## Holdout macro-F1 progression")
    md.append("")
    md.append("| Stage | Model | Holdout macro-F1 |")
    md.append("| --- | --- | --- |")
    md.append("| Bake-off baseline | LightGBM (default) | 0.247 |")
    o = summary["phases"]["optuna"].get("holdout_metrics", {})
    if o.get("macro_f1"):
        md.append(f"| Phase 6.1 Optuna | LightGBM (tuned) | "
                  f"{o['macro_f1']['point_estimate']:.3f} "
                  f"[CI {o['macro_f1']['ci_lo']:.3f}, {o['macro_f1']['ci_hi']:.3f}] |")
    s = summary["phases"]["stacking"].get("holdout_metrics", {})
    if s.get("macro_f1"):
        md.append(f"| Phase 6.2 Stacking | LightGBM + XGB + ChemBERTa | "
                  f"{s['macro_f1']['point_estimate']:.3f} "
                  f"[CI {s['macro_f1']['ci_lo']:.3f}, {s['macro_f1']['ci_hi']:.3f}] |")
    t = summary["phases"]["threshold"].get("holdout_metrics_with_tuned_thresholds", {})
    if t.get("macro_f1_tuned_thresholds"):
        md.append(f"| Phase 6.4 Per-class thresholds | LightGBM (tuned + thr) | "
                  f"{t['macro_f1_tuned_thresholds']['point_estimate']:.3f} "
                  f"[CI {t['macro_f1_tuned_thresholds']['ci_lo']:.3f}, "
                  f"{t['macro_f1_tuned_thresholds']['ci_hi']:.3f}] |")
    md.append("")

    # Focal loss is CV-only (no holdout in current impl)
    f = summary["phases"]["focal"]
    md.append("## Focal-loss XGBoost (scaffold-CV)")
    md.append("")
    md.append(f"- CV macro-F1: **{f.get('scaffold_cv_macro_f1_mean', 0):.3f} ± "
              f"{f.get('scaffold_cv_macro_f1_std', 0):.3f}** "
              f"(γ={f.get('objective', '').split('=')[-1].rstrip(')')})")
    md.append(f"- Per-class F1 mean: {f.get('per_class_f1_mean_across_folds', {})}")
    md.append("")

    md.append("## MLflow audit trail")
    md.append("")
    mlf = summary["phases"]["mlflow"]
    md.append(f"- Logged **{mlf['n_runs_logged']}** runs to `{mlf['mlruns_dir']}`")
    md.append(f"- Inspect: `mlflow ui --backend-store-uri file://{mlf['mlruns_dir']}`")
    md.append("")

    (OUTPUT_DIR / "advanced_summary.md").write_text("\n".join(md))
    logger.info("Wrote advanced summary -> %s", OUTPUT_DIR / "advanced_summary.md")
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "all":
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        out = run_all(optuna_trials=n_trials)
        print("\n=== Phase 6 complete ===")
        for phase, data in out["phases"].items():
            print(f"  {phase}: {list(data.keys())[:5]}")
    elif cmd == "tune":
        from cancerag.ml.hyperopt import run_optuna
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        out = run_optuna(n_trials=n_trials)
        print(f"Best CV macro-F1: {out['best_cv_macro_f1']:.4f}")
    elif cmd == "stack":
        from cancerag.ml.ensemble import run_stacking
        out = run_stacking()
        print(f"Stacking OOF macro-F1: {out['stacking_oof_macro_f1']:.4f}")
    elif cmd == "focal":
        from cancerag.ml.ensemble import run_focal_loss
        gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
        out = run_focal_loss(gamma=gamma)
        print(f"Focal-loss CV macro-F1: {out['scaffold_cv_macro_f1_mean']:.4f}")
    elif cmd == "threshold":
        from cancerag.ml.hyperopt import optimize_thresholds
        out = optimize_thresholds()
        print(f"Threshold-tuned OOF macro-F1: {out['best_oof_macro_f1']:.4f}")
    elif cmd == "mlflow":
        from cancerag.ml.mlflow_logging import log_all_runs
        out = log_all_runs()
        print(f"Logged {out['n_runs_logged']} runs")
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m cancerag.ml.advanced [all|tune|stack|focal|threshold|mlflow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
