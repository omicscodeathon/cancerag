"""Locate and load the run artefacts every figure is drawn from.

Figures read artefacts only; they never retrain or rewrite anything. The data
root defaults to ``<repo>/data`` and can be overridden with ``--data`` or the
``CANCERAG_DATA_DIR`` environment variable (useful in a git worktree, where the
gitignored artefacts live in the main checkout).

``ml_models_preQCfix_backup/`` is deliberately never read: it holds the
superseded pre-leakage-fix run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "manuscript" / "figures"


class MissingArtefact(FileNotFoundError):
    """Raised when a figure's source artefact is not on disk."""


def default_data_dir() -> Path:
    env = os.environ.get("CANCERAG_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "data"


@dataclass
class Artefacts:
    data_dir: Path = field(default_factory=default_data_dir)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).resolve()

    # ---- paths ---------------------------------------------------------
    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models(self) -> Path:
        return self.processed / "ml_models"

    def path(self, rel: str) -> Path:
        """Resolve ``rel`` (relative to ``data/``) and fail loudly if absent."""
        if "preQCfix" in rel:
            raise ValueError("ml_models_preQCfix_backup is superseded; never plot it")
        p = self.data_dir / rel
        if not p.exists():
            raise MissingArtefact(f"artefact not found: {p}")
        return p

    # ---- generic loaders -----------------------------------------------
    def json(self, rel: str):
        return json.loads(self.path(rel).read_text())

    def csv(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.path(rel), **kw)

    def parquet(self, rel: str, **kw) -> pd.DataFrame:
        return pd.read_parquet(self.path(rel), **kw)

    # ---- shared, cached datasets ---------------------------------------
    @cached_property
    def train(self) -> pd.DataFrame:
        return self.parquet("processed/ml_ready_dataset.parquet")

    @cached_property
    def holdout(self) -> pd.DataFrame:
        return self.parquet("holdout/dataset_holdout.parquet")

    @cached_property
    def modelled(self) -> pd.DataFrame:
        """All 615 modelled rows (train-eligible + temporal holdout)."""
        tr = self.train.assign(split="train-eligible")
        ho = self.holdout.assign(split="temporal holdout")
        return pd.concat([tr, ho], ignore_index=True)

    @cached_property
    def model_meta(self) -> dict:
        return self.json("processed/ml_models/model_meta.json")

    # ---- the selected model --------------------------------------------
    # Which learner wins is a decision the run makes, not a constant. A figure
    # that names one in a literal path is right only until the bake-off picks
    # something else -- and then it does not fail with "the model changed", it
    # fails with "file not found", which reads like a missing artefact.
    @cached_property
    def selected_model(self) -> str:
        """The learner `model_training` chose, e.g. ``"lightgbm"``."""
        return self.json("processed/ml_models/selection_decision.json")["chosen"]

    @cached_property
    def calibrated_model_rel(self) -> str:
        """Path (relative to ``data/``) of the selected model's calibrated fit."""
        return f"processed/ml_models/{self.selected_model}_final_calibrated.joblib"

    @cached_property
    def receptor_names(self) -> dict[str, str]:
        """UniProt accession -> BiasDB receptor name (from binding_sites.json)."""
        sites = self.json("processed/binding_sites.json")["binding_sites"]
        return {s["uniprot"]: s.get("biasdb_name") or s["uniprot"] for s in sites}


# Short, conventional receptor names for axis labels; anything not listed falls
# back to the BiasDB name from binding_sites.json.
SHORT_RECEPTOR = {
    "P14416": "D2 dopamine",
    "P35372": "μ-opioid",
    "P41145": "κ-opioid",
    "P07550": "β2-adrenoceptor",
    "Q92847": "Ghrelin",
    "P08172": "M2 muscarinic",
    "Q9H3N8": "Histamine H4",
    "P0DMS8": "Adenosine A3",
    "P34972": "CB2 cannabinoid",
    "P35348": "α1A-adrenoceptor",
    "P08588": "β1-adrenoceptor",
    "P41146": "NOP (nociceptin)",
    "P18825": "α2C-adrenoceptor",
    "P11229": "M1 muscarinic",
    "Q9NQS5": "GPR84",
    "P08908": "5-HT1A",
    "P21728": "D1 dopamine",
    "P41143": "δ-opioid",
    "P43116": "EP2 prostanoid",
    "P34969": "5-HT7",
    "P28335": "5-HT2C",
    "P21462": "FPR1",
}


def receptor_label(art: Artefacts, uniprot: str) -> str:
    return SHORT_RECEPTOR.get(uniprot) or art.receptor_names.get(uniprot, uniprot)
