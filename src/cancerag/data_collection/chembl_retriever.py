"""
ChEMBL agonist retriever.

Stage 02 wiring (post-decision):
- **Registry-strict** (decision: strict mode). Receptor names are resolved to
  UniProt accessions via ``ReceptorRegistry``. Unknown receptors raise
  ``UnknownReceptorError`` instead of silently picking the first ChEMBL
  search hit (which used to drift across ChEMBL releases).
- **Activity-driven** filtering. Wires the ``chembl_activity_types``,
  ``chembl_min_confidence_score``, and ``chembl_activity_threshold_nm``
  config keys (which were previously dead code). The mechanism-table-only
  retrieval path is replaced by an activity-table query joined to the
  molecule API.
- ChEMBL output is **not** the labelled training set (that's BiasDB-only,
  per Stage 02 decision). These CSVs are the inference-time
  "nearest-training-ligand" reference set and the unlabelled pool for any
  later PU-learning experiments. Each row carries
  ``label_status="unlabeled"`` — never a hardcoded ``bias_category``.
- Per-output ``.meta.json`` sidecar with provenance.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

from cancerag.data_collection.provenance import write_meta
from cancerag.data_collection.registry import ReceptorRegistry
from cancerag.utils.network import NetworkRetrier, NetworkRetrySettings

logger = logging.getLogger(__name__)


# Default config knobs (all overridable via config.yaml::data_collection)
DEFAULT_ACTIVITY_TYPES = ("EC50", "pEC50", "Ki", "IC50")
DEFAULT_MIN_CONFIDENCE = 8
DEFAULT_THRESHOLD_NM = 1000.0


class UnknownReceptorError(KeyError):
    """Raised when a receptor name is not in the canonical registry and
    strict mode is enabled (Stage 02 decision)."""


class ChEMBLRetrievalError(RuntimeError):
    """Raised when a ChEMBL fetch fails after retries."""


class ChEMBLRetriever:
    """Retrieves *unlabelled* agonist data from ChEMBL keyed on UniProt.

    The output is intentionally tagged ``label_status="unlabeled"`` so it
    is never confused with the labelled BiasDB training set.
    """

    def __init__(
        self,
        output_dir: str | Path,
        registry: ReceptorRegistry,
        *,
        network_config: dict | None = None,
        activity_types: tuple[str, ...] = DEFAULT_ACTIVITY_TYPES,
        min_confidence_score: int = DEFAULT_MIN_CONFIDENCE,
        activity_threshold_nm: float = DEFAULT_THRESHOLD_NM,
        max_per_receptor: int | None = None,
        strict: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = registry
        self.activity_types = tuple(activity_types)
        self.min_confidence_score = int(min_confidence_score)
        self.activity_threshold_nm = float(activity_threshold_nm)
        self.max_per_receptor = max_per_receptor
        self.strict = bool(strict)

        self.target_api = new_client.target
        self.activity_api = new_client.activity
        self.molecule_api = new_client.molecule

        self.retry_settings = NetworkRetrySettings.from_config(network_config)
        self.network_retrier = NetworkRetrier(self.retry_settings, logger=logger)

    @classmethod
    def from_config(
        cls,
        output_dir: str | Path,
        config: dict,
        *,
        registry: ReceptorRegistry | None = None,
    ) -> "ChEMBLRetriever":
        """Build a retriever from the project config block."""
        dc = config.get("data_collection", {})
        if registry is None:
            registry = ReceptorRegistry.load()
        return cls(
            output_dir=output_dir,
            registry=registry,
            network_config=config.get("network"),
            activity_types=tuple(
                dc.get("chembl_activity_types", DEFAULT_ACTIVITY_TYPES)
            ),
            min_confidence_score=int(
                dc.get("chembl_min_confidence_score", DEFAULT_MIN_CONFIDENCE)
            ),
            activity_threshold_nm=float(
                dc.get("chembl_activity_threshold_nm", DEFAULT_THRESHOLD_NM)
            ),
            max_per_receptor=dc.get("chembl_max_agonists_per_receptor"),
            strict=True,
        )

    def _run_with_retry(
        self,
        description: str,
        func,
        exceptions: tuple[type[BaseException], ...] = (
            requests.exceptions.RequestException,
        ),
    ):
        return self.network_retrier.run(description, func, exceptions)

    def _resolve_target_id(self, receptor_name: str) -> tuple[str, str] | None:
        """Resolve receptor name → (UniProt, ChEMBL target_id) via the registry.

        Returns ``None`` if not in the registry and strict mode is off; raises
        ``UnknownReceptorError`` if not in the registry and strict is on.
        """
        row = self.registry.by_biasdb_name(receptor_name)
        if row is None:
            if self.strict:
                raise UnknownReceptorError(
                    f"Receptor {receptor_name!r} is not in data/registry/"
                    f"receptors.tsv. Add a row pinning it to a UniProt "
                    f"accession before running ChEMBL retrieval (strict mode)."
                )
            logger.warning(
                "Receptor %r not in registry; skipping (strict=False)",
                receptor_name,
            )
            return None
        uniprot = str(row["uniprot"])
        chembl_target_id = str(row.get("chembl_target_id") or "").strip()
        if not chembl_target_id:
            if self.strict:
                raise UnknownReceptorError(
                    f"Receptor {receptor_name!r} has no chembl_target_id in "
                    f"the registry."
                )
            return None
        return uniprot, chembl_target_id

    def _fetch_activities(self, target_id: str) -> list[dict[str, Any]]:
        """Pull activity rows for the target via the ChEMBL activity API.

        Filters: assay_type='F' (functional), standard_relation='=',
        standard_type ∈ self.activity_types, standard_value <= threshold,
        confidence_score >= self.min_confidence_score.
        """

        def _fetch():
            qs = self.activity_api.filter(
                target_chembl_id=target_id,
                assay_type="F",
                standard_relation="=",
                standard_type__in=list(self.activity_types),
                standard_value__lte=self.activity_threshold_nm,
                standard_units="nM",
            )
            return list(qs)

        try:
            rows = self._run_with_retry(
                f"ChEMBL activity fetch for {target_id}", _fetch
            )
        except requests.exceptions.RequestException as exc:
            raise ChEMBLRetrievalError(
                f"ChEMBL activity fetch failed for {target_id}: {exc}"
            ) from exc

        # Confidence score filter applied client-side (the API filter for
        # `confidence_score__gte` is honored inconsistently across releases).
        filtered = [
            r for r in rows
            if r.get("confidence_score") is not None
            and int(r["confidence_score"]) >= self.min_confidence_score
        ]
        return filtered

    def _fetch_molecule_smiles(self, chembl_id: str) -> str | None:
        def _fetch():
            return self.molecule_api.get(chembl_id)

        try:
            record = self._run_with_retry(
                f"ChEMBL molecule fetch {chembl_id}", _fetch
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Molecule fetch failed for %s: %s", chembl_id, exc)
            return None
        if not record:
            return None
        structures = record.get("molecule_structures") or {}
        return structures.get("canonical_smiles")

    def _build_dataframe(
        self,
        activities: list[dict[str, Any]],
        receptor_name: str,
        uniprot: str,
        target_id: str,
    ) -> pd.DataFrame:
        """One row per unique molecule with summarized activity context."""
        rows: list[dict] = []
        seen_molecules: set[str] = set()
        for act in activities:
            mol_id = act.get("molecule_chembl_id")
            if not mol_id or mol_id in seen_molecules:
                continue
            seen_molecules.add(mol_id)
            smiles = self._fetch_molecule_smiles(mol_id)
            if not smiles:
                continue
            rows.append(
                {
                    "canonical_smiles": smiles,
                    "ligand_name": mol_id,
                    "molecule_chembl_id": mol_id,
                    "receptor_subtype": receptor_name,
                    "receptor_uniprot": uniprot,
                    "chembl_target_id": target_id,
                    "standard_type": act.get("standard_type"),
                    "standard_value": act.get("standard_value"),
                    "standard_units": act.get("standard_units"),
                    "assay_type": act.get("assay_type"),
                    "confidence_score": act.get("confidence_score"),
                    "source": "ChEMBL",
                    # Stage 02 decision: ChEMBL rows are NOT labelled with a
                    # bias category. They are the unlabelled pool / reference
                    # set, kept separate from the labelled training table.
                    "label_status": "unlabeled",
                    "bias_category": None,
                }
            )
            if self.max_per_receptor and len(rows) >= self.max_per_receptor:
                break
        return pd.DataFrame(rows)

    def _output_path(self, receptor_name: str, uniprot: str) -> Path:
        slug = (
            receptor_name.replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .lower()
        )
        return self.output_dir / f"{slug}__{uniprot}__unlabeled.csv"

    def run(self, receptor_list: list[str]) -> dict:
        """Fetch unlabelled agonist data for each named receptor.

        Returns a summary dict; each per-receptor CSV is written under
        ``output_dir`` with a sibling ``.meta.json`` sidecar.
        """
        summary: dict = {
            "total_receptors": len(receptor_list),
            "total_records": 0,
            "receptors": [],
            "config": {
                "activity_types": list(self.activity_types),
                "min_confidence_score": self.min_confidence_score,
                "activity_threshold_nm": self.activity_threshold_nm,
                "max_per_receptor": self.max_per_receptor,
                "strict": self.strict,
            },
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Starting ChEMBL retrieval for %d receptors (strict=%s)",
            len(receptor_list), self.strict,
        )

        for receptor_name in tqdm(receptor_list, desc="Fetching ChEMBL data"):
            try:
                resolved = self._resolve_target_id(receptor_name)
            except UnknownReceptorError as exc:
                summary["receptors"].append(
                    {"receptor": receptor_name, "status": "unknown_receptor",
                     "records": 0, "error": str(exc)}
                )
                logger.error("%s", exc)
                # In strict mode, a single missing receptor halts the run so
                # the registry stays the source of truth.
                raise
            if resolved is None:
                summary["receptors"].append(
                    {"receptor": receptor_name, "status": "skipped_unknown",
                     "records": 0}
                )
                continue
            uniprot, target_id = resolved

            out_path = self._output_path(receptor_name, uniprot)
            if out_path.exists():
                logger.info(
                    "ChEMBL data for %s (%s) already exists at %s; skipping",
                    receptor_name, uniprot, out_path,
                )
                continue

            activities = self._fetch_activities(target_id)
            df = self._build_dataframe(
                activities, receptor_name, uniprot, target_id
            )

            if df.empty:
                logger.warning(
                    "No qualifying activity rows for %s (%s, %s)",
                    receptor_name, uniprot, target_id,
                )
                summary["receptors"].append(
                    {"receptor": receptor_name, "uniprot": uniprot,
                     "target_id": target_id, "status": "no_data",
                     "records": 0}
                )
                continue

            df.to_csv(out_path, index=False)
            write_meta(
                out_path,
                source_url=f"chembl-target:{target_id}",
                source_version="chembl-webresource-client",
                query_params={
                    "target_chembl_id": target_id,
                    "uniprot": uniprot,
                    "assay_type": "F",
                    "standard_relation": "=",
                    "standard_type__in": list(self.activity_types),
                    "standard_value__lte": self.activity_threshold_nm,
                    "min_confidence_score": self.min_confidence_score,
                },
                row_count=len(df),
                extra={"label_status": "unlabeled"},
            )
            summary["total_records"] += len(df)
            summary["receptors"].append(
                {
                    "receptor": receptor_name, "uniprot": uniprot,
                    "target_id": target_id, "status": "saved",
                    "records": int(len(df)), "file": str(out_path),
                }
            )

        summary_path = self.output_dir / "chembl_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("ChEMBL retrieval summary written to %s", summary_path)
        return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    test_receptors = ["5-HT1A receptor", "D2 receptor"]
    registry = ReceptorRegistry.load()
    retriever = ChEMBLRetriever(
        output_dir="data/raw/chembl_test", registry=registry
    )
    retriever.run(test_receptors)
