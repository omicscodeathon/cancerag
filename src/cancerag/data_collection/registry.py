"""
Canonical receptor registry.

The registry is a TSV keyed by UniProt accession that all data retrievers
join through. It removes the brittle name-based receptor matching that the
previous pipeline relied on (where ChEMBL `target.search()` returned the
first hit for a string like "5-HT1A receptor", and that result could change
between ChEMBL releases).

Stage 01 fix — see improvements/01_data_collection.md F1.1, F1.2, F1.6.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("data/registry/receptors.tsv")

REQUIRED_COLUMNS = (
    "uniprot",
    "gene_symbol",
    "biasdb_name",
    "chembl_target_id",
    "gpcrdb_id",
    "gpcrdb_class",
    "gpcrdb_family",
    "preferred_pdb_active",
    "preferred_pdb_inactive",
    "alphafold_id",
    "notes",
)


class RegistryError(ValueError):
    """Raised when the registry file is malformed."""


class ReceptorRegistry:
    """In-memory view of the canonical receptor registry."""

    def __init__(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise RegistryError(f"Registry missing required columns: {missing}")
        if df["uniprot"].duplicated().any():
            dups = df.loc[df["uniprot"].duplicated(), "uniprot"].tolist()
            raise RegistryError(f"Duplicate UniProt accessions in registry: {dups}")
        if df["uniprot"].isna().any():
            raise RegistryError("Registry contains rows with empty uniprot column")
        self._df = df.set_index("uniprot", drop=False)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_REGISTRY_PATH) -> "ReceptorRegistry":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Receptor registry not found at {path}")
        df = pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")
        logger.info("Loaded receptor registry from %s (%d rows)", path, len(df))
        return cls(df)

    def by_uniprot(self, uniprot: str) -> pd.Series:
        if uniprot not in self._df.index:
            raise KeyError(f"UniProt {uniprot!r} not found in registry")
        return self._df.loc[uniprot]

    def by_biasdb_name(self, biasdb_name: str) -> pd.Series | None:
        match = self._df[self._df["biasdb_name"].str.lower() == biasdb_name.lower()]
        if match.empty:
            return None
        if len(match) > 1:
            raise RegistryError(
                f"Multiple registry rows for biasdb_name={biasdb_name!r}"
            )
        return match.iloc[0]

    def all_uniprots(self) -> list[str]:
        return self._df["uniprot"].tolist()

    def __iter__(self) -> Iterator[pd.Series]:
        for _, row in self._df.iterrows():
            yield row

    def __len__(self) -> int:
        return len(self._df)

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df.reset_index(drop=True).copy()
