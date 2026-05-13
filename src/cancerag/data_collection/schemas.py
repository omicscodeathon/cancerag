"""
Pydantic schemas for external data payloads.

Catches schema drift in BiasDB / ChEMBL responses by validating each row,
rather than the previous behaviour of mapping a hardcoded positional column
list onto whatever JSON came back (which silently misaligned the label
vector if the source ever reordered columns).

Stage 01 fix — see improvements/01_data_collection.md F1.3.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BiasDBRow(BaseModel):
    """One row of the BiasDB JSON response.

    The response is a list-of-lists; we map each inner list onto these named
    fields. If BiasDB adds or removes columns we want to fail loudly here, not
    silently misalign every downstream label.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ligand_name: str | None = None
    smiles: str | None = None
    smiles_duplicate: str | None = None
    receptor_family: str | None = None
    receptor: str | None = None
    receptor_subtype: str | None = None
    bias_category: str | None = None
    bias_pathway: str | None = None
    reference_ligand: str | None = None
    assay_1: str | None = None
    assay_2: str | None = None
    publication_title: str | None = None
    author: str | None = None
    doi: str | None = None
    pmid: str | None = None
    year: int | None = None
    molecular_weight: float | None = None
    logp: float | None = None
    hba: float | None = None
    hbd: float | None = None
    rings: float | None = None
    tpsa: float | None = None

    @field_validator("year", mode="before")
    @classmethod
    def _coerce_year(cls, v: Any) -> int | None:
        if v in (None, "", "None"):
            return None
        try:
            return int(str(v).strip())
        except (TypeError, ValueError):
            return None

    @field_validator(
        "molecular_weight", "logp", "hba", "hbd", "rings", "tpsa", mode="before"
    )
    @classmethod
    def _coerce_float(cls, v: Any) -> float | None:
        if v in (None, "", "None"):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_positional(cls, row: list[Any], headers: tuple[str, ...]) -> "BiasDBRow":
        """Convert a positional list (current BiasDB API shape) to a named row."""
        if len(row) != len(headers):
            raise ValueError(
                f"BiasDB row arity mismatch: expected {len(headers)}, got {len(row)}"
            )
        return cls(**dict(zip(headers, row)))


BIASDB_HEADER_ORDER: tuple[str, ...] = (
    "ligand_name",
    "smiles",
    "smiles_duplicate",
    "receptor_family",
    "receptor",
    "receptor_subtype",
    "bias_category",
    "bias_pathway",
    "reference_ligand",
    "assay_1",
    "assay_2",
    "publication_title",
    "author",
    "doi",
    "pmid",
    "year",
    "molecular_weight",
    "logp",
    "hba",
    "hbd",
    "rings",
    "tpsa",
)


class SchemaDriftError(ValueError):
    """Raised when a BiasDB / ChEMBL payload no longer matches the expected schema."""


def parse_biasdb_payload(payload: list[Any]) -> list[BiasDBRow]:
    """Validate every row of a BiasDB payload; raise on the first failure."""
    if not isinstance(payload, list):
        raise SchemaDriftError(
            f"BiasDB payload must be a list, got {type(payload).__name__}"
        )

    rows: list[BiasDBRow] = []
    for i, raw in enumerate(payload):
        try:
            if isinstance(raw, dict):
                rows.append(BiasDBRow(**raw))
            elif isinstance(raw, list):
                rows.append(BiasDBRow.from_positional(raw, BIASDB_HEADER_ORDER))
            else:
                raise SchemaDriftError(
                    f"BiasDB row {i} has unexpected type {type(raw).__name__}"
                )
        except Exception as exc:
            raise SchemaDriftError(f"BiasDB row {i} failed validation: {exc}") from exc
    return rows
