"""
BiasDB retriever.

Stage 01 fixes applied (see improvements/01_data_collection.md):
- Validates the JSON payload via `parse_biasdb_payload` (Pydantic schema)
  rather than mapping a positional column header list onto whatever JSON the
  endpoint returns. Schema drift now raises loudly instead of misaligning
  the label vector silently.
- Raises `BiasDBRetrievalError` on failure instead of calling `sys.exit(1)`,
  so the function is usable from notebooks, tests, and the inference app.
- Writes a `<artifact>.meta.json` sidecar with source URL, fetch timestamp,
  query params, SHA-256, and row count.
- Removed the module-level `logging.basicConfig`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from cancerag.data_collection.provenance import write_meta
from cancerag.data_collection.schemas import (
    BIASDB_HEADER_ORDER,
    SchemaDriftError,
    parse_biasdb_payload,
)
from cancerag.utils.network import (
    NetworkRetrier,
    NetworkRetrySettings,
    create_retry_session,
)

logger = logging.getLogger(__name__)

BIASDB_URL = "https://biasdb.drug-design.de/data_0/query?user_query=default_query"


class BiasDBRetrievalError(RuntimeError):
    """Raised when the BiasDB retrieval fails or returns an invalid payload."""


def download_biasdb_data(
    output_path: str,
    network_config: dict | None = None,
    *,
    write_provenance: bool = True,
) -> pd.DataFrame:
    """Fetch the complete BiasDB dataset and save it as a CSV.

    Idempotent: returns the cached CSV if it already exists.

    Raises:
        BiasDBRetrievalError: when the network call fails after retries, or the
            response is not valid JSON, or the schema does not match the
            expected BiasDB layout.
    """
    output_path = str(output_path)
    if os.path.exists(output_path):
        logger.info(
            "BiasDB data already exists at %s. Loading existing data...", output_path
        )
        try:
            df = pd.read_csv(output_path)
            logger.info("Loaded %d existing BiasDB records", len(df))
            return df
        except Exception as exc:
            logger.warning(
                "Could not load existing BiasDB data: %s. Re-downloading...", exc
            )

    logger.info("Fetching data from BiasDB URL: %s", BIASDB_URL)

    retry_settings = NetworkRetrySettings.from_config(network_config)
    retrier = NetworkRetrier(retry_settings, logger=logger)
    session = create_retry_session(retry_settings, allowed_methods=["GET"])

    def _fetch() -> Any:
        response = session.get(BIASDB_URL, timeout=60)
        response.raise_for_status()
        return response.json()

    try:
        raw_payload = retrier.run(
            "BiasDB download", _fetch, (requests.exceptions.RequestException,)
        )
    except requests.exceptions.RequestException as exc:
        raise BiasDBRetrievalError(
            f"BiasDB download failed after retries: {exc}"
        ) from exc
    except ValueError as exc:
        raise BiasDBRetrievalError("BiasDB response was not valid JSON") from exc

    try:
        rows = parse_biasdb_payload(raw_payload)
    except SchemaDriftError as exc:
        raise BiasDBRetrievalError(f"BiasDB schema drift: {exc}") from exc

    if not rows:
        logger.warning("BiasDB query returned no rows")
        df = pd.DataFrame(columns=list(BIASDB_HEADER_ORDER))
    else:
        df = pd.DataFrame([r.model_dump() for r in rows])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        logger.info("Saved %d BiasDB records to %s", len(df), output_path)
    except OSError as exc:
        raise BiasDBRetrievalError(
            f"Could not write BiasDB CSV to {output_path}: {exc}"
        ) from exc

    if write_provenance:
        write_meta(
            output_path,
            source_url=BIASDB_URL,
            source_version="biasdb-default-query",
            query_params={"user_query": "default_query"},
            row_count=len(df),
        )

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    download_biasdb_data(str(output_dir / "biasdb_data.csv"))
