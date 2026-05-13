"""
Provenance sidecars for retrieved artifacts.

Every retriever writes `<artifact>.meta.json` next to the data file, recording
the source URL, fetch time, source version, query params, SHA-256 of the
payload, and row count. Without this, the manuscript "reproducibility" claim is
unverifiable: BiasDB content changes, ChEMBL releases monthly, PDB updates
weekly.

Stage 01 fix — see improvements/01_data_collection.md F1.4, F1.5.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_meta(
    artifact_path: Path | str,
    *,
    source_url: str,
    source_version: str | None = None,
    query_params: dict[str, Any] | None = None,
    row_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write `<artifact>.meta.json` next to the artifact and return its path."""
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found at {artifact_path}")

    meta = {
        "artifact_path": str(artifact_path),
        "source_url": source_url,
        "source_version": source_version,
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query_params": query_params or {},
        "sha256": sha256_of(artifact_path),
        "row_count": row_count,
    }
    if extra:
        meta.update(extra)

    meta_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta_path


def read_meta(artifact_path: Path | str) -> dict[str, Any]:
    artifact_path = Path(artifact_path)
    meta_path = artifact_path.with_suffix(artifact_path.suffix + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta sidecar not found at {meta_path}")
    return json.loads(meta_path.read_text())


def verify_meta(artifact_path: Path | str) -> bool:
    """Return True if the recorded SHA-256 matches the artifact on disk."""
    meta = read_meta(artifact_path)
    return meta["sha256"] == sha256_of(Path(artifact_path))
