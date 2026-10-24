"""Verify every prepared receptor chain against its canonical UniProt sequence.

``prepare_receptor`` runs an identity gate on the chain it writes, but the
gate needs the canonical sequence, and when that cannot be fetched it records
"unverified" (``None``) rather than failing. The 54 sidecars in the shipped
set predate the gate altogether and carry no identity fields; seven repaired
receptors have no sidecar at all. This module re-runs the same gate
(:func:`receptor_preprocessor.verify_chain_identity`, unchanged) offline
against a FASTA cache, writes the numbers into each sidecar, and writes one
summary JSON with per-receptor values and their median and minimum.

Run:
    python -m cancerag.preprocessing.chain_verification
    python -m cancerag.preprocessing.chain_verification --no-write-sidecars
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import statistics
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from Bio.PDB import PDBParser

from cancerag.preprocessing.receptor_preprocessor import (
    MIN_CHAIN_COVERAGE,
    MIN_CHAIN_IDENTITY,
    _chain_one_letter,
    _sha256_of_path,
    verify_chain_identity,
)

logger = logging.getLogger(__name__)

RECEPTOR_DIR = Path("data/processed/receptors")
CACHE_DIR = Path("data/raw/uniprot")
SUMMARY_PATH = Path("data/processed/receptor_chain_verification.json")
REPAIR_REPORT_PATH = Path("data/processed/receptor_repair_report.json")
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
SIDECAR_SUFFIX = ".prep.meta.json"
# Derived files that sit next to the prepared receptor but are not receptors.
DERIVED_SUFFIXES = (".prolif_H.pdb",)


def _http_fetch(acc: str) -> str:
    with urllib.request.urlopen(UNIPROT_FASTA_URL.format(acc=acc), timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse_fasta(text: str) -> tuple[str, str]:
    """Return ``(header, sequence)`` of the first record."""
    header, seq, seen = "", [], False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            if seen:
                break
            header, seen = line[1:], True
        elif seen and line:
            seq.append(line)
    return header, "".join(seq)


def canonical_sequence(
    acc: str, cache_dir: Path | str = CACHE_DIR, *, refresh: bool = False,
    fetch: Callable[[str], str] = _http_fetch,
) -> tuple[str, str]:
    """Canonical UniProt sequence for ``acc``, cached as ``<acc>.fasta``.

    Returns ``(header, sequence)``. The header carries the sequence version
    (``SV=``), which is recorded so a later UniProt revision is detectable.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{acc}.fasta"
    if refresh or not path.exists():
        text = fetch(acc)
        header, seq = parse_fasta(text)
        if not seq:
            raise ValueError(f"empty UniProt FASTA for {acc}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    header, seq = parse_fasta(path.read_text())
    if not seq:
        raise ValueError(f"empty cached FASTA {path}")
    return header, seq


def _sequence_version(header: str) -> int | None:
    for tok in header.split():
        if tok.startswith("SV="):
            try:
                return int(tok[3:])
            except ValueError:
                return None
    return None


def prepared_receptors(receptor_dir: Path | str = RECEPTOR_DIR) -> list[Path]:
    """Every prepared receptor PDB (``<UniProt>.pdb``), derived files excluded."""
    return sorted(
        p for p in Path(receptor_dir).glob("*.pdb")
        if not p.name.endswith(DERIVED_SUFFIXES)
    )


def verify_receptor(pdb_path: Path, acc: str, canonical: str) -> dict:
    """Run the prepare-time identity gate on one prepared receptor."""
    structure = PDBParser(QUIET=True).get_structure("rec", str(pdb_path))
    chains = [c.id for m in structure for c in m]
    n_res = len(_chain_one_letter(structure, chains[0]).replace("X", "")) if chains else 0
    identity, coverage = verify_chain_identity(pdb_path, acc, canonical=canonical)
    return {
        "uniprot": acc,
        "pdb": str(pdb_path),
        "chains_in_file": chains,
        "verified_chain": chains[0] if chains else None,
        "n_chain_residues": n_res,
        "canonical_length": len(canonical),
        "chain_identity_to_uniprot": round(identity, 4),
        "chain_coverage": round(coverage, 4),
        "passes_gate": bool(
            identity >= MIN_CHAIN_IDENTITY and coverage >= MIN_CHAIN_COVERAGE
        ),
    }


def _stats(values: list[float]) -> dict:
    if not values:
        return {"median": None, "min": None, "max": None}
    return {"median": float(statistics.median(values)),
            "min": float(min(values)), "max": float(max(values))}


def _repair_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return {r["acc"]: r for r in json.loads(path.read_text())}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def update_sidecar(
    sidecar: Path, record: dict, provenance: dict, *, backup_suffix: str | None,
) -> None:
    """Write identity and coverage into an existing sidecar, backing it up
    once first. Other fields are left as they are."""
    if backup_suffix:
        backup = sidecar.with_name(sidecar.name + backup_suffix)
        if not backup.exists():
            shutil.copy2(sidecar, backup)
    meta = json.loads(sidecar.read_text())
    meta["chain_identity_to_uniprot"] = record["chain_identity_to_uniprot"]
    meta["chain_coverage"] = record["chain_coverage"]
    meta["chain_verification"] = {
        **provenance,
        "verified_chain": record["verified_chain"],
        "passes_gate": record["passes_gate"],
    }
    sidecar.write_text(json.dumps(meta, indent=2, sort_keys=True))


def run(
    *, receptor_dir: Path | str = RECEPTOR_DIR, cache_dir: Path | str = CACHE_DIR,
    summary_path: Path | str = SUMMARY_PATH,
    repair_report: Path | str = REPAIR_REPORT_PATH,
    write_sidecars: bool = True, backup_suffix: str | None = None,
    refresh: bool = False, fetch: Callable[[str], str] = _http_fetch,
) -> dict:
    receptor_dir, summary_path = Path(receptor_dir), Path(summary_path)
    now = datetime.now(UTC).isoformat()
    repairs = _repair_index(Path(repair_report))
    records, failures = [], []
    for pdb in prepared_receptors(receptor_dir):
        acc = pdb.stem
        try:
            header, seq = canonical_sequence(acc, cache_dir, refresh=refresh,
                                             fetch=fetch)
        except (OSError, ValueError) as exc:  # network, HTTP or empty FASTA
            logger.warning("no canonical sequence for %s: %s", acc, exc)
            failures.append({"uniprot": acc, "error": str(exc)})
            continue
        rec = verify_receptor(pdb, acc, seq)
        rec["uniprot_header"] = header
        rec["uniprot_sequence_version"] = _sequence_version(header)
        sidecar = pdb.with_name(pdb.name + SIDECAR_SUFFIX)
        rec["sidecar"] = str(sidecar) if sidecar.exists() else None
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            rec["sidecar_kept_chain"] = meta.get("kept_chain")
            rec["sidecar_output_sha_matches_file"] = (
                meta.get("output_sha256") == _sha256_of_path(pdb)
            )
            if write_sidecars:
                update_sidecar(sidecar, rec, {
                    "verified_at_utc": now,
                    "canonical_source": UNIPROT_FASTA_URL.format(acc=acc),
                    "uniprot_sequence_version": rec["uniprot_sequence_version"],
                    "gate": "receptor_preprocessor.verify_chain_identity",
                    "min_identity": MIN_CHAIN_IDENTITY,
                    "min_coverage": MIN_CHAIN_COVERAGE,
                }, backup_suffix=backup_suffix)
        else:
            repair = repairs.get(acc)
            rec["sidecar_missing_reason"] = (
                f"receptor replaced by the chain repair (chosen PDB "
                f"{repair.get('chosen_pdb')}, chain {repair.get('chain')}); the "
                "repair wrote the PDB without a prep sidecar"
                if repair else "no prep sidecar found"
            )
        records.append(rec)

    ids = [r["chain_identity_to_uniprot"] for r in records]
    covs = [r["chain_coverage"] for r in records]
    with_sidecar = [r for r in records if r["sidecar"]]
    summary = {
        "schema_version": 1,
        "generated_at_utc": now,
        "generator": "python -m cancerag.preprocessing.chain_verification",
        "gate": {
            "function": "cancerag.preprocessing.receptor_preprocessor."
                        "verify_chain_identity",
            "alignment": "Bio.Align.PairwiseAligner global, free end gaps, "
                         "match +2, mismatch -1, gap open -10, extend -0.5",
            "identity": "matches / aligned residue pairs",
            "coverage": "aligned residue pairs / residues in the prepared chain",
            "min_identity": MIN_CHAIN_IDENTITY,
            "min_coverage": MIN_CHAIN_COVERAGE,
        },
        "canonical_source": UNIPROT_FASTA_URL,
        "n_prepared_receptors": len(records) + len(failures),
        "n_verified": len(records),
        "n_with_sidecar": len(with_sidecar),
        "n_without_sidecar": len(records) - len(with_sidecar),
        "n_below_gate": sum(not r["passes_gate"] for r in records),
        "identity": _stats(ids),
        "coverage": _stats(covs),
        "below_gate": [r["uniprot"] for r in records if not r["passes_gate"]],
        "failures": failures,
        "receptors": records,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--receptor-dir", default=str(RECEPTOR_DIR))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--summary", default=str(SUMMARY_PATH))
    ap.add_argument("--repair-report", default=str(REPAIR_REPORT_PATH))
    ap.add_argument("--no-write-sidecars", action="store_true")
    ap.add_argument("--backup-suffix", default=None,
                    help="copy each sidecar to <name><suffix> before editing "
                         "(skipped if the backup already exists)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download FASTA even when cached")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    s = run(receptor_dir=args.receptor_dir, cache_dir=args.cache_dir,
            summary_path=args.summary, repair_report=args.repair_report,
            write_sidecars=not args.no_write_sidecars,
            backup_suffix=args.backup_suffix, refresh=args.refresh)
    print(f"verified {s['n_verified']}/{s['n_prepared_receptors']} receptors "
          f"({s['n_with_sidecar']} with sidecar); "
          f"identity median {s['identity']['median']:.3f} min {s['identity']['min']:.3f}; "
          f"coverage median {s['coverage']['median']:.3f} min {s['coverage']['min']:.3f}; "
          f"below gate: {s['below_gate'] or 'none'}")


if __name__ == "__main__":
    main()
