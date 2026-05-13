import glob
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Conserved Na+ ion stabilizes the inactive state of class A GPCRs.
# Other small-molecule cofactors and structural Mg2+ should also be considered.
RETAIN_HET_RESNAMES_DEFAULT = frozenset({"NA"})

# Chain identifiers commonly used for non-receptor partners in GPCR structures.
# Removing these prevents Vina from docking into the wrong protein.
COMMON_NON_RECEPTOR_CHAINS = frozenset(
    {"B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"}
)

# --- RECEPTOR CLEANING ---


class ReceptorPreprocessor:
    """
    Cleans raw PDB files to prepare them for docking.

    This class is responsible for:
    1. Finding all raw PDB files downloaded from the PDB.
    2. Removing all non-protein molecules (water, ligands, ions).
    3. Saving the cleaned, protein-only PDB structures to a processed directory.
    """

    def __init__(self, config: dict):
        """
        Initializes the ReceptorPreprocessor.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.config = config
        self.paths = config["paths"]
        self.pdb_raw_dir = self.paths["pdb_summary"]
        self.pdb_processed_dir = os.path.join(self.paths["processed_data"], "receptors")
        os.makedirs(self.pdb_processed_dir, exist_ok=True)
        self.parser = PDBParser(QUIET=True)

    class ProteinSelect(Select):
        """A Bio.PDB Select class to keep only standard protein residues."""

        def accept_residue(self, residue):
            # The residue ID tuple is ('HETATM', residue_number, insertion_code) for heteroatoms
            # and (' ', residue_number, insertion_code) for standard residues.
            return residue.id[0] == " "

    def _clean_pdb_file(self, input_path: str, output_path: str):
        """
        Reads a PDB file, removes non-protein atoms, and saves the result.
        """
        try:
            structure = self.parser.get_structure("receptor", input_path)
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_path, self.ProteinSelect())
        except Exception as e:
            logger.error(f"Could not process PDB file {input_path}: {e}")

    def run(self):
        """
        Stage 03 — registry-strict, per-receptor PDB selection + preparation.

        Pipeline:
            for each UniProt in the registry that appears in unified_ligands.csv:
                1. Locate the receptor's cached PDB folder.
                2. Score every PDB in the folder (resolution, ligand
                   presence, active-state hint, fusion-protein penalty).
                3. Pick the top-scoring PDB.
                4. Run `prepare_receptor()` -> data/processed/receptors/<uniprot>.pdb
                   plus a `.prep.meta.json` sidecar.
                5. If no PDB is cached, mark for AlphaFold fallback (a
                   placeholder row in the audit report; the AF download
                   itself is a separate Stage 03 sub-step).

        Emits:
            data/registry/preferred_pdbs.tsv  (auto-selection + manual override)
            data/processed/receptors/<UNIPROT>.pdb (per receptor)
            data/processed/receptors/<UNIPROT>.pdb.prep.meta.json
            data/processed/receptor_audit.md  (reviewer-facing summary)
        """
        # Lazy imports to keep the legacy class importable without these.
        from cancerag.data_collection.registry import ReceptorRegistry
        from cancerag.preprocessing.alphafold_fetcher import (
            AlphaFoldFetchError,
            fetch_alphafold_pdb,
        )
        from cancerag.preprocessing.pdb_selector import (
            select_best_pdb,
            write_preferred_tsv,
        )

        # AlphaFold fallback config: enabled by default per Stage 03 decision.
        af_cfg = self.config.get("preprocessing", {})
        use_alphafold_fallback = bool(
            af_cfg.get("alphafold_fallback", True)
        )
        plddt_threshold = float(af_cfg.get("alphafold_plddt_threshold", 70.0))
        af_cache_dir = Path(self.paths.get(
            "alphafold_cache",
            os.path.join(self.paths.get("raw_data", "data/raw"), "alphafold"),
        ))

        logger.info("Starting receptor preprocessing (Stage 03 wiring)...")

        registry_path = self.paths.get("registry", "data/registry/receptors.tsv")
        registry = ReceptorRegistry.load(registry_path)

        # Optionally restrict to receptors that actually appear in the curated
        # ligand table — otherwise we waste cycles preparing receptors with
        # zero downstream rows.
        unified_path = os.path.join(
            self.paths.get("processed_data", "data/processed"),
            "unified_ligands.csv",
        )
        relevant_uniprots: set[str] = set()
        if os.path.exists(unified_path):
            try:
                import pandas as pd

                df = pd.read_csv(unified_path)
                if "receptor_uniprot" in df.columns:
                    relevant_uniprots = set(
                        df["receptor_uniprot"].dropna().astype(str).unique()
                    )
                    logger.info(
                        "Restricting receptor curation to %d UniProts that "
                        "appear in unified_ligands.csv",
                        len(relevant_uniprots),
                    )
            except Exception as e:
                logger.warning(
                    "Could not load %s for UniProt filtering: %s", unified_path, e
                )

        rows: list[dict] = []
        prep_summaries: list[dict] = []
        af_fallback_uniprots: list[tuple[str, str]] = []

        # The legacy cache uses sanitized folder names rather than UniProt.
        # Build a candidate-folder lookup: try a few naming conventions.
        for _, reg_row in registry.dataframe.iterrows():
            uniprot = str(reg_row["uniprot"])
            biasdb_name = str(reg_row["biasdb_name"])
            if relevant_uniprots and uniprot not in relevant_uniprots:
                continue

            # Look for cached PDBs in either the new uniprot-keyed location or
            # the legacy sanitized-name location.
            candidate_dirs = [
                Path(self.pdb_raw_dir) / uniprot,
                Path(self.pdb_raw_dir) / _legacy_folder_name(biasdb_name),
            ]
            candidate_dirs = [d for d in candidate_dirs if d.exists()]

            best = None
            for d in candidate_dirs:
                cand = select_best_pdb(d, prefer_state="active")
                if cand and (best is None or cand.score > best.score):
                    best = cand

            if best is None:
                af_fallback_uniprots.append((uniprot, biasdb_name))
                rows.append({
                    "uniprot": uniprot,
                    "biasdb_name": biasdb_name,
                    "selected_pdb": "",
                    "resolution": None,
                    "state": "",
                    "ligand_resname": "",
                    "score": "",
                    "reason": "no_pdb_cached_use_alphafold_fallback",
                    "manual_override": "",
                })
                # Try AlphaFold fallback if enabled.
                if use_alphafold_fallback:
                    try:
                        af_meta = fetch_alphafold_pdb(
                            uniprot, af_cache_dir,
                            plddt_threshold=plddt_threshold,
                        )
                        if af_meta["passes_gate"]:
                            output_pdb = (
                                Path(self.pdb_processed_dir) / f"{uniprot}.pdb"
                            )
                            try:
                                meta = prepare_receptor(
                                    input_pdb=Path(af_meta["output_pdb"]),
                                    output_pdb=output_pdb,
                                    target_uniprot=uniprot,
                                    structure_source="alphafold",
                                )
                                prep_summaries.append({
                                    "uniprot": uniprot,
                                    "biasdb_name": biasdb_name,
                                    "input_pdb_id": f"AF-{uniprot}",
                                    "output_path": str(output_pdb),
                                    "kept_chain": meta["kept_chain"],
                                    "dropped_chains": meta["dropped_chains"],
                                    "het_residues_kept": meta["het_residues_kept"],
                                    "het_residues_dropped": meta["het_residues_dropped"],
                                    "waters_dropped": meta["waters_dropped"],
                                    "altloc_atoms_dropped": meta["altloc_atoms_dropped"],
                                    "input_sha256": meta["input_sha256"],
                                    "output_sha256": meta["output_sha256"],
                                    "status": (
                                        f"ok_alphafold_pLDDT={af_meta['mean_plddt']:.1f}"
                                    ),
                                })
                            except ValueError as e:
                                prep_summaries.append({
                                    "uniprot": uniprot,
                                    "biasdb_name": biasdb_name,
                                    "input_pdb_id": f"AF-{uniprot}",
                                    "status": f"af_prep_failed: {e}",
                                })
                        else:
                            prep_summaries.append({
                                "uniprot": uniprot,
                                "biasdb_name": biasdb_name,
                                "input_pdb_id": f"AF-{uniprot}",
                                "status": (
                                    f"af_low_pLDDT={af_meta['mean_plddt']:.1f}_"
                                    f"below_{plddt_threshold}"
                                ),
                            })
                    except AlphaFoldFetchError as e:
                        prep_summaries.append({
                            "uniprot": uniprot,
                            "biasdb_name": biasdb_name,
                            "input_pdb_id": f"AF-{uniprot}",
                            "status": f"af_fetch_failed: {e}",
                        })
                continue

            rows.append({
                "uniprot": uniprot,
                "biasdb_name": biasdb_name,
                "selected_pdb": best.pdb_id,
                "resolution": best.resolution if best.resolution != float("inf") else "",
                "state": best.detected_state,
                "ligand_resname": best.detected_ligand_resname or "",
                "score": f"{best.score:.2f}",
                "reason": best.reason,
                "manual_override": "",
            })

            # Curate the chosen structure.
            output_pdb = Path(self.pdb_processed_dir) / f"{uniprot}.pdb"
            try:
                meta = prepare_receptor(
                    input_pdb=best.path,
                    output_pdb=output_pdb,
                    target_uniprot=uniprot,
                    structure_source="pdb",
                )
                prep_summaries.append({
                    "uniprot": uniprot,
                    "biasdb_name": biasdb_name,
                    "input_pdb_id": best.pdb_id,
                    "output_path": str(output_pdb),
                    "kept_chain": meta["kept_chain"],
                    "dropped_chains": meta["dropped_chains"],
                    "het_residues_kept": meta["het_residues_kept"],
                    "het_residues_dropped": meta["het_residues_dropped"],
                    "waters_dropped": meta["waters_dropped"],
                    "altloc_atoms_dropped": meta["altloc_atoms_dropped"],
                    "input_sha256": meta["input_sha256"],
                    "output_sha256": meta["output_sha256"],
                    "status": "ok",
                })
            except ValueError as e:
                logger.error("prepare_receptor failed for %s (%s): %s",
                             uniprot, best.pdb_id, e)
                # The PDB the picker chose is unusable (wrong protein,
                # tiny fragment, etc.). Fall back to AlphaFold.
                fell_back_ok = False
                if use_alphafold_fallback:
                    try:
                        af_meta = fetch_alphafold_pdb(
                            uniprot, af_cache_dir,
                            plddt_threshold=plddt_threshold,
                        )
                        if af_meta["passes_gate"]:
                            output_pdb = (
                                Path(self.pdb_processed_dir) / f"{uniprot}.pdb"
                            )
                            try:
                                meta = prepare_receptor(
                                    input_pdb=Path(af_meta["output_pdb"]),
                                    output_pdb=output_pdb,
                                    target_uniprot=uniprot,
                                    structure_source="alphafold",
                                )
                                prep_summaries.append({
                                    "uniprot": uniprot,
                                    "biasdb_name": biasdb_name,
                                    "input_pdb_id": f"AF-{uniprot}",
                                    "output_path": str(output_pdb),
                                    "kept_chain": meta["kept_chain"],
                                    "dropped_chains": meta["dropped_chains"],
                                    "het_residues_kept": meta["het_residues_kept"],
                                    "het_residues_dropped": meta["het_residues_dropped"],
                                    "waters_dropped": meta["waters_dropped"],
                                    "altloc_atoms_dropped": meta["altloc_atoms_dropped"],
                                    "input_sha256": meta["input_sha256"],
                                    "output_sha256": meta["output_sha256"],
                                    "status": (
                                        f"ok_alphafold_after_pdb_failed_"
                                        f"pLDDT={af_meta['mean_plddt']:.1f}"
                                    ),
                                })
                                fell_back_ok = True
                            except ValueError as af_e:
                                prep_summaries.append({
                                    "uniprot": uniprot,
                                    "biasdb_name": biasdb_name,
                                    "input_pdb_id": f"AF-{uniprot}",
                                    "status": f"af_prep_failed: {af_e}",
                                })
                    except AlphaFoldFetchError as af_e:
                        logger.warning("AlphaFold fallback failed for %s: %s",
                                       uniprot, af_e)
                if not fell_back_ok:
                    prep_summaries.append({
                        "uniprot": uniprot,
                        "biasdb_name": biasdb_name,
                        "input_pdb_id": best.pdb_id,
                        "status": f"prep_failed: {e}",
                    })

        # Emit preferred_pdbs.tsv
        preferred_path = Path("data/registry/preferred_pdbs.tsv")
        write_preferred_tsv(rows, preferred_path)
        logger.info("Wrote %d preferred-PDB rows to %s", len(rows), preferred_path)

        # Emit receptor_audit.md
        audit_path = Path(self.paths.get("processed_data", "data/processed")) \
            / "receptor_audit.md"
        self._emit_receptor_audit(rows, prep_summaries, af_fallback_uniprots, audit_path)
        logger.info("Wrote receptor audit to %s", audit_path)

    def _emit_receptor_audit(
        self,
        rows: list[dict],
        prep_summaries: list[dict],
        af_fallback: list[tuple[str, str]],
        out_path: Path,
    ) -> None:
        """Reviewer-facing receptor summary (analog of dataset_audit.md)."""
        n_total = len(rows)
        n_pdb = sum(1 for r in rows if r["selected_pdb"])
        n_af = len(af_fallback)
        # `ok` from PDB-only path; `ok_alphafold...` from the AF fallback paths.
        n_prep_ok = sum(
            1 for s in prep_summaries
            if str(s.get("status", "")).startswith("ok")
        )
        n_prep_ok_pdb = sum(
            1 for s in prep_summaries if s.get("status") == "ok"
        )
        n_prep_ok_af = sum(
            1 for s in prep_summaries
            if str(s.get("status", "")).startswith("ok_alphafold")
        )

        def _md_table(records: list[dict], cols: list[str]) -> str:
            head = "| " + " | ".join(cols) + " |"
            sep = "| " + " | ".join("---" for _ in cols) + " |"
            body = [
                "| " + " | ".join(str(r.get(c, "") if r.get(c) is not None else "")
                                  for c in cols) + " |"
                for r in records
            ]
            return "\n".join([head, sep, *body])

        md = []
        md.append("# CancerAg receptor audit (Stage 03)")
        md.append("")
        md.append(
            f"_Generated: {datetime.now(timezone.utc).isoformat()}_  "
        )
        md.append(
            f"_Registry rows requested: {n_total} ; PDB selected: {n_pdb} ; "
            f"no PDB cached: {n_af} ; "
            f"prepared OK: {n_prep_ok} ({n_prep_ok_pdb} from PDB, "
            f"{n_prep_ok_af} from AlphaFold fallback)_  "
        )
        md.append("")
        md.append("## Per-receptor PDB selection")
        md.append("")
        md.append(_md_table(
            rows,
            ["uniprot", "biasdb_name", "selected_pdb", "resolution",
             "state", "ligand_resname", "score", "reason"],
        ))
        md.append("")

        if af_fallback:
            md.append("## Receptors with no cached PDB (need AlphaFold fallback)")
            md.append("")
            md.append("| uniprot | biasdb_name |")
            md.append("| --- | --- |")
            for u, n in af_fallback:
                md.append(f"| {u} | {n} |")
            md.append("")

        md.append("## Per-receptor preparation summary")
        md.append("")
        md.append(_md_table(
            prep_summaries,
            ["uniprot", "biasdb_name", "input_pdb_id", "kept_chain",
             "dropped_chains", "het_residues_kept", "het_residues_dropped",
             "waters_dropped", "altloc_atoms_dropped", "status"],
        ))
        md.append("")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(md))


def _legacy_folder_name(biasdb_name: str) -> str:
    """Mirror the sanitization the legacy receptor_retriever used to build
    folder names from BiasDB receptor names."""
    return (
        biasdb_name.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .lower()
    )


# --- BINDING SITE EXTRACTION (for later use) ---


def extract_binding_site(pdb_file: str, ligand_name: str = None, padding: float = 5.0):
    """
    Calculates the binding site center and dimensions from a co-crystallized ligand.

    Args:
        pdb_file (str): Path to the input PDB file (the raw, not cleaned one).
        ligand_name (str, optional): The 3-letter residue name of the ligand.
                                     If None, the first non-solvent/non-ion
                                     heteroatom will be used. Defaults to None.
        padding (float, optional): Extra padding (in Angstroms) to add to the
                                   bounding box dimensions. Defaults to 5.0.

    Returns:
        dict: A dictionary containing the center and size of the binding site box.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("receptor", pdb_file)
    except Exception as e:
        logger.error(f"Could not parse PDB file {pdb_file}: {e}")
        return None

    # Use the curated single source of truth — see preprocessing/het_resnames
    # — instead of the old 13-name local list that missed lipids, detergents,
    # glycans, buffers, and cofactors.
    from cancerag.preprocessing.het_resnames import LIGAND_AUTO_DETECT_IGNORE

    ligand_atoms = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].startswith("H_"):
                    res_name = residue.get_resname().strip().upper()
                    if ligand_name and res_name == ligand_name.upper():
                        ligand_atoms.extend(list(residue.get_atoms()))
                        break
                    elif not ligand_name and res_name not in LIGAND_AUTO_DETECT_IGNORE:
                        logger.info(
                            f"Auto-detecting ligand in {pdb_file}. Found '{res_name}'."
                        )
                        ligand_atoms.extend(list(residue.get_atoms()))
                        break
            if ligand_atoms:
                break
        if ligand_atoms:
            break

    if not ligand_atoms:
        logger.warning(
            f"No suitable ligand found in {pdb_file}. Cannot define binding site."
        )
        return None

    coords = np.array([atom.get_coord() for atom in ligand_atoms])
    center = np.mean(coords, axis=0)
    min_coords, max_coords = np.min(coords, axis=0), np.max(coords, axis=0)
    size = (max_coords - min_coords) + (2 * padding)

    binding_site = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }
    logger.info(
        f"Calculated binding site for {pdb_file}: Center {center.round(2)}, Size {size.round(2)}"
    )
    return binding_site


# --- STAGE 03 — receptor curation v2 ----------------------------------------
# (see improvements/03_receptor_curation.md)
#
# The legacy `ReceptorPreprocessor.ProteinSelect` keeps every standard residue
# but drops every HETATM — including the conserved Na+ ion that stabilizes the
# inactive state of class A GPCRs and modified residues like MSE. It also
# keeps every protein chain, so G-protein subunits and nanobody fragments end
# up in the docking input. The functions below replace that behaviour without
# breaking the legacy API.


class ReceptorChainSelect(Select):
    """Bio.PDB selector that retains a single receptor chain (and optional
    explicit HETATM resnames such as the conserved Na+ ion), and drops
    alternate locations to a single canonical altloc."""

    def __init__(
        self,
        keep_chain: str,
        retain_het_resnames: frozenset[str] = RETAIN_HET_RESNAMES_DEFAULT,
        altloc: str = "A",
    ) -> None:
        self.keep_chain = keep_chain
        self.retain_het_resnames = frozenset(
            r.upper() for r in retain_het_resnames
        )
        self.altloc = altloc

    def accept_chain(self, chain) -> int:  # noqa: D401  (Bio.PDB API)
        return 1 if chain.id == self.keep_chain else 0

    def accept_residue(self, residue) -> int:
        if residue.id[0] == " ":
            return 1
        if residue.id[0].startswith("H_"):
            return 1 if residue.get_resname().strip().upper() in self.retain_het_resnames else 0
        # Waters (residue.id[0] == 'W') are dropped.
        return 0

    def accept_atom(self, atom) -> int:
        # Drop everything except the chosen altloc (' ' means no altloc set).
        if atom.altloc not in (" ", "", self.altloc):
            return 0
        return 1


def _sha256_of_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_UNIPROT_SEQ_CACHE: dict[str, str | None] = {}


def _fetch_uniprot_sequence(uniprot: str) -> str | None:
    """Fetch the canonical UniProt sequence (FASTA). Cached in-memory."""
    if uniprot in _UNIPROT_SEQ_CACHE:
        return _UNIPROT_SEQ_CACHE[uniprot]
    try:
        import urllib.request
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta"
        with urllib.request.urlopen(url, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        seq = "".join(
            line.strip() for line in text.splitlines() if not line.startswith(">")
        )
        _UNIPROT_SEQ_CACHE[uniprot] = seq or None
    except Exception as exc:  # network failure -> rescue can't run; fall through
        logger.warning(
            "uniprot fasta fetch failed for %s: %s", uniprot, exc
        )
        _UNIPROT_SEQ_CACHE[uniprot] = None
    return _UNIPROT_SEQ_CACHE[uniprot]


def _chain_one_letter(structure, chain_id: str) -> str:
    chars: list[str] = []
    for model in structure:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                chars.append(_THREE_TO_ONE.get(
                    residue.get_resname().strip().upper(), "X"
                ))
            break
        break
    return "".join(chars)


def _detect_chain_by_sequence(
    pdb_path: Path | str,
    target_uniprot: str,
    chain_lengths: dict[str, int],
) -> str | None:
    """Pick the chain whose residue sequence aligns best to the canonical
    UniProt sequence. Used when DBREF is missing UNP records (engineered
    construct with PDB self-references)."""
    target_seq = _fetch_uniprot_sequence(target_uniprot)
    if not target_seq:
        return None
    target_set = set(target_seq)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", str(pdb_path))
    best_chain: str | None = None
    best_id: float = 0.0
    for chain_id, length in chain_lengths.items():
        if length < 200:
            continue
        chain_seq = _chain_one_letter(structure, chain_id)
        if not chain_seq:
            continue
        # Crude metric: longest common subsequence ≈ identity. Use a sliding
        # window of chain_seq over target_seq to find best contiguous match.
        # For GPCR receptor chains, ~80% of residues are present in the UniProt
        # sequence (with occasional gaps). For Gα, sequence overlap with a
        # GPCR target is essentially zero (~5% by chance).
        in_target = sum(1 for c in chain_seq if c in target_set) / len(chain_seq)
        # Quick filter — if the residue alphabet alone doesn't match, skip
        # detailed scoring.
        if in_target < 0.85:
            continue
        # Count contiguous-window identity.
        from difflib import SequenceMatcher
        ident = SequenceMatcher(None, chain_seq, target_seq).ratio()
        if ident > best_id:
            best_id, best_chain = ident, chain_id
    # Require ≥30% identity to claim a match (engineered constructs typically
    # land at 60-90%; non-receptor chains land at <10%).
    if best_chain is not None and best_id >= 0.30:
        logger.info(
            "sequence-rescue picked chain %s for %s at identity=%.2f",
            best_chain, target_uniprot, best_id,
        )
        return best_chain
    return None


def _parse_dbref_chain_to_uniprot(
    pdb_path: Path | str,
) -> dict[str, list[str]]:
    """Parse `DBREF` records and return {chain_id: [uniprot_acc, ...]}.

    Each chain may carry several DBREFs — e.g. a fusion construct with BRIL
    (`P0ABE7`) at the N-terminus followed by the actual receptor. We keep
    the full ordered list so the caller can choose the one that matches the
    target UniProt rather than `setdefault`-ing on the first hit.

    Only `UNP` references count — `PDB` self-references (engineered
    constructs) are ignored.
    """
    mapping: dict[str, list[str]] = {}
    for raw in Path(pdb_path).read_text().splitlines():
        # DBREF  <pdb> <chain> <seqB> <seqE> UNP    <acc>     <name>      ...
        if raw.startswith("DBREF  "):
            parts = raw.split()
            if len(parts) >= 7 and parts[5] == "UNP":
                chain = parts[2]
                acc = parts[6]
                mapping.setdefault(chain, []).append(acc)
        elif raw.startswith("DBREF1 "):
            # Multi-line variant — left unhandled; we fall back to length
            # heuristic if no DBREF UNP matches the target.
            continue
    return mapping


def detect_receptor_chain(
    pdb_path: Path | str,
    prefer: str = "A",
    target_uniprot: str | None = None,
) -> str | None:
    """Return the chain id that looks like the receptor.

    UniProt-anchored selection: if ``target_uniprot`` is given and any chain's
    DBREF maps to it, return that chain (this is the authoritative answer for
    GPCR-Gprotein complexes where chain A is the Gα and the receptor sits on
    chain R / D / F / etc.).

    Otherwise fall back to the length heuristic: prefer ``prefer`` if it has
    at least 200 residues, else pick the longest chain ≥ 200 residues.
    Returns None if no chain qualifies.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", str(pdb_path))
    chain_lengths: dict[str, int] = {}
    for model in structure:
        for chain in model:
            n = sum(1 for r in chain if r.id[0] == " ")
            chain_lengths[chain.id] = chain_lengths.get(chain.id, 0) + n
        break  # only the first model
    if not chain_lengths:
        return None

    if target_uniprot:
        dbref = _parse_dbref_chain_to_uniprot(pdb_path)
        candidates = [
            (c, chain_lengths.get(c, 0))
            for c, accs in dbref.items()
            if target_uniprot in accs and c in chain_lengths
        ]
        if candidates:
            # Pick the longest chain that maps to the target UniProt.
            candidates.sort(key=lambda kv: kv[1], reverse=True)
            return candidates[0][0]
        # No DBREF match — try sequence-identity rescue against the chain
        # SEQRES, in case the receptor chain is an engineered construct
        # whose DBREF only points at PDB self-refs.
        rescue = _detect_chain_by_sequence(
            pdb_path, target_uniprot, chain_lengths
        )
        if rescue is not None:
            return rescue

    if prefer in chain_lengths and chain_lengths[prefer] >= 200:
        return prefer
    longest, n = max(chain_lengths.items(), key=lambda kv: kv[1])
    return longest if n >= 200 else None


def prepare_receptor(
    input_pdb: Path | str,
    output_pdb: Path | str,
    *,
    keep_chain: str | None = None,
    target_uniprot: str | None = None,
    retain_het_resnames: frozenset[str] = RETAIN_HET_RESNAMES_DEFAULT,
    altloc: str = "A",
    structure_source: str = "pdb",
    write_meta: bool = True,
) -> dict:
    """Curate a single receptor PDB and emit a `.prep.meta.json` sidecar.

    Returns the metadata dict that is written to disk (also returned in
    memory so tests can inspect it without re-reading the file).

    Idempotent by content: a fresh meta sidecar that matches the current
    input SHA-256 short-circuits re-prep.
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    input_sha = _sha256_of_path(input_pdb)

    meta_path = output_pdb.with_suffix(output_pdb.suffix + ".prep.meta.json")
    if write_meta and output_pdb.exists() and meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text())
            if existing_meta.get("input_sha256") == input_sha:
                logger.info(
                    "receptor prep cache hit for %s (input sha matches)", input_pdb
                )
                return existing_meta
        except json.JSONDecodeError:
            logger.warning("Stale meta sidecar at %s; re-preparing", meta_path)

    chosen_chain = keep_chain or detect_receptor_chain(
        input_pdb, target_uniprot=target_uniprot
    )
    if chosen_chain is None:
        raise ValueError(
            f"prepare_receptor: no chain >=200 residues in {input_pdb}; "
            "explicit keep_chain required"
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", str(input_pdb))

    # Inventory what we are dropping for the metadata sidecar.
    dropped_chains: list[str] = []
    waters_dropped = 0
    het_kept: list[str] = []
    het_dropped: list[str] = []
    altlocs_dropped = 0
    for model in structure:
        for chain in model:
            if chain.id != chosen_chain:
                dropped_chains.append(chain.id)
                continue
            for residue in chain:
                hetflag = residue.id[0]
                resname = residue.get_resname().strip().upper()
                if hetflag == "W":
                    waters_dropped += 1
                elif hetflag.startswith("H_"):
                    if resname in retain_het_resnames:
                        het_kept.append(resname)
                    else:
                        het_dropped.append(resname)
                for atom in residue:
                    if atom.altloc not in (" ", "", altloc):
                        altlocs_dropped += 1
        break

    selector = ReceptorChainSelect(
        keep_chain=chosen_chain,
        retain_het_resnames=retain_het_resnames,
        altloc=altloc,
    )
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_pdb), selector)

    meta = {
        "input_pdb": str(input_pdb),
        "output_pdb": str(output_pdb),
        "input_sha256": input_sha,
        "output_sha256": _sha256_of_path(output_pdb),
        "structure_source": structure_source,
        "kept_chain": chosen_chain,
        "dropped_chains": sorted(set(dropped_chains)),
        "retain_het_resnames": sorted(retain_het_resnames),
        "het_residues_kept": sorted(set(het_kept)),
        "het_residues_dropped": sorted(set(het_dropped)),
        "waters_dropped": waters_dropped,
        "altloc_policy": altloc,
        "altloc_atoms_dropped": altlocs_dropped,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "biopython_PDBIO_select",
    }
    if write_meta:
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return meta
