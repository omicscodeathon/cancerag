"""
Single source of truth for HETATM residue names that should be ignored when
auto-detecting a co-crystallized orthosteric ligand.

Stage 04 fix — see improvements/04_binding_site_definition.md F4.8.

The previous pipeline duplicated a 13-name list across
`active_site_identifier.py` and `receptor_preprocessor.py`, and the list
covered only waters, common ions, and a handful of buffers. It did NOT cover
the lipids, detergents, glycans, and crystallization additives that are
present in essentially every GPCR PDB structure — meaning the legacy
"first HETATM" picker frequently selected cholesterol, oleic acid, or a PEG
fragment instead of the actual orthosteric ligand.
"""

from __future__ import annotations

from typing import Final


# Solvents and waters
WATERS: Final[frozenset[str]] = frozenset({"HOH", "WAT", "DOD", "TIP", "TIP3"})

# Monoatomic ions and small inorganic groups
IONS: Final[frozenset[str]] = frozenset(
    {
        "NA", "K", "MG", "CA", "ZN", "MN", "CL", "BR", "I", "F",
        "FE", "CU", "NI", "CO", "CD", "HG", "PB", "AL", "BA", "SR",
        "RB", "CS", "LI",
        "SO4", "PO4", "PO3", "NO3", "CO3", "ACT",  # acetate
    }
)

# Cryoprotectants, crystallization additives, common buffers
BUFFERS: Final[frozenset[str]] = frozenset(
    {
        "GOL", "EDO", "MPD", "BU2", "DMS", "DMSO", "MES", "TRS", "TRIS",
        "HEPES", "PIPES", "BIS", "BIC", "PG4", "PG5", "PEG", "PE3", "PE4",
        "PEU", "P6G", "1PE", "FMT", "BME", "MRD", "BTB", "EPE", "CIT",
        "TLA",  # tartrate
    }
)

# Lipids, detergents, fatty acids — pervasive in GPCR cryo-EM/X-ray
LIPIDS_AND_DETERGENTS: Final[frozenset[str]] = frozenset(
    {
        "CLR",  # cholesterol
        "CHS",  # cholesteryl hemisuccinate
        "OLA", "OLC", "OLE",  # oleic acid / monoolein variants
        "PLM",  # palmitic acid
        "MYR",  # myristic acid
        "STE",  # stearic acid
        "PEE", "PEF", "PCW", "POV", "POP", "PSF", "PEH",  # phospholipids
        "LMT", "DDM", "OGA", "BOG", "OG",  # n-alkyl maltosides / glucosides
        "LFA", "LMU", "LMG", "LMN",
        "LPE", "LPP",
        "Y01",  # cholesteryl analog
        "ALY",  # acetyl-lysine derivative seen as additive
    }
)

# Glycans (often on extracellular loops — not the orthosteric ligand)
GLYCANS: Final[frozenset[str]] = frozenset(
    {
        "NAG", "MAN", "BMA", "FUC", "GAL", "GLC", "BGC", "FUL",
        "SIA", "NDG", "NGA",
    }
)

# Common cofactors that should be retained for some receptors but never
# treated as "the ligand" for auto-detection of the orthosteric box.
COFACTORS: Final[frozenset[str]] = frozenset(
    {
        "GTP", "GDP", "GNP", "GSP",  # G-protein nucleotides
        "ATP", "ADP", "AMP",
        "NAD", "NAP", "NDP",
        "FAD", "FMN",
    }
)


# Modified / non-standard amino acid residues that show up as HETATM in PDB
# files (selenomethionine, phospho-serines, methylated residues, etc.).
# They are part of the protein, not a ligand.
MODIFIED_AMINO_ACIDS: Final[frozenset[str]] = frozenset(
    {
        "MSE",  # selenomethionine — extremely common in protein crystals
        "SEP",  # phospho-serine
        "TPO",  # phospho-threonine
        "PTR",  # phospho-tyrosine
        "MLY", "M3L",  # methylated lysines
        "ALY",          # acetyl-lysine
        "FME",          # N-formyl-methionine
        "CME", "OCS", "CSO",  # modified cysteines
        "HYP",          # hydroxyproline
        "PCA",          # pyroglutamic acid
        "KCX",          # carbamoyl-lysine
        "LLP",          # lysine-pyridoxal-5'-phosphate
        "NEP",          # methyl-histidine
    }
)

# Additional buffer / cryoprotectant resnames found in real GPCR PDBs that
# the initial list missed.
EXTRA_BUFFERS: Final[frozenset[str]] = frozenset(
    {
        "PGE", "PG0", "PG6",  # polyethylene glycol fragments
        "BJM",                 # crystallographic buffer
        "UNX", "UNK", "UNL",   # unknown atom / residue / ligand placeholders
    }
)


# The full ignore list applied by `active_site_identifier` and the PDB
# selector when auto-detecting the co-crystal orthosteric ligand from
# HETATM residues.
LIGAND_AUTO_DETECT_IGNORE: Final[frozenset[str]] = (
    WATERS
    | IONS
    | BUFFERS
    | LIPIDS_AND_DETERGENTS
    | GLYCANS
    | COFACTORS
    | MODIFIED_AMINO_ACIDS
    | EXTRA_BUFFERS
)


def is_ignorable_het(resname: str) -> bool:
    """Return True if a HETATM residue name should be skipped during
    orthosteric-ligand auto-detection."""
    if not resname:
        return True
    return resname.strip().upper() in LIGAND_AUTO_DETECT_IGNORE
