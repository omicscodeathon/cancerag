#!/usr/bin/env python
"""
Standalone script to download AlphaFold structures for missing receptors.

Usage:
    python download_alphafold_structures.py
    python download_alphafold_structures.py --receptors "5HT1A receptor" "MC4 receptor"
    python download_alphafold_structures.py --all --force
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cancerag.data_collection.alphafold_retriever import AlphaFoldRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# List of receptors commonly missing from PDB
MISSING_RECEPTORS = [
    "5HT1A receptor",
    "5HT1B receptor",
    "5HT2B receptor",
    "5HT7 receptor",
    "A2B receptor",
    "S1P1 receptor",
    "NTS1 receptor",
    "MC4 receptor",
    "MC5 receptor",
    "OXE receptor",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download AlphaFold predicted structures for GPCRs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download default missing receptors
    python download_alphafold_structures.py

    # Download specific receptors
    python download_alphafold_structures.py --receptors "5HT1A receptor" "D3 receptor"

    # Download all mapped receptors
    python download_alphafold_structures.py --all

    # Force re-download existing structures
    python download_alphafold_structures.py --force

    # Custom output directory
    python download_alphafold_structures.py --output-dir data/alphafold_structures
        """,
    )

    parser.add_argument(
        "--receptors", nargs="+", help="Specific receptor names to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all receptors in the UniProt mapping",
    )
    parser.add_argument(
        "--output-dir",
        default="data/pdb",
        help="Output directory for structures (default: data/pdb)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--list-receptors",
        action="store_true",
        help="List all available receptor mappings and exit",
    )

    args = parser.parse_args()

    # Initialize retriever
    retriever = AlphaFoldRetriever(output_dir=args.output_dir)

    # List receptors if requested
    if args.list_receptors:
        print("\nAvailable Receptor → UniProt Mappings:")
        print("=" * 60)
        for receptor, uniprot in sorted(retriever.RECEPTOR_UNIPROT_MAP.items()):
            print(f"  {receptor:<35} → {uniprot}")
        print("=" * 60)
        print(f"\nTotal: {len(retriever.RECEPTOR_UNIPROT_MAP)} receptors")
        return

    # Determine which receptors to download
    if args.all:
        receptors = list(retriever.RECEPTOR_UNIPROT_MAP.keys())
        logger.info(f"Downloading all {len(receptors)} mapped receptors...")
    elif args.receptors:
        receptors = args.receptors
    else:
        receptors = MISSING_RECEPTORS
        logger.info("Downloading commonly missing receptors...")

    # Download structures
    print("\n" + "=" * 60)
    print("AlphaFold Structure Download")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Number of receptors: {len(receptors)}")
    print(f"Force re-download: {args.force}")
    print("=" * 60 + "\n")

    results = retriever.download_for_receptors(receptors, force_download=args.force)

    # Print detailed results
    print("\n" + "=" * 60)
    print("Download Results:")
    print("=" * 60)

    successful = []
    failed = []
    skipped = []

    for receptor, path in results.items():
        if path:
            if retriever.stats["total_skipped"] > 0 and Path(path).exists():
                status = "⊘ Skipped (exists)"
                skipped.append(receptor)
            else:
                status = "✓ Success"
                successful.append(receptor)
            print(f"  {status}: {receptor}")
            print(f"            → {path}")
        else:
            status = "✗ Failed"
            failed.append(receptor)
            print(f"  {status}: {receptor}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total receptors: {len(receptors)}")
    print(f"  ✓ Downloaded: {len(successful)}")
    print(f"  ⊘ Skipped: {len(skipped)}")
    print(f"  ✗ Failed: {len(failed)}")
    print("=" * 60)

    # Print failed receptors with suggestions
    if failed:
        print("\nFailed Downloads (no UniProt mapping or AlphaFold prediction):")
        for receptor in failed:
            print(f"  - {receptor}")
        print("\nTip: You can add custom mappings with:")
        print("  retriever.add_uniprot_mapping('Receptor Name', 'UNIPROT_ID')")

    # Print summary file location
    summary_file = Path(args.output_dir) / "alphafold_summary.json"
    if summary_file.exists():
        print(f"\nDetailed summary saved to: {summary_file}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
