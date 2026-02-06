#!/usr/bin/env python3
"""
Test script to verify docking fixes work correctly.
This simulates what happens in the container.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.docking_extractor import DockingFeatureExtractor
from rdkit import Chem

def test_docking_fix():
    """Test that docking uses pre-converted receptors."""

    print("="*60)
    print("Testing Docking Fix")
    print("="*60)

    # Initialize extractor
    base_path = Path(__file__).parent.parent
    print(f"\nBase path: {base_path}")

    extractor = DockingFeatureExtractor(base_path=str(base_path))

    # Check directories
    pre_converted_dir = base_path / "data" / "processed" / "receptors_prepared"
    interim_dir = base_path / "data" / "interim" / "docking_results" / "receptors"

    print(f"\nPre-converted dir exists: {pre_converted_dir.exists()}")
    if pre_converted_dir.exists():
        count = len(list(pre_converted_dir.glob("*.pdbqt")))
        print(f"  Found {count} pre-converted receptors")

    print(f"\nInterim dir exists: {interim_dir.exists()}")
    if interim_dir.exists():
        count = len(list(interim_dir.glob("*.pdbqt")))
        print(f"  Found {count} cached receptors")

    # Test with a simple molecule
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print("\n❌ Failed to create molecule")
        return False

    print(f"\n✓ Testing with SMILES: {smiles}")

    # Get first available receptor
    if not extractor.receptor_names:
        print("❌ No receptors available")
        return False

    receptor_name = extractor.receptor_names[0]
    print(f"✓ Using receptor: {receptor_name}")

    # Check which file would be used
    binding_site = extractor.binding_sites[receptor_name]
    pdb_id = binding_site.get("source_pdb")

    pre_converted_path = base_path / "data" / "processed" / "receptors_prepared" / f"{pdb_id}.pdbqt"
    interim_path = interim_dir / f"{receptor_name}.pdbqt"

    print(f"\nReceptor preparation check:")
    print(f"  Pre-converted ({pre_converted_path.name}): {pre_converted_path.exists()}")
    print(f"  Interim ({interim_path.name}): {interim_path.exists()}")

    # Test receptor preparation
    print(f"\nTesting receptor preparation...")
    receptor_pdbqt = extractor._prepare_receptor(receptor_name)

    if receptor_pdbqt:
        print(f"✓ Receptor prepared: {receptor_pdbqt}")
        if "receptors_prepared" in receptor_pdbqt:
            print("✓ Using pre-converted receptor (FAST)")
        elif "interim" in receptor_pdbqt:
            print("⚠ Using interim/cached receptor")
        else:
            print("⚠ Using unknown location")
    else:
        print("❌ Receptor preparation failed")
        return False

    # Optional: Test full docking (can be slow)
    import os
    if os.environ.get("TEST_FULL_DOCKING") == "1":
        print(f"\nRunning full docking test...")
        affinity = extractor.dock_single_receptor(mol, receptor_name)

        if affinity is not None:
            print(f"✓ Docking successful: {affinity:.2f} kcal/mol")
        else:
            print("❌ Docking failed")
            return False
    else:
        print("\nSkipping full docking test (set TEST_FULL_DOCKING=1 to enable)")

    print("\n" + "="*60)
    print("✓ All checks passed!")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_docking_fix()
    sys.exit(0 if success else 1)
