#!/usr/bin/env python3
"""
Script to prepare only the selected high-yield receptors for deployment.
This reduces the receptor data from ~214MB (306 files) to ~32MB (46 files).
"""

import json
import shutil
from pathlib import Path

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    structure_summary = base_dir / "data/processed/structure_selection_summary.json"
    receptors_dir = base_dir / "data/processed/receptors"
    deployment_dir = base_dir / "data/processed/receptors_selected"

    # Read the selected structures
    print("Reading structure selection summary...")
    with open(structure_summary, 'r') as f:
        selected = json.load(f)

    # Extract PDB IDs
    pdb_ids = [info['selected_pdb'] for receptor, info in selected.items()]
    print(f"Found {len(pdb_ids)} selected receptors")

    # Create deployment directory
    deployment_dir.mkdir(exist_ok=True)

    # Copy selected receptor files
    copied_count = 0
    missing_files = []

    for pdb_id in pdb_ids:
        # Copy both .pdb and .pdbqt files
        for ext in ['.pdb', '.pdbqt']:
            source_file = receptors_dir / f"{pdb_id}{ext}"
            if source_file.exists():
                dest_file = deployment_dir / f"{pdb_id}{ext}"
                shutil.copy2(source_file, dest_file)
                copied_count += 1
            else:
                missing_files.append(f"{pdb_id}{ext}")

    print(f"\nCopied {copied_count} receptor files to {deployment_dir}")

    if missing_files:
        print(f"\nWarning: {len(missing_files)} files not found:")
        for f in missing_files[:10]:  # Show first 10
            print(f"  - {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    # Calculate size reduction
    import subprocess
    original_size = subprocess.check_output(['du', '-sh', str(receptors_dir)]).decode().split()[0]
    selected_size = subprocess.check_output(['du', '-sh', str(deployment_dir)]).decode().split()[0]

    print(f"\nSize comparison:")
    print(f"  Original (all 306 receptors): {original_size}")
    print(f"  Selected (46 receptors): {selected_size}")
    print(f"\nReduction: {306 - len(pdb_ids)} receptors removed")

    # Create a list of selected receptors for reference
    selected_list_file = deployment_dir / "selected_receptors.txt"
    with open(selected_list_file, 'w') as f:
        f.write("# Selected high-yield GPCR receptors for deployment\n")
        f.write(f"# Total: {len(pdb_ids)} receptors\n\n")
        for receptor, info in sorted(selected.items()):
            f.write(f"{info['selected_pdb']}\t{receptor}\t"
                   f"Resolution: {info['resolution']}\t"
                   f"Score: {info['score']}\n")

    print(f"\nCreated receptor list: {selected_list_file}")
    print("\nDone! Update Dockerfile to use data/processed/receptors_selected/")

if __name__ == "__main__":
    main()
