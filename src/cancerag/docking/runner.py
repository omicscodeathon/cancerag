import os
from cancerag.utils.receptor_mapper import ReceptorMapper
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def _run_single_docking_task(task: dict) -> dict:
    """
    Executes a single AutoDock Vina docking command.

    Args:
        task (dict): Contains the command and metadata for a single docking run.

    Returns:
        dict: The original task dict with 'success' and 'error' keys added.
    """
    try:
        # Execute the command using shell=True because the command is a single string
        # with shell features like redirection.
        subprocess.run(
            task["cmd"],
            check=True,
            shell=True,
            capture_output=True,  # Still capture output to check for errors if files aren't created
            text=True,
        )
        # Vina might return exit code 0 even if it fails, so we check for output files.
        if os.path.exists(task["out_file"]) and os.path.getsize(task["out_file"]) > 0:
            return {"success": True, **task}
        else:
            # If output file isn't created, read the log to find out why.
            error_message = "Output file not created or is empty."
            if os.path.exists(task["log_file"]):
                with open(task["log_file"], "r") as f:
                    error_message += f"\nLog content:\n{f.read()}"
            return {"success": False, "error": error_message, **task}
    except subprocess.CalledProcessError as e:
        # This catches non-zero exit codes. The error is in stderr.
        error_message = e.stderr.strip()
        return {
            "success": False,
            "error": f"Vina Execution Error: {error_message}",
            **task,
        }
    except Exception as e:
        # This catches other exceptions, like file not found if 'vina' isn't in PATH.
        return {"success": False, "error": f"Unexpected Python Error: {e}", **task}


def run_docking_multiprocess(
    prepared_receptors: dict,
    prepared_ligands: list,
    binding_sites: dict,
    output_dir: str,
    num_cpu: int,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    ligand_receptor_mapping: dict = None,
) -> dict:
    """
    Runs docking for ligands against their SPECIFIC target receptors only.
    
    This is the CORRECT approach for biased agonism - each ligand is docked
    only against its target receptor, not against all receptors.

    Args:
        prepared_receptors (dict): Receptors ready for docking (name -> path).
        prepared_ligands (list): Ligands ready for docking.
        binding_sites (dict): Binding site info for each receptor.
        output_dir (str): Main directory for docking results.
        num_cpu (int): Number of CPUs to use.
        exhaustiveness (int): Vina's exhaustiveness parameter.
        num_modes (int): Vina's num_modes parameter.
        ligand_receptor_mapping (dict): Maps ligand names to their target receptors.

    Returns:
        dict: A dictionary containing the raw results for each receptor.
    """
    all_results = {}
    
    # Check if we have ligand-receptor mapping (corrected approach)
    if ligand_receptor_mapping is not None:
        print(f"Starting CORRECTED docking with {num_cpu} processes...")
        print("🎯 Each ligand will be docked against its SPECIFIC target receptor only!")
        
        # Group ligands by their target receptor
        ligands_by_receptor = {}
        for ligand in prepared_ligands:
            ligand_name = ligand['name']
            if ligand_name in ligand_receptor_mapping:
                target_receptor = ligand_receptor_mapping[ligand_name]
                if target_receptor not in ligands_by_receptor:
                    ligands_by_receptor[target_receptor] = []
                ligands_by_receptor[target_receptor].append(ligand)
            else:
                print(f"⚠️  Warning: No target receptor found for ligand {ligand_name}")
        
        print("\n📊 DOCKING SUMMARY:")
        print(f"Total ligands: {len(prepared_ligands)}")
        print(f"Receptors with ligands: {len(ligands_by_receptor)}")
        
        total_dockings = sum(len(ligands) for ligands in ligands_by_receptor.values())
        print(f"Total docking calculations: {total_dockings}")
        print(f"Reduction from naive approach: {len(prepared_ligands) * len(prepared_receptors) - total_dockings:,} fewer dockings!")
        
        # Process each receptor
        for receptor_name, receptor_ligands in ligands_by_receptor.items():
            if receptor_name not in prepared_receptors:
                print(f"⚠️  Warning: Receptor {receptor_name} not found in prepared receptors")
                continue
                
            if receptor_name not in binding_sites:
                print(f"⚠️  Warning: No binding site found for receptor {receptor_name}")
                continue
                
            print(f"\n🧬 Docking {len(receptor_ligands)} ligands against {receptor_name}...")
            
            receptor_pdbqt = prepared_receptors[receptor_name]
            binding_site = binding_sites[receptor_name]
            receptor_outdir = os.path.join(output_dir, receptor_name)
            os.makedirs(receptor_outdir, exist_ok=True)

            docking_tasks = []
            for ligand in receptor_ligands:
                out_file = os.path.join(receptor_outdir, f"{ligand['name']}_docked.pdbqt")
                log_file = os.path.join(receptor_outdir, f"{ligand['name']}_log.txt")

                # Skip if docking results already exist
                if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                    print(f"    - Skipping {ligand['name']} (results already exist)")
                    continue

                # The command is now a single f-string for clarity and direct shell execution.
                # Redirection `>` and `2>&1` handles capturing all output to the log file.
                cmd = (
                    f"vina --receptor '{receptor_pdbqt}' --ligand '{ligand['pdbqt_file']}' "
                    f"--center_x {binding_site['center_x']} --center_y {binding_site['center_y']} --center_z {binding_site['center_z']} "
                    f"--size_x {binding_site['size_x']} --size_y {binding_site['size_y']} --size_z {binding_site['size_z']} "
                    f"--out '{out_file}' --exhaustiveness {exhaustiveness} --num_modes {num_modes} --cpu 1 "
                    f"> '{log_file}' 2>&1"
                )

                docking_tasks.append(
                    {
                        "cmd": cmd,
                        "ligand_name": ligand["name"],
                        "out_file": out_file,
                        "log_file": log_file,
                        "receptor_name": receptor_name,
                    }
                )

            # Execute docking tasks in parallel
            with ProcessPoolExecutor(max_workers=num_cpu) as executor:
                results = list(
                    tqdm(
                        executor.map(_run_single_docking_task, docking_tasks),
                        total=len(docking_tasks),
                        desc=f"Docking {receptor_name}",
                    )
                )

            # Store results
            all_results[receptor_name] = results
            
            # Count successes and failures
            successes = sum(1 for r in results if r["success"])
            failures = len(results) - successes
            print(f"  ✅ {successes} successful, ❌ {failures} failed")

        return all_results
        
    else:
        # Fallback to original approach (for backward compatibility)
        print(f"Starting docking with {num_cpu} processes...")
        print("⚠️  WARNING: Using naive approach - docking all ligands against all receptors!")

        for receptor_name, receptor_pdbqt in prepared_receptors.items():
            print(f"\nDocking against {receptor_name}...")
            binding_site = binding_sites[receptor_name]
            receptor_outdir = os.path.join(output_dir, receptor_name)
            os.makedirs(receptor_outdir, exist_ok=True)

            docking_tasks = []
            for ligand in prepared_ligands:
                out_file = os.path.join(receptor_outdir, f"{ligand['name']}_docked.pdbqt")
                log_file = os.path.join(receptor_outdir, f"{ligand['name']}_log.txt")

                # Skip if docking results already exist
                if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
                    print(f"    - Skipping {ligand['name']} (results already exist)")
                    continue

                cmd = (
                    f"vina --receptor '{receptor_pdbqt}' --ligand '{ligand['pdbqt_file']}' "
                    f"--center_x {binding_site['center_x']} --center_y {binding_site['center_y']} --center_z {binding_site['center_z']} "
                    f"--size_x {binding_site['size_x']} --size_y {binding_site['size_y']} --size_z {binding_site['size_z']} "
                    f"--out '{out_file}' --exhaustiveness {exhaustiveness} --num_modes {num_modes} --cpu 1 "
                    f"> '{log_file}' 2>&1"
                )

                docking_tasks.append(
                    {
                        "cmd": cmd,
                        "ligand_name": ligand["name"],
                        "out_file": out_file,
                        "log_file": log_file,
                        "receptor_name": receptor_name,
                    }
                )

            # Execute docking tasks in parallel
            with ProcessPoolExecutor(max_workers=num_cpu) as executor:
                results = list(
                    tqdm(
                        executor.map(_run_single_docking_task, docking_tasks),
                        total=len(docking_tasks),
                        desc=f"Docking {receptor_name}",
                    )
                )

            all_results[receptor_name] = results

        return all_results


def create_ligand_receptor_mapping(ligands_df):
    """
    Create a mapping from ligand names to their target receptors.
    
    Args:
        ligands_df (pd.DataFrame): DataFrame with ligand data including receptor_subtype column
        
    Returns:
        dict: Maps ligand names to receptor names
    """
    mapping = {}
    # Reuse the same normalization logic as the pipeline utilities to stay in sync with main.py
    mapper = ReceptorMapper()
    
    for _, row in ligands_df.iterrows():
        ligand_name = row.get('ligand_name', f"ligand_{row.name}")
        receptor_subtype = row.get('receptor_subtype', '')
        
        if receptor_subtype:
            normalized_receptor = mapper._normalize_receptor_name(receptor_subtype)
            mapping[ligand_name] = normalized_receptor
    
    return mapping