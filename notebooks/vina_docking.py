import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium", app_title="Vina Docking")


@app.cell
def _():
    import marimo as mo
    import os
    import subprocess
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem, PandasTools
    import multiprocessing
    from tqdm import tqdm
    import glob
    import shutil
    import matplotlib.pyplot as plt
    import seaborn as sns
    from concurrent.futures import ProcessPoolExecutor
    import json
    import re
    from Bio.PDB import PDBParser, PDBIO
    import sys

    class ERDockingPipeline:
        """
        Pipeline for molecular docking against different estrogen receptor conformations.
        Supports AutoDock Vina as the primary docking engine with options for alternative tools.
        """

        def __init__(self, 
                     ligand_file, 
                     output_dir="docking_results",
                     receptor_structures=None, 
                     binding_sites=None,
                     num_cpu=None,
                     docking_software="vina"):
            """
            Initialize the docking pipeline.

            Parameters:
            -----------
            ligand_file : str
                Path to SDF file containing prepared ligands
            output_dir : str
                Directory to store docking results
            receptor_structures : dict
                Dictionary mapping receptor names to PDB file paths
                If None, default structures will be used
            binding_sites : dict
                Dictionary mapping receptor names to binding site definitions
                If None, default binding sites will be used
            num_cpu : int
                Number of CPU cores to use for parallel processing
                If None, will use available cores - 1
            docking_software : str
                Docking software to use ('vina', 'smina', 'gnina')
            """
            self.ligand_file = ligand_file
            self.output_dir = output_dir
            self.docking_software = docking_software.lower()

            # Set default receptor structures if not provided
            if receptor_structures is None:
                self.receptor_structures = {
                    'ER_alpha': '../data/pdb/er_alpha/1A52.pdb',  # Replace with actual PDB IDs
                    'ER_beta': '../data/pdb/er_beta/1NDE.pdb',    # e.g., 1A52 for ER-alpha
                    'ER_nuclear': '../data/pdb/er_complex/3OMO.pdb'
                }
            else:
                self.receptor_structures = receptor_structures

            # Define default binding sites if not provided
            if binding_sites is None:
                # These are example coordinates - replace with actual binding site coordinates
                self.binding_sites = {
                    'ER_alpha': {
                        'center_x': 30.45, 'center_y': 5.82, 'center_z': 24.33,
                        'size_x': 20.0, 'size_y': 20.0, 'size_z': 20.0
                    },
                    'ER_beta': {
                        'center_x': 11.15, 'center_y': 15.98, 'center_z': 9.76,
                        'size_x': 20.0, 'size_y': 20.0, 'size_z': 20.0
                    },
                    'ER_nuclear': {
                        'center_x': -5.44, 'center_y': 20.01, 'center_z': 13.29,
                        'size_x': 20.0, 'size_y': 20.0, 'size_z': 20.0
                    }
                }
            else:
                self.binding_sites = binding_sites

            # Set number of CPU cores
            if num_cpu is None:
                self.num_cpu = max(1, multiprocessing.cpu_count() - 1)
            else:
                self.num_cpu = num_cpu

            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            # Store ligands and results
            self.ligands = None
            self.docking_results = {}

            # Check if required software is installed
            self.check_dependencies()

        def check_dependencies(self):
            """Check if required software dependencies are installed and install if missing."""
            missing_deps = []
        
            # Check AutoDock Vina
            if self.docking_software == "vina":
                try:
                    result = subprocess.run(["vina", "--help"], 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, 
                                           text=True)
                    if result.returncode != 0:
                        missing_deps.append("vina")
                        print("Warning: AutoDock Vina not found in PATH.")
                except FileNotFoundError:
                    missing_deps.append("vina")
                    print("Warning: AutoDock Vina not found.")
        
            # Check Open Babel for format conversion
            try:
                result = subprocess.run(["obabel", "-H"], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, 
                                       text=True)
                if result.returncode != 0:
                    missing_deps.append("openbabel")
                    print("Warning: Open Babel not found in PATH.")
            except FileNotFoundError:
                missing_deps.append("openbabel")
                print("Warning: Open Babel not found.")
        
            # If any dependencies are missing, try to install them
            if missing_deps:
                print("Attempting to install missing dependencies...")
                self._install_dependencies(missing_deps)
            else:
                print("All required dependencies are installed.")

        def _install_dependencies(self, missing_deps):
            """Try to install missing dependencies using available package managers."""
            # Check which package managers are available
            package_managers = []
        
            # Check for conda
            try:
                subprocess.run(["conda", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
                package_managers.append("conda")
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        
            # Check for poetry
            try:
                subprocess.run(["poetry", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
                package_managers.append("poetry")
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        
            # Check for pip
            try:
                subprocess.run([sys.executable, "-m", "pip", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
                package_managers.append("pip")
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        
            if not package_managers:
                print("No package managers (conda, poetry, or pip) found. Please install the dependencies manually.")
                return
        
            # Map dependencies to package names in different package managers
            package_map = {
                "vina": {
                    "conda": "autodock-vina",
                    "pip": "vina",  # This might not exist on PyPI
                    "poetry": "vina"  # This might not exist as a poetry package
                },
                "openbabel": {
                    "conda": "openbabel",
                    "pip": "openbabel",
                    "poetry": "openbabel"
                }
            }
        
            # Try to install using the available package managers
            for manager in package_managers:
                print(f"Attempting to install with {manager}...")
                success = True
            
                for dep in missing_deps:
                    if dep not in package_map or manager not in package_map[dep]:
                        print(f"No known {manager} package for {dep}. Skipping.")
                        continue
                    
                    package_name = package_map[dep][manager]
                    try:
                        if manager == "conda":
                            subprocess.run(["conda", "install", "-y", package_name], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          check=True)
                        elif manager == "pip":
                            subprocess.run([sys.executable, "-m", "pip", "install", package_name], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          check=True)
                        elif manager == "poetry":
                            subprocess.run(["poetry", "add", package_name], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, 
                                          check=True)
                        print(f"Successfully installed {dep} using {manager}.")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to install {dep} using {manager}: {e}")
                        success = False
            
                if success:
                    print(f"All dependencies successfully installed using {manager}.")
                    return
        
            print("Could not install all dependencies. Please install them manually:")
            for dep in missing_deps:
                print(f"  - {dep}")
            
        def load_ligands(self):
            """Load ligands from SDF file."""
            print(f"Loading ligands from {self.ligand_file}...")

            # Use RDKit to load the molecules
            mols = []
            for mol in Chem.SDMolSupplier(self.ligand_file):
                if mol is not None:
                    mols.append(mol)

            print(f"Loaded {len(mols)} valid molecules")
            self.ligands = mols
            return mols

        def prepare_receptors(self):
            """Prepare receptor structures for docking."""
            prepared_receptors = {}

            for receptor_name, pdb_file in self.receptor_structures.items():
                output_pdbqt = os.path.join(self.output_dir, f"{receptor_name}.pdbqt")

                # Check if PDB file exists
                if not os.path.exists(pdb_file):
                    print(f"Error: Receptor PDB file {pdb_file} not found.")
                    print(f"You can download it from the PDB website or use a different structure.")
                    continue

                # Convert PDB to PDBQT using Open Babel
                print(f"Preparing {receptor_name} structure...")
                try:
                    cmd = [
                        "obabel", 
                        pdb_file, 
                        "-O", output_pdbqt, 
                        "-xr"  # Add hydrogens and compute partial charges
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    prepared_receptors[receptor_name] = output_pdbqt
                    print(f"Prepared {receptor_name} saved to {output_pdbqt}")
                except subprocess.CalledProcessError as e:
                    print(f"Error preparing {receptor_name}: {e}")
                    continue

            self.prepared_receptors = prepared_receptors
            return prepared_receptors

        def prepare_ligands(self):
            """Convert ligands to PDBQT format for docking."""
            if self.ligands is None:
                self.load_ligands()

            # Create a directory for ligand PDBQT files
            ligand_dir = os.path.join(self.output_dir, "ligands")
            os.makedirs(ligand_dir, exist_ok=True)

            prepared_ligands = []

            print("Converting ligands to PDBQT format...")
            for idx, mol in enumerate(tqdm(self.ligands)):
                # Get molecule name or ID
                if mol.HasProp("ChEMBL_ID"):
                    mol_name = mol.GetProp("ChEMBL_ID")
                else:
                    mol_name = f"ligand_{idx}"

                # Create temporary mol file
                mol_file = os.path.join(ligand_dir, f"{mol_name}.mol")
                pdbqt_file = os.path.join(ligand_dir, f"{mol_name}.pdbqt")

                # Write mol file
                Chem.MolToMolFile(mol, mol_file)

                # Convert to PDBQT using Open Babel
                try:
                    cmd = [
                        "obabel", 
                        mol_file, 
                        "-O", pdbqt_file, 
                        "-xh",  # Add hydrogens
                        "--partialcharge", "gasteiger"  # Add Gasteiger charges
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    # Store information about the prepared ligand
                    prepared_ligands.append({
                        "mol_idx": idx,
                        "name": mol_name,
                        "pdbqt_file": pdbqt_file
                    })
                except subprocess.CalledProcessError:
                    print(f"Error preparing ligand {mol_name}")
                    continue

            print(f"Prepared {len(prepared_ligands)} ligands for docking")
            self.prepared_ligands = prepared_ligands
            return prepared_ligands

        def run_docking(self, receptor_names=None, exhaustiveness=8, num_modes=9):
            """
            Run molecular docking for selected receptors.

            Parameters:
            -----------
            receptor_names : list
                List of receptor names to dock against (default: all prepared receptors)
            exhaustiveness : int
                Search exhaustiveness parameter for AutoDock Vina
            num_modes : int
                Number of binding modes to generate
            """
            # Prepare receptors if not already done
            if not hasattr(self, 'prepared_receptors'):
                self.prepare_receptors()

            # Prepare ligands if not already done
            if not hasattr(self, 'prepared_ligands'):
                self.prepare_ligands()

            # Select which receptors to dock against
            if receptor_names is None:
                receptor_names = list(self.prepared_receptors.keys())
            else:
                # Filter to only include prepared receptors
                receptor_names = [r for r in receptor_names if r in self.prepared_receptors]

            if not receptor_names:
                print("Error: No valid receptors selected for docking.")
                return

            print(f"Starting docking against {len(receptor_names)} receptors: {', '.join(receptor_names)}")

            all_results = {}

            # Process each receptor
            for receptor_name in receptor_names:
                receptor_pdbqt = self.prepared_receptors[receptor_name]
                binding_site = self.binding_sites[receptor_name]

                print(f"\nDocking against {receptor_name}...")

                # Create output directory for this receptor
                receptor_outdir = os.path.join(self.output_dir, receptor_name)
                os.makedirs(receptor_outdir, exist_ok=True)

                # Create docking tasks
                docking_tasks = []
                for ligand in self.prepared_ligands:
                    ligand_pdbqt = ligand["pdbqt_file"]
                    ligand_name = ligand["name"]
                    out_file = os.path.join(receptor_outdir, f"{ligand_name}_docked.pdbqt")
                    log_file = os.path.join(receptor_outdir, f"{ligand_name}_log.txt")

                    # Create Vina command
                    cmd = [
                        "vina",
                        "--receptor", receptor_pdbqt,
                        "--ligand", ligand_pdbqt,
                        "--center_x", str(binding_site["center_x"]),
                        "--center_y", str(binding_site["center_y"]),
                        "--center_z", str(binding_site["center_z"]),
                        "--size_x", str(binding_site["size_x"]),
                        "--size_y", str(binding_site["size_y"]),
                        "--size_z", str(binding_site["size_z"]),
                        "--out", out_file,
                        "--log", log_file,
                        "--exhaustiveness", str(exhaustiveness),
                        "--num_modes", str(num_modes),
                        "--cpu", "1"  # Each task will use 1 CPU
                    ]

                    docking_tasks.append({
                        "cmd": cmd,
                        "ligand_name": ligand_name,
                        "out_file": out_file,
                        "log_file": log_file
                    })

                # Run docking tasks in parallel
                results = []
                with ProcessPoolExecutor(max_workers=self.num_cpu) as executor:
                    futures = []
                    for task in docking_tasks:
                        futures.append(executor.submit(self._run_docking_task, task))

                    # Collect results as they complete
                    for future in tqdm(futures, total=len(futures), desc=f"Docking against {receptor_name}"):
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except Exception as e:
                            print(f"Error in docking task: {e}")

                # Parse and store results
                receptor_results = self._parse_docking_results(receptor_name, results)
                all_results[receptor_name] = receptor_results

                # Print summary for this receptor
                self._print_docking_summary(receptor_name, receptor_results)

            # Store all results
            self.docking_results = all_results
            return all_results

        def _run_docking_task(self, task):
            """Run a single docking task."""
            try:
                subprocess.run(task["cmd"], check=True, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)

                # Check if output file exists
                if os.path.exists(task["out_file"]) and os.path.exists(task["log_file"]):
                    return {"success": True, **task}
                else:
                    return {"success": False, "error": "Output files not created", **task}
            except subprocess.CalledProcessError as e:
                return {
                    "success": False, 
                    "error": f"Process error: {e}", 
                    **task
                }
            except Exception as e:
                return {
                    "success": False, 
                    "error": f"Exception: {e}", 
                    **task
                }

        def _parse_docking_results(self, receptor_name, results):
            """Parse docking results to extract binding affinities and poses."""
            parsed_results = []

            for result in results:
                if not result["success"]:
                    continue

                ligand_name = result["ligand_name"]
                log_file = result["log_file"]

                # Parse log file to extract binding affinities
                try:
                    affinities = []
                    mode_pattern = re.compile(r'^\s*\d+\s+([-\d\.]+)\s+')

                    with open(log_file, 'r') as f:
                        content = f.read()

                        # Extract binding modes section
                        modes_section = content.split("-----+")
                        if len(modes_section) >= 3:
                            modes_text = modes_section[2].strip()
                            for line in modes_text.split('\n'):
                                match = mode_pattern.match(line)
                                if match:
                                    affinity = float(match.group(1))
                                    affinities.append(affinity)

                    # Get best (lowest) binding affinity
                    best_affinity = min(affinities) if affinities else None

                    parsed_results.append({
                        "ligand_name": ligand_name,
                        "best_affinity": best_affinity,
                        "all_affinities": affinities,
                        "out_file": result["out_file"],
                        "log_file": log_file
                    })
                except Exception as e:
                    print(f"Error parsing results for {ligand_name}: {e}")
                    continue

            # Sort by best binding affinity
            parsed_results.sort(key=lambda x: x["best_affinity"] if x["best_affinity"] is not None else float('inf'))

            return parsed_results

        def _print_docking_summary(self, receptor_name, results):
            """Print summary of docking results for a receptor."""
            if not results:
                print(f"No successful docking results for {receptor_name}")
                return

            print(f"\nTop 10 compounds for {receptor_name} by binding affinity (kcal/mol):")
            print("-" * 60)
            print(f"{'Rank':<6}{'Ligand Name':<20}{'Binding Affinity':<20}")
            print("-" * 60)

            for i, result in enumerate(results[:10]):
                print(f"{i+1:<6}{result['ligand_name']:<20}{result['best_affinity']:<20.2f}")

        def compare_receptor_affinities(self, top_n=50):
            """
            Compare binding affinities across different receptors to identify biased ligands.

            Parameters:
            -----------
            top_n : int
                Number of top compounds to analyze

            Returns:
            --------
            pd.DataFrame with comparative binding affinities
            """
            if not self.docking_results:
                print("No docking results available. Run docking first.")
                return None

            # Collect binding data for each ligand across receptors
            ligand_data = {}
            receptor_names = list(self.docking_results.keys())

            for receptor, results in self.docking_results.items():
                for result in results:
                    ligand_name = result["ligand_name"]
                    affinity = result["best_affinity"]

                    if ligand_name not in ligand_data:
                        ligand_data[ligand_name] = {r: None for r in receptor_names}

                    ligand_data[ligand_name][receptor] = affinity

            # Convert to DataFrame
            df = pd.DataFrame.from_dict(ligand_data, orient='index')

            # Calculate bias scores
            if len(receptor_names) >= 2:
                for i, receptor1 in enumerate(receptor_names):
                    for receptor2 in receptor_names[i+1:]:
                        bias_col = f"bias_{receptor1}_vs_{receptor2}"
                        df[bias_col] = df[receptor1] - df[receptor2]

            # Sort by different criteria
            sorted_dfs = {}

            # Sort by individual receptor affinity
            for receptor in receptor_names:
                sorted_dfs[f"top_{receptor}"] = df.sort_values(by=receptor).head(top_n)

            # Sort by bias scores if available
            bias_cols = [col for col in df.columns if col.startswith("bias_")]
            for bias_col in bias_cols:
                # Get receptors from bias column name
                parts = bias_col.split('_')
                receptor1 = parts[1]
                receptor2 = parts[3]

                # Sort for bias toward receptor1 (negative values)
                sorted_dfs[f"biased_toward_{receptor1}"] = df.sort_values(by=bias_col).head(top_n)

                # Sort for bias toward receptor2 (positive values)
                sorted_dfs[f"biased_toward_{receptor2}"] = df.sort_values(by=bias_col, ascending=False).head(top_n)

            # Print summary of biased compounds
            self._print_bias_summary(df, receptor_names)

            # Save full results
            results_file = os.path.join(self.output_dir, "affinity_comparison.csv")
            df.to_csv(results_file)
            print(f"Full comparison saved to {results_file}")

            return df, sorted_dfs

        def _print_bias_summary(self, df, receptor_names):
            """Print summary of biased compounds."""
            print("\nBiased Compound Analysis:")
            print("-" * 80)

            # Check for each bias pair
            for i, receptor1 in enumerate(receptor_names):
                for receptor2 in receptor_names[i+1:]:
                    bias_col = f"bias_{receptor1}_vs_{receptor2}"

                    if bias_col in df.columns:
                        # Get top 5 compounds biased toward each receptor
                        toward_receptor1 = df.sort_values(by=bias_col).head(5)
                        toward_receptor2 = df.sort_values(by=bias_col, ascending=False).head(5)

                        print(f"\nTop 5 compounds biased toward {receptor1} vs {receptor2}:")
                        for idx, row in toward_receptor1.iterrows():
                            bias_value = row[bias_col]
                            print(f"  {idx}: {receptor1} ({row[receptor1]:.2f}) vs {receptor2} ({row[receptor2]:.2f}), Bias: {bias_value:.2f}")

                        print(f"\nTop 5 compounds biased toward {receptor2} vs {receptor1}:")
                        for idx, row in toward_receptor2.iterrows():
                            bias_value = row[bias_col]
                            print(f"  {idx}: {receptor2} ({row[receptor2]:.2f}) vs {receptor1} ({row[receptor1]:.2f}), Bias: {bias_value:.2f}")

        def visualize_results(self):
            """Generate visualizations of docking results."""
            if not self.docking_results:
                print("No docking results available. Run docking first.")
                return

            # Create visualization directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # Plot distribution of binding affinities for each receptor
            plt.figure(figsize=(10, 6))
            for receptor, results in self.docking_results.items():
                affinities = [r["best_affinity"] for r in results if r["best_affinity"] is not None]
                if affinities:
                    sns.kdeplot(affinities, label=receptor)

            plt.xlabel("Binding Affinity (kcal/mol)")
            plt.ylabel("Density")
            plt.title("Distribution of Binding Affinities Across Receptors")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            affinity_dist_file = os.path.join(viz_dir, "affinity_distribution.png")
            plt.savefig(affinity_dist_file, dpi=300, bbox_inches='tight')
            plt.close()

            # Create scatter plots for each receptor pair to visualize bias
            receptor_names = list(self.docking_results.keys())
            if len(receptor_names) >= 2:
                for i, receptor1 in enumerate(receptor_names):
                    for receptor2 in receptor_names[i+1:]:
                        self._plot_receptor_comparison(receptor1, receptor2, viz_dir)

            print(f"Visualizations saved to {viz_dir}")

        def _plot_receptor_comparison(self, receptor1, receptor2, viz_dir):
            """Create scatter plot comparing binding affinities between two receptors."""
            # Collect data for compounds that were docked to both receptors
            data = []

            # Get results for each receptor
            results1 = {r["ligand_name"]: r["best_affinity"] for r in self.docking_results[receptor1]}
            results2 = {r["ligand_name"]: r["best_affinity"] for r in self.docking_results[receptor2]}

            # Find common ligands
            common_ligands = set(results1.keys()).intersection(set(results2.keys()))

            # Collect data points
            x_values = []
            y_values = []
            ligand_names = []

            for ligand in common_ligands:
                if results1[ligand] is not None and results2[ligand] is not None:
                    x_values.append(results1[ligand])
                    y_values.append(results2[ligand])
                    ligand_names.append(ligand)

            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(x_values, y_values, alpha=0.7)

            # Add diagonal line (equal binding)
            min_val = min(min(x_values), min(y_values))
            max_val = max(max(x_values), max(y_values))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

            # Add labels for potentially biased compounds
            for i, ligand in enumerate(ligand_names):
                # Check if binding difference is significant (> 2 kcal/mol)
                if abs(x_values[i] - y_values[i]) > 2.0:
                    plt.annotate(ligand, (x_values[i], y_values[i]), 
                                 fontsize=8, alpha=0.7)

            plt.xlabel(f"{receptor1} Binding Affinity (kcal/mol)")
            plt.ylabel(f"{receptor2} Binding Affinity (kcal/mol)")
            plt.title(f"Binding Affinity Comparison: {receptor1} vs {receptor2}")
            plt.grid(True, linestyle='--', alpha=0.5)

            # Highlighting regions of bias
            plt.axvspan(min_val, max_val, min_val, min_val + 1, alpha=0.1, color='green', 
                       label=f"Biased toward {receptor1}")
            plt.axvspan(min_val, min_val + 1, min_val, max_val, alpha=0.1, color='blue', 
                       label=f"Biased toward {receptor2}")

            plt.legend()

            comparison_file = os.path.join(viz_dir, f"comparison_{receptor1}_vs_{receptor2}.png")
            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
            plt.close()

        def extract_biased_candidates(self, bias_threshold=2.0, output_sdf=None):
            """
            Extract potentially biased ligands for further analysis.

            Parameters:
            -----------
            bias_threshold : float
                Minimum difference in binding affinity (kcal/mol) to consider a ligand biased
            output_sdf : str
                Path to save extracted biased ligands (default: biased_candidates.sdf in output_dir)

            Returns:
            --------
            dict with categorized biased ligands
            """
            if not self.docking_results:
                print("No docking results available. Run docking first.")
                return None

            # Calculate bias between receptors
            comparison_df, sorted_dfs = self.compare_receptor_affinities(top_n=100)

            # Identify biased compounds
            biased_candidates = {}
            receptor_names = list(self.docking_results.keys())

            for i, receptor1 in enumerate(receptor_names):
                for receptor2 in receptor_names[i+1:]:
                    bias_col = f"bias_{receptor1}_vs_{receptor2}"

                    if bias_col in comparison_df.columns:
                        # Find compounds biased toward receptor1
                        toward_r1 = comparison_df[
                            (comparison_df[bias_col] < -bias_threshold) & 
                            (comparison_df[receptor1].notna()) & 
                            (comparison_df[receptor2].notna())
                        ]

                        # Find compounds biased toward receptor2
                        toward_r2 = comparison_df[
                            (comparison_df[bias_col] > bias_threshold) & 
                            (comparison_df[receptor1].notna()) & 
                            (comparison_df[receptor2].notna())
                        ]

                        if not toward_r1.empty:
                            key = f"biased_toward_{receptor1}_vs_{receptor2}"
                            biased_candidates[key] = list(toward_r1.index)

                        if not toward_r2.empty:
                            key = f"biased_toward_{receptor2}_vs_{receptor1}"
                            biased_candidates[key] = list(toward_r2.index)

            # Output biased candidates to SDF if requested
            if output_sdf is None:
                output_sdf = os.path.join(self.output_dir, "biased_candidates.sdf")

            # Combine all biased candidates
            all_biased = set()
            for candidates in biased_candidates.values():
                all_biased.update(candidates)

            # Extract molecules from original SDF
            if self.ligands is not None:
                writer = Chem.SDWriter(output_sdf)

                # Create name-to-mol mapping
                mol_map = {}
                for idx, mol in enumerate(self.ligands):
                    if mol.HasProp("ChEMBL_ID"):
                        name = mol.GetProp("ChEMBL_ID")
                    else:
                        name = f"ligand_{idx}"
                    mol_map[name] = mol

                # Write biased molecules to SDF
                for name in all_biased:
                    if name in mol_map:
                        mol = mol_map[name]

                        # Add bias information to properties
                        for bias_cat, candidates in biased_candidates.items():
                            if name in candidates:
                                mol.SetProp("BiasCategory", bias_cat)
                                # Extract receptor names from bias category
                                parts = bias_cat.split('_')
                                favored_receptor = parts[2]
                                vs_receptor = parts[4]

                                # Add specific binding values
                                if favored_receptor in comparison_df.columns and vs_receptor in comparison_df.columns:
                                    favored_value = comparison_df.loc[name, favored_receptor]
                                    vs_value = comparison_df.loc[name, vs_receptor]
                                    bias_value = comparison_df.loc[name, bias_col]

                                    mol.SetProp(f"Affinity_{favored_receptor}", f"{favored_value:.2f}")
                                    mol.SetProp(f"Affinity_{vs_receptor}", f"{vs_value:.2f}")
                                    mol.SetProp("BiasValue", f"{bias_value:.2f}")

                        writer.Write(mol)

                writer.close()
                print(f"Saved {len(all_biased)} biased candidates to {output_sdf}")

            return biased_candidates

        def analyze_structural_features(self, biased_candidates=None):
            """
            Analyze structural features of biased ligands to identify patterns.

            Parameters:
            -----------
            biased_candidates : dict
                Dictionary of biased candidates as returned by extract_biased_candidates
                If None, will extract biased candidates

            Returns:
            --------
            DataFrame with structural feature analysis
            """
            if biased_candidates is None:
                biased_candidates = self.extract_biased_candidates()

            if not biased_candidates:
                print("No biased candidates identified.")
                return None

            # Create name-to-mol mapping
            mol_map = {}
            for idx, mol in enumerate(self.ligands):
                if mol.HasProp("ChEMBL_ID"):
                    name = mol.GetProp("ChEMBL_ID")
                else:
                    name = f"ligand_{idx}"
                mol_map[name] = mol

            # Calculate descriptors for each category of biased compounds
            results = {}
            for bias_cat, compounds in biased_candidates.items():
                # Extract receptor names from bias category
                parts = bias_cat.split('_')
                if len(parts) >= 5:
                    favored_receptor = parts[2]

                    # Get molecules
                    mols = [mol_map[name] for name in compounds if name in mol_map]

                    if mols:
                        # Calculate basic properties
                        props = {
                            'MolWt': [],
                            'LogP': [],
                            'NumHDonors': [],
                            'NumHAcceptors': [],
                            'NumRotatableBonds': [],
                            'NumAromaticRings': []
                        }

                        for mol in mols:
                            props['MolWt'].append(Chem.Descriptors.MolWt(mol))
                            props['LogP'].append(Chem.Descriptors.MolLogP(mol))
                            props['NumHDonors'].append(Chem.Descriptors.NumHDonors(mol))
                            props['NumHAcceptors'].append(Chem.Descriptors.NumHAcceptors(mol))
                            props['NumRotatableBonds'].append(Chem.Descriptors.NumRotatableBonds(mol))
                            props['NumAromaticRings'].append(Chem.Lipinski.NumAromaticRings(mol))

                        # Calculate means and standard deviations
                        stats = {}
                        for prop, values in props.items():
                            stats[f"{prop}_mean"] = np.mean(values)
                            stats[f"{prop}_std"] = np.std(values)

                        # Count common substructures
                        substructures = {
                            'phenol': Chem.MolFromSmarts('c[OH]'),
                            'carboxylic_acid': Chem.MolFromSmarts('C(=O)[OH]'),
                            'amine': Chem.MolFromSmarts('[NH2]'),
                            'amide': Chem.MolFromSmarts('C(=O)[NH]'),
                            'sulfonamide': Chem.MolFromSmarts('S(=O)(=O)[NH]')
                        }

                        for name, patt in substructures.items():
                            counts = [len(mol.GetSubstructMatches(patt)) for mol in mols]
                            stats[f"{name}_count"] = sum(counts)
                            stats[f"{name}_percent"] = sum(1 for c in counts if c > 0) / len(counts) * 100

                        stats['compound_count'] = len(mols)
                        results[bias_cat] = stats

            # Convert to DataFrame for easier comparison
            df = pd.DataFrame.from_dict(results, orient='index')

            # Save results to CSV
            output_file = os.path.join(self.output_dir, "structural_analysis.csv")
            df.to_csv(output_file)
            print(f"Structural analysis saved to {output_file}")

            # Print summary
            print("\nStructural Feature Summary for Biased Compounds:")
            print("-" * 80)

            for bias_cat in results:
                print(f"\n{bias_cat} ({results[bias_cat]['compound_count']} compounds):")
                print(f"  Average MW: {results[bias_cat]['MolWt_mean']:.1f} ± {results[bias_cat]['MolWt_std']:.1f}")
                print(f"  Average LogP: {results[bias_cat]['LogP_mean']:.1f} ± {results[bias_cat]['LogP_std']:.1f}")
                print(f"  H-bond donors: {results[bias_cat]['NumHDonors_mean']:.1f} ± {results[bias_cat]['NumHDonors_std']:.1f}")
                print(f"  H-bond acceptors: {results[bias_cat]['NumHAcceptors_mean']:.1f} ± {results[bias_cat]['NumHAcceptors_std']:.1f}")
                print(f"  Rotatable bonds: {results[bias_cat]['NumRotatableBonds_mean']:.1f} ± {results[bias_cat]['NumRotatableBonds_std']:.1f}")

                # Show key substructures
                substructs = []
                for substr in ['phenol', 'carboxylic_acid', 'amine', 'amide', 'sulfonamide']:
                    if results[bias_cat][f"{substr}_percent"] > 30:  # Threshold for reporting
                        substructs.append(f"{substr} ({results[bias_cat][f'{substr}_percent']:.0f}%)")

                if substructs:
                    print(f"  Common substructures: {', '.join(substructs)}")

            return df

        def export_poses(self, top_n=10, output_dir=None):
            """
            Export top binding poses for visualization in PyMOL or similar.

            Parameters:
            -----------
            top_n : int
                Number of top poses to export per receptor
            output_dir : str
                Directory to save poses (default: poses/ in output_dir)
            """
            if not self.docking_results:
                print("No docking results available. Run docking first.")
                return

            if output_dir is None:
                output_dir = os.path.join(self.output_dir, "poses")

            os.makedirs(output_dir, exist_ok=True)

            for receptor_name, results in self.docking_results.items():
                # Create receptor directory
                receptor_dir = os.path.join(output_dir, receptor_name)
                os.makedirs(receptor_dir, exist_ok=True)

                # Copy receptor structure
                receptor_pdbqt = self.prepared_receptors[receptor_name]
                receptor_pdb = os.path.join(receptor_dir, f"{receptor_name}.pdb")

                # Convert PDBQT to PDB for better visualization
                try:
                    cmd = ["obabel", receptor_pdbqt, "-O", receptor_pdb]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except Exception as e:
                    print(f"Error converting receptor {receptor_name}: {e}")
                    continue

                # Export top poses
                for i, result in enumerate(results[:top_n]):
                    if i >= top_n:
                        break

                    ligand_name = result["ligand_name"]
                    pose_pdbqt = result["out_file"]
                    pose_pdb = os.path.join(receptor_dir, f"{ligand_name}_pose.pdb")

                    # Convert PDBQT to PDB
                    try:
                        cmd = ["obabel", pose_pdbqt, "-O", pose_pdb]
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception as e:
                        print(f"Error converting pose for {ligand_name}: {e}")
                        continue

                    # Create PyMOL script for this ligand-receptor pair
                    pymol_script = os.path.join(receptor_dir, f"{ligand_name}_view.pml")
                    with open(pymol_script, 'w') as f:
                        f.write(f"""# PyMOL script for {ligand_name} - {receptor_name}
    load {receptor_name}.pdb, receptor
    load {ligand_name}_pose.pdb, ligand
    hide everything
    show cartoon, receptor
    show sticks, ligand
    color cyan, ligand
    zoom ligand, 5
    show sticks, receptor and (ligand around 5.0)
    color green, receptor and (ligand around 5.0)
    label ligand, name
    set label_size, 0.8
    set ray_shadows, 0
    set antialias, 2
    bg_color white
    """)

                print(f"Exported top {min(top_n, len(results))} poses for {receptor_name} to {receptor_dir}")

            # Create convenience script to view all
            all_script = os.path.join(output_dir, "view_all.pml")
            with open(all_script, 'w') as f:
                f.write("# PyMOL script to view all poses\n")
                for receptor_name in self.docking_results.keys():
                    receptor_dir = os.path.join(output_dir, receptor_name)
                    script_files = glob.glob(os.path.join(receptor_dir, "*_view.pml"))
                    for script in script_files:
                        f.write(f"@{script}\n")

            print(f"\nExported all poses to {output_dir}")
            print(f"Use PyMOL to view individual poses or use {all_script} to view all")

        def generate_report(self):
            """Generate a comprehensive report of the docking results."""
            if not self.docking_results:
                print("No docking results available. Run docking first.")
                return

            report_dir = os.path.join(self.output_dir, "report")
            os.makedirs(report_dir, exist_ok=True)

            # Create HTML report
            report_file = os.path.join(report_dir, "docking_report.html")

            # Generate visualizations if not already done
            if not os.path.exists(os.path.join(self.output_dir, "visualizations")):
                self.visualize_results()

            # Extract biased candidates if not already done
            biased_candidates = None
            if hasattr(self, 'biased_candidates'):
                biased_candidates = self.biased_candidates
            else:
                biased_candidates = self.extract_biased_candidates()
                self.biased_candidates = biased_candidates

            # Analyze structural features
            if not os.path.exists(os.path.join(self.output_dir, "structural_analysis.csv")):
                self.analyze_structural_features(biased_candidates)

            # Generate HTML report
            with open(report_file, 'w') as f:
                f.write("""<!DOCTYPE html>
    <html>
    <head>
        <title>ER Docking Results Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            h1, h2, h3 { color: #333366; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .figure { margin: 20px 0; text-align: center; }
            .figure img { max-width: 100%; border: 1px solid #ddd; }
            .caption { font-style: italic; margin-top: 8px; }
        </style>
    </head>
    <body>
        <h1>Estrogen Receptor Docking Analysis Report</h1>
        <p>Generated on """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>

        <h2>Summary</h2>
        <p>This report summarizes the results of molecular docking against different estrogen receptor conformations.</p>
        <p>Total ligands processed: """ + str(len(self.prepared_ligands)) + """</p>
        <p>Receptors analyzed: """ + ", ".join(self.prepared_receptors.keys()) + """</p>
    """)

                # Add top compounds for each receptor
                f.write("<h2>Top Compounds by Receptor</h2>\n")

                for receptor_name, results in self.docking_results.items():
                    f.write(f"<h3>{receptor_name}</h3>\n")

                    if results:
                        f.write("<table>\n")
                        f.write("<tr><th>Rank</th><th>Ligand Name</th><th>Binding Affinity (kcal/mol)</th></tr>\n")

                        for i, result in enumerate(results[:10]):
                            f.write(f"<tr><td>{i+1}</td><td>{result['ligand_name']}</td><td>{result['best_affinity']:.2f}</td></tr>\n")

                        f.write("</table>\n")
                    else:
                        f.write("<p>No results available for this receptor.</p>\n")

                # Add visualizations
                f.write("<h2>Binding Affinity Distributions</h2>\n")
                f.write('<div class="figure">\n')
                f.write('  <img src="../visualizations/affinity_distribution.png" alt="Binding affinity distributions">\n')
                f.write('  <div class="caption">Figure 1: Distribution of binding affinities across different receptors.</div>\n')
                f.write('</div>\n')

                # Add receptor comparisons
                f.write("<h2>Receptor Comparisons</h2>\n")
                receptor_names = list(self.docking_results.keys())
                if len(receptor_names) >= 2:
                    for i, receptor1 in enumerate(receptor_names):
                        for receptor2 in receptor_names[i+1:]:
                            img_path = f"../visualizations/comparison_{receptor1}_vs_{receptor2}.png"
                            if os.path.exists(os.path.join(self.output_dir, img_path.replace("../", ""))):
                                f.write('<div class="figure">\n')
                                f.write(f'  <img src="{img_path}" alt="Comparison of {receptor1} vs {receptor2}">\n')
                                f.write(f'  <div class="caption">Figure: Binding affinity comparison between {receptor1} and {receptor2}.</div>\n')
                                f.write('</div>\n')

                # Add biased candidates summary
                if biased_candidates:
                    f.write("<h2>Biased Ligands Analysis</h2>\n")

                    for bias_cat, candidates in biased_candidates.items():
                        if candidates:
                            parts = bias_cat.split('_')
                            if len(parts) >= 5:
                                favored_receptor = parts[2]
                                vs_receptor = parts[4]

                                f.write(f"<h3>Ligands biased toward {favored_receptor} vs {vs_receptor}</h3>\n")
                                f.write(f"<p>Number of compounds: {len(candidates)}</p>\n")

                                if len(candidates) > 0:
                                    f.write("<table>\n")
                                    f.write(f"<tr><th>Ligand</th><th>{favored_receptor} Affinity</th><th>{vs_receptor} Affinity</th><th>Bias</th></tr>\n")

                                    # Load comparison data
                                    comparison_file = os.path.join(self.output_dir, "affinity_comparison.csv")
                                    if os.path.exists(comparison_file):
                                        comparison_df = pd.read_csv(comparison_file, index_col=0)
                                        bias_col = f"bias_{favored_receptor}_vs_{vs_receptor}"

                                        for ligand in candidates[:10]:  # Show top 10
                                            if ligand in comparison_df.index and bias_col in comparison_df.columns:
                                                aff1 = comparison_df.loc[ligand, favored_receptor]
                                                aff2 = comparison_df.loc[ligand, vs_receptor]
                                                bias = comparison_df.loc[ligand, bias_col]

                                                f.write(f"<tr><td>{ligand}</td><td>{aff1:.2f}</td><td>{aff2:.2f}</td><td>{bias:.2f}</td></tr>\n")

                                    f.write("</table>\n")

                # Conclusion
                f.write("""
        <h2>Conclusion</h2>
        <p>This analysis has identified compounds with differential binding preferences for specific estrogen receptor subtypes.
        These compounds may serve as leads for developing selective estrogen receptor modulators (SERMs) with improved isoform selectivity.</p>

        <h2>Next Steps</h2>
        <ul>
            <li>Experimental validation of top candidates using binding assays</li>
            <li>Further molecular dynamics simulations to analyze binding stability</li>
            <li>Structure-activity relationship (SAR) analysis to optimize leads</li>
        </ul>

        <p><i>Analysis performed using ERDockingPipeline</i></p>
    </body>
    </html>
    """)

            print(f"\nGenerated comprehensive report at {report_file}")

            # Create CSV summary of all results
            summary_file = os.path.join(report_dir, "docking_summary.csv")
            summary_data = []

            for receptor_name, results in self.docking_results.items():
                for result in results:
                    data = {
                        "Receptor": receptor_name,
                        "Ligand": result["ligand_name"],
                        "Binding_Affinity": result["best_affinity"],
                        "Output_File": result["out_file"]
                    }
                    summary_data.append(data)

            pd.DataFrame(summary_data).to_csv(summary_file, index=False)
            print(f"Summary CSV saved to {summary_file}")

            return report_file

    # Example usage
    if __name__ == "__main__":
        # Initialize the pipeline
        pipeline = ERDockingPipeline(
            ligand_file="/home/halleluyah/Documents/Programming Projects/Bioinformatics/cancerag/notebooks/new_er_docking_ready.sdf",
            output_dir="docking_results"
        )

        # Run the complete workflow
        pipeline.prepare_receptors()
        pipeline.prepare_ligands()
        pipeline.run_docking()
        pipeline.compare_receptor_affinities()
        pipeline.visualize_results()
        biased_candidates = pipeline.extract_biased_candidates()
        pipeline.analyze_structural_features(biased_candidates)
        pipeline.export_poses()
        pipeline.generate_report()
    return (
        AllChem,
        Chem,
        ERDockingPipeline,
        PDBIO,
        PDBParser,
        PandasTools,
        ProcessPoolExecutor,
        biased_candidates,
        glob,
        json,
        mo,
        multiprocessing,
        np,
        os,
        pd,
        pipeline,
        plt,
        re,
        shutil,
        sns,
        subprocess,
        sys,
        tqdm,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
