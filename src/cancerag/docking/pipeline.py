import os
import multiprocessing
from rdkit import Chem
from . import preparation, runner, analysis, reporting


class DockingPipeline:
    """
    Orchestrates the molecular docking pipeline by delegating tasks to specialized modules.
    """

    def __init__(
        self,
        ligand_file: str,
        receptor_structures: dict,
        binding_sites: dict,
        output_dir: str = "docking_results",
        num_cpu: int = None,
    ):
        """
        Initializes the pipeline.

        Args:
            ligand_file (str): Path to the input SDF file with prepared ligands.
            receptor_structures (dict): Maps receptor names to their PDB file paths.
            binding_sites (dict): Maps receptor names to their binding site definitions.
            output_dir (str): Directory to store all docking-related results.
            num_cpu (int, optional): Number of CPUs for parallel processing. Defaults to auto-detect.
        """
        self.ligand_file = ligand_file
        self.receptor_structures = receptor_structures
        self.binding_sites = binding_sites
        self.output_dir = output_dir

        if num_cpu is None:
            self.num_cpu = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.num_cpu = num_cpu

        os.makedirs(self.output_dir, exist_ok=True)

        # To hold the state of the pipeline
        self.ligands = None
        self.prepared_receptors = None
        self.prepared_ligands = None
        self.raw_docking_results = None
        self.parsed_docking_results = None
        self.affinity_df = None

    def run_pipeline(self):
        """Executes the entire docking and analysis workflow step-by-step."""
        print("--- Starting Docking Pipeline ---")

        # 1. Load and Prepare Receptors and Ligands
        print("\nStep 1: Loading and Preparing Inputs...")
        self.ligands = list(Chem.SDMolSupplier(self.ligand_file))
        if not self.ligands:
            print("  - ERROR: No valid ligands loaded from SDF file. Halting.")
            return

        self.prepared_receptors = preparation.prepare_receptors(
            self.receptor_structures, self.output_dir
        )
        self.prepared_ligands = preparation.prepare_ligands(
            self.ligands, self.output_dir
        )

        if not self.prepared_receptors or not self.prepared_ligands:
            print("  - ERROR: Preparation of receptors or ligands failed. Halting.")
            return

        # 2. Run Docking (CORRECTED APPROACH)
        print("\nStep 2: Running Docking Simulations...")
        
        # Load ligand data to create receptor mapping
        import pandas as pd
        ligands_df = pd.read_csv('data/processed/drug_like_ligands_clean.csv')
        ligand_receptor_mapping = runner.create_ligand_receptor_mapping(ligands_df)
        
        self.raw_docking_results = runner.run_docking_multiprocess(
            prepared_receptors=self.prepared_receptors,
            prepared_ligands=self.prepared_ligands,
            binding_sites=self.binding_sites,
            output_dir=self.output_dir,
            num_cpu=self.num_cpu,
            ligand_receptor_mapping=ligand_receptor_mapping,
        )

        # 3. Parse and Analyze Results
        print("\nStep 3: Parsing and Analyzing Docking Results...")
        self.parsed_docking_results = {
            receptor: analysis.parse_docking_results(results)
            for receptor, results in self.raw_docking_results.items()
        }
        self.affinity_df = analysis.compare_receptor_affinities(
            self.parsed_docking_results
        )

        # Save the final affinity comparison dataframe
        affinity_csv_path = os.path.join(self.output_dir, "affinity_comparison.csv")
        self.affinity_df.to_csv(affinity_csv_path)
        print(f"  - Saved final affinity comparison to {affinity_csv_path}")

        # 4. Generate Report and Visualizations
        print("\nStep 4: Generating Reports and Visualizations...")
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)

        reporting.visualize_affinity_distribution(self.parsed_docking_results, viz_dir)
        reporting.visualize_receptor_comparison(self.affinity_df, viz_dir)
        reporting.generate_html_report(
            self.parsed_docking_results, self.affinity_df, self.output_dir
        )

        print("--- Docking Pipeline Finished Successfully ---")
