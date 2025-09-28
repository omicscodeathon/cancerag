import os
import yaml


from cancerag.data_collection import (
    biasdb_retriever,
    receptor_retriever,
    chembl_retriever,
)
from cancerag.preprocessing import receptor_preprocessor
from cancerag.preprocessing.ligand_preprocessor import LigandPreprocessor
from cancerag.features import molecular_descriptors
from cancerag.features import active_site_identifier
from cancerag.docking import run_docking
from cancerag.utils.unbiased_agonist_adder import UnbiasedAgonistAdder
# from cancerag.docking.pipeline import DockingPipeline


def run_pipeline(config_path: str):
    """
    Orchestrates the entire GPCR signaling bias prediction pipeline.
    """
    # 1. Load Configuration
    print("1. Loading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    paths = config["paths"]
    os.makedirs(paths["pdb_summary"], exist_ok=True)

    # --- STAGE 1: DATA COLLECTION ---
    print("\n2. Starting Data Collection Stage...")

    # 2.1. Get primary data from BiasDB
    print("   - Fetching data from BiasDB...")
    biasdb_df = biasdb_retriever.download_biasdb_data(paths["biasdb_input"])
    if biasdb_df.empty:
        print("   - ERROR: Failed to retrieve data from BiasDB. Halting pipeline.")
        return
    print(f"   - Loaded {len(biasdb_df)} records from BiasDB.")

    # 2.2. Get unique receptors and download PDB structures for each
    print("   - Initializing PDB structure retrieval...")
    unique_receptors = biasdb_df["receptor_subtype"].dropna().unique()
    print(f"   - Found {len(unique_receptors)} unique receptors to process.")

    # Initialize and run the receptor retriever
    retriever = receptor_retriever.ReceptorRetriever(
        output_dir=paths["pdb_summary"],
        max_downloads=config["data_collection"]["max_pdb_files_per_receptor"],
        force_redownload=False,  # This could be a command-line argument in the future
    )
    retriever.run(unique_receptors)
    print(
        f"   - PDB retrieval complete. Summary is at {os.path.join(paths['pdb_summary'], 'summary.json')}"
    )

    # 2.3. Fetch agonist ligands from ChEMBL for each receptor
    # COMMENTED OUT FOR SIMPLIFIED PIPELINE - Will add unbiased agonists later
    # print("\n   - Initializing ChEMBL agonist data retrieval...")
    # chembl_dir = paths["chembl_raw"]
    # # The retriever will handle directory creation, so we just ensure the parent exists
    # os.makedirs(os.path.dirname(chembl_dir), exist_ok=True)
    #
    # chembl_retriever_instance = chembl_retriever.ChEMBLRetriever(output_dir=chembl_dir)
    # chembl_retriever_instance.run(unique_receptors)
    # print("   - ChEMBL data retrieval complete.")

    print("\nData Collection stage complete.")

    # --- OPTIONAL: ADD UNBIASED AGONISTS ---
    # Uncomment the following lines to add unbiased agonists as a separate class
    # print("\n2.4. Adding Unbiased Agonists from ChEMBL...")
    # unbiased_adder = UnbiasedAgonistAdder(config)
    # unbiased_adder.run()
    # print("   - Unbiased agonist addition complete.")

    # --- STAGE 2: LIGAND PREPROCESSING ---
    print("\n3. Starting Ligand Preprocessing Stage...")

    # Initialize and run the ligand preprocessor
    ligand_processor = LigandPreprocessor(config)
    ligand_processor.run()

    print("   - Ligand preprocessing complete.")

    # --- STAGE 3: RECEPTOR PREPROCESSING ---
    print("\n4. Starting Receptor Preprocessing Stage...")

    # Initialize and run the receptor preprocessor
    receptor_processor = receptor_preprocessor.ReceptorPreprocessor(config)
    receptor_processor.run()

    print("   - Receptor preprocessing complete.")

    # --- STAGE 4: FEATURE EXTRACTION (Ligand Descriptors) ---
    print("\n5. Starting Feature Extraction Stage (Molecular Descriptors)...")

    # Initialize and run the molecular descriptor calculator
    descriptor_calculator = molecular_descriptors.MolecularDescriptorCalculator(config)
    descriptor_calculator.run()

    print("   - Molecular descriptor calculation complete.")

    print("\n6. Starting Feature Extraction Stage (Active Site Identification)...")

    # Initialize and run the active site identifier
    active_site_identifier_instance = active_site_identifier.ActiveSiteIdentifier(
        config
    )
    active_site_identifier_instance.run()

    print("   - Active site identification complete.")

    # --- STAGE 5: MOLECULAR DOCKING ---
    print("\n7. Starting Molecular Docking Stage...")

    # Run the entire docking pipeline
    run_docking.run_docking_stage(config)

    print("   - Molecular docking complete.")

    print("\n---")
    print("Pipeline halted after docking as per current implementation plan.")
    print("Next steps: Implement final data collation and machine learning stages.")

    # The rest of the pipeline (preprocessing, docking, ML) will be connected here
    # once the data collection is fully implemented and produces a unified ligand set.


if __name__ == "__main__":
    config_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "config.yaml"
    )
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found at {config_file}")
    else:
        run_pipeline(config_file)
