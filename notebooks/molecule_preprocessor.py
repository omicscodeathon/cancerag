import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium", app_title="Molecule Preprocessing Pipeline")


@app.cell
def _(mo):
    mo.md(r"""### Importing the necessary Libraries""")
    return


@app.cell
def _():
    # Import libraries
    import marimo as mo
    from chembl_webresource_client.new_client import new_client
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    import pandas as pd
    import logging
    from tqdm import tqdm
    import time
    import random
    import os
    from datetime import datetime
    import requests.exceptions
    return (
        Chem,
        Descriptors,
        datetime,
        logging,
        mo,
        new_client,
        os,
        pd,
        random,
        rdMolDescriptors,
        requests,
        time,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md(r"""### Analysis of this data""")
    return


@app.cell
def _():
    path = "/home/halleluyah/Documents/Programming Projects/Bioinformatics/cancerag/data/estrogen_receptor_bioactivity_20250204_012600.csv"
    return (path,)


@app.cell
def _(path, pd):
    df=pd.read_csv(path)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _():
    from rdkit.Chem import Lipinski, AllChem, PandasTools
    from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
    from collections import defaultdict
    import numpy as np
    import warnings

    import warnings
    warnings.simplefilter("ignore", category=DeprecationWarning)
    return (
        AllChem,
        FilterCatalog,
        FilterCatalogParams,
        Lipinski,
        PandasTools,
        defaultdict,
        np,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""### Pipeline To Screen unwanted molecules""")
    return


@app.cell
def _(
    AllChem,
    Chem,
    Descriptors,
    FilterCatalog,
    FilterCatalogParams,
    Lipinski,
    PandasTools,
    defaultdict,
    logging,
    np,
    os,
    pd,
    rdMolDescriptors,
    tqdm,
):
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    class NewMoleculePreprocessor:
        """
        A class to preprocess molecules retrieved from ChEMBL for molecular docking.
        Implements filtering based on Lipinski's rules, PAINS filters, and activity thresholds.
        """

        def __init__(self, input_file, target_name="Estrogen Receptor", activity_threshold=1000):
            """
            Initialize the preprocessor with the file containing ChEMBL molecules.

            Parameters:
            -----------
            input_file : str
                Path to the CSV or SDF file containing molecules from ChEMBL
            target_name : str
                Name of the target protein (default: "Estrogen Receptor")
            activity_threshold : float
                Activity threshold in nM (default: 1000 nM = 1 μM)
            """
            self.input_file = input_file
            self.target_name = target_name
            self.activity_threshold = activity_threshold
            self.molecules = None
            self.filtered_molecules = None

            # Initialize PAINS filter
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_filter = FilterCatalog(params)

            # Load the molecules
            self._load_molecules()

        def _load_molecules(self):
            """Load molecules from the input file."""
            file_ext = os.path.splitext(self.input_file)[1].lower()

            if file_ext == '.csv':
                self.molecules = pd.read_csv(self.input_file)
                # Assuming the SMILES column exists, convert to RDKit molecules
                if 'canonical_smiles' in self.molecules.columns:
                    try:
                        # Try to add molecules column
                        PandasTools.AddMoleculeColumnToFrame(self.molecules, 'canonical_smiles', 'Molecule')

                        # Verify molecules were created correctly
                        valid_count = self.molecules['Molecule'].apply(lambda x: x is not None and isinstance(x, Chem.rdchem.Mol)).sum()
                        logger.info(f"Successfully created {valid_count} valid RDKit molecules out of {len(self.molecules)} entries")

                        # Remove rows with invalid molecules
                        invalid_count = len(self.molecules) - valid_count
                        if invalid_count > 0:
                            self.molecules = self.molecules[self.molecules['Molecule'].apply(lambda x: x is not None and isinstance(x, Chem.rdchem.Mol))]
                            logger.info(f"Removed {invalid_count} entries with invalid SMILES")

                    except Exception as e:
                        logger.error(f"Error creating molecules from SMILES: {e}")
                        raise
                else:
                    raise ValueError("CSV file must contain a 'SMILES' column")

            elif file_ext == '.sdf':
                # Use RDKit's PandasTools to read SDF
                try:
                    self.molecules = PandasTools.LoadSDF(self.input_file)
                    logger.info(f"Loaded SDF file with {len(self.molecules)} molecules")
                except Exception as e:
                    logger.error(f"Error loading SDF file: {e}")
                    raise

            else:
                raise ValueError("Input file must be .csv or .sdf format")

            logger.info(f"Loaded {len(self.molecules)} molecules from {self.input_file}")

        def apply_activity_filter(self, activity_column='standard_value', unit_column='standard_units'):
            """
            Filter molecules based on activity against the target.

            Parameters:
            -----------
            activity_column : str
                Column name containing activity values
            unit_column : str
                Column name containing activity units
            """
            if self.molecules is None:
                raise ValueError("No molecules loaded")

            # Make a copy to avoid modifying the original
            filtered_df = self.molecules.copy()

            # Standardize units to nM if needed
            def convert_to_nm(value, unit):
                if pd.isna(value) or pd.isna(unit):
                    return np.nan

                if unit.lower() in ['nm', 'nanomolar']:
                    return value
                elif unit.lower() in ['μm', 'um', 'micromolar']:
                    return value * 1000
                elif unit.lower() in ['mm', 'millimolar']:
                    return value * 1000000
                elif unit.lower() in ['pm', 'picomolar']:
                    return value / 1000
                else:
                    return np.nan

            # Apply conversion if required columns exist
            if activity_column in filtered_df.columns and unit_column in filtered_df.columns:
                filtered_df['Activity_nM'] = filtered_df.apply(
                    lambda row: convert_to_nm(row[activity_column], row[unit_column]), axis=1
                )

                # Filter by activity threshold (lower values are more active)
                prev_count = len(filtered_df)
                filtered_df = filtered_df[filtered_df['Activity_nM'] <= self.activity_threshold]
                logger.info(f"Activity filter: {len(filtered_df)}/{prev_count} molecules remain after applying {self.activity_threshold} nM threshold")
            else:
                logger.warning(f"Could not apply activity filter as {activity_column} or {unit_column} columns are missing")

            self.filtered_molecules = filtered_df
            return filtered_df

        def apply_lipinski_filters(self, strict=False):
            """
            Apply Lipinski's Rule of Five filters.

            Parameters:
            -----------
            strict : bool
                If True, molecules must pass all rules
                If False, molecules can violate one rule (default pharmaceutical standard)
            """
            if self.filtered_molecules is None:
                if self.molecules is None:
                    raise ValueError("No molecules loaded")
                self.filtered_molecules = self.molecules.copy()

            # Make a copy to avoid modifying the original
            filtered_df = self.filtered_molecules.copy()

            # Calculate Lipinski properties
            props = defaultdict(list)

            for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Calculating Lipinski properties"):
                mol = row['ROMol'] if 'ROMol' in filtered_df.columns else row['Molecule']

                if mol is None or not isinstance(mol, Chem.rdchem.Mol):
                    # Skip entries without valid molecules
                    for prop in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'Rotatable_Bonds', 'Lipinski_Violations']:
                        props[prop].append(None)
                    continue

                try:
                    # Calculate properties
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    hbd = Lipinski.NumHDonors(mol)
                    hba = Lipinski.NumHAcceptors(mol)
                    tpsa = Descriptors.TPSA(mol)
                    rotatable_bonds = Descriptors.NumRotatableBonds(mol)

                    # Count Lipinski violations
                    violations = 0
                    if mw > 500: violations += 1
                    if logp > 5: violations += 1
                    if hbd > 5: violations += 1
                    if hba > 10: violations += 1

                    # Store properties
                    props['MW'].append(mw)
                    props['LogP'].append(logp)
                    props['HBD'].append(hbd)
                    props['HBA'].append(hba)
                    props['TPSA'].append(tpsa)
                    props['Rotatable_Bonds'].append(rotatable_bonds)
                    props['Lipinski_Violations'].append(violations)
                except Exception as e:
                    logger.warning(f"Error calculating properties for molecule {idx}: {e}")
                    for prop in ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'Rotatable_Bonds', 'Lipinski_Violations']:
                        props[prop].append(None)

            # Add properties to dataframe
            for prop, values in props.items():
                filtered_df[prop] = values

            # Remove rows with None properties (invalid molecules)
            prev_count = len(filtered_df)
            filtered_df = filtered_df.dropna(subset=['Lipinski_Violations'])
            logger.info(f"Removed {prev_count - len(filtered_df)} molecules with invalid properties")

            # Apply Lipinski filter based on violation count
            max_violations = 0 if strict else 1
            prev_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df['Lipinski_Violations'] <= max_violations]

            logger.info(f"Lipinski filter ({'strict' if strict else 'relaxed'}): {len(filtered_df)}/{prev_count} molecules remain")
            self.filtered_molecules = filtered_df
            return filtered_df

        def apply_pains_filter(self):
            """Remove molecules containing PAINS substructures."""
            if self.filtered_molecules is None:
                if self.molecules is None:
                    raise ValueError("No molecules loaded")
                self.filtered_molecules = self.molecules.copy()

            # Make a copy to avoid modifying the original
            filtered_df = self.filtered_molecules.copy()

            # Check each molecule for PAINS patterns
            pains_flags = []

            for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Checking PAINS patterns"):
                mol = row['ROMol'] if 'ROMol' in filtered_df.columns else row['Molecule']

                if mol is None or not isinstance(mol, Chem.rdchem.Mol):
                    pains_flags.append(True)  # Flag as problematic if no molecule
                    continue

                try:
                    # Check if molecule contains PAINS patterns
                    has_pains = self.pains_filter.HasMatch(mol)
                    pains_flags.append(has_pains)
                except Exception as e:
                    logger.warning(f"Error checking PAINS for molecule {idx}: {e}")
                    pains_flags.append(True)  # Flag as problematic if error occurs

            # Add PAINS flag to dataframe
            filtered_df['Has_PAINS'] = pains_flags

            # Remove molecules with PAINS patterns
            prev_count = len(filtered_df)
            filtered_df = filtered_df[~filtered_df['Has_PAINS']]

            logger.info(f"PAINS filter: {len(filtered_df)}/{prev_count} molecules remain after removing PAINS compounds")
            self.filtered_molecules = filtered_df
            return filtered_df

        def apply_additional_filters(self, tpsa_max=140, rotatable_bonds_max=10):
            """
            Apply additional drug-likeness filters.

            Parameters:
            -----------
            tpsa_max : float
                Maximum topological polar surface area (Å²)
            rotatable_bonds_max : int
                Maximum number of rotatable bonds
            """
            if self.filtered_molecules is None:
                if self.molecules is None:
                    raise ValueError("No molecules loaded")
                self.filtered_molecules = self.molecules.copy()

            # Apply filters
            filtered_df = self.filtered_molecules.copy()

            if 'TPSA' in filtered_df.columns:
                prev_count = len(filtered_df)
                filtered_df = filtered_df[filtered_df['TPSA'] <= tpsa_max]
                logger.info(f"TPSA filter: {len(filtered_df)}/{prev_count} molecules remain after applying {tpsa_max} Å² threshold")

            if 'Rotatable_Bonds' in filtered_df.columns:
                prev_count = len(filtered_df)
                filtered_df = filtered_df[filtered_df['Rotatable_Bonds'] <= rotatable_bonds_max]
                logger.info(f"Rotatable bonds filter: {len(filtered_df)}/{prev_count} molecules remain after applying {rotatable_bonds_max} threshold")

            self.filtered_molecules = filtered_df
            return filtered_df

        def generate_3d_structures(self, output_sdf="docking_ready_molecules.sdf"):
            """
            Generate 3D structures for filtered molecules.

            Parameters:
            -----------
            output_sdf : str
                Path to save the output SDF file with 3D coordinates

            Returns:
            --------
            str : Path to the generated SDF file
            """
            if self.filtered_molecules is None:
                raise ValueError("No filtered molecules available. Run filtering steps first.")

            # Create a list for valid molecules
            mols_for_docking = []
            skipped_count = 0

            logger.info("Generating 3D structures...")
            for idx, row in tqdm(self.filtered_molecules.iterrows(), total=len(self.filtered_molecules)):
                mol = row['ROMol'] if 'ROMol' in self.filtered_molecules.columns else row['Molecule']

                # Skip invalid molecules
                if mol is None or not isinstance(mol, Chem.rdchem.Mol):
                    logger.warning(f"Invalid molecule object at index {idx}, type: {type(mol)}")
                    skipped_count += 1
                    continue

                try:
                    # Make a copy of the molecule
                    mol_copy = Chem.Mol(mol)

                    # Generate 3D coordinates
                    mol_with_h = Chem.AddHs(mol_copy)
                    success = AllChem.EmbedMolecule(mol_with_h, randomSeed=42)

                    if success == -1:
                        logger.warning(f"Could not generate 3D coordinates for molecule {idx}")
                        skipped_count += 1
                        continue

                    # Energy minimize the structure
                    try:
                        AllChem.UFFOptimizeMolecule(mol_with_h)
                    except Exception as e:
                        logger.warning(f"Energy minimization failed for molecule {idx}: {e}")
                        # Continue anyway with the embedded but not minimized structure

                    # Add properties as SD data
                    for prop in ['ChEMBL_ID', 'MW', 'LogP', 'Activity_nM', "molecule_chembl_id"]:
                        if prop in row and not pd.isna(row[prop]):
                            mol_with_h.SetProp(prop, str(row[prop]))

                    mols_for_docking.append(mol_with_h)
                except Exception as e:
                    logger.warning(f"Error processing molecule {idx}: {e}")
                    skipped_count += 1
                    continue

            logger.info(f"Skipped {skipped_count} molecules due to errors or invalid structures")

            # Check if we have any valid molecules
            if not mols_for_docking:
                logger.error("No valid molecules for docking after 3D structure generation")
                return None

            # Write molecules to SDF file
            logger.info(f"Writing {len(mols_for_docking)} molecules to {output_sdf}")
            with Chem.SDWriter(output_sdf) as writer:
                for mol in mols_for_docking:
                    try:
                        writer.write(mol)
                    except Exception as e:
                        logger.warning(f"Error writing molecule to SDF: {e}")

            return output_sdf

        def cluster_molecules(self, n_clusters=50):
            """
            Cluster molecules to ensure structural diversity.

            Parameters:
            -----------
            n_clusters : int
                Number of clusters to create

            Returns:
            --------
            pd.DataFrame : Dataframe with cluster assignments
            """
            from sklearn.cluster import KMeans

            if self.filtered_molecules is None:
                raise ValueError("No filtered molecules available. Run filtering steps first.")

            # Calculate Morgan fingerprints
            fingerprints = []
            valid_indices = []

            logger.info("Calculating molecular fingerprints...")
            for idx, row in tqdm(self.filtered_molecules.iterrows(), total=len(self.filtered_molecules)):
                mol = row['ROMol'] if 'ROMol' in self.filtered_molecules.columns else row['Molecule']

                if mol is None or not isinstance(mol, Chem.rdchem.Mol):
                    continue

                try:
                    # Use rdMolDescriptors instead of AllChem for Morgan fingerprints
                    # This avoids the deprecation warning
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    array = np.zeros((1,))
                    Chem.DataStructs.ConvertToNumpyArray(fp, array)
                    fingerprints.append(array)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.warning(f"Error calculating fingerprint for molecule {idx}: {e}")

            if not fingerprints:
                logger.error("No valid fingerprints generated. Clustering cannot proceed.")
                return self.filtered_molecules

            # Convert to numpy array
            fingerprints_array = np.array(fingerprints)

            # Perform clustering
            actual_n_clusters = min(n_clusters, len(fingerprints_array))
            logger.info(f"Clustering {len(fingerprints_array)} molecules into {actual_n_clusters} clusters...")

            try:
                kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42)
                clusters = kmeans.fit_predict(fingerprints_array)
            except Exception as e:
                logger.error(f"Error during clustering: {e}")
                # Return original dataframe if clustering fails
                return self.filtered_molecules

            # Add cluster information to dataframe
            cluster_df = pd.DataFrame({
                'Original_Index': valid_indices,
                'Cluster': clusters
            })

            # Merge with original dataframe
            result_df = pd.merge(
                self.filtered_molecules.reset_index(),
                cluster_df,
                left_on='index',  # Changed from left_index to left_on
                right_on='Original_Index',
                how='right'
            )

            logger.info(f"Clustered molecules into {len(set(clusters))} groups")

            # Update filtered molecules
            self.filtered_molecules = result_df
            return result_df

        def select_diverse_subset(self, n_per_cluster=1, max_total=500):
            """
            Select a diverse subset of molecules for docking.

            Parameters:
            -----------
            n_per_cluster : int
                Number of molecules to select from each cluster
            max_total : int
                Maximum total number of molecules to select

            Returns:
            --------
            pd.DataFrame : Dataframe with selected molecules
            """
            if 'Cluster' not in self.filtered_molecules.columns:
                logger.info("Clustering molecules first...")
                self.cluster_molecules()

            # Group by cluster
            grouped = self.filtered_molecules.groupby('Cluster')

            # Select top molecules by activity from each cluster
            selected_mols = []

            for cluster, group in grouped:
                # Sort by activity (if available) and take top n
                if 'Activity_nM' in group.columns:
                    sorted_group = group.sort_values('Activity_nM')
                else:
                    # If no activity data, just take random n
                    sorted_group = group

                selected = sorted_group.head(n_per_cluster)
                selected_mols.append(selected)

            # Combine selected molecules
            result = pd.concat(selected_mols)

            # Limit to max_total if needed
            if len(result) > max_total:
                if 'Activity_nM' in result.columns:
                    result = result.sort_values('Activity_nM').head(max_total)
                else:
                    result = result.head(max_total)

            logger.info(f"Selected {len(result)} diverse molecules for docking")
            return result

        def run_full_pipeline(self, output_sdf="docking_ready_molecules.sdf", max_molecules=500):
            """
            Run the complete preprocessing pipeline.

            Parameters:
            -----------
            output_sdf : str
                Path to save the output SDF file
            max_molecules : int
                Maximum number of molecules to select for docking

            Returns:
            --------
            str : Path to the generated SDF file
            """
            logger.info("Starting full preprocessing pipeline...")

            # Check if we have valid molecules
            if self.molecules is None or len(self.molecules) == 0:
                logger.error("No valid molecules loaded. Cannot proceed with pipeline.")
                return None

            # Apply filters
            try:
                self.apply_activity_filter()
                self.apply_lipinski_filters(strict=False)
                self.apply_pains_filter()
                self.apply_additional_filters()
            except Exception as e:
                logger.error(f"Error during filtering steps: {e}")
                return None

            # Check if we have molecules after filtering
            if self.filtered_molecules is None or len(self.filtered_molecules) == 0:
                logger.error("No molecules remain after filtering steps. Cannot proceed.")
                return None

            # Cluster and select diverse subset
            try:
                self.cluster_molecules()
                final_selection = self.select_diverse_subset(max_total=max_molecules)
            except Exception as e:
                logger.error(f"Error during clustering and selection: {e}")
                # If clustering fails, just take the top molecules by activity
                if 'Activity_nM' in self.filtered_molecules.columns:
                    final_selection = self.filtered_molecules.sort_values('Activity_nM').head(max_molecules)
                else:
                    final_selection = self.filtered_molecules.head(max_molecules)

            # Store the final selection
            self.filtered_molecules = final_selection

            # Generate 3D structures and save
            try:
                output_file = self.generate_3d_structures(output_sdf)
                return output_file
            except Exception as e:
                logger.error(f"Error during 3D structure generation: {e}")
                return None


    # Example usage
    if __name__ == "__main__":
        input_file = "/home/halleluyah/Documents/Programming Projects/Bioinformatics/cancerag/data/estrogen_receptor_bioactivity_20250204_012600.csv"

        try:
            # Initialize the preprocessor
            new_preprocessor = NewMoleculePreprocessor(
                input_file=input_file,
                target_name="Estrogen Receptor",
                activity_threshold=1000  # 1 μM
            )

            # Run the full pipeline
            new_output_file = new_preprocessor.run_full_pipeline(
                output_sdf="new_er_docking_ready.sdf",
                max_molecules=500
            )

            if new_output_file:
                logger.info(f"Preprocessing complete. Docking-ready molecules saved to {new_output_file}")
            else:
                logger.error("Pipeline failed to produce output file.")
        except Exception as e:
            logger.error(f"An error occurred during pipeline execution: {e}")
    return (
        NewMoleculePreprocessor,
        input_file,
        logger,
        new_output_file,
        new_preprocessor,
    )


@app.cell
def _(mo):
    mo.md(r"""### Visualisation of the results and storing of visualisation""")
    return


@app.cell
def _(AllChem, Chem, os, pd):
    # import os
    import glob
    # from rdkit import Chem
    # from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import IPythonConsole
    # from rdkit.Chem import PandasTools
    # import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    # import numpy as np

    # Function to read molecules from an SDF file
    def read_molecules_from_sdf(sdf_file):
        """
        Read molecules from an SDF file and extract their properties.

        Args:
            sdf_file (str): Path to the SDF file

        Returns:
            tuple: (molecules list, properties dictionary)
        """
        molecules = []
        properties = {}

        # Read the SDF file
        suppl = Chem.SDMolSupplier(sdf_file)

        for mol in suppl:
            if mol is not None:
                # Store the molecule
                molecules.append(mol)

                # Extract ChEMBL ID and other properties
                mol_properties = {}
                for prop_name in mol.GetPropNames():
                    mol_properties[prop_name] = mol.GetProp(prop_name)

                # Use ChEMBL ID as the key if available, otherwise use molecule index
                key = mol_properties.get('molecule_chembl_id', f"Molecule_{len(molecules)}")
                properties[key] = mol_properties

        return molecules, properties

    # Function to visualize molecules
    def visualize_molecules(molecules, properties, output_file=None, num_per_row=10, molsPerPage=50):
        """
        Visualize molecules and their ChEMBL IDs.

        Args:
            molecules (list): List of RDKit molecule objects
            properties (dict): Dictionary of molecule properties
            output_file (str): Path to save the visualization (optional)
            num_per_row (int): Number of molecules per row
            molsPerPage (int): Number of molecules per page
        """
        # Add 2D coordinates if not present
        for mol in molecules:
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)

        # Prepare molecule legends (ChEMBL IDs)
        legends = []
        activity_values = []

        for mol in molecules:
            # Get molecule ChEMBL ID
            if mol.HasProp('molecule_chembl_id'):
                chembl_id = mol.GetProp('molecule_chembl_id')
                legend = f"{chembl_id}"

                # Add activity if available
                if mol.HasProp('Activity_nM'):
                    activity = mol.GetProp('Activity_nM')
                    legend += f"\nActivity: {activity} nM"
                    try:
                        activity_values.append(float(activity))
                    except ValueError:
                        activity_values.append(None)
                else:
                    activity_values.append(None)
            else:
                legend = "Unknown"
                activity_values.append(None)

            legends.append(legend)

        # Create the visualization
        if output_file:
            # Save to file
            img = Draw.MolsToGridImage(
                molecules,
                molsPerRow=num_per_row,
                subImgSize=(900, 900),
                legends=legends,
                useSVG=True
            )
            with open(output_file, 'w') as f:
                f.write(img.data)
            print(f"Visualization saved to {output_file}")
        else:
            # Display in the notebook/output
            img = Draw.MolsToGridImage(
                molecules,
                molsPerRow=num_per_row,
                subImgSize=(900, 900),
                legends=legends
            )
            return img

    # Function to create a DataFrame with molecule information
    def create_molecule_dataframe(molecules, properties):
        """
        Create a DataFrame with molecule information.

        Args:
            molecules (list): List of RDKit molecule objects
            properties (dict): Dictionary of molecule properties

        Returns:
            pandas.DataFrame: DataFrame with molecule information
        """
        data = []

        for i, mol in enumerate(molecules):
            mol_data = {}

            # Convert the molecule to SMILES string
            smiles = Chem.MolToSmiles(mol) if mol is not None else ""
            mol_data['SMILES'] = smiles

            # Find the corresponding properties
            chembl_id = mol.GetProp('molecule_chembl_id') if mol.HasProp('molecule_chembl_id') else f"Molecule_{i+1}"
            mol_data['molecule_chembl_id'] = chembl_id

            mol_props = properties.get(chembl_id, {})

            # Add all properties to the data
            for prop_name, prop_value in mol_props.items():
                mol_data[prop_name] = prop_value

            data.append(mol_data)

        # Create DataFrame
        df = pd.DataFrame(data)

        return df

    # Simple function to display molecule with ChEMBL ID
    def display_molecules(sdf_file):
        """
        Display molecules from an SDF file.

        Args:
            sdf_file (str): Path to the SDF file
        """
        # Read molecules from SDF file
        molecules, properties = read_molecules_from_sdf(sdf_file)

        if not molecules:
            print(f"No molecules found in {sdf_file}")
            return

        # Display molecule information
        print(f"Found {len(molecules)} molecules in {sdf_file}")

        # Get ChEMBL IDs
        chembl_ids = []
        for mol in molecules:
            if mol.HasProp('molecule_chembl_id'):
                chembl_ids.append(mol.GetProp('molecule_chembl_id'))
            else:
                chembl_ids.append("Unknown")

        print("ChEMBL IDs:", chembl_ids)

        # Create visualization
        img = visualize_molecules(molecules, properties)
        return img

    # Main function to process SDF files
    def process_sdf_files(sdf_files, output_dir='molecule_visualizations'):
        """
        Process multiple SDF files and generate visualizations.

        Args:
            sdf_files (list): List of paths to SDF files
            output_dir (str): Directory to save the output files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        all_molecules = []
        all_properties = {}

        # Process each SDF file
        for sdf_file in sdf_files:
            print(f"Processing {sdf_file}...")
            molecules, properties = read_molecules_from_sdf(sdf_file)

            # Add molecules and properties to the overall collection
            all_molecules.extend(molecules)
            all_properties.update(properties)

            # Generate visualization for this file
            base_name = os.path.basename(sdf_file).split('.')[0]
            output_file = os.path.join(output_dir, f"{base_name}_visualization.svg")
            visualize_molecules(molecules, properties, output_file)

        # Create a DataFrame with all molecule information
        df = create_molecule_dataframe(all_molecules, all_properties)

        # Save the DataFrame to a CSV file
        csv_file = os.path.join(output_dir, "molecule_information.csv")
        df.to_csv(csv_file, index=False)
        print(f"Molecule information saved to {csv_file}")

        # Generate combined visualization
        combined_output_file = os.path.join(output_dir, "all_molecules_visualization.svg")
        visualize_molecules(all_molecules, all_properties, combined_output_file)

        return all_molecules, all_properties, df

    # Simplified function to just read and display molecules
    def simple_display(sdf_file):
        """
        Simple function to just read and display molecules from an SDF file.

        Args:
            sdf_file (str): Path to the SDF file
        """
        try:
            # Read the SDF file
            suppl = Chem.SDMolSupplier(sdf_file)
            molecules = [mol for mol in suppl if mol is not None]

            if not molecules:
                print(f"No molecules found in {sdf_file}")
                return

            # Get ChEMBL IDs and create legends
            legends = []
            for mol in molecules:
                if mol.HasProp('molecule_chembl_id'):
                    chembl_id = mol.GetProp('molecule_chembl_id')
                    if mol.HasProp('Activity_nM'):
                        activity = mol.GetProp('Activity_nM')
                        legends.append(f"{chembl_id}\nActivity: {activity} nM")
                    else:
                        legends.append(chembl_id)
                else:
                    legends.append("Unknown")

            # Compute 2D coordinates if needed
            for mol in molecules:
                if mol.GetNumConformers() == 0:
                    AllChem.Compute2DCoords(mol)

            # Create and display the visualization
            img = Draw.MolsToGridImage(
                molecules,
                molsPerRow=1,
                subImgSize=(500, 500),
                legends=legends
            )

            return img

        except Exception as e:
            print(f"Error processing SDF file: {e}")
            return None

    # Example usage
    if __name__ == "__main__":
        # Option 1: Process a single SDF file
        sdf_file = "new_er_docking_ready.sdf"  # Change to your SDF file path

        # Use the simplified function for direct display
        img = simple_display(sdf_file)
        # print(img)

        molecules, properties = read_molecules_from_sdf(sdf_file)
        visualize_molecules(molecules, properties, "molecule_visualization.svg")
        sdf_df = create_molecule_dataframe(molecules, properties)
        sdf_df.to_csv("sorted_molecule_info.csv")
    return (
        Draw,
        IPythonConsole,
        cm,
        create_molecule_dataframe,
        display_molecules,
        glob,
        img,
        molecules,
        plt,
        process_sdf_files,
        properties,
        read_molecules_from_sdf,
        sdf_df,
        sdf_file,
        simple_display,
        visualize_molecules,
    )


@app.cell
def _(pd):
    sorted_df = pd.read_csv("sorted_molecule_info.csv")
    return (sorted_df,)


@app.cell
def _(sorted_df):
    sorted_df
    return


if __name__ == "__main__":
    app.run()
