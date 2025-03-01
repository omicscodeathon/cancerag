import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium", app_title="Visualisation of the PDB Files")


@app.cell
def _(json, os, tqdm):
    # import os
    import glob
    # import json
    import numpy as np
    import matplotlib.pyplot as plt
    # from tqdm import tqdm

    # -------------- VISUALIZATION OPTION 1: PYMOL PYTHON API --------------

    def visualize_with_pymol(pdb_path, output_image=None, show_surface=True, show_cartoon=True, 
                             highlight_ligands=True, color_by="secondary"):
        """
        Visualize a protein structure using PyMOL's Python API.
    
        Args:
            pdb_path (str): Path to the PDB file
            output_image (str): Path to save the image (if None, just displays)
            show_surface (bool): Whether to show protein surface
            show_cartoon (bool): Whether to show protein cartoon
            highlight_ligands (bool): Whether to highlight non-protein ligands
            color_by (str): How to color the protein ('secondary', 'chain', 'rainbow', 'b-factor')
        """
        # You need to have PyMOL installed and importable
        try:
            import pymol
            from pymol import cmd
        except ImportError:
            print("PyMOL not installed or not importable. Install with: pip install pymol-open-source")
            return None
    
        # Launch PyMOL in quiet mode
        pymol.finish_launching(['pymol', '-qc'])
    
        # Clear any existing objects
        cmd.delete('all')
    
        # Load the PDB file
        protein_name = os.path.basename(pdb_path).split('.')[0]
        cmd.load(pdb_path, protein_name)
    
        # Basic setup
        cmd.bg_color('white')
        cmd.remove('solvent')  # Remove water molecules
    
        # Display options
        if show_cartoon:
            cmd.show('cartoon')
        else:
            cmd.hide('cartoon')
        
        if show_surface:
            cmd.show('surface', 'polymer')
    
        # Color the protein based on the specified method
        if color_by == 'secondary':
            cmd.color('marine', 'ss h')  # Alpha helices
            cmd.color('forest', 'ss s')  # Beta sheets
            cmd.color('wheat', 'ss l+""')  # Loops
        elif color_by == 'chain':
            cmd.util.chainbow()
        elif color_by == 'rainbow':
            cmd.spectrum('count', 'rainbow')
        elif color_by == 'b-factor':
            cmd.spectrum('b', 'blue_white_red')
    
        # Highlight ligands if requested
        if highlight_ligands:
            cmd.select('ligands', 'not polymer')
            cmd.show('sticks', 'ligands')
            cmd.color('yellow', 'ligands')
            cmd.show('spheres', 'ligands')
            cmd.set('sphere_transparency', 0.5, 'ligands')
    
        # Set some good defaults for nice rendering
        cmd.set('ray_opaque_background', 0)
        cmd.set('ray_trace_mode', 1)
        cmd.set('ray_shadows', 0)
        cmd.set('depth_cue', 0)
        cmd.set('stick_radius', 0.15)
        cmd.set('sphere_scale', 0.25)
        cmd.set('cartoon_fancy_helices', 1)
    
        # Orient the view
        cmd.orient()
        cmd.zoom(buffer=1.0)
    
        # Ray trace for high quality
        cmd.ray(1200, 900)
    
        # If output path is provided, save the image
        if output_image:
            cmd.png(output_image, width=1200, height=900, dpi=300, ray=1)
            print(f"Image saved to {output_image}")
        else:
            # If no output path, just show the image
            cmd.show()
    
        return cmd

    # -------------- VISUALIZATION OPTION 2: NGLVIEW (FOR JUPYTER) --------------

    def visualize_with_nglview(pdb_path, show_surface=True, show_cartoon=True, 
                               highlight_ligands=True, color_scheme="chainname"):
        """
        Visualize a protein structure using NGLView in a Jupyter notebook.
    
        Args:
            pdb_path (str): Path to the PDB file
            show_surface (bool): Whether to show protein surface
            show_cartoon (bool): Whether to show protein cartoon
            highlight_ligands (bool): Whether to highlight non-protein ligands
            color_scheme (str): How to color the protein ('chainname', 'residueindex', 'secondary', etc.)
    
        Returns:
            nglview.NGLWidget: Interactive widget for Jupyter notebook
        """
        try:
            import nglview as nv
        except ImportError:
            print("NGLView not installed. Install with: pip install nglview")
            return None
    
        # Create viewer and load structure
        view = nv.show_file(pdb_path)
    
        # Clear default representation
        view.clear_representations()
    
        # Add requested representations
        if show_cartoon:
            view.add_representation('cartoon', selection='polymer', color=color_scheme)
    
        if show_surface:
            view.add_representation('surface', selection='polymer', opacity=0.7, color=color_scheme)
    
        if highlight_ligands:
            view.add_representation('licorice', selection='not polymer and not water', color='element')
            view.add_representation('spacefill', selection='not polymer and not water', opacity=0.6, color='element')
    
        # Set view orientation and camera
        view.center()
        view._remote_call('setSize', target='Widget', args=['800px', '600px'])
    
        return view

    # -------------- VISUALIZATION OPTION 3: PY3DMOL --------------

    def visualize_with_py3dmol(pdb_path, width=800, height=600, show_surface=True, 
                               show_cartoon=True, highlight_ligands=True, color_scheme="spectrum"):
        """
        Visualize a protein structure using Py3DMol in a Jupyter notebook.
    
        Args:
            pdb_path (str): Path to the PDB file
            width (int): Width of the viewer
            height (int): Height of the viewer
            show_surface (bool): Whether to show protein surface
            show_cartoon (bool): Whether to show protein cartoon
            highlight_ligands (bool): Whether to highlight non-protein ligands
            color_scheme (str): How to color the protein ('spectrum', 'chain', etc.)
    
        Returns:
            py3Dmol.view: Interactive view for Jupyter notebook
        """
        try:
            import py3Dmol
        except ImportError:
            print("Py3DMol not installed. Install with: pip install py3Dmol")
            return None
    
        # Read PDB file
        with open(pdb_path, 'r') as f:
            pdb_data = f.read()
    
        # Create viewer
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_data, 'pdb')
    
        # Add requested representations
        if show_cartoon:
            if color_scheme == "spectrum":
                view.setStyle({'model': -1, 'chain': 'A', 'elem': 'C'}, 
                             {'cartoon': {'color': 'spectrum'}})
            elif color_scheme == "chain":
                view.setStyle({'model': -1}, {'cartoon': {'colorscheme': 'chain'}})
            else:
                view.setStyle({'model': -1}, {'cartoon': {}})
    
        if show_surface:
            view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'color': 'white'})
    
        if highlight_ligands:
            view.setStyle({'model': -1, 'resn': ['EST', 'E2B', 'OHT', 'RAL', 'DES', 'GEN', 'AIT']}, 
                         {'stick': {'colorscheme': 'yellowCarbon', 'radius': 0.3}})
            view.setStyle({'model': -1, 'hetflag': True, 'not': {'resn': 'HOH'}}, 
                         {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})
    
        # Set view orientation and zoom
        view.zoomTo()
    
        return view

    # -------------- BATCH VISUALIZATION --------------

    def batch_visualize(pdb_dir, output_dir, method='pymol'):
        """
        Process all PDB files in a directory and create visualizations.
    
        Args:
            pdb_dir (str): Directory containing PDB files
            output_dir (str): Directory to save visualizations
            method (str): Visualization method ('pymol', 'static_images')
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
        # Get all PDB files in the directory and its subdirectories
        pdb_files = []
        for root, dirs, files in os.walk(pdb_dir):
            for file in files:
                if file.endswith('.pdb'):
                    pdb_files.append(os.path.join(root, file))
    
        print(f"Found {len(pdb_files)} PDB files")
    
        if method == 'pymol':
            try:
                import pymol
            except ImportError:
                print("PyMOL not installed. Install with: pip install pymol-open-source")
                return
        
            # Process each file with PyMOL
            for pdb_file in tqdm(pdb_files, desc="Generating visualizations"):
                base_name = os.path.basename(pdb_file).split('.')[0]
                output_path = os.path.join(output_dir, f"{base_name}.png")
            
                # Skip if already processed
                if os.path.exists(output_path):
                    continue
                
                try:
                    visualize_with_pymol(pdb_file, output_path)
                except Exception as e:
                    print(f"Error processing {pdb_file}: {str(e)}")
    
        elif method == 'static_images':
            # Create a grid of comparison images using matplotlib
            # This is a simpler alternative if PyMOL is not available
            er_types = ['er_alpha', 'er_beta', 'er_complex']
        
            # Group files by ER type
            grouped_files = {}
            for er_type in er_types:
                grouped_files[er_type] = [f for f in pdb_files if er_type in f.lower()]
        
            # Create a comparison figure
            fig, axs = plt.subplots(len(er_types), min(3, max([len(files) for files in grouped_files.values()])), 
                                    figsize=(15, 12))
        
            for i, er_type in enumerate(er_types):
                files = grouped_files[er_type][:3]  # Take up to 3 examples
                for j, pdb_file in enumerate(files):
                    # Create a simple representation using matplotlib
                    # (This is just a placeholder - in reality you'd need to extract coordinates)
                    axs[i, j].text(0.5, 0.5, f"{er_type}\n{os.path.basename(pdb_file)}", 
                                 ha='center', va='center')
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                    axs[i, j].set_title(os.path.basename(pdb_file).split('.')[0])
        
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'er_comparison.png'), dpi=300)
            plt.close()

    # -------------- INTERACTIVE VISUALIZATION APP --------------

    def create_visualization_app(pdb_dir):
        """
        Create a simple interactive app to visualize PDB files.
        For use in Jupyter notebooks.
    
        Args:
            pdb_dir (str): Directory containing PDB files
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("ipywidgets not installed. Install with: pip install ipywidgets")
            return
    
        # Get all PDB files
        pdb_files = []
        for root, dirs, files in os.walk(pdb_dir):
            for file in files:
                if file.endswith('.pdb'):
                    pdb_files.append(os.path.join(root, file))
    
        # Sort by type and name
        pdb_files.sort()
    
        # Create dropdown for file selection
        file_dropdown = widgets.Dropdown(
            options=[(os.path.basename(f), f) for f in pdb_files],
            description='PDB File:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='50%')
        )
    
        # Create visualization method dropdown
        method_dropdown = widgets.Dropdown(
            options=[('PyMOL', 'pymol'), ('NGLView', 'nglview'), ('Py3DMol', 'py3dmol')],
            description='Method:',
            style={'description_width': 'initial'}
        )
    
        # Create display options
        show_surface = widgets.Checkbox(value=True, description='Show Surface')
        show_cartoon = widgets.Checkbox(value=True, description='Show Cartoon')
        highlight_ligands = widgets.Checkbox(value=True, description='Highlight Ligands')
    
        # Create output area
        output = widgets.Output()
    
        # Define update function
        def update_view(change=None):
            output.clear_output()
            with output:
                pdb_path = file_dropdown.value
                method = method_dropdown.value
            
                if method == 'pymol':
                    try:
                        visualize_with_pymol(
                            pdb_path, 
                            show_surface=show_surface.value,
                            show_cartoon=show_cartoon.value,
                            highlight_ligands=highlight_ligands.value
                        )
                        print("PyMOL visualization complete. If using Jupyter Lab, you might need to check the PyMOL window.")
                    except Exception as e:
                        print(f"Error with PyMOL: {str(e)}")
            
                elif method == 'nglview':
                    try:
                        view = visualize_with_nglview(
                            pdb_path,
                            show_surface=show_surface.value,
                            show_cartoon=show_cartoon.value,
                            highlight_ligands=highlight_ligands.value
                        )
                        display(view)
                    except Exception as e:
                        print(f"Error with NGLView: {str(e)}")
            
                elif method == 'py3dmol':
                    try:
                        view = visualize_with_py3dmol(
                            pdb_path,
                            show_surface=show_surface.value,
                            show_cartoon=show_cartoon.value,
                            highlight_ligands=highlight_ligands.value
                        )
                        display(view)
                    except Exception as e:
                        print(f"Error with Py3DMol: {str(e)}")
    
        # Connect the update function to the widgets
        file_dropdown.observe(update_view, names='value')
        method_dropdown.observe(update_view, names='value')
        show_surface.observe(update_view, names='value')
        show_cartoon.observe(update_view, names='value')
        highlight_ligands.observe(update_view, names='value')
    
        # Create button for updating
        update_button = widgets.Button(
            description='Update Visualization',
            button_style='info'
        )
        update_button.on_click(update_view)
    
        # Create the layout
        controls = widgets.VBox([
            file_dropdown,
            widgets.HBox([method_dropdown, update_button]),
            widgets.HBox([show_surface, show_cartoon, highlight_ligands])
        ])
    
        # Display the app
        display(controls, output)
    
        # Initial update
        update_view()

    # -------------- ANALYSIS FUNCTIONS --------------

    def compare_er_subtypes(pdb_dir, output_dir="analysis_results"):
        """
        Analyze and compare the structures of different ER subtypes.
    
        Args:
            pdb_dir (str): Directory containing PDB files
            output_dir (str): Directory to save analysis results
        """
        try:
            import Bio.PDB
        except ImportError:
            print("BioPython not installed. Install with: pip install biopython")
            return
    
        os.makedirs(output_dir, exist_ok=True)
    
        # Get all PDB files and group by type
        er_types = ['er_alpha', 'er_beta', 'er_complex']
        grouped_files = {er_type: [] for er_type in er_types}
    
        for root, dirs, files in os.walk(pdb_dir):
            for file in files:
                if file.endswith('.pdb'):
                    path = os.path.join(root, file)
                    # Determine which group it belongs to
                    for er_type in er_types:
                        if er_type in path.lower():
                            grouped_files[er_type].append(path)
                            break
    
        # Initialize parser
        parser = Bio.PDB.PDBParser(QUIET=True)
    
        # Store data for analysis
        binding_site_residues = {er_type: [] for er_type in er_types}
        helix_counts = {er_type: [] for er_type in er_types}
        sheet_counts = {er_type: [] for er_type in er_types}
    
        # Process each file
        for er_type, files in grouped_files.items():
            print(f"Analyzing {er_type} structures...")
        
            for pdb_file in tqdm(files[:10]):  # Limit to 10 files per type for speed
                try:
                    # Parse structure
                    structure_id = os.path.basename(pdb_file).split('.')[0]
                    structure = parser.get_structure(structure_id, pdb_file)
                
                    # Count secondary structure elements (this is simplified - real analysis would be more complex)
                    helix_count = 0
                    sheet_count = 0
                
                    # Simplified: count residues in helices and sheets based on CA atom B-factor
                    # (In real analysis, you would use DSSP or similar)
                    for model in structure:
                        for chain in model:
                            for residue in chain:
                                if 'CA' in residue:
                                    ca_atom = residue['CA']
                                    # Just a placeholder - real analysis would use DSSP
                                    if ca_atom.get_bfactor() > 50:
                                        helix_count += 1
                                    elif ca_atom.get_bfactor() < 30:
                                        sheet_count += 1
                
                    helix_counts[er_type].append(helix_count)
                    sheet_counts[er_type].append(sheet_count)
                
                    # Find binding site residues (simplified example)
                    # For real analysis, you would need to identify the binding site more rigorously
                    binding_residues = []
                    for model in structure:
                        for chain in model:
                            for residue in chain:
                                # Basic distance-based approach (simplified)
                                # In reality, you would identify ligands and measure distances properly
                                if residue.get_resname() not in ['HOH', 'WAT']:
                                    for atom in residue:
                                        # Check if any atom is at the center of the protein 
                                        # (This is just a placeholder for demonstration)
                                        if atom.get_coord()[0]**2 + atom.get_coord()[1]**2 + atom.get_coord()[2]**2 < 100:
                                            binding_residues.append(residue.get_id()[1])
                                            break
                
                    binding_site_residues[er_type].append(binding_residues)
                
                except Exception as e:
                    print(f"Error processing {pdb_file}: {str(e)}")
    
        # Create some simple visualizations of the analysis
        plt.figure(figsize=(10, 6))
    
        # Plot average helix counts
        avg_helix = [np.mean(helix_counts[er_type]) if helix_counts[er_type] else 0 for er_type in er_types]
        avg_sheet = [np.mean(sheet_counts[er_type]) if sheet_counts[er_type] else 0 for er_type in er_types]
    
        x = np.arange(len(er_types))
        width = 0.35
    
        plt.bar(x - width/2, avg_helix, width, label='Avg. Helix Count')
        plt.bar(x + width/2, avg_sheet, width, label='Avg. Sheet Count')
    
        plt.xlabel('ER Subtype')
        plt.ylabel('Count')
        plt.title('Secondary Structure Comparison')
        plt.xticks(x, [er_type.replace('er_', 'ER-').upper() for er_type in er_types])
        plt.legend()
    
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'secondary_structure_comparison.png'), dpi=300)
        plt.close()
    
        # Save the analysis data
        analysis_data = {
            'helix_counts': helix_counts,
            'sheet_counts': sheet_counts,
            'binding_site_residues': binding_site_residues
        }
    
        with open(os.path.join(output_dir, 'analysis_data.json'), 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
        print(f"Analysis complete. Results saved to {output_dir}")

    # -------------- MAIN VISUALIZATION FUNCTIONS --------------

    def visualize_single_structure(pdb_path, output_path=None, method='pymol'):
        """
        Visualize a single PDB structure using the specified method.
    
        Args:
            pdb_path (str): Path to the PDB file
            output_path (str): Path to save the output image (if applicable)
            method (str): Visualization method ('pymol', 'nglview', 'py3dmol')
        """
        if method == 'pymol':
            return visualize_with_pymol(pdb_path, output_path)
        elif method == 'nglview':
            return visualize_with_nglview(pdb_path)
        elif method == 'py3dmol':
            return visualize_with_py3dmol(pdb_path)
        else:
            print(f"Unknown visualization method: {method}")
            return None

    def main():
        """
        Main function to demonstrate usage of the visualization functions.
    
        Uncomment the function you want to use.
        """
        # Define paths
        pdb_dir = "/home/halleluyah/Documents/Programming Projects/Bioinformatics/cancerag/data/pdb/er_alpha"
        output_dir = "visualizations"
    
        # Example: Visualize a single PDB file with PyMOL
        # Example PDB file (if you already have one downloaded)
        # sample_pdb = "data/pdb/er_alpha/1A52.pdb"
        # visualize_single_structure(sample_pdb, f"{output_dir}/1A52.png", method='pymol')
    
        # Example: Batch visualize all PDB files in the directory
        # batch_visualize(pdb_dir, output_dir, method='py3dmol')
    
        # Example: Create an interactive visualization app for Jupyter notebooks
        create_visualization_app(pdb_dir)
    
        # Example: Run comparative analysis on different ER subtypes
        # compare_er_subtypes(pdb_dir, output_dir=f"{output_dir}/analysis")
    
        print("To use this code, uncomment one of the examples in the main function.")
        print("Or import the functions you need into your own scripts or notebooks.")

    if __name__ == "__main__":
        main()
    return (
        batch_visualize,
        compare_er_subtypes,
        create_visualization_app,
        glob,
        main,
        np,
        plt,
        visualize_single_structure,
        visualize_with_nglview,
        visualize_with_py3dmol,
        visualize_with_pymol,
    )


if __name__ == "__main__":
    app.run()
