"""
Beautiful Interactive CancerAg Inference App

Interactive interface with receptor selection, 3D visualization, and comprehensive results.
"""

import logging
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.docking_extractor import DockingFeatureExtractor
from src.inference_pipeline import InferencePipeline
from src.molecular_visualizer import MolecularVisualizer
from src.predictor import load_predictor
from src.receptor_manager import ReceptorManager
from src.receptor_processor import ReceptorProcessor
from src.result_visualizer import ResultVisualizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from gradio_molecule3d import Molecule3D

    HAS_MOLECULE3D = True
except ImportError:
    HAS_MOLECULE3D = False
    logger.warning(
        "gradio_molecule3d not available. Install with: pip install gradio-molecule3d"
    )

# Global instances
_predictor = None
_pipeline = None
_receptor_manager = None
_visualizer = None
_result_visualizer = None
_docking_extractor = None
_receptor_processor = None


def initialize_app():
    """Initialize all components."""
    global \
        _predictor, \
        _pipeline, \
        _receptor_manager, \
        _visualizer, \
        _result_visualizer, \
        _docking_extractor, \
        _receptor_processor

    try:
        base_path = Path(__file__).parent.parent

        # Initialize components
        logger.info("Initializing app components...")
        _predictor = load_predictor(model_name="random_forest")
        _pipeline = InferencePipeline(
            _predictor, base_path=str(base_path), enable_docking=False
        )
        _receptor_manager = ReceptorManager(base_path=str(base_path))
        _visualizer = MolecularVisualizer()
        _result_visualizer = ResultVisualizer()
        _docking_extractor = DockingFeatureExtractor(base_path=str(base_path))
        _receptor_processor = ReceptorProcessor(base_path=str(base_path))

        logger.info("App initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize app: {e}", exc_info=True)
        return False


def get_receptor_list() -> List[Tuple[str, str]]:
    """Get list of available receptors for dropdown."""
    if _receptor_manager is None:
        return []
    receptors = _receptor_manager.get_available_receptors()
    return [(r["display_name"], r["name"]) for r in receptors]


def get_receptor_pdb_path(receptor_name: str) -> Optional[str]:
    """Get PDB file path for a receptor."""
    if not receptor_name or _receptor_manager is None:
        return None

    receptor = _receptor_manager.get_receptor_by_name(receptor_name)
    if not receptor or not receptor["available"]:
        return None

    pdb_path = receptor["pdb_path"]
    if pdb_path and Path(pdb_path).exists():
        return pdb_path
    return None


def process_pdb_id(pdb_id: str, progress=gr.Progress()) -> Tuple[Optional[str], str]:
    """
    Process a PDB ID input.

    Returns:
        Tuple of (pdb_path, info_html)
    """
    if not pdb_id or not pdb_id.strip():
        return (
            None,
            '<div style="text-align:center;padding:20px;color:var(--body-text-color, #666);">Enter a PDB ID</div>',
        )

    if _receptor_processor is None:
        return (
            None,
            '<div style="text-align:center;padding:20px;color:var(--error-color, #e74c3c);">Receptor processor not initialized</div>',
        )

    pdb_path, binding_info, status_msg = _receptor_processor.fetch_pdb_by_id(
        pdb_id.strip(), progress_callback=progress
    )

    if pdb_path and Path(pdb_path).exists():
        info = f"""
        <div style="margin-top:15px;padding:15px;background:var(--card-background, #f8f9fa);border-radius:8px;border:1px solid var(--border-color, #ddd);color:var(--body-text-color, #333);">
            <strong>PDB ID:</strong> {binding_info.get("pdb_id", pdb_id)}<br>
            <strong>Source:</strong> {binding_info.get("source", "PDB")}<br>
            <strong>Binding Site:</strong> Identified from co-crystallized ligand<br>
            <small style="opacity:0.7;">{status_msg}</small>
        </div>
        """
        return pdb_path, info
    else:
        return (
            None,
            f'<div style="text-align:center;padding:20px;color:var(--error-color, #e74c3c);">{status_msg}</div>',
        )


def process_uploaded_file(
    uploaded_file, progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """
    Process an uploaded PDB file.

    Returns:
        Tuple of (pdb_path, info_html)
    """
    if uploaded_file is None:
        return (
            None,
            '<div style="text-align:center;padding:20px;color:var(--body-text-color, #666);">Upload a PDB file</div>',
        )

    if _receptor_processor is None:
        return (
            None,
            '<div style="text-align:center;padding:20px;color:var(--error-color, #e74c3c);">Receptor processor not initialized</div>',
        )

    file_path = (
        uploaded_file.name if hasattr(uploaded_file, "name") else str(uploaded_file)
    )

    pdb_path, binding_info, status_msg = _receptor_processor.process_uploaded_file(
        file_path, progress_callback=progress
    )

    if pdb_path and Path(pdb_path).exists():
        info = f"""
        <div style="margin-top:15px;padding:15px;background:var(--card-background, #f8f9fa);border-radius:8px;border:1px solid var(--border-color, #ddd);color:var(--body-text-color, #333);">
            <strong>File:</strong> {Path(file_path).name}<br>
            <strong>Source:</strong> Uploaded<br>
            <strong>Binding Site:</strong> Identified from co-crystallized ligand<br>
            <small style="opacity:0.7;">{status_msg}</small>
        </div>
        """
        return pdb_path, info
    else:
        return (
            None,
            f'<div style="text-align:center;padding:20px;color:var(--error-color, #e74c3c);">{status_msg}</div>',
        )


def visualize_receptor(receptor_name: str) -> Tuple[Optional[str], str]:
    """
    Get receptor PDB path and info for visualization from existing structures.

    Returns:
        Tuple of (pdb_path, info_html)
    """
    if not receptor_name or _receptor_manager is None:
        return (
            None,
            '<div style="text-align:center;padding:50px;color:var(--body-text-color, #666);">Select a receptor to visualize</div>',
        )

    receptor = _receptor_manager.get_receptor_by_name(receptor_name)
    if not receptor or not receptor["available"]:
        return (
            None,
            '<div style="text-align:center;padding:50px;color:var(--error-color, #e74c3c);">Receptor structure not available</div>',
        )

    pdb_path = receptor["pdb_path"]
    if pdb_path and Path(pdb_path).exists():
        info = f"""
        <div style="margin-top:15px;padding:15px;background:var(--card-background, #f8f9fa);border-radius:8px;border:1px solid var(--border-color, #ddd);color:var(--body-text-color, #333);">
            <strong>Receptor:</strong> {receptor["display_name"]}<br>
            <strong>PDB ID:</strong> {receptor["pdb_id"]}<br>
            <strong>Binding Site:</strong> Defined from co-crystallized ligand
        </div>
        """
        return pdb_path, info
    else:
        return (
            None,
            '<div style="text-align:center;padding:50px;color:var(--error-color, #e74c3c);">Receptor structure file not found</div>',
        )


def get_receptor_path_from_input(
    input_method: str,
    receptor_name: Optional[str] = None,
    pdb_id: Optional[str] = None,
    uploaded_file: Optional[Any] = None,
) -> Tuple[Optional[str], str]:
    """
    Unified function to get receptor path based on input method.

    Args:
        input_method: "existing", "pdb_id", or "upload"
        receptor_name: For "existing" method
        pdb_id: For "pdb_id" method
        uploaded_file: For "upload" method

    Returns:
        Tuple of (pdb_path, info_html)
    """
    if input_method == "existing":
        return visualize_receptor(receptor_name or "")
    elif input_method == "pdb_id":
        return process_pdb_id(pdb_id or "")
    elif input_method == "upload":
        return process_uploaded_file(uploaded_file)
    else:
        return (
            None,
            '<div style="text-align:center;padding:20px;color:var(--body-text-color, #666);">Select an input method</div>',
        )


def visualize_ligand(smiles: str) -> str:
    """Visualize ligand from SMILES."""
    if not smiles or not smiles.strip():
        return '<div style="text-align:center;padding:50px;color:var(--body-text-color, #666);">Enter a SMILES string to visualize</div>'

    try:
        from src.molecule_processor import MoleculeProcessor

        processor = MoleculeProcessor()

        is_valid, error_msg = processor.validate_smiles(smiles)
        if not is_valid:
            return f'<div style="text-align:center;padding:50px;color:var(--error-color, #e74c3c);">Invalid SMILES: {error_msg}</div>'

        mol = processor.standardize_molecule(smiles)
        if mol is None:
            return '<div style="text-align:center;padding:50px;color:var(--error-color, #e74c3c);">Failed to process molecule</div>'

        html = _visualizer.visualize_ligand_3d(mol, width=400, height=400)
        canonical = processor.get_canonical_smiles(mol)
        info = f"""
        <div style="margin-top:15px;padding:15px;background:var(--card-background, #f8f9fa);border-radius:8px;border:1px solid var(--border-color, #ddd);color:var(--body-text-color, #333);">
            <strong>Canonical SMILES:</strong><br>
            <code style="background:var(--input-background, #fff);padding:5px;border-radius:4px;display:block;margin-top:5px;color:var(--body-text-color, #333);border:1px solid var(--border-color, #ddd);">{canonical}</code>
        </div>
        """
        return html + info
    except Exception as e:
        logger.error(f"Ligand visualization error: {e}")
        return f'<div style="text-align:center;padding:50px;color:var(--error-color, #e74c3c);">Error: {str(e)}</div>'


def run_prediction(
    smiles: str, receptor_path: Optional[str], run_docking: bool, progress=gr.Progress()
) -> Tuple[str, str, str, str, str, str, Optional[str]]:
    """
    Run complete prediction pipeline with progress tracking.

    Returns:
        Tuple of (prediction_result, probabilities_html, descriptors_html, docking_html, comparison_html, graphs_html)
    """
    if not smiles or not smiles.strip():
        return ("⚠️ Please enter a SMILES string", "", "", "", "", "")

    if not receptor_path:
        return (
            "⚠️ Please select or provide a receptor structure",
            "",
            "",
            "",
            "",
            "",
            None,
        )

    try:
        progress(0.1, desc="Validating SMILES...")

        # Process ligand
        from src.molecule_processor import MoleculeProcessor

        processor = MoleculeProcessor()

        is_valid, error_msg = processor.validate_smiles(smiles)
        if not is_valid:
            return (f"❌ Invalid SMILES: {error_msg}", "", "", "", "", "")

        progress(0.2, desc="Standardizing molecule...")
        mol = processor.standardize_molecule(smiles)
        if mol is None:
            return ("❌ Failed to process molecule", "", "", "", "", "")

        progress(0.3, desc="Extracting molecular descriptors...")
        # Extract molecular descriptors
        from src.feature_extractor import FeatureExtractor

        feature_extractor = FeatureExtractor()
        features_df = feature_extractor.extract_features(mol)

        # Get receptor display name
        receptor_display_name = Path(receptor_path).stem if receptor_path else "Unknown"

        # Perform docking if requested
        docking_scores = {}
        docking_html = ""
        if run_docking and _docking_extractor and receptor_path:
            try:
                progress(0.4, desc="Preparing ligand for docking...")
                logger.info(f"Running docking for {receptor_display_name}...")

                progress(
                    0.5, desc=f"Running docking against {receptor_display_name}..."
                )
                # Try to dock using custom receptor path
                affinity = _docking_extractor.dock_custom_receptor_path(
                    mol, receptor_path
                )

                progress(0.7, desc="Processing docking results...")

                if affinity is not None:
                    docking_scores[receptor_display_name] = affinity

                    # Create docking visualization
                    fig = _result_visualizer.plot_docking_results(
                        {receptor_display_name: affinity}
                    )
                    import base64
                    import io

                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                    buffer.seek(0)
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    docking_html = f"""
                    <div style="padding:20px;background:var(--card-background, #f8f9fa);border-radius:8px;border:1px solid var(--border-color, #ddd);">
                        <h3 style="margin-top:0;color:var(--heading-color, #333);">⚗️ Docking Results</h3>
                        <div style="background:var(--input-background, #fff);padding:15px;border-radius:8px;margin-bottom:15px;border:1px solid var(--border-color, #ddd);">
                            <strong style="color:var(--body-text-color, #333);">Binding Affinity:</strong> <span style="font-size:20px;color:#2ecc71;font-weight:bold;">{affinity:.2f} kcal/mol</span><br>
                            <small style="color:var(--body-text-color, #666);opacity:0.8;">More negative values indicate stronger binding</small>
                        </div>
                        <img src="data:image/png;base64,{img_str}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);background:var(--input-background, #fff);padding:5px;"/>
                    </div>
                    """
                    plt.close(fig)

                    # Add docking feature to feature vector (for other receptors, use default)
                    all_docking_features = {
                        name: -5.0 for name in _docking_extractor.receptor_names
                    }
                    # Use a generic key for custom receptors
                    all_docking_features["custom_receptor"] = affinity
                    docking_features_df = pd.DataFrame([all_docking_features])
                    features_df = pd.concat([features_df, docking_features_df], axis=1)
                else:
                    docking_html = f'<div style="padding:20px;background:var(--warning-background, #fff3cd);border-radius:8px;color:var(--warning-text, #856404);border:1px solid var(--border-color, #ffc107);">⚠️ Docking failed for {receptor_display_name}. Using default values.</div>'
                    # Add default docking features
                    all_docking_features = {
                        name: -5.0 for name in _docking_extractor.receptor_names
                    }
                    docking_features_df = pd.DataFrame([all_docking_features])
                    features_df = pd.concat([features_df, docking_features_df], axis=1)
            except Exception as e:
                logger.warning(f"Docking failed: {e}")
                docking_html = f'<div style="padding:20px;background:var(--warning-background, #fff3cd);border-radius:8px;color:var(--warning-text, #856404);border:1px solid var(--border-color, #ffc107);">⚠️ Docking error: {str(e)}</div>'
                # Add default docking features
                all_docking_features = {
                    name: -5.0 for name in _docking_extractor.receptor_names
                }
                docking_features_df = pd.DataFrame([all_docking_features])
                features_df = pd.concat([features_df, docking_features_df], axis=1)

        progress(0.8, desc="Making prediction...")
        # Make prediction
        predicted_class, probabilities = _predictor.predict(features_df)

        progress(0.9, desc="Generating visualizations...")

        # Format prediction result with dark mode support
        prediction_html = f"""
        <div style="text-align:center;padding:30px;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);border-radius:12px;color:white;box-shadow:0 4px 15px rgba(0,0,0,0.2);">
            <h2 style="margin:0 0 10px 0;font-size:28px;color:white;">🎯 Predicted Bias Category</h2>
            <div style="font-size:36px;font-weight:bold;margin:15px 0;color:white;">{predicted_class}</div>
            <div style="font-size:14px;opacity:0.9;color:white;">Based on molecular descriptors and docking analysis</div>
        </div>
        """

        # Create probabilities visualization
        prob_fig = _result_visualizer.plot_class_probabilities(probabilities)
        import base64
        import io

        buffer = io.BytesIO()
        prob_fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        buffer.seek(0)
        prob_img_str = base64.b64encode(buffer.getvalue()).decode()
        probabilities_html = f'<img src="data:image/png;base64,{prob_img_str}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);"/>'
        plt.close(prob_fig)

        # Extract and display descriptors
        key_descriptors = {
            "MW": features_df.get("MW", [None])[0],
            "LogP": features_df.get("LogP", [None])[0],
            "HBD": features_df.get("HBD", [None])[0],
            "HBA": features_df.get("HBA", [None])[0],
            "TPSA": features_df.get("TPSA", [None])[0],
            "Rotatable_Bonds": features_df.get("Rotatable_Bonds", [None])[0],
            "QED": features_df.get("qed", [None])[0],
            "SPS": features_df.get("SPS", [None])[0],
        }
        # Filter out None values
        key_descriptors = {k: v for k, v in key_descriptors.items() if v is not None}
        descriptors_html = _result_visualizer.create_descriptors_table(key_descriptors)

        # Create comparison view with structure visualization
        comparison_html = _visualizer.create_comparison_view(mol, receptor_path)

        # Add structure visualization to results if Molecule3D is available
        structure_visualization = None
        if HAS_MOLECULE3D and receptor_path and Path(receptor_path).exists():
            structure_visualization = receptor_path

        # Create descriptor radar chart
        if key_descriptors:
            radar_fig = _result_visualizer.plot_descriptors_radar(key_descriptors)
            buffer = io.BytesIO()
            radar_fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
            buffer.seek(0)
            radar_img_str = base64.b64encode(buffer.getvalue()).decode()
            graphs_html = f"""
            <div style="margin-bottom:20px;">
                <h3 style="color:var(--heading-color, #333);margin-bottom:15px;">📊 Molecular Descriptors Profile</h3>
                <img src="data:image/png;base64,{radar_img_str}" style="max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);background:var(--input-background, #fff);padding:5px;"/>
            </div>
            """
            plt.close(radar_fig)
        else:
            graphs_html = ""

        progress(1.0, desc="Complete!")

        return (
            prediction_html,
            probabilities_html,
            descriptors_html,
            docking_html,
            comparison_html,
            graphs_html,
            structure_visualization,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return (f"❌ Error: {str(e)}", "", "", "", "", "", None)


# Initialize app
logger.info("Initializing app...")
if not initialize_app():
    logger.error("Failed to initialize app. Some features may not work.")

# Custom CSS for dark mode support, progress bar visibility, and 3Dmol.js compatibility
custom_css = """
<style>
:root {
    --card-background: var(--background-fill-secondary, #f8f9fa);
    --input-background: var(--background-fill-primary, #ffffff);
    --body-text-color: var(--body-text-color, #333333);
    --heading-color: var(--body-text-color, #333333);
    --border-color: var(--border-color-primary, #e0e0e0);
    --error-color: #e74c3c;
    --warning-background: #fff3cd;
    --warning-text: #856404;
}

/* Ensure 3Dmol.js viewers are properly styled */
.receptor-viewer {
    position: relative;
}

.receptor-viewer canvas {
    border-radius: 8px;
}

/* Progress bar styling for better visibility */
.progress-bar-container {
    background: var(--background-fill-secondary, #f0f0f0) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin: 10px 0 !important;
    border: 1px solid var(--border-color-primary, #ddd) !important;
}

.progress-bar {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    height: 8px !important;
    border-radius: 4px !important;
    transition: width 0.3s ease !important;
}

.progress-text {
    color: var(--body-text-color, #333) !important;
    font-weight: 500 !important;
    margin-top: 5px !important;
    font-size: 14px !important;
}

/* Dark mode progress bar */
.dark .progress-bar-container {
    background: var(--background-fill-secondary, #2a2a2a) !important;
    border-color: var(--border-color-primary, #444) !important;
}

.dark .progress-text {
    color: var(--body-text-color, #e0e0e0) !important;
}

/* Gradio progress component styling */
.gr-progress {
    background: var(--background-fill-secondary, #f0f0f0) !important;
    border: 1px solid var(--border-color-primary, #ddd) !important;
    border-radius: 8px !important;
    padding: 15px !important;
    margin: 15px 0 !important;
}

.gr-progress .progress-bar {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    height: 10px !important;
    border-radius: 5px !important;
}

.gr-progress .progress-text {
    color: var(--body-text-color, #333) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin-top: 8px !important;
}
</style>

<!-- Preload 3Dmol.js in the main page context -->
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.0.2/build/3Dmol-min.js" async></script>
"""

# Create Gradio interface with dark mode support
with gr.Blocks(
    title="🧬 CancerAg: Interactive Biased Agonism Prediction",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="gray"),
) as app:
    gr.HTML(custom_css)
    gr.Markdown(
        """
        # 🧬 CancerAg: Interactive Biased Agonism Prediction
        
        **Predict GPCR ligand signaling bias with interactive receptor selection and 3D visualization**
        
        Select a receptor, visualize structures, and predict bias categories using machine learning.
        """
    )

    with gr.Tabs():
        with gr.Tab("🔬 Interactive Prediction"):
            gr.Markdown("### Step 1: Select Receptor")

            # Receptor input method selection
            receptor_input_method = gr.Radio(
                choices=["existing", "pdb_id", "upload"],
                value="existing",
                label="Receptor Input Method",
                info="Choose how to provide the receptor structure",
                elem_id="receptor_method",
            )

            # Store the selected receptor path (hidden state)
            current_receptor_path = gr.State(value=None)

            with gr.Row():
                with gr.Column(scale=1):
                    # Tab 1: Existing receptors
                    with gr.Group(visible=True) as existing_group:
                        receptor_dropdown = gr.Dropdown(
                            choices=get_receptor_list(),
                            label="Available Receptors",
                            info="Select from pre-processed GPCR receptors",
                            interactive=True,
                        )

                    # Tab 2: PDB ID input
                    with gr.Group(visible=False) as pdb_id_group:
                        pdb_id_input = gr.Textbox(
                            label="PDB ID",
                            placeholder="e.g., 1F88, 3SN6",
                            info="Enter a PDB identifier to fetch and process",
                            interactive=True,
                        )
                        fetch_pdb_btn = gr.Button(
                            "🔍 Fetch & Process PDB", variant="secondary"
                        )

                    # Tab 3: File upload
                    with gr.Group(visible=False) as upload_group:
                        pdb_file_upload = gr.File(
                            label="Upload PDB File",
                            file_types=[".pdb"],
                            info="Upload your own PDB structure file",
                        )
                        process_upload_btn = gr.Button(
                            "📤 Process Uploaded File", variant="secondary"
                        )

                    # Receptor visualization (shared across all methods)
                    if HAS_MOLECULE3D:
                        receptor_visualization = Molecule3D(
                            label="Receptor Structure",
                            reps=[
                                {"model": 0, "style": "cartoon", "color": "spectrum"}
                            ],
                            height=400,
                        )
                        receptor_info = gr.HTML(
                            label="Receptor Information",
                            value='<div style="text-align:center;padding:20px;color:var(--body-text-color, #666);">Select a receptor input method</div>',
                        )
                    else:
                        receptor_visualization = gr.HTML(
                            label="Receptor Structure",
                            value='<div style="text-align:center;padding:50px;color:var(--body-text-color, #666);">Select a receptor input method</div>',
                        )
                        receptor_info = gr.HTML()

                with gr.Column(scale=1):
                    gr.Markdown("### Step 2: Enter Ligand")
                    smiles_input = gr.Textbox(
                        label="SMILES String",
                        placeholder="Enter SMILES string, e.g., CCO (ethanol)",
                        lines=3,
                        info="Enter the molecular structure in SMILES notation",
                    )
                    ligand_visualization = gr.HTML(
                        label="Ligand Structure",
                        value='<div style="text-align:center;padding:50px;color:var(--body-text-color, #666);">Enter a SMILES string to visualize</div>',
                    )

            gr.Markdown("### Step 3: Run Analysis")

            with gr.Row():
                run_docking_checkbox = gr.Checkbox(
                    label="Perform Docking Analysis",
                    value=True,
                    info="Dock the ligand against the selected receptor (takes 1-2 minutes)",
                )
                predict_btn = gr.Button(
                    "🚀 Predict Bias Category",
                    variant="primary",
                    size="lg",
                )

            # Update UI visibility based on input method
            def update_input_visibility(method: str):
                """Show/hide input groups based on selected method."""
                return (
                    gr.update(visible=(method == "existing")),
                    gr.update(visible=(method == "pdb_id")),
                    gr.update(visible=(method == "upload")),
                )

            receptor_input_method.change(
                fn=update_input_visibility,
                inputs=receptor_input_method,
                outputs=[existing_group, pdb_id_group, upload_group],
            )

            # Update visualizations for each input method
            if HAS_MOLECULE3D:

                def update_receptor_visualization(
                    method: str, receptor_name: str, pdb_id: str, uploaded_file
                ) -> Tuple[Optional[str], str, Optional[str]]:
                    """Update receptor visualization using Molecule3D component."""
                    pdb_path, info = get_receptor_path_from_input(
                        method, receptor_name, pdb_id, uploaded_file
                    )
                    # Molecule3D expects file path as string or None
                    return pdb_path if pdb_path else None, info, pdb_path

                # Existing receptors
                receptor_dropdown.change(
                    fn=lambda name: update_receptor_visualization(
                        "existing", name, "", None
                    ),
                    inputs=receptor_dropdown,
                    outputs=[
                        receptor_visualization,
                        receptor_info,
                        current_receptor_path,
                    ],
                )

                # PDB ID
                fetch_pdb_btn.click(
                    fn=lambda pdb_id: update_receptor_visualization(
                        "pdb_id", "", pdb_id, None
                    ),
                    inputs=pdb_id_input,
                    outputs=[
                        receptor_visualization,
                        receptor_info,
                        current_receptor_path,
                    ],
                )

                # File upload
                process_upload_btn.click(
                    fn=lambda file: update_receptor_visualization(
                        "upload", "", "", file
                    ),
                    inputs=pdb_file_upload,
                    outputs=[
                        receptor_visualization,
                        receptor_info,
                        current_receptor_path,
                    ],
                )
            else:

                def update_receptor_html(
                    method: str, receptor_name: str, pdb_id: str, uploaded_file
                ) -> Tuple[str, Optional[str]]:
                    """Update receptor visualization using HTML fallback."""
                    _, info = get_receptor_path_from_input(
                        method, receptor_name, pdb_id, uploaded_file
                    )
                    pdb_path, _ = get_receptor_path_from_input(
                        method, receptor_name, pdb_id, uploaded_file
                    )
                    return info, pdb_path

                receptor_dropdown.change(
                    fn=lambda name: update_receptor_html("existing", name, "", None),
                    inputs=receptor_dropdown,
                    outputs=[receptor_visualization, current_receptor_path],
                )

                fetch_pdb_btn.click(
                    fn=lambda pdb_id: update_receptor_html("pdb_id", "", pdb_id, None),
                    inputs=pdb_id_input,
                    outputs=[receptor_visualization, current_receptor_path],
                )

                process_upload_btn.click(
                    fn=lambda file: update_receptor_html("upload", "", "", file),
                    inputs=pdb_file_upload,
                    outputs=[receptor_visualization, current_receptor_path],
                )

            smiles_input.change(
                fn=visualize_ligand,
                inputs=smiles_input,
                outputs=ligand_visualization,
            )

            # Results section
            gr.Markdown("### 📊 Results")

            with gr.Row():
                prediction_output = gr.HTML(label="Prediction")

            # Structure visualization in results (if Molecule3D available)
            if HAS_MOLECULE3D:
                with gr.Row():
                    result_structure_viewer = Molecule3D(
                        label="Receptor Structure (Results)",
                        reps=[{"model": 0, "style": "cartoon", "color": "spectrum"}],
                        height=400,
                        visible=False,
                    )

            with gr.Row():
                with gr.Column():
                    probabilities_output = gr.HTML(label="Class Probabilities")
                with gr.Column():
                    docking_output = gr.HTML(label="Docking Results")

            with gr.Row():
                comparison_output = gr.HTML(label="Ligand-Receptor Comparison")

            with gr.Row():
                graphs_output = gr.HTML(label="Additional Visualizations")

            with gr.Row():
                descriptors_output = gr.HTML(label="Molecular Descriptors")

            # Connect prediction button
            if HAS_MOLECULE3D:
                predict_btn.click(
                    fn=run_prediction,
                    inputs=[smiles_input, current_receptor_path, run_docking_checkbox],
                    outputs=[
                        prediction_output,
                        probabilities_output,
                        descriptors_output,
                        docking_output,
                        comparison_output,
                        graphs_output,
                        result_structure_viewer,
                    ],
                )
            else:
                predict_btn.click(
                    fn=lambda smiles, path, docking: run_prediction(
                        smiles, path, docking
                    )[:-1],
                    inputs=[smiles_input, current_receptor_path, run_docking_checkbox],
                    outputs=[
                        prediction_output,
                        probabilities_output,
                        descriptors_output,
                        docking_output,
                        comparison_output,
                        graphs_output,
                    ],
                )

            # Example molecules
            gr.Markdown("### 💡 Example Molecules")
            example_smiles = [
                "CCO",  # Ethanol
                "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
                "CN1CCC[C@H]1c2cccnc2",  # Nicotine
            ]
            gr.Examples(
                examples=[[smiles] for smiles in example_smiles],
                inputs=smiles_input,
                label="Click to load example",
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About CancerAg
                
                **CancerAg** is a machine learning platform for predicting biased agonism in GPCR ligands.
                
                ### Features
                - 🎯 **Bias Prediction**: Predict signaling bias categories using trained ML models
                - 🔬 **Interactive Receptor Selection**: Choose from 50+ available GPCR structures
                - 🧪 **3D Visualization**: View receptor and ligand structures in 3D
                - ⚗️ **Molecular Docking**: Perform docking analysis to predict binding affinities
                - 📊 **Comprehensive Analysis**: View descriptors, probabilities, and visualizations
                
                ### Supported Bias Categories
                - **G protein** - G protein-biased signaling
                - **β Arrestin** - β-arrestin-biased signaling
                - **ERK** - ERK pathway bias
                - **G protein selectivity** - Selective G protein coupling
                - **Agonist** - Non-biased agonist
                
                ### Model Performance
                - **Test Accuracy**: ~77%
                - **F1-Score**: ~0.60 (macro-averaged)
                - **Features**: 200+ molecular descriptors + docking affinities
                
                ### Citation
                If you use CancerAg in your research, please cite:
                
                > CancerAg: A Machine Learning Platform for GPCR Biased Agonism Prediction
                
                ---
                
                **Note**: Predictions should be validated experimentally. This tool is for research purposes.
                """
            )

    gr.Markdown(
        """
        ---
        <div style="text-align:center;color:#666;padding:20px;">
            <p>🧬 CancerAg - GPCR Biased Agonism Prediction Platform</p>
            <p style="font-size:12px;">Built with ❤️ for computational drug discovery</p>
        </div>
        """
    )

if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
