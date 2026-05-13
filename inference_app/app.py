"""
Beautiful Interactive CancerAg Inference App

Interactive interface with receptor selection, 3D visualization, and comprehensive results.
Uses pure Gradio components for a clean, intuitive interface.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Suppress sklearn feature name warnings (features are in correct order)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.docking_extractor import DockingFeatureExtractor
from src.inference_pipeline import InferencePipeline
from src.molecular_visualizer import MolecularVisualizer
from src.predictor import load_predictor, ModernBiasPredictor
from src.receptor_manager import ReceptorManager
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
_predictor = None              # legacy BiasPredictor (kept for fallback)
_modern_predictor = None       # new ModernBiasPredictor (Phase 5)
_pipeline = None
_receptor_manager = None
_visualizer = None
_result_visualizer = None
_docking_extractor = None


def initialize_app():
    """Initialize all components."""
    global \
        _predictor, \
        _modern_predictor, \
        _pipeline, \
        _receptor_manager, \
        _visualizer, \
        _result_visualizer, \
        _docking_extractor

    try:
        base_path = Path(__file__).parent.parent

        logger.info("Initializing app components...")
        # Phase 5: prefer ModernBiasPredictor (sklearn-Pipeline artifacts)
        try:
            _modern_predictor = ModernBiasPredictor(repo_root=base_path)
            logger.info(
                "ModernBiasPredictor loaded: model=%s sha256=%s",
                _modern_predictor.model_name,
                (_modern_predictor.model_sha256 or "")[:8],
            )
        except Exception as exc:
            logger.warning(
                "ModernBiasPredictor unavailable (%s); falling back to legacy",
                exc,
            )
            _modern_predictor = None

        # Legacy predictor kept as a fallback when modern artifacts missing.
        try:
            _predictor = load_predictor(model_name="random_forest")
        except Exception as exc:
            logger.warning("Legacy predictor unavailable: %s", exc)
            _predictor = None

        _pipeline = (
            InferencePipeline(
                _predictor, base_path=str(base_path), enable_docking=False
            )
            if _predictor is not None
            else None
        )
        _receptor_manager = ReceptorManager(base_path=str(base_path))
        _visualizer = MolecularVisualizer()
        _result_visualizer = ResultVisualizer()
        _docking_extractor = DockingFeatureExtractor(base_path=str(base_path))

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


def visualize_receptor(receptor_name: str) -> Tuple[Optional[str], str]:
    """Get receptor PDB path and info for visualization."""
    if not receptor_name or _receptor_manager is None:
        return (None, "")

    receptor = _receptor_manager.get_receptor_by_name(receptor_name)
    if not receptor or not receptor["available"]:
        return (None, "❌ Receptor not available")

    pdb_path = receptor["pdb_path"]
    binding_site = receptor.get("binding_site", {})
    
    if pdb_path and Path(pdb_path).exists():
        info = f"**{receptor['display_name']}** • PDB: `{receptor['pdb_id']}` • Binding site from: {binding_site.get('ligand_name', 'N/A')}"
        return pdb_path, info
    else:
        return (None, "❌ Structure file not found")


def validate_and_visualize_ligand(smiles: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Validate and visualize ligand from SMILES."""
    if not smiles or not smiles.strip():
        return (None, "", None)

    try:
        from src.molecule_processor import MoleculeProcessor

        processor = MoleculeProcessor()

        is_valid, error_msg = processor.validate_smiles(smiles)
        if not is_valid:
            return (None, f"❌ {error_msg}", None)

        mol = processor.standardize_molecule(smiles)
        if mol is None:
            return (None, "❌ Failed to process molecule", None)

        from rdkit import Chem
        from rdkit.Chem import Draw, Descriptors
        import tempfile
        
        img = Draw.MolToImage(mol, size=(350, 350))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as tmp:
            img_path = tmp.name
            img.save(img_path)
        
        canonical = processor.get_canonical_smiles(mol)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        info_md = f"✅ Valid • MW: {mw:.1f} • LogP: {logp:.2f}"
        
        return (img_path, info_md, canonical)
    except Exception as e:
        logger.error(f"Ligand visualization error: {e}")
        return (None, f"❌ {str(e)}", None)


def assess_drug_likeness(mol) -> Dict:
    """Assess drug-likeness using multiple rules."""
    from rdkit.Chem import Descriptors, Lipinski
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    
    # Lipinski's Rule of 5
    lipinski_violations = sum([
        mw > 500,
        logp > 5,
        hbd > 5,
        hba > 10
    ])
    lipinski_pass = lipinski_violations <= 1
    
    # Veber's rules (oral bioavailability)
    veber_pass = rot_bonds <= 10 and tpsa <= 140
    
    # Lead-likeness
    lead_like = 250 <= mw <= 350 and logp <= 3.5 and rot_bonds <= 7
    
    return {
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "tpsa": tpsa,
        "rot_bonds": rot_bonds,
        "lipinski_violations": lipinski_violations,
        "lipinski_pass": lipinski_pass,
        "veber_pass": veber_pass,
        "lead_like": lead_like,
    }


def run_prediction(
    smiles: str, receptor_name: Optional[str], run_docking: bool, progress=gr.Progress()
) -> Tuple:
    """Run complete prediction pipeline with progress tracking."""

    # Default empty returns (12 outputs — added AD banner + SHAP table)
    empty_return = (
        "",  # prediction_output
        {},  # probabilities_output
        "",  # drug_likeness_md
        "",  # interpretation_md
        "",  # docking_md
        None,  # radar_chart
        gr.update(visible=False),  # results_section
        gr.update(visible=False),  # loading_section
        None,  # descriptors_df
        "",  # binding_interpretation
        "",  # ad_banner_md
        None,  # shap_df
    )
    
    if not smiles or not smiles.strip() or not receptor_name:
        return empty_return

    receptor = _receptor_manager.get_receptor_by_name(receptor_name) if _receptor_manager else None
    if not receptor or not receptor["available"]:
        return empty_return

    try:
        progress(0.05, desc="🔬 Validating molecule...")

        from src.molecule_processor import MoleculeProcessor
        processor = MoleculeProcessor()

        is_valid, error_msg = processor.validate_smiles(smiles)
        if not is_valid:
            return (
                f"❌ **Invalid SMILES:** {error_msg}",
                {}, "", "", "", None, gr.update(visible=True), gr.update(visible=False), None, "", "", None
            )

        progress(0.15, desc="⚙️ Processing molecule...")
        mol = processor.standardize_molecule(smiles)
        if mol is None:
            return (
                "❌ **Error:** Failed to process molecule",
                {}, "", "", "", None, gr.update(visible=True), gr.update(visible=False), None, "", "", None
            )

        # Drug-likeness assessment
        progress(0.20, desc="💊 Assessing drug-likeness...")
        drug_props = assess_drug_likeness(mol)

        progress(0.25, desc="📊 Extracting descriptors...")
        from src.feature_extractor import FeatureExtractor
        feature_extractor = FeatureExtractor()
        features_df = feature_extractor.extract_features(mol)

        receptor_display_name = receptor["display_name"]
        binding_site = receptor.get("binding_site", {})

        # Docking
        docking_affinity_value = None
        docking_md = ""
        binding_interpretation = ""
        
        if run_docking and _docking_extractor:
            try:
                progress(0.35, desc=f"🧪 Preparing receptor...")

                logger.info(f"="*60)
                logger.info(f"DOCKING: {receptor_name}")
                logger.info(f"Receptor: {receptor_display_name} (PDB: {receptor['pdb_id']})")
                logger.info(f"Binding site: ({binding_site.get('center_x', 0):.1f}, {binding_site.get('center_y', 0):.1f}, {binding_site.get('center_z', 0):.1f})")

                progress(0.50, desc=f"⚗️ Docking (30-60s)...")

                affinity = _docking_extractor.dock_single_receptor(mol, receptor_name)

                progress(0.70, desc="📈 Analyzing results...")

                if affinity is not None:
                    docking_affinity_value = affinity
                    
                    logger.info(f"DOCKING RESULT: {affinity:.2f} kcal/mol")
                    logger.info(f"="*60)
                    
                    # Binding strength interpretation
                    if affinity < -9.0:
                        strength = "🟢 Excellent"
                        strength_desc = "Very high predicted binding affinity"
                    elif affinity < -7.0:
                        strength = "🟡 Good"
                        strength_desc = "Strong predicted binding"
                    elif affinity < -5.0:
                        strength = "🟠 Moderate"
                        strength_desc = "Moderate binding predicted"
                    else:
                        strength = "🔴 Weak"
                        strength_desc = "Low binding affinity"
                    
                    docking_md = f"""### ⚗️ Docking Results

| Metric | Value |
|--------|-------|
| **Binding Affinity** | **{affinity:.2f} kcal/mol** |
| **Strength** | {strength} |
| **Target** | {receptor_display_name} ({receptor['pdb_id']}) |
| **Binding Site Ligand** | {binding_site.get('ligand_name', 'N/A')} |

> *{strength_desc}. More negative = stronger binding.*
"""
                    binding_interpretation = f"{strength} ({affinity:.2f} kcal/mol)"

                    # Add docking features
                    all_docking_features = {name: -5.0 for name in _docking_extractor.receptor_names}
                    all_docking_features[receptor_name] = affinity
                    docking_features_df = pd.DataFrame([all_docking_features])
                    features_df = pd.concat([features_df, docking_features_df], axis=1)
                else:
                    docking_md = "⚠️ Docking failed - using default values"
                    binding_interpretation = "N/A (docking failed)"
                    all_docking_features = {name: -5.0 for name in _docking_extractor.receptor_names}
                    docking_features_df = pd.DataFrame([all_docking_features])
                    features_df = pd.concat([features_df, docking_features_df], axis=1)
                    
            except Exception as e:
                logger.error(f"Docking exception: {e}", exc_info=True)
                docking_md = f"❌ Docking error: {str(e)[:50]}"
                binding_interpretation = "Error"
                all_docking_features = {name: -5.0 for name in _docking_extractor.receptor_names}
                docking_features_df = pd.DataFrame([all_docking_features])
                features_df = pd.concat([features_df, docking_features_df], axis=1)

        progress(0.80, desc="🤖 Running prediction...")
        ad_banner_md = ""
        shap_df = None
        # Phase 5: prefer ModernBiasPredictor when available — uses the
        # sklearn-Pipeline artifacts and returns AD + SHAP top-5.
        receptor_uniprot = receptor.get("uniprot") or receptor.get("uniprot_id")
        if _modern_predictor is not None and receptor_uniprot:
            modern_result = _modern_predictor.predict(
                smiles, str(receptor_uniprot), log_audit=True
            )
            predicted_class = modern_result["predicted_class"]
            probabilities = modern_result["probabilities"]
            ad = modern_result["applicability"]
            conf_lbl = modern_result["confidence"]
            top_shap = modern_result["top_shap"]
            if not ad.get("in_domain", True):
                ad_banner_md = (
                    f"### ⚠️ Out-of-Domain Warning\n\n"
                    f"Nearest-neighbor Tanimoto = "
                    f"**{ad['nearest_neighbor_tanimoto']:.2f}** "
                    f"(threshold {ad['threshold']:.2f}). "
                    f"This molecule is structurally far from the training set; "
                    f"treat the prediction with extra caution."
                )
            else:
                ad_banner_md = (
                    f"### ✅ In Applicability Domain\n\n"
                    f"Nearest-neighbor Tanimoto = "
                    f"**{ad['nearest_neighbor_tanimoto']:.2f}** "
                    f"(≥ {ad['threshold']:.2f}). Confidence: **{conf_lbl}**."
                )
            if top_shap:
                shap_df = pd.DataFrame(
                    [{"Feature": n, "SHAP value": f"{v:+.4f}"} for n, v in top_shap]
                )
        elif _predictor is not None:
            predicted_class, probabilities = _predictor.predict(features_df)
        else:
            return (
                "❌ **Error:** No predictor available",
                {}, "", "", "", None, gr.update(visible=True), gr.update(visible=False), None, "", "", None
            )

        progress(0.90, desc="🎨 Generating analysis...")

        # Main prediction result
        probabilities_dict = {k: float(v) for k, v in probabilities.items()}
        max_prob = max(probabilities_dict.values()) if probabilities_dict else 0
        
        if max_prob >= 0.7:
            conf_level = "High confidence"
            conf_emoji = "🟢"
        elif max_prob >= 0.5:
            conf_level = "Moderate confidence"
            conf_emoji = "🟡"
        else:
            conf_level = "Low confidence"
            conf_emoji = "🟠"

        prediction_md = f"""# 🎯 {predicted_class}

{conf_emoji} **{conf_level}** ({max_prob*100:.0f}%)

Predicted signaling bias for this ligand against **{receptor_display_name}**"""

        # Drug-likeness analysis
        lipinski_status = "✅ Pass" if drug_props["lipinski_pass"] else f"❌ {drug_props['lipinski_violations']} violations"
        veber_status = "✅ Pass" if drug_props["veber_pass"] else "❌ Fail"
        lead_status = "✅ Yes" if drug_props["lead_like"] else "❌ No"

        drug_likeness_md = f"""### 💊 Drug-Likeness Assessment

| Rule | Status | Details |
|------|--------|---------|
| **Lipinski's Ro5** | {lipinski_status} | MW≤500, LogP≤5, HBD≤5, HBA≤10 |
| **Veber's Rules** | {veber_status} | RotBonds≤10, TPSA≤140 |
| **Lead-Like** | {lead_status} | 250≤MW≤350, LogP≤3.5, RotBonds≤7 |

**Key Properties:**
- MW: {drug_props['mw']:.1f} | LogP: {drug_props['logp']:.2f} | TPSA: {drug_props['tpsa']:.1f}
- HBD: {drug_props['hbd']} | HBA: {drug_props['hba']} | RotBonds: {drug_props['rot_bonds']}
"""

        # Result interpretation
        interpretation_md = f"""### 🔍 What This Means

**{predicted_class}** indicates this ligand may preferentially signal through the **{predicted_class.lower()} pathway** when bound to {receptor_display_name}.

| Aspect | Finding |
|--------|---------|
| **Predicted Bias** | {predicted_class} |
| **Confidence** | {max_prob*100:.0f}% ({conf_level.lower()}) |
| **Binding** | {binding_interpretation if binding_interpretation else 'Docking disabled'} |
| **Drug-likeness** | {'Favorable' if drug_props['lipinski_pass'] else 'Needs optimization'} |

> **Note:** This is a computational prediction. Experimental validation is recommended.
"""

        # Descriptors table
        descriptors_data = [
            {"Property": "Molecular Weight", "Value": f"{drug_props['mw']:.1f}", "Rule": "≤500 (Lipinski)"},
            {"Property": "LogP", "Value": f"{drug_props['logp']:.2f}", "Rule": "≤5 (Lipinski)"},
            {"Property": "H-Bond Donors", "Value": str(drug_props['hbd']), "Rule": "≤5 (Lipinski)"},
            {"Property": "H-Bond Acceptors", "Value": str(drug_props['hba']), "Rule": "≤10 (Lipinski)"},
            {"Property": "TPSA (Å²)", "Value": f"{drug_props['tpsa']:.1f}", "Rule": "≤140 (Veber)"},
            {"Property": "Rotatable Bonds", "Value": str(drug_props['rot_bonds']), "Rule": "≤10 (Veber)"},
        ]
        if docking_affinity_value:
            descriptors_data.append({
                "Property": "Binding Affinity", 
                "Value": f"{docking_affinity_value:.2f} kcal/mol",
                "Rule": "<-7 (good)"
            })
        descriptors_df = pd.DataFrame(descriptors_data)

        # Radar chart
        radar_img_path = None
        try:
            radar_data = {
                "MW": drug_props['mw'],
                "LogP": drug_props['logp'],
                "HBD": drug_props['hbd'],
                "HBA": drug_props['hba'],
                "TPSA": drug_props['tpsa'],
                "RotBonds": drug_props['rot_bonds'],
            }
            radar_fig = _result_visualizer.plot_descriptors_radar(radar_data)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as tmp:
                radar_img_path = tmp.name
                radar_fig.savefig(radar_img_path, format="png", dpi=150, bbox_inches="tight")
            plt.close(radar_fig)
        except Exception as e:
            logger.warning(f"Could not generate radar chart: {e}")

        progress(1.0, desc="✅ Complete!")

        return (
            prediction_md,
            probabilities_dict,
            drug_likeness_md,
            interpretation_md,
            docking_md,
            radar_img_path,
            gr.update(visible=True),   # Show results
            gr.update(visible=False),  # Hide loading
            descriptors_df,
            binding_interpretation,
            ad_banner_md,
            shap_df,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return (
            f"❌ **Error:** {str(e)}",
            {}, "", "", "", None, gr.update(visible=True), gr.update(visible=False), None, "", "", None
        )


def start_loading():
    """Show loading state and hide results."""
    return (gr.update(visible=True), gr.update(visible=False))


# Initialize app
logger.info("Initializing app...")
if not initialize_app():
    logger.error("Failed to initialize app. Some features may not work.")


# ==================== GRADIO UI ====================
with gr.Blocks(
    title="CancerAg: Biased Agonism Prediction",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
) as app:
    
    # Header
    gr.Markdown(
        """
        # 🧬 CancerAg
        ### Predict GPCR Ligand Signaling Bias
        """
    )

    with gr.Tabs() as main_tabs:
        
        # ==================== PREDICTION TAB ====================
        with gr.Tab("🔬 Predict", id="predict-tab"):
            
            with gr.Row(equal_height=False):
                
                # ========== INPUT COLUMN ==========
                with gr.Column(scale=1, min_width=400):
                    
                    # Receptor Selection
                    with gr.Group():
                        gr.Markdown("### 1️⃣ Select Receptor")
                        
                        receptor_dropdown = gr.Dropdown(
                            choices=get_receptor_list(),
                            label="Target Receptor",
                            info="43 GPCR structures available",
                            interactive=True,
                        )
                        
                        receptor_info = gr.Markdown(value="")
                        current_receptor_name = gr.State(value=None)
                    
                    # Ligand Input
                    with gr.Group():
                        gr.Markdown("### 2️⃣ Enter Molecule")
                        
                        smiles_input = gr.Textbox(
                            label="SMILES",
                            placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                            lines=1,
                        )
                        
                        ligand_info = gr.Markdown(value="")
                        
                        ligand_image = gr.Image(
                            label="Structure",
                            type="filepath",
                            height=180,
                            show_label=False,
                        )
                        
                        canonical_smiles_state = gr.State(value=None)
                        
                        with gr.Row():
                            ex_btn1 = gr.Button("Aspirin", size="sm", variant="secondary")
                            ex_btn2 = gr.Button("Ibuprofen", size="sm", variant="secondary")
                            ex_btn3 = gr.Button("Nicotine", size="sm", variant="secondary")
                    
                    # Run
                    with gr.Group():
                        gr.Markdown("### 3️⃣ Run")
                        
                        run_docking_checkbox = gr.Checkbox(
                            label="🧪 Enable Docking",
                            value=True,
                            info="30-60s per prediction",
                        )
                        
                        predict_btn = gr.Button(
                            "🚀 Predict Bias",
                            variant="primary",
                            size="lg",
                        )
                
                # ========== RECEPTOR VIEW ==========
                with gr.Column(scale=1, min_width=400):
                    
                    if HAS_MOLECULE3D:
                        gr.Markdown("### Receptor 3D Structure")
                        receptor_visualization = Molecule3D(
                            label="3D View",
                            reps=[{"model": 0, "style": "cartoon", "color": "spectrum"}],
                            height=450,
                            show_label=False,
                        )
                    else:
                        receptor_visualization = gr.Markdown("*3D viewer requires `gradio-molecule3d`*")
            
            # ==================== LOADING ====================
            with gr.Column(visible=False) as loading_section:
                gr.Markdown("---")
                gr.Markdown(
                    """
                    ## ⏳ Analyzing...

                    Please wait while we process your molecule. Docking typically takes 30-60 seconds.
                    """
                )
            
            # ==================== RESULTS ====================
            with gr.Column(visible=False) as results_section:
                gr.Markdown("---")
                
                with gr.Row():
                    # Prediction
                    with gr.Column(scale=1):
                        prediction_output = gr.Markdown(value="")

                    # Probabilities
                    with gr.Column(scale=1):
                        probabilities_output = gr.Label(
                            label="Class Probabilities (calibrated)",
                            num_top_classes=5,
                        )

                # Applicability-Domain banner (Phase 5)
                ad_banner_md = gr.Markdown(value="")

                # Top-5 SHAP feature contributions (Phase 5)
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🔍 Top-5 Feature Contributions (SHAP)")
                        shap_output = gr.DataFrame(
                            headers=["Feature", "SHAP value"],
                            interactive=False,
                        )
                
                with gr.Row():
                    # Interpretation
                    with gr.Column(scale=1):
                        interpretation_md = gr.Markdown(value="")
                    
                    # Drug-likeness
                    with gr.Column(scale=1):
                        drug_likeness_md = gr.Markdown(value="")
                
                # Docking results
                docking_md = gr.Markdown(value="")
                
                with gr.Row():
                    # Radar chart
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Molecular Profile")
                        radar_chart = gr.Image(
                            type="filepath",
                            height=300,
                            show_label=False,
                        )
                    
                    # Properties table
                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Properties")
                        descriptors_output = gr.DataFrame(
                            headers=["Property", "Value", "Rule"],
                            interactive=False,
                        )
                
                # Hidden state for binding interpretation
                binding_interpretation_state = gr.State(value="")

            # ==================== EVENTS ====================
            
            ex_btn1.click(fn=lambda: "CC(=O)OC1=CC=CC=C1C(=O)O", outputs=smiles_input)
            ex_btn2.click(fn=lambda: "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", outputs=smiles_input)
            ex_btn3.click(fn=lambda: "CN1CCC[C@H]1c2cccnc2", outputs=smiles_input)
            
            if HAS_MOLECULE3D:
                def update_receptor(receptor_name: str):
                    pdb_path, info_md = visualize_receptor(receptor_name)
                    return pdb_path if pdb_path else None, info_md, receptor_name

                receptor_dropdown.change(
                    fn=update_receptor,
                    inputs=receptor_dropdown,
                    outputs=[receptor_visualization, receptor_info, current_receptor_name],
                )
            else:
                receptor_dropdown.change(
                    fn=lambda n: (visualize_receptor(n)[1], n),
                    inputs=receptor_dropdown,
                    outputs=[receptor_info, current_receptor_name],
                )

            smiles_input.change(
                fn=validate_and_visualize_ligand,
                inputs=smiles_input,
                outputs=[ligand_image, ligand_info, canonical_smiles_state],
            )

            predict_btn.click(
                fn=start_loading,
                outputs=[loading_section, results_section],
            ).then(
                fn=run_prediction,
                inputs=[smiles_input, current_receptor_name, run_docking_checkbox],
                outputs=[
                    prediction_output,
                    probabilities_output,
                    drug_likeness_md,
                    interpretation_md,
                    docking_md,
                    radar_chart,
                    results_section,
                    loading_section,
                    descriptors_output,
                    binding_interpretation_state,
                    ad_banner_md,
                    shap_output,
                ],
            )

        # ==================== ABOUT ====================
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## About CancerAg
                
                Machine learning platform for predicting GPCR ligand signaling bias.
                
                ### Bias Categories
                | Category | Description |
                |----------|-------------|
                | **G protein** | G protein-biased signaling |
                | **β Arrestin** | β-arrestin-biased signaling |
                | **ERK** | ERK pathway bias |
                | **G protein selectivity** | Selective G protein coupling |
                | **Agonist** | Balanced agonist |
                
                ### Model Info
                - **Algorithm:** Random Forest
                - **Accuracy:** ~77%
                - **Features:** 200+ descriptors + docking
                - **Receptors:** 43 validated structures
                
                > ⚠️ Predictions should be validated experimentally.
                """
            )

    gr.Markdown("---\n**CancerAg** • GPCR Bias Prediction")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=port,
        allowed_paths=[
            "/app/data/processed/receptors",
            "/app/data/processed",
            "/app/logs",
            "/app/temp_receptors",
        ],
    )
