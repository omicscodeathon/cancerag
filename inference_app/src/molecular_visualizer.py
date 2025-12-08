"""
Molecular Visualization for Inference App

Creates 3D molecular visualizations using py3Dmol or alternative methods.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

try:
    import py3Dmol

    HAS_PY3DMOL = True
except ImportError:
    HAS_PY3DMOL = False
    logger.warning("py3Dmol not available. Install with: pip install py3dmol")

try:
    from rdkit.Chem import Draw

    HAS_RDKIT_DRAW = True
except ImportError:
    HAS_RDKIT_DRAW = False


class MolecularVisualizer:
    """Creates molecular visualizations for the web interface."""

    def __init__(self):
        """Initialize the visualizer."""
        self.has_3d = HAS_PY3DMOL

    def visualize_ligand_3d(
        self, mol: Chem.Mol, width: int = 400, height: int = 400
    ) -> str:
        """
        Create 3D visualization HTML for a ligand.
        Falls back to 2D if py3Dmol not available.

        Args:
            mol: RDKit molecule object
            width: Width of visualization
            height: Height of visualization

        Returns:
            HTML string with embedded visualization
        """
        # Always use 2D for now (more reliable in Gradio)
        # Can enable 3D later if py3Dmol is properly configured
        return self.visualize_ligand_2d(mol, width, height)

    def visualize_ligand_2d(
        self, mol: Chem.Mol, width: int = 400, height: int = 400
    ) -> str:
        """
        Create 2D visualization HTML for a ligand.

        Args:
            mol: RDKit molecule object
            width: Width of visualization
            height: Height of visualization

        Returns:
            HTML string with embedded 2D image
        """
        try:
            if not HAS_RDKIT_DRAW:
                return f'<div style="width:{width}px;height:{height}px;display:flex;align-items:center;justify-content:center;border:1px solid #ccc;">Molecule visualization not available</div>'

            # Generate 2D image
            img = Draw.MolToImage(mol, size=(width, height))

            # Convert to base64
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            # Create HTML with dark mode support
            html = f"""
            <div style="text-align:center;">
                <img src="data:image/png;base64,{img_str}" 
                     style="max-width:{width}px;max-height:{height}px;border:1px solid var(--border-color, #ddd);border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);background:var(--input-background, #fff);padding:10px;" 
                     alt="Molecule Structure"/>
            </div>
            """
            return html

        except Exception as e:
            logger.error(f"2D visualization failed: {e}")
            return f'<div style="width:{width}px;height:{height}px;display:flex;align-items:center;justify-content:center;border:1px solid #ccc;color:#666;">Visualization Error</div>'

    def visualize_receptor_3d(
        self, pdb_path: str, width: int = 600, height: int = 500
    ) -> str:
        """
        Create 3D visualization HTML for a receptor using py3Dmol.
        Uses write_html() to create standalone HTML and embeds it in iframe.

        Args:
            pdb_path: Path to PDB file
            width: Width of visualization
            height: Height of visualization

        Returns:
            HTML string with embedded 3D visualization
        """
        try:
            # Extract PDB ID from path
            pdb_id = Path(pdb_path).stem

            if HAS_PY3DMOL:
                # Read PDB file
                with open(pdb_path, "r") as f:
                    pdb_data = f.read()

                # Create py3Dmol viewer
                viewer = py3Dmol.view(width=width, height=height)
                viewer.addModel(pdb_data, "pdb")
                viewer.setStyle({"cartoon": {"color": "spectrum"}})
                viewer.setBackgroundColor("0xeeeeee")
                viewer.zoomTo()
                viewer.render()

                # Get the HTML structure from py3Dmol
                py3dmol_html = viewer._make_html()

                # Extract viewer ID
                import re

                viewer_id_match = re.search(r'id="([^"]+)"', py3dmol_html)
                viewer_id = (
                    viewer_id_match.group(1)
                    if viewer_id_match
                    else f"viewer_{id(viewer)}"
                )

                # Create HTML that ensures 3Dmol.js loads in parent window context
                # The key is to load the script in the parent document, not in an iframe
                import json

                pdb_data_escaped = json.dumps(pdb_data)

                html_output = f'''
                <div class="receptor-viewer" style="width:100%;max-width:{width}px;">
                    <!-- Ensure 3Dmol.js is loaded in parent window -->
                    <script>
                        (function() {{
                            // Load 3Dmol.js in parent window if not already loaded
                            var loadScript = function(src, callback) {{
                                // Check if already loaded
                                if (typeof $3Dmol !== 'undefined') {{
                                    if (callback) callback();
                                    return;
                                }}
                                
                                // Check if script tag already exists
                                var existing = document.querySelector('script[src="' + src + '"]');
                                if (existing) {{
                                    existing.addEventListener('load', callback);
                                    return;
                                }}
                                
                                var script = document.createElement('script');
                                script.src = src;
                                script.async = true;
                                script.onload = function() {{
                                    console.log('3Dmol.js loaded from', src);
                                    if (callback) callback();
                                }};
                                script.onerror = function() {{
                                    console.error('Failed to load 3Dmol.js from', src);
                                }};
                                document.head.appendChild(script);
                            }};
                            
                            // Try loading from primary CDN
                            loadScript('https://cdn.jsdelivr.net/npm/3dmol@2.0.2/build/3Dmol-min.js', function() {{
                                initViewer();
                            }});
                            
                            function initViewer() {{
                                var maxRetries = 50;
                                var retryCount = 0;
                                
                                function tryInit() {{
                                    retryCount++;
                                    
                                    if (typeof $3Dmol === 'undefined') {{
                                        if (retryCount < maxRetries) {{
                                            setTimeout(tryInit, 100);
                                        }} else {{
                                            console.error('3Dmol.js failed to load after', maxRetries, 'retries');
                                            showFallback();
                                        }}
                                        return;
                                    }}
                                    
                                    try {{
                                        var viewerId = "{viewer_id}";
                                        var element = document.getElementById(viewerId);
                                        
                                        if (!element) {{
                                            if (retryCount < maxRetries) {{
                                                setTimeout(tryInit, 100);
                                            }} else {{
                                                showFallback();
                                            }}
                                            return;
                                        }}
                                        
                                        // Hide warning message if present
                                        var warningId = viewerId.replace('3dmolviewer_', '3dmolwarning_');
                                        var warning = document.getElementById(warningId);
                                        if (warning) {{
                                            warning.style.display = 'none';
                                        }}
                                        
                                        // Create or get viewer
                                        var viewer = window['viewer_' + viewerId];
                                        if (!viewer) {{
                                            viewer = $3Dmol.createViewer(element, {{
                                                defaultcolors: $3Dmol.rasmolElementColors
                                            }});
                                            window['viewer_' + viewerId] = viewer;
                                        }}
                                        
                                        // Add model and style
                                        var pdbData = {pdb_data_escaped};
                                        viewer.addModel(pdbData, "pdb");
                                        viewer.setStyle({{cartoon: {{color: "spectrum"}}}});
                                        viewer.setBackgroundColor(0xeeeeee);
                                        viewer.zoomTo();
                                        viewer.render();
                                        
                                        console.log('3Dmol viewer initialized successfully');
                                    }} catch (error) {{
                                        console.error('Error initializing viewer:', error);
                                        showFallback();
                                    }}
                                }}
                                
                                // Start initialization
                                if (document.readyState === 'loading') {{
                                    document.addEventListener('DOMContentLoaded', tryInit);
                                }} else {{
                                    setTimeout(tryInit, 100);
                                }}
                            }}
                            
                            function showFallback() {{
                                var viewerId = "{viewer_id}";
                                var element = document.getElementById(viewerId);
                                if (element) {{
                                    element.innerHTML = '<div style="padding:30px;text-align:center;background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);border-radius:12px;color:white;"><h4 style="margin:0 0 10px 0;color:white;">🧬 Receptor Structure</h4><p style="margin:5px 0;"><strong>PDB ID:</strong> {pdb_id}</p><p style="margin:5px 0;font-size:14px;opacity:0.9;">Structure loaded and ready for docking</p></div>';
                                }}
                            }}
                        }})();
                    </script>
                    
                    <!-- py3Dmol's HTML structure -->
                    {py3dmol_html}
                    
                    <div style="margin-top:10px;padding:10px;background:var(--card-background, #f8f9fa);border-radius:8px;color:var(--body-text-color, #333);border:1px solid var(--border-color, #ddd);">
                        <small><strong>PDB ID:</strong> {pdb_id}</small>
                        <br><small style="opacity:0.7;">💡 If 3D viewer doesn't load, the structure is still ready for docking analysis</small>
                    </div>
                </div>
                '''
                return html_output
            else:
                # Fallback: Show info with PDB file content as base64
                with open(pdb_path, "r") as f:
                    pdb_content = f.read()[:1000]  # First 1000 chars

                html = f"""
                <div class="receptor-viewer" style="width:100%;max-width:{width}px;padding:20px;background:var(--card-background, #f8f9fa);border-radius:12px;color:var(--body-text-color, #333);border:1px solid var(--border-color, #ddd);">
                    <h4 style="margin-top:0;color:var(--heading-color, #333);">🧬 Receptor Structure</h4>
                    <p><strong>PDB ID:</strong> {pdb_id}</p>
                    <p><strong>File:</strong> {Path(pdb_path).name}</p>
                    <p style="font-size:14px;opacity:0.8;">3D structure loaded and ready for docking</p>
                    <div style="margin-top:15px;padding:10px;background:var(--input-background, #fff);border-radius:6px;font-family:monospace;font-size:11px;max-height:200px;overflow:auto;">
                        <pre style="margin:0;color:var(--body-text-color, #333);">{pdb_content}...</pre>
                    </div>
                    <p style="margin-top:10px;font-size:12px;opacity:0.7;">💡 Install py3dmol for 3D visualization: pip install py3dmol</p>
                </div>
                """
                return html

        except Exception as e:
            logger.error(f"Receptor visualization failed: {e}")
            return f'<div style="width:{width}px;height:{height}px;display:flex;align-items:center;justify-content:center;border:1px solid var(--border-color, #ccc);color:var(--error-color, #e74c3c);padding:20px;border-radius:8px;">Receptor visualization error<br><small>{str(e)}</small></div>'

    def create_comparison_view(
        self, ligand_mol: Chem.Mol, receptor_pdb: Optional[str] = None
    ) -> str:
        """
        Create a side-by-side comparison view.

        Args:
            ligand_mol: RDKit molecule object
            receptor_pdb: Optional path to receptor PDB file

        Returns:
            HTML string with comparison view
        """
        ligand_html = self.visualize_ligand_3d(ligand_mol, width=300, height=300)

        if receptor_pdb:
            receptor_html = self.visualize_receptor_3d(
                receptor_pdb, width=300, height=300
            )
        else:
            receptor_html = '<div style="width:300px;height:300px;display:flex;align-items:center;justify-content:center;border:1px solid #ccc;color:#666;">No receptor selected</div>'

        html = f"""
        <div style="display:flex;gap:20px;justify-content:center;flex-wrap:wrap;">
            <div style="text-align:center;">
                <h4 style="margin-bottom:10px;color:var(--heading-color, #333);">Ligand</h4>
                {ligand_html}
            </div>
            <div style="text-align:center;">
                <h4 style="margin-bottom:10px;color:var(--heading-color, #333);">Receptor</h4>
                {receptor_html}
            </div>
        </div>
        """
        return html
