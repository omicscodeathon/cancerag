"""
Result Visualization for Inference App

Creates beautiful visualizations for prediction results, descriptors, and docking scores.
"""

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


class ResultVisualizer:
    """Creates visualizations for prediction results."""

    def __init__(self):
        """Initialize the visualizer."""
        self.color_palette = sns.color_palette("husl", 10)

    def plot_class_probabilities(
        self, probabilities: Dict[str, float], figsize: tuple = (8, 6)
    ) -> Figure:
        """
        Create a bar chart of class probabilities.

        Args:
            probabilities: Dictionary of class names to probabilities
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by probability
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        classes = [x[0] for x in sorted_probs]
        probs = [x[1] for x in sorted_probs]

        # Create bar chart
        bars = ax.barh(classes, probs, color=self.color_palette[: len(classes)])

        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(
                prob + 0.01,
                i,
                f"{prob:.1%}",
                va="center",
                fontweight="bold",
                fontsize=11,
            )

        ax.set_xlabel("Probability", fontsize=12, fontweight="bold")
        ax.set_ylabel("Bias Category", fontsize=12, fontweight="bold")
        ax.set_title("Prediction Probabilities", fontsize=14, fontweight="bold", pad=20)
        ax.set_xlim(0, max(probs) * 1.2)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig

    def plot_docking_results(
        self, docking_scores: Dict[str, float], figsize: tuple = (10, 6)
    ) -> Figure:
        """
        Create a bar chart of docking scores.

        Args:
            docking_scores: Dictionary of receptor names to binding affinities
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by binding affinity (more negative = better)
        sorted_scores = sorted(docking_scores.items(), key=lambda x: x[1])
        receptors = [x[0].replace("_", " ").title() for x in sorted_scores]
        scores = [x[1] for x in sorted_scores]

        # Color based on binding strength
        colors = [
            "#2ecc71" if s < -7 else "#f39c12" if s < -6 else "#e74c3c" for s in scores
        ]

        bars = ax.barh(receptors, scores, color=colors)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(
                score - 0.2,
                i,
                f"{score:.2f} kcal/mol",
                va="center",
                fontweight="bold",
                fontsize=9,
                color="white" if score < -6 else "black",
            )

        ax.set_xlabel("Binding Affinity (kcal/mol)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Receptor", fontsize=12, fontweight="bold")
        ax.set_title(
            "Docking Results - Binding Affinities",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.axvline(x=-6, color="gray", linestyle="--", alpha=0.5, label="Good binding")
        ax.legend()
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig

    def plot_descriptors_radar(
        self, descriptors: Dict[str, float], figsize: tuple = (8, 8)
    ) -> Figure:
        """
        Create a radar chart of key molecular descriptors.

        Args:
            descriptors: Dictionary of descriptor names to values
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Select key descriptors
        key_descriptors = [
            "MW",
            "LogP",
            "HBD",
            "HBA",
            "TPSA",
            "Rotatable_Bonds",
        ]

        # Normalize values for radar chart (0-1 scale)
        values = []
        labels = []
        max_values = {
            "MW": 1000,
            "LogP": 10,
            "HBD": 10,
            "HBA": 20,
            "TPSA": 200,
            "Rotatable_Bonds": 20,
        }

        for desc in key_descriptors:
            if desc in descriptors:
                val = descriptors[desc]
                max_val = max_values.get(desc, val * 2)
                normalized = min(val / max_val, 1.0) if max_val > 0 else 0
                values.append(normalized)
                labels.append(desc)

        if not values:
            # Create empty figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No descriptor data available", ha="center", va="center")
            return fig

        # Create radar chart
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]

        ax.plot(angles, values, "o-", linewidth=2, color="#3498db", label="Ligand")
        ax.fill(angles, values, alpha=0.25, color="#3498db")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            "Molecular Descriptors Profile", fontsize=14, fontweight="bold", pad=20
        )

        plt.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 15,
        figsize: tuple = (10, 6),
    ) -> Figure:
        """
        Create a bar chart of top feature importances.

        Args:
            feature_importance: Dictionary of feature names to importance scores
            top_n: Number of top features to show
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get top N features
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:top_n]
        features = [x[0] for x in sorted_features]
        importances = [x[1] for x in sorted_features]

        bars = ax.barh(features, importances, color=self.color_palette[: len(features)])

        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(
                imp + max(importances) * 0.01,
                i,
                f"{imp:.4f}",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Importance Score", fontsize=12, fontweight="bold")
        ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold", pad=20
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        return fig

    def create_descriptors_table(self, descriptors: Dict[str, float]) -> str:
        """
        Create an HTML table of molecular descriptors.

        Args:
            descriptors: Dictionary of descriptor names to values

        Returns:
            HTML string with formatted table
        """
        # Group descriptors by category
        categories = {
            "Basic Properties": ["MW", "LogP", "TPSA"],
            "Hydrogen Bonding": ["HBD", "HBA"],
            "Flexibility": ["Rotatable_Bonds"],
            "Drug-likeness": ["QED", "SPS", "Lipinski_Violations"],
        }

        html = '<div style="overflow-x:auto;">'
        html += '<table style="width:100%;border-collapse:collapse;margin:20px 0;background:var(--card-background, #fff);border-radius:8px;overflow:hidden;">'

        for category, desc_list in categories.items():
            html += f'<tr style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);color:white;"><th colspan="2" style="padding:12px;text-align:left;font-weight:bold;">{category}</th></tr>'
            for desc in desc_list:
                if desc in descriptors:
                    value = descriptors[desc]
                    # Format value based on type
                    if isinstance(value, float):
                        if abs(value) < 0.01:
                            formatted = f"{value:.2e}"
                        else:
                            formatted = f"{value:.2f}"
                    else:
                        formatted = str(value)

                    html += f"""
                    <tr style="border-bottom:1px solid var(--border-color, #ddd);background:var(--input-background, #fff);">
                        <td style="padding:8px;font-weight:500;color:var(--body-text-color, #333);">{desc}</td>
                        <td style="padding:8px;text-align:right;color:var(--body-text-color, #555);">{formatted}</td>
                    </tr>
                    """

        html += "</table></div>"
        return html
