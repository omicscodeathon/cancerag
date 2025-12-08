"""
Clustering Validation Visualization and Report Generation

Creates comprehensive visualizations and analysis reports for clustering results,
including comparison with true labels and detailed metrics.
"""

import json
import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClusteringVisualizer:
    """
    Creates visualizations and reports for clustering validation.
    """

    def __init__(self, config: Dict):
        """
        Initialize visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = config["paths"]
        self.results_dir = os.path.join(self.paths["reports"], "clustering_analysis")
        self.figures_dir = os.path.join(self.paths["figures"], "clustering_analysis")
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_results(self) -> Dict:
        """Load clustering results from JSON file."""
        results_path = os.path.join(self.results_dir, "clustering_results.json")
        with open(results_path, "r") as f:
            results = json.load(f)
        logger.info(f"Loaded clustering results from {results_path}")
        return results

    def load_assignments(self) -> pd.DataFrame:
        """Load cluster assignments."""
        assignments_path = os.path.join(self.results_dir, "cluster_assignments.csv")
        df = pd.read_csv(assignments_path)
        logger.info(f"Loaded cluster assignments from {assignments_path}")
        return df

    def plot_elbow_silhouette(self, results: Dict) -> None:
        """
        Plot elbow and silhouette curves side by side.

        Args:
            results: Clustering results dictionary
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Elbow plot
        elbow = results["elbow"]
        ax1.plot(
            elbow["k_values"],
            elbow["inertias"],
            marker="o",
            linewidth=2,
            markersize=8,
        )
        ax1.axvline(
            x=elbow["optimal_k"],
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal k = {elbow['optimal_k']}",
        )
        ax1.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax1.set_ylabel("Inertia", fontsize=12)
        ax1.set_title("Elbow Method for Optimal k", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Silhouette plot
        silhouette = results["silhouette"]
        ax2.plot(
            silhouette["k_values"],
            silhouette["scores"],
            marker="o",
            linewidth=2,
            markersize=8,
            color="green",
        )
        ax2.axvline(
            x=silhouette["optimal_k"],
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Optimal k = {silhouette['optimal_k']}",
        )
        ax2.set_xlabel("Number of Clusters (k)", fontsize=12)
        ax2.set_ylabel("Silhouette Score", fontsize=12)
        ax2.set_title(
            "Silhouette Analysis for Optimal k", fontsize=14, fontweight="bold"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "elbow_silhouette_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved elbow-silhouette plot to {save_path}")
        plt.close()

    def plot_cluster_composition(self, results: Dict) -> None:
        """
        Plot cluster composition showing distribution of true labels.

        Args:
            results: Clustering results dictionary
        """
        composition = pd.DataFrame.from_dict(
            results["cluster_composition"], orient="index"
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Stacked bar chart
        composition.plot(kind="bar", stacked=True, ax=ax1, colormap="tab10")
        ax1.set_xlabel("Cluster", fontsize=12)
        ax1.set_ylabel("Number of Samples", fontsize=12)
        ax1.set_title("Cluster Composition (Stacked)", fontsize=14, fontweight="bold")
        ax1.legend(title="Bias Category", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3, axis="y")

        # Heatmap
        sns.heatmap(
            composition,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            ax=ax2,
            cbar_kws={"label": "Count"},
        )
        ax2.set_xlabel("Bias Category", fontsize=12)
        ax2.set_ylabel("Cluster", fontsize=12)
        ax2.set_title("Cluster Composition (Heatmap)", fontsize=14, fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(self.figures_dir, "cluster_composition.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved cluster composition plot to {save_path}")
        plt.close()

    def plot_dimensionality_reduction(
        self, results: Dict, assignments: pd.DataFrame
    ) -> None:
        """
        Plot PCA, UMAP, and t-SNE projections.

        Args:
            results: Clustering results dictionary
            assignments: Cluster assignments DataFrame
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Color maps
        cluster_colors = plt.cm.tab10(assignments["cluster"])

        # Create label to color mapping for consistency
        unique_labels = assignments["true_label"].unique()
        n_labels = len(unique_labels)
        label_cmap = plt.cm.tab20 if n_labels > 10 else plt.cm.tab10
        label_to_color = {
            label: label_cmap(i / n_labels) for i, label in enumerate(unique_labels)
        }
        true_colors = [label_to_color[label] for label in assignments["true_label"]]

        # PCA - Clusters
        pca_data = np.array(results["pca_2d"])
        axes[0, 0].scatter(
            pca_data[:, 0], pca_data[:, 1], c=cluster_colors, alpha=0.6, s=50
        )
        axes[0, 0].set_title(
            "PCA Projection (Colored by Cluster)", fontsize=12, fontweight="bold"
        )
        axes[0, 0].set_xlabel("PC1")
        axes[0, 0].set_ylabel("PC2")

        # PCA - True Labels
        axes[1, 0].scatter(
            pca_data[:, 0], pca_data[:, 1], c=true_colors, alpha=0.6, s=50
        )
        axes[1, 0].set_title(
            "PCA Projection (Colored by True Labels)", fontsize=12, fontweight="bold"
        )
        axes[1, 0].set_xlabel("PC1")
        axes[1, 0].set_ylabel("PC2")

        # UMAP - Clusters
        if "umap_2d" in results:
            umap_data = np.array(results["umap_2d"])
            axes[0, 1].scatter(
                umap_data[:, 0], umap_data[:, 1], c=cluster_colors, alpha=0.6, s=50
            )
            axes[0, 1].set_title(
                "UMAP Projection (Colored by Cluster)",
                fontsize=12,
                fontweight="bold",
            )
            axes[0, 1].set_xlabel("UMAP1")
            axes[0, 1].set_ylabel("UMAP2")

            # UMAP - True Labels
            axes[1, 1].scatter(
                umap_data[:, 0], umap_data[:, 1], c=true_colors, alpha=0.6, s=50
            )
            axes[1, 1].set_title(
                "UMAP Projection (Colored by True Labels)",
                fontsize=12,
                fontweight="bold",
            )
            axes[1, 1].set_xlabel("UMAP1")
            axes[1, 1].set_ylabel("UMAP2")
        else:
            axes[0, 1].text(0.5, 0.5, "UMAP not available", ha="center", va="center")
            axes[1, 1].text(0.5, 0.5, "UMAP not available", ha="center", va="center")

        # t-SNE - Clusters
        if "tsne_2d" in results:
            tsne_data = np.array(results["tsne_2d"])
            tsne_indices = results.get("tsne_indices", list(range(len(tsne_data))))

            axes[0, 2].scatter(
                tsne_data[:, 0],
                tsne_data[:, 1],
                c=[cluster_colors[i] for i in tsne_indices],
                alpha=0.6,
                s=50,
            )
            axes[0, 2].set_title(
                "t-SNE Projection (Colored by Cluster)",
                fontsize=12,
                fontweight="bold",
            )
            axes[0, 2].set_xlabel("t-SNE1")
            axes[0, 2].set_ylabel("t-SNE2")

            # t-SNE - True Labels
            axes[1, 2].scatter(
                tsne_data[:, 0],
                tsne_data[:, 1],
                c=[true_colors[i] for i in tsne_indices],
                alpha=0.6,
                s=50,
            )
            axes[1, 2].set_title(
                "t-SNE Projection (Colored by True Labels)",
                fontsize=12,
                fontweight="bold",
            )
            axes[1, 2].set_xlabel("t-SNE1")
            axes[1, 2].set_ylabel("t-SNE2")
        else:
            axes[0, 2].text(0.5, 0.5, "t-SNE not available", ha="center", va="center")
            axes[1, 2].text(0.5, 0.5, "t-SNE not available", ha="center", va="center")

        # Add legend for true labels
        legend_elements = [
            Patch(facecolor=label_to_color[label], label=label)
            for label in unique_labels
        ]
        fig.legend(
            handles=legend_elements,
            loc="center right",
            title="True Bias Categories",
            bbox_to_anchor=(1.12, 0.5),
        )

        plt.tight_layout()
        save_path = os.path.join(
            self.figures_dir, "dimensionality_reduction_comparison.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved dimensionality reduction plots to {save_path}")
        plt.close()

    def generate_validation_report(
        self, results: Dict, assignments: pd.DataFrame
    ) -> None:
        """
        Generate comprehensive validation report.

        Args:
            results: Clustering results dictionary
            assignments: Cluster assignments DataFrame
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLUSTERING VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Summary
        report_lines.append("## SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Samples: {len(assignments)}")
        report_lines.append(
            f"Optimal Number of Clusters: {results['silhouette']['optimal_k']}"
        )
        report_lines.append(
            f"Number of True Classes: {len(assignments['true_label'].unique())}"
        )
        report_lines.append("")

        # Optimal k determination
        report_lines.append("## OPTIMAL K DETERMINATION")
        report_lines.append("-" * 80)
        report_lines.append(
            f"Elbow Method suggests k = {results['elbow']['optimal_k']}"
        )
        report_lines.append(
            f"Silhouette Analysis suggests k = {results['silhouette']['optimal_k']}"
        )
        report_lines.append(
            f"Selected k = {results['silhouette']['optimal_k']} (based on silhouette)"
        )
        report_lines.append("")

        # Validation metrics
        report_lines.append("## CLUSTERING VALIDATION METRICS")
        report_lines.append("-" * 80)
        validation = results["validation"]
        report_lines.append(f"Silhouette Score: {validation['silhouette_score']:.4f}")
        report_lines.append("  - Range: [-1, 1], Higher is better")
        report_lines.append(
            "  - Interpretation: "
            + (
                "Poor separation"
                if validation["silhouette_score"] < 0.3
                else (
                    "Moderate separation"
                    if validation["silhouette_score"] < 0.5
                    else "Good separation"
                )
            )
        )
        report_lines.append("")
        report_lines.append(
            f"Calinski-Harabasz Score: {validation['calinski_harabasz_score']:.4f}"
        )
        report_lines.append("  - Higher is better (well-defined clusters)")
        report_lines.append("")
        report_lines.append(
            f"Davies-Bouldin Index: {validation['davies_bouldin_score']:.4f}"
        )
        report_lines.append("  - Lower is better (less cluster overlap)")
        report_lines.append("")

        # Comparison with true labels
        report_lines.append("## COMPARISON WITH TRUE BIAS LABELS")
        report_lines.append("-" * 80)
        if "label_comparison" in results:
            ari = results["label_comparison"]["adjusted_rand_index"]
            report_lines.append(f"Adjusted Rand Index (ARI): {ari:.4f}")
            report_lines.append("  - Range: [-1, 1], Higher is better")
            report_lines.append("  - 0.0 means random clustering")
            report_lines.append("  - 1.0 means perfect match")
            report_lines.append("")

            # Percentage similarity
            similarity_pct = max(0, ari * 100)  # Convert to percentage, cap at 0
            report_lines.append(f"Similarity to True Labels: {similarity_pct:.2f}%")
            report_lines.append("")

            # Interpretation
            report_lines.append("### INTERPRETATION:")
            if ari < 0.1:
                interpretation = (
                    "The discovered clusters show very weak alignment with the known bias "
                    "categories. This suggests that:\n"
                    "  1. The molecular features alone may not be sufficient to distinguish "
                    "bias types\n"
                    "  2. Additional features (e.g., receptor-specific features, interaction "
                    "patterns) may be needed\n"
                    "  3. The bias categories may not form natural clusters in the current "
                    "feature space\n"
                    "  4. Supervised learning approaches are more appropriate for this "
                    "classification task"
                )
            elif ari < 0.3:
                interpretation = (
                    "The clusters show weak but detectable alignment with bias categories. "
                    "Some feature groups may be capturing bias-related patterns, but "
                    "supervised learning will likely be more effective."
                )
            elif ari < 0.5:
                interpretation = (
                    "The clusters show moderate alignment with bias categories, suggesting "
                    "that molecular features contain useful information for distinguishing "
                    "bias types."
                )
            else:
                interpretation = (
                    "The clusters show strong alignment with bias categories, indicating "
                    "that the molecular features effectively capture bias-related patterns."
                )

            report_lines.append(interpretation)
        report_lines.append("")

        # Cluster composition
        report_lines.append("## CLUSTER COMPOSITION")
        report_lines.append("-" * 80)
        composition = pd.DataFrame.from_dict(
            results["cluster_composition"], orient="index"
        )
        report_lines.append(composition.to_string())
        report_lines.append("")

        # Class distribution
        report_lines.append("## TRUE LABEL DISTRIBUTION")
        report_lines.append("-" * 80)
        label_dist = assignments["true_label"].value_counts().sort_index()
        for label, count in label_dist.items():
            pct = (count / len(assignments)) * 100
            report_lines.append(f"{label}: {count} ({pct:.1f}%)")
        report_lines.append("")

        # Recommendations
        report_lines.append("## RECOMMENDATIONS")
        report_lines.append("-" * 80)
        if (
            "label_comparison" in results
            and results["label_comparison"]["adjusted_rand_index"] < 0.2
        ):
            report_lines.append(
                "1. The low ARI suggests unsupervised clustering doesn't naturally "
                "separate bias types"
            )
            report_lines.append(
                "2. Continue with supervised learning approaches (as done with Random Forest)"
            )
            report_lines.append(
                "3. Consider feature engineering to create bias-specific descriptors"
            )
            report_lines.append(
                "4. Explore receptor-ligand interaction patterns as additional features"
            )
            report_lines.append(
                "5. Investigate if certain feature subsets show better clustering for "
                "specific bias types"
            )
        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_path = os.path.join(self.results_dir, "validation_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))
        logger.info(f"Saved validation report to {report_path}")

        # Also print to console
        print("\n".join(report_lines))

    def run_visualization_pipeline(self) -> None:
        """Run complete visualization pipeline."""
        logger.info("Starting clustering visualization pipeline...")

        # Load data
        results = self.load_results()
        assignments = self.load_assignments()

        # Generate all plots
        logger.info("Generating elbow/silhouette plots...")
        self.plot_elbow_silhouette(results)

        logger.info("Generating cluster composition plots...")
        self.plot_cluster_composition(results)

        logger.info("Generating dimensionality reduction plots...")
        self.plot_dimensionality_reduction(results, assignments)

        # Generate report
        logger.info("Generating validation report...")
        self.generate_validation_report(results, assignments)

        logger.info("Clustering visualization pipeline complete!")


def main():
    """Main entry point."""
    import yaml

    # Load config
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run visualization
    visualizer = ClusteringVisualizer(config)
    visualizer.run_visualization_pipeline()


if __name__ == "__main__":
    main()
