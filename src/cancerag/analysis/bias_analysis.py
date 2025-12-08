"""
Bias Analysis Module

Analyzes bias patterns, distributions, and relationships in the dataset.
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class BiasAnalyzer:
    """
    Analyzes bias patterns and distributions in the dataset.
    """

    def __init__(self, output_dir: str = "results/analysis"):
        """
        Initialize the bias analyzer.

        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def analyze_bias_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of bias labels.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing bias distribution...")

        results = {}

        # Basic distribution
        bias_counts = df["primary_bias_label"].value_counts()
        bias_percentages = df["primary_bias_label"].value_counts(normalize=True) * 100

        results["bias_counts"] = bias_counts.to_dict()
        results["bias_percentages"] = bias_percentages.to_dict()

        # Class imbalance assessment
        total_samples = len(df)
        max_class_size = bias_counts.max()
        min_class_size = bias_counts.min()

        results["class_imbalance_ratio"] = max_class_size / min_class_size
        results["total_samples"] = total_samples
        results["num_classes"] = len(bias_counts)

        # Create visualization
        self._plot_bias_distribution(bias_counts, bias_percentages)

        logger.info(f"Bias distribution: {results['bias_counts']}")
        logger.info(f"Class imbalance ratio: {results['class_imbalance_ratio']:.2f}")

        return results

    def analyze_receptor_bias_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze bias patterns across different receptors.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing receptor bias patterns...")

        results = {}

        # Bias distribution by receptor family
        family_bias = (
            df.groupby(["receptor_family", "primary_bias_label"])
            .size()
            .unstack(fill_value=0)
        )
        results["family_bias_matrix"] = family_bias.to_dict()

        # Bias distribution by individual receptor
        receptor_bias = (
            df.groupby(["receptor", "primary_bias_label"]).size().unstack(fill_value=0)
        )
        results["receptor_bias_matrix"] = receptor_bias.to_dict()

        # Most biased receptors
        receptor_bias_pct = receptor_bias.div(receptor_bias.sum(axis=1), axis=0) * 100
        results["receptor_bias_percentages"] = receptor_bias_pct.to_dict()

        # Create visualizations
        self._plot_receptor_bias_heatmap(family_bias, "receptor_family")
        self._plot_receptor_bias_heatmap(receptor_bias.head(15), "receptor")

        return results

    def analyze_bias_pathways(self, df: pd.DataFrame) -> Dict:
        """
        Analyze bias pathway patterns.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing bias pathway patterns...")

        results = {}

        # Pathway distribution
        pathway_counts = df["bias_pathway"].value_counts()
        results["pathway_counts"] = pathway_counts.to_dict()

        # Bias category vs pathway analysis
        bias_pathway_crosstab = pd.crosstab(df["bias_category"], df["bias_pathway"])
        results["bias_pathway_crosstab"] = bias_pathway_crosstab.to_dict()

        # Create visualization
        self._plot_bias_pathway_analysis(bias_pathway_crosstab)

        return results

    def analyze_bias_consistency(self, df: pd.DataFrame) -> Dict:
        """
        Analyze consistency of bias labels across the dataset.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing bias consistency...")

        results = {}

        # Check for ligands with multiple bias labels
        ligand_bias_counts = df.groupby("ligand_name")["primary_bias_label"].nunique()
        multi_bias_ligands = ligand_bias_counts[ligand_bias_counts > 1]

        results["multi_bias_ligands"] = len(multi_bias_ligands)
        results["consistency_percentage"] = (
            1 - len(multi_bias_ligands) / len(df)
        ) * 100

        # Receptor count analysis
        receptor_count_stats = df["receptor_count"].describe()
        results["receptor_count_stats"] = receptor_count_stats.to_dict()

        logger.info(f"Bias consistency: {results['consistency_percentage']:.1f}%")

        return results

    def _plot_bias_distribution(
        self, bias_counts: pd.Series, bias_percentages: pd.Series
    ) -> None:
        """Create bias distribution plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot
        bars = ax1.bar(
            bias_counts.index,
            bias_counts.values,
            color=sns.color_palette("husl", len(bias_counts)),
        )
        ax1.set_title("Bias Label Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Bias Label")
        ax1.set_ylabel("Number of Ligands")
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, count, pct in zip(bars, bias_counts.values, bias_percentages.values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{count}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
            )

        # Pie chart
        ax2.pie(
            bias_counts.values,
            labels=bias_counts.index,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title(
            "Bias Label Distribution (Percentage)", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "bias_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_receptor_bias_heatmap(
        self, bias_matrix: pd.DataFrame, level: str
    ) -> None:
        """Create receptor bias heatmap."""
        plt.figure(figsize=(12, 8))

        # Normalize by row to show percentages
        bias_matrix_norm = bias_matrix.div(bias_matrix.sum(axis=1), axis=0) * 100

        sns.heatmap(
            bias_matrix_norm,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Percentage (%)"},
        )

        plt.title(
            f"Bias Distribution by {level.title()}", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Bias Label")
        plt.ylabel(f"{level.title()}")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"bias_heatmap_{level}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_bias_pathway_analysis(self, crosstab: pd.DataFrame) -> None:
        """Create bias pathway analysis plot."""
        plt.figure(figsize=(12, 8))

        sns.heatmap(
            crosstab, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"}
        )

        plt.title("Bias Category vs Pathway Crosstab", fontsize=14, fontweight="bold")
        plt.xlabel("Bias Pathway")
        plt.ylabel("Bias Category")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "bias_pathway_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete bias analysis.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Complete analysis results
        """
        logger.info("Starting comprehensive bias analysis...")

        results = {}

        # Run all analysis methods
        results["distribution"] = self.analyze_bias_distribution(df)
        results["receptor_patterns"] = self.analyze_receptor_bias_patterns(df)
        results["pathways"] = self.analyze_bias_pathways(df)
        results["consistency"] = self.analyze_bias_consistency(df)

        logger.info("Bias analysis completed successfully!")

        return results
