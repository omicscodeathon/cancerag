"""
Statistical Analysis Module

Performs comprehensive statistical analysis on the dataset.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Performs comprehensive statistical analysis on the dataset.
    """

    def __init__(self, output_dir: str = "results/analysis"):
        """
        Initialize the statistical analyzer.

        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

    def analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Analyze correlations between features.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing feature correlations...")

        results = {}

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove columns with too many missing values
        missing_threshold = 0.5
        valid_cols = []
        for col in numeric_cols:
            if df[col].isnull().sum() / len(df) < missing_threshold:
                valid_cols.append(col)

        # Limit to manageable number of features for correlation analysis
        if len(valid_cols) > 100:
            # Select most important features (basic descriptors + some RDKit)
            basic_descriptors = [
                "molecular_weight",
                "logp",
                "hba",
                "hbd",
                "rings",
                "tpsa",
            ]
            rdkit_cols = [
                col
                for col in valid_cols
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "Max",
                        "Min",
                        "Mol",
                        "Num",
                        "BCUT",
                        "Chi",
                        "Kappa",
                        "PEOE",
                        "SMR",
                        "EState",
                    ]
                )
            ]

            selected_cols = (
                basic_descriptors + rdkit_cols[:50]
            )  # Limit to 50 RDKit features
            valid_cols = [col for col in selected_cols if col in valid_cols]

        # Calculate correlation matrix
        corr_matrix = df[valid_cols].corr()

        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )

        results["correlation_matrix"] = corr_matrix.to_dict()
        results["high_correlation_pairs"] = high_corr_pairs
        results["num_features_analyzed"] = len(valid_cols)

        # Create visualization
        self._plot_correlation_heatmap(corr_matrix)

        return results

    def analyze_distributions(self, df: pd.DataFrame) -> Dict:
        """
        Analyze distributions of features.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing feature distributions...")

        results = {}

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Analyze distributions for key features
        key_features = ["molecular_weight", "logp", "hba", "hbd", "rings", "tpsa"]
        distribution_stats = {}

        for feature in key_features:
            if feature in numeric_cols:
                data = df[feature].dropna()

                # Normality tests
                shapiro_stat, shapiro_p = stats.shapiro(
                    data.sample(min(5000, len(data)))
                )
                kstest_stat, kstest_p = stats.kstest(
                    data, "norm", args=(data.mean(), data.std())
                )

                distribution_stats[feature] = {
                    "mean": data.mean(),
                    "std": data.std(),
                    "skewness": stats.skew(data),
                    "kurtosis": stats.kurtosis(data),
                    "shapiro_stat": shapiro_stat,
                    "shapiro_p": shapiro_p,
                    "is_normal_shapiro": shapiro_p > 0.05,
                    "kstest_stat": kstest_stat,
                    "kstest_p": kstest_p,
                    "is_normal_kstest": kstest_p > 0.05,
                }

        results["distribution_stats"] = distribution_stats

        # Create visualizations
        self._plot_distribution_analysis(df, key_features)

        return results

    def analyze_bias_differences(self, df: pd.DataFrame) -> Dict:
        """
        Analyze statistical differences between bias groups.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing bias group differences...")

        results = {}

        # Get numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        key_features = ["molecular_weight", "logp", "hba", "hbd", "rings", "tpsa"]

        # Statistical tests for each feature
        bias_groups = df["primary_bias_label"].unique()
        statistical_tests = {}

        for feature in key_features:
            if feature in numeric_cols:
                # Prepare data for each bias group
                group_data = []
                group_names = []

                for bias in bias_groups:
                    group_data.append(
                        df[df["primary_bias_label"] == bias][feature].dropna()
                    )
                    group_names.append(bias)

                # Kruskal-Wallis test (non-parametric ANOVA)
                try:
                    kruskal_stat, kruskal_p = kruskal(*group_data)
                    statistical_tests[feature] = {
                        "kruskal_stat": kruskal_stat,
                        "kruskal_p": kruskal_p,
                        "significant_difference": kruskal_p < 0.05,
                        "group_means": {
                            name: data.mean()
                            for name, data in zip(group_names, group_data)
                        },
                        "group_stds": {
                            name: data.std()
                            for name, data in zip(group_names, group_data)
                        },
                    }
                except Exception as e:
                    logger.warning(
                        f"Could not perform Kruskal-Wallis test for {feature}: {e}"
                    )
                    statistical_tests[feature] = {"error": str(e)}

        results["statistical_tests"] = statistical_tests

        # Create visualizations
        self._plot_bias_group_differences(df, key_features)

        return results

    def analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing missing data patterns...")

        results = {}

        # Overall missing data statistics
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100

        results["overall_missing"] = {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": missing_percentage,
        }

        # Missing data by column
        missing_by_column = df.isnull().sum()
        missing_by_column_pct = (missing_by_column / len(df)) * 100

        results["missing_by_column"] = missing_by_column.to_dict()
        results["missing_by_column_pct"] = missing_by_column_pct.to_dict()

        # Columns with high missing data
        high_missing_cols = missing_by_column_pct[missing_by_column_pct > 50].to_dict()
        results["high_missing_columns"] = high_missing_cols

        # Missing data patterns
        missing_patterns = {}
        for col in df.columns:
            if missing_by_column_pct[col] > 0:
                missing_patterns[col] = {
                    "missing_count": missing_by_column[col],
                    "missing_percentage": missing_by_column_pct[col],
                    "data_type": str(df[col].dtype),
                }

        results["missing_patterns"] = missing_patterns

        # Create visualizations
        self._plot_missing_data_analysis(missing_by_column_pct)

        return results

    def analyze_feature_importance(self, df: pd.DataFrame) -> Dict:
        """
        Analyze feature importance using mutual information.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing feature importance...")

        results = {}

        # Get numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Remove columns with too many missing values
        missing_threshold = 0.3
        valid_cols = []
        for col in numeric_cols:
            if df[col].isnull().sum() / len(df) < missing_threshold:
                valid_cols.append(col)

        # Limit features for computational efficiency
        if len(valid_cols) > 100:
            # Select key features
            key_features = ["molecular_weight", "logp", "hba", "hbd", "rings", "tpsa"]
            rdkit_cols = [
                col
                for col in valid_cols
                if any(
                    col.startswith(prefix)
                    for prefix in [
                        "Max",
                        "Min",
                        "Mol",
                        "Num",
                        "BCUT",
                        "Chi",
                        "Kappa",
                        "PEOE",
                        "SMR",
                        "EState",
                    ]
                )
            ]

            selected_cols = key_features + rdkit_cols[:50]
            valid_cols = [col for col in selected_cols if col in valid_cols]

        # Prepare data
        X = df[valid_cols].fillna(df[valid_cols].median())
        y = df["primary_bias_label"]

        # Calculate mutual information
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)

            # Create feature importance dataframe
            feature_importance = pd.DataFrame(
                {"feature": valid_cols, "mutual_info": mi_scores}
            ).sort_values("mutual_info", ascending=False)

            results["feature_importance"] = feature_importance.to_dict("records")
            results["top_features"] = feature_importance.head(20).to_dict("records")

            # Create visualization
            self._plot_feature_importance(feature_importance.head(20))

        except Exception as e:
            logger.warning(f"Could not calculate mutual information: {e}")
            results["error"] = str(e)

        return results

    def _plot_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> None:
        """Create correlation heatmap."""
        plt.figure(figsize=(12, 10))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"label": "Correlation Coefficient"},
        )

        plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(
            self.output_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_distribution_analysis(
        self, df: pd.DataFrame, features: List[str]
    ) -> None:
        """Create distribution analysis plots."""
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = [
            ax for row in axes for ax in (row if isinstance(row, np.ndarray) else [row])
        ]

        for i, feature in enumerate(features):
            if i < len(axes) and feature in df.columns:
                # Histogram
                data = df[feature].dropna()
                axes[i].hist(
                    data, bins=30, alpha=0.7, color="skyblue", edgecolor="black"
                )
                axes[i].set_title(
                    f"{feature.replace('_', ' ').title()} Distribution",
                    fontweight="bold",
                )
                axes[i].set_xlabel(feature.replace("_", " ").title())
                axes[i].set_ylabel("Frequency")
                axes[i].grid(True, alpha=0.3)

                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                axes[i].axvline(
                    mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}"
                )
                axes[i].axvline(
                    mean_val + std_val,
                    color="orange",
                    linestyle=":",
                    alpha=0.7,
                    label=f"+1σ: {mean_val + std_val:.2f}",
                )
                axes[i].axvline(
                    mean_val - std_val,
                    color="orange",
                    linestyle=":",
                    alpha=0.7,
                    label=f"-1σ: {mean_val - std_val:.2f}",
                )
                axes[i].legend()

        # Remove empty subplots
        for i in range(len(features), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "distribution_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_bias_group_differences(
        self, df: pd.DataFrame, features: List[str]
    ) -> None:
        """Create bias group difference plots."""
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = [
            ax for row in axes for ax in (row if isinstance(row, np.ndarray) else [row])
        ]

        for i, feature in enumerate(features):
            if i < len(axes) and feature in df.columns:
                # Box plot by bias group
                sns.boxplot(data=df, x="primary_bias_label", y=feature, ax=axes[i])
                axes[i].set_title(
                    f"{feature.replace('_', ' ').title()} by Bias Group",
                    fontweight="bold",
                )
                axes[i].set_xlabel("Bias Label")
                axes[i].set_ylabel(feature.replace("_", " ").title())
                axes[i].tick_params(axis="x", rotation=45)

        # Remove empty subplots
        for i in range(len(features), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "bias_group_differences.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_missing_data_analysis(self, missing_by_column_pct: pd.Series) -> None:
        """Create missing data analysis plots."""
        # Plot 1: Missing data percentage
        plt.figure(figsize=(15, 8))

        # Sort by missing percentage
        missing_sorted = missing_by_column_pct.sort_values(ascending=False)

        # Plot top 20 columns with missing data
        top_missing = missing_sorted[missing_sorted > 0].head(20)

        if len(top_missing) > 0:
            bars = plt.bar(
                range(len(top_missing)), top_missing.values, color="lightcoral"
            )
            plt.title(
                "Top 20 Columns with Missing Data", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Column")
            plt.ylabel("Missing Data Percentage (%)")
            plt.xticks(
                range(len(top_missing)), top_missing.index, rotation=45, ha="right"
            )

            # Add value labels on bars
            for bar, pct in zip(bars, top_missing.values):
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "missing_data_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_feature_importance(self, feature_importance: pd.DataFrame) -> None:
        """Create feature importance plot."""
        plt.figure(figsize=(12, 8))

        bars = plt.barh(
            range(len(feature_importance)),
            feature_importance["mutual_info"],
            color="lightgreen",
            alpha=0.7,
        )

        plt.yticks(
            range(len(feature_importance)),
            [feat.replace("_", " ").title() for feat in feature_importance["feature"]],
        )
        plt.xlabel("Mutual Information Score")
        plt.title(
            "Top 20 Features by Mutual Information", fontsize=14, fontweight="bold"
        )

        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, feature_importance["mutual_info"])):
            plt.text(
                score + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                ha="left",
                va="center",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "feature_importance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete statistical analysis.

        Args:
            df (pd.DataFrame): Unified dataset

        Returns:
            Dict: Complete analysis results
        """
        logger.info("Starting comprehensive statistical analysis...")

        results = {}

        # Run all analysis methods
        results["correlations"] = self.analyze_correlations(df)
        results["distributions"] = self.analyze_distributions(df)
        results["bias_differences"] = self.analyze_bias_differences(df)
        results["missing_data"] = self.analyze_missing_data(df)
        results["feature_importance"] = self.analyze_feature_importance(df)

        logger.info("Statistical analysis completed successfully!")

        return results
