"""
Unsupervised Clustering Analysis for Biased Agonist Identification

This module implements clustering analysis to discover natural groupings
in the feature space without using bias labels. This helps validate that
our molecular features can distinguish between different bias types.

Usage:
    from cancerag.ml.clustering import run_clustering_analysis
    results = run_clustering_analysis(config)
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Optional imports with fallbacks
try:
    from umap import UMAP

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE

    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """
    Performs unsupervised clustering analysis on molecular features.

    Discovers natural groupings in the data and validates that features
    can distinguish between different bias types without supervision.
    """

    def __init__(self, config: dict):
        """
        Initialize the ClusteringAnalyzer.

        Args:
            config (dict): The project's configuration dictionary.
        """
        self.paths = config["paths"]
        self.dataset_path = os.path.join(
            self.paths["processed_data"], "unified_ml_dataset.csv"
        )
        self.output_dir = os.path.join(self.paths["reports"], "clustering_analysis")
        os.makedirs(self.output_dir, exist_ok=True)

        self.random_state = config.get("ml_model", {}).get("random_state", 42)

    def _load_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the unified ML dataset.

        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset not found: {self.dataset_path}")
            logger.error("Run dataset assembly first.")
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        logger.info(f"Loading dataset from {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)

        # Separate features and labels
        label_col = "bias_category"  # Adjust based on your column name

        if label_col not in df.columns:
            # Try alternative label columns
            for col in ["label", "target", "class", "bias_type"]:
                if col in df.columns:
                    label_col = col
                    break

        if label_col not in df.columns:
            logger.warning(
                "No label column found. Running unsupervised clustering only."
            )
            labels = pd.Series([0] * len(df))
            features = df.select_dtypes(include=[np.number])
        else:
            labels = df[label_col]
            features = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

        # Remove non-numeric columns
        features = features.select_dtypes(include=[np.number])

        logger.info(
            f"Loaded {len(features)} samples with {len(features.columns)} features"
        )

        return features, labels

    def _preprocess_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for clustering.

        Args:
            features: Raw feature DataFrame

        Returns:
            Preprocessed features
        """
        # Remove columns that are all NaN
        nan_cols = features.columns[features.isna().all()].tolist()
        if nan_cols:
            logger.info(f"Removing {len(nan_cols)} columns that are entirely NaN")
            features = features.drop(columns=nan_cols)

        # Remove columns with constant values (including all zeros)
        constant_cols = []
        for col in features.columns:
            if features[col].nunique() <= 1:
                constant_cols.append(col)
        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant-value columns")
            features = features.drop(columns=constant_cols)

        # Handle remaining missing values
        if features.isnull().any().any():
            logger.info("Filling remaining missing values with column medians")
            features = features.fillna(features.median())
            # If still have NaN (median was NaN), fill with 0
            if features.isnull().any().any():
                logger.info("Filling any remaining NaN with 0")
                features = features.fillna(0)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index,
        )

        return features_scaled

    def _elbow_method(self, features: pd.DataFrame, k_range: range = None) -> Dict:
        """
        Determine optimal number of clusters using elbow method.

        Args:
            features: Preprocessed features
            k_range: Range of k values to test

        Returns:
            Dictionary with k values and inertias
        """
        if k_range is None:
            k_range = range(2, 11)

        logger.info("Computing elbow method...")
        inertias = []
        k_values = []

        for k in tqdm(k_range, desc="Elbow method"):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
            k_values.append(k)

        # Calculate rate of change
        diffs = np.diff(inertias)
        second_diffs = np.diff(diffs)

        # Optimal k is where second derivative is maximum (sharpest elbow)
        if len(second_diffs) > 0:
            optimal_k_elbow = k_values[np.argmax(second_diffs) + 1]
        else:
            optimal_k_elbow = k_values[0]

        return {
            "k_values": k_values,
            "inertias": inertias,
            "optimal_k": optimal_k_elbow,
        }

    def _silhouette_analysis(
        self, features: pd.DataFrame, k_range: range = None
    ) -> Dict:
        """
        Perform silhouette analysis to find optimal k.

        Args:
            features: Preprocessed features
            k_range: Range of k values to test

        Returns:
            Dictionary with k values and silhouette scores
        """
        if k_range is None:
            k_range = range(2, 11)

        logger.info("Computing silhouette scores...")
        scores = []
        k_values = []

        for k in tqdm(k_range, desc="Silhouette analysis"):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores.append(score)
            k_values.append(k)

        # Optimal k is where silhouette score is maximum
        optimal_k_silhouette = k_values[np.argmax(scores)]

        return {
            "k_values": k_values,
            "scores": scores,
            "optimal_k": optimal_k_silhouette,
        }

    def _cluster_validation(
        self, features: pd.DataFrame, labels: np.ndarray, k: int
    ) -> Dict[str, float]:
        """
        Compute validation metrics for clustering.

        Args:
            features: Preprocessed features
            labels: Cluster labels
            k: Number of clusters

        Returns:
            Dictionary of validation metrics
        """
        metrics = {}

        # Silhouette score (higher is better, range: -1 to 1)
        metrics["silhouette_score"] = silhouette_score(features, labels)

        # Calinski-Harabasz score (higher is better)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(features, labels)

        # Davies-Bouldin index (lower is better)
        metrics["davies_bouldin_score"] = davies_bouldin_score(features, labels)

        return metrics

    def _compute_pca_projection(
        self, features: pd.DataFrame, n_components: int = 2
    ) -> np.ndarray:
        """
        Compute PCA projection for visualization.

        Args:
            features: Preprocessed features
            n_components: Number of principal components

        Returns:
            Projected features
        """
        logger.info(f"Computing PCA ({n_components} components)...")
        pca = PCA(n_components=n_components, random_state=self.random_state)
        projected = pca.fit_transform(features)

        logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

        return projected

    def _compute_umap_projection(
        self, features: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15
    ) -> Optional[np.ndarray]:
        """
        Compute UMAP projection for visualization.

        Args:
            features: Preprocessed features
            n_components: Number of dimensions
            n_neighbors: Number of neighbors

        Returns:
            Projected features or None if UMAP not available
        """
        if not HAS_UMAP:
            logger.warning("UMAP not available, skipping UMAP projection")
            return None

        logger.info(f"Computing UMAP ({n_components} components)...")
        umap = UMAP(
            n_components=n_components,
            random_state=self.random_state,
            n_neighbors=n_neighbors,
        )
        projected = umap.fit_transform(features)

        return projected

    def _compute_tsne_projection(
        self, features: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0
    ) -> Optional[np.ndarray]:
        """
        Compute t-SNE projection for visualization.

        Args:
            features: Preprocessed features
            n_components: Number of dimensions
            perplexity: Perplexity parameter

        Returns:
            Projected features or None if t-SNE not available
        """
        if not HAS_TSNE:
            logger.warning("t-SNE not available, skipping t-SNE projection")
            return None

        logger.info(f"Computing t-SNE ({n_components} components)...")

        # Sample for t-SNE if too many points (it's slow)
        if len(features) > 1000:
            logger.info(f"Sampling {1000} points for t-SNE (faster)")
            sample_indices = np.random.choice(len(features), 1000, replace=False)
            features_sample = features.iloc[sample_indices]
        else:
            features_sample = features
            sample_indices = np.arange(len(features))

        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=self.random_state,
            max_iter=1000,
        )
        projected = tsne.fit_transform(features_sample.values)

        return projected, sample_indices

    def _compare_with_labels(
        self, cluster_labels: np.ndarray, true_labels: pd.Series
    ) -> Dict[str, float]:
        """
        Compare clustering results with true bias labels.

        Args:
            cluster_labels: Predicted cluster assignments
            true_labels: True bias category labels

        Returns:
            Dictionary with comparison metrics
        """
        # Encode true labels if they're strings
        if true_labels.dtype == object:
            unique_labels = true_labels.unique()
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            encoded_labels = true_labels.map(label_map).values
        else:
            encoded_labels = true_labels.values

        # Adjusted Rand Index (higher is better, range: -1 to 1)
        ari = adjusted_rand_score(encoded_labels, cluster_labels)

        return {"adjusted_rand_index": ari}

    def run_clustering_analysis(self) -> Dict:
        """
        Execute complete clustering analysis pipeline.

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 80)
        logger.info("STARTING CLUSTERING ANALYSIS")
        logger.info("=" * 80)

        # Load dataset
        features, labels = self._load_dataset()

        # Preprocess features
        features_scaled = self._preprocess_features(features)

        results = {}

        # 1. Elbow method
        logger.info("\n" + "-" * 80)
        logger.info("ELBOW METHOD")
        logger.info("-" * 80)
        elbow_results = self._elbow_method(features_scaled)
        results["elbow"] = elbow_results
        logger.info(f"Optimal k (elbow method): {elbow_results['optimal_k']}")

        # 2. Silhouette analysis
        logger.info("\n" + "-" * 80)
        logger.info("SILHOUETTE ANALYSIS")
        logger.info("-" * 80)
        silhouette_results = self._silhouette_analysis(features_scaled)
        results["silhouette"] = silhouette_results
        logger.info(f"Optimal k (silhouette): {silhouette_results['optimal_k']}")

        # 3. Select optimal k (use silhouette as it's more reliable)
        optimal_k = silhouette_results["optimal_k"]
        logger.info(f"\nSelected optimal k = {optimal_k}")

        # 4. Perform final clustering
        logger.info("\n" + "-" * 80)
        logger.info(f"FINAL CLUSTERING (k={optimal_k})")
        logger.info("-" * 80)
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # 5. Validation metrics
        validation_metrics = self._cluster_validation(
            features_scaled, cluster_labels, optimal_k
        )
        results["validation"] = validation_metrics
        logger.info("\nValidation Metrics:")
        for metric, value in validation_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # 6. Compare with true labels if available
        if len(labels.unique()) > 1:
            comparison = self._compare_with_labels(cluster_labels, labels)
            results["label_comparison"] = comparison
            logger.info("\nComparison with True Labels:")
            for metric, value in comparison.items():
                logger.info(f"  {metric}: {value:.4f}")

        # 7. Dimensionality reduction for visualization
        logger.info("\n" + "-" * 80)
        logger.info("DIMENSIONALITY REDUCTION")
        logger.info("-" * 80)

        # PCA
        pca_projected = self._compute_pca_projection(features_scaled, n_components=2)
        results["pca_2d"] = pca_projected.tolist()

        # UMAP
        umap_projected = self._compute_umap_projection(features_scaled, n_components=2)
        if umap_projected is not None:
            results["umap_2d"] = umap_projected.tolist()

        # t-SNE
        tsne_result = self._compute_tsne_projection(features_scaled, n_components=2)
        if tsne_result is not None:
            if isinstance(tsne_result, tuple):
                tsne_projected, tsne_indices = tsne_result
            else:
                tsne_projected = tsne_result
                tsne_indices = np.arange(len(tsne_projected))
            results["tsne_2d"] = tsne_projected.tolist()
            results["tsne_indices"] = tsne_indices.tolist()

        # 8. Cluster composition analysis
        logger.info("\n" + "-" * 80)
        logger.info("CLUSTER COMPOSITION ANALYSIS")
        logger.info("-" * 80)

        cluster_comp = pd.DataFrame(
            {
                "cluster": cluster_labels,
                "bias_category": labels.values if labels.dtype == object else labels,
            }
        )
        composition = (
            cluster_comp.groupby("cluster")["bias_category"]
            .value_counts()
            .unstack(fill_value=0)
        )
        results["cluster_composition"] = composition.to_dict("index")

        logger.info("\nCluster Composition:")
        logger.info(composition)

        # 9. Save results
        self._save_results(results, cluster_labels, labels)

        logger.info("\n" + "=" * 80)
        logger.info("CLUSTERING ANALYSIS COMPLETE")
        logger.info("=" * 80)

        return results

    def _save_results(
        self, results: Dict, cluster_labels: np.ndarray, true_labels: pd.Series
    ):
        """
        Save clustering results to files.

        Args:
            results: All clustering results
            cluster_labels: Cluster assignments
            true_labels: True labels
        """
        # Save full results JSON
        results_path = os.path.join(self.output_dir, "clustering_results.json")
        with open(results_path, "w") as f:
            # Remove numpy arrays for JSON serialization
            json_results = {}
            for key, value in results.items():
                if key.endswith("_2d") or key == "tsne_indices":
                    json_results[key] = value
                elif isinstance(value, dict):
                    json_results[key] = {
                        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                        for k, v in value.items()
                    }
                elif isinstance(value, (int, float)):
                    json_results[key] = float(value)
                else:
                    json_results[key] = value

            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to: {results_path}")

        # Save cluster assignments
        assignments_path = os.path.join(self.output_dir, "cluster_assignments.csv")
        assignments_df = pd.DataFrame(
            {"cluster": cluster_labels, "true_label": true_labels.values}
        )
        assignments_df.to_csv(assignments_path, index=False)
        logger.info(f"Assignments saved to: {assignments_path}")


def run_clustering_analysis(config: dict) -> Dict:
    """
    Run complete clustering analysis.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with clustering results
    """
    analyzer = ClusteringAnalyzer(config)
    results = analyzer.run_clustering_analysis()
    return results


def main():
    """Main entry point for standalone execution."""
    import yaml

    # Load config
    config_path = "configs/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Run clustering analysis
    _ = run_clustering_analysis(config)
    print("\nClustering analysis complete!")


if __name__ == "__main__":
    main()
