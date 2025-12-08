"""
Unit tests for clustering analysis.
"""

import numpy as np
import pandas as pd
import pytest

from cancerag.ml.clustering import ClusteringAnalyzer


@pytest.mark.unit
class TestClusteringAnalysis:
    """Test suite for clustering analysis."""

    def test_clustering_analyzer_init(self, test_config):
        """Test ClusteringAnalyzer initialization."""
        analyzer = ClusteringAnalyzer(test_config)

        assert analyzer is not None
        assert hasattr(analyzer, "random_state")

    def test_elbow_method(self, test_config):
        """Test elbow method calculation."""
        analyzer = ClusteringAnalyzer(test_config)

        # Create dummy data
        n_samples = 100
        n_features = 10
        X = np.random.rand(n_samples, n_features)
        X_df = pd.DataFrame(X)

        # Calculate elbow
        elbow_results = analyzer._elbow_method(X_df, k_range=range(2, 6))

        assert "k_values" in elbow_results
        assert "inertias" in elbow_results
        assert "optimal_k" in elbow_results
        assert elbow_results["optimal_k"] >= 2

    def test_silhouette_analysis(self, test_config):
        """Test silhouette analysis."""
        analyzer = ClusteringAnalyzer(test_config)

        # Create dummy data
        n_samples = 100
        n_features = 10
        X = np.random.rand(n_samples, n_features)
        X_df = pd.DataFrame(X)

        # Calculate silhouette
        silhouette_results = analyzer._silhouette_analysis(X_df, k_range=range(2, 6))

        assert "k_values" in silhouette_results
        assert "scores" in silhouette_results
        assert "optimal_k" in silhouette_results
        assert silhouette_results["optimal_k"] >= 2

    def test_cluster_validation_metrics(self, test_config):
        """Test cluster validation metrics calculation."""
        from sklearn.cluster import KMeans

        analyzer = ClusteringAnalyzer(test_config)

        # Create dummy data
        n_samples = 100
        n_features = 10
        X = np.random.rand(n_samples, n_features)
        X_df = pd.DataFrame(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_df)

        # Calculate metrics
        metrics = analyzer._cluster_validation(X_df, labels, k=3)

        assert "silhouette_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert -1 <= metrics["silhouette_score"] <= 1

    def test_pca_projection(self, test_config):
        """Test PCA projection."""
        analyzer = ClusteringAnalyzer(test_config)

        # Create dummy data
        n_samples = 100
        n_features = 20
        X = np.random.rand(n_samples, n_features)
        X_df = pd.DataFrame(X)

        # Project to 2D
        projected = analyzer._compute_pca_projection(X_df, n_components=2)

        assert projected.shape == (n_samples, 2)
        assert isinstance(projected, np.ndarray)
