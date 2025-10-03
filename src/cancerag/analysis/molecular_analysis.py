"""
Molecular Descriptor Analysis Module

Analyzes molecular descriptors and their relationships with bias.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class MolecularAnalyzer:
    """
    Analyzes molecular descriptors and their relationships with bias.
    """
    
    def __init__(self, output_dir: str = "results/analysis"):
        """
        Initialize the molecular analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def get_molecular_descriptors(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of molecular descriptor columns.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            List[str]: List of molecular descriptor column names
        """
        # Basic descriptors
        basic_descriptors = ['molecular_weight', 'logp', 'hba', 'hbd', 'rings', 'tpsa']
        
        # RDKit descriptors (all columns that start with specific prefixes)
        rdkit_prefixes = [
            'Max', 'Min', 'qed', 'SPS', 'Mol', 'Num', 'BCUT', 'Avg', 'Balaban', 
            'Bertz', 'Chi', 'Hall', 'Ipc', 'Kappa', 'Labute', 'PEOE', 'SMR', 
            'SlogP', 'EState', 'VSA', 'Fraction', 'Heavy', 'NHOH', 'NO', 
            'NumAliphatic', 'NumAmide', 'NumAromatic', 'NumAtom', 'NumBridge', 
            'NumHAcceptors', 'NumHDonors', 'NumHetero', 'NumRotatable', 
            'NumSaturated', 'NumSpiro', 'NumUnspecified', 'Phi', 'Ring', 'fr_'
        ]
        
        rdkit_descriptors = []
        for col in df.columns:
            if any(col.startswith(prefix) for prefix in rdkit_prefixes):
                rdkit_descriptors.append(col)
        
        return basic_descriptors + rdkit_descriptors
    
    def analyze_basic_properties(self, df: pd.DataFrame) -> Dict:
        """
        Analyze basic molecular properties.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing basic molecular properties...")
        
        results = {}
        basic_props = ['molecular_weight', 'logp', 'hba', 'hbd', 'rings', 'tpsa']
        
        # Statistical summary
        for prop in basic_props:
            if prop in df.columns:
                results[prop] = {
                    'mean': df[prop].mean(),
                    'std': df[prop].std(),
                    'min': df[prop].min(),
                    'max': df[prop].max(),
                    'median': df[prop].median(),
                    'q25': df[prop].quantile(0.25),
                    'q75': df[prop].quantile(0.75)
                }
        
        # Create visualizations
        self._plot_basic_properties_distribution(df, basic_props)
        self._plot_basic_properties_correlation(df, basic_props)
        
        return results
    
    def analyze_descriptor_bias_correlations(self, df: pd.DataFrame) -> Dict:
        """
        Analyze correlations between molecular descriptors and bias.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing descriptor-bias correlations...")
        
        results = {}
        molecular_descriptors = self.get_molecular_descriptors(df)
        
        # Get numeric descriptors only
        numeric_descriptors = []
        for desc in molecular_descriptors:
            if desc in df.columns and df[desc].dtype in ['float64', 'int64']:
                numeric_descriptors.append(desc)
        
        # Calculate correlations with bias (using bias as numeric)
        bias_numeric = pd.Categorical(df['primary_bias_label']).codes
        correlations = {}
        
        for desc in numeric_descriptors:
            if not df[desc].isna().all():
                corr, p_value = stats.pearsonr(df[desc].fillna(df[desc].median()), bias_numeric)
                correlations[desc] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'abs_correlation': abs(corr)
                }
        
        # Sort by absolute correlation
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1]['abs_correlation'], reverse=True)
        
        results['top_correlations'] = dict(sorted_correlations[:20])  # Top 20
        results['all_correlations'] = correlations
        
        # Create visualization
        self._plot_descriptor_bias_correlations(sorted_correlations[:20])
        
        return results
    
    def analyze_descriptor_distributions(self, df: pd.DataFrame) -> Dict:
        """
        Analyze distributions of molecular descriptors.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing descriptor distributions...")
        
        results = {}
        molecular_descriptors = self.get_molecular_descriptors(df)
        
        # Get numeric descriptors
        numeric_descriptors = []
        for desc in molecular_descriptors:
            if desc in df.columns and df[desc].dtype in ['float64', 'int64']:
                numeric_descriptors.append(desc)
        
        # Analyze distributions
        distribution_stats = {}
        for desc in numeric_descriptors[:20]:  # Limit to first 20 for performance
            if not df[desc].isna().all():
                # Normality test
                shapiro_stat, shapiro_p = stats.shapiro(df[desc].dropna().sample(min(5000, len(df))))
                
                distribution_stats[desc] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'is_normal': shapiro_p > 0.05,
                    'skewness': stats.skew(df[desc].dropna()),
                    'kurtosis': stats.kurtosis(df[desc].dropna())
                }
        
        results['distribution_stats'] = distribution_stats
        
        # Create visualization
        self._plot_descriptor_distributions(df, numeric_descriptors[:12])
        
        return results
    
    def analyze_descriptor_clustering(self, df: pd.DataFrame) -> Dict:
        """
        Analyze clustering of molecular descriptors.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing descriptor clustering...")
        
        results = {}
        molecular_descriptors = self.get_molecular_descriptors(df)
        
        # Get numeric descriptors
        numeric_descriptors = []
        for desc in molecular_descriptors:
            if desc in df.columns and df[desc].dtype in ['float64', 'int64']:
                numeric_descriptors.append(desc)
        
        # Prepare data for clustering
        desc_data = df[numeric_descriptors].fillna(df[numeric_descriptors].median())
        
        # Standardize features
        scaler = StandardScaler()
        desc_data_scaled = scaler.fit_transform(desc_data)
        
        # PCA analysis
        pca = PCA()
        pca_result = pca.fit_transform(desc_data_scaled)
        
        results['pca_explained_variance'] = pca.explained_variance_ratio_.tolist()
        results['pca_cumulative_variance'] = np.cumsum(pca.explained_variance_ratio_).tolist()
        results['n_components_95'] = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        
        # Create visualization
        self._plot_pca_analysis(pca_result, pca.explained_variance_ratio_, df['primary_bias_label'])
        
        return results
    
    def _plot_basic_properties_distribution(self, df: pd.DataFrame, basic_props: List[str]) -> None:
        """Create basic properties distribution plots."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, prop in enumerate(basic_props):
            if prop in df.columns and i < len(axes):
                # Histogram
                axes[i].hist(df[prop].dropna(), bins=30, alpha=0.7, color=sns.color_palette("husl", 1)[0])
                axes[i].set_title(f'{prop.replace("_", " ").title()} Distribution', fontweight='bold')
                axes[i].set_xlabel(prop.replace("_", " ").title())
                axes[i].set_ylabel('Frequency')
        
        # Remove empty subplots
        for i in range(len(basic_props), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'basic_properties_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_basic_properties_correlation(self, df: pd.DataFrame, basic_props: List[str]) -> None:
        """Create basic properties correlation matrix."""
        # Get available basic properties
        available_props = [prop for prop in basic_props if prop in df.columns]
        
        if len(available_props) > 1:
            plt.figure(figsize=(10, 8))
            
            corr_matrix = df[available_props].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            
            plt.title('Basic Molecular Properties Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'basic_properties_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_descriptor_bias_correlations(self, top_correlations: List[Tuple]) -> None:
        """Create descriptor-bias correlation plot."""
        if not top_correlations:
            return
        
        descriptors = [item[0] for item in top_correlations]
        correlations = [item[1]['correlation'] for item in top_correlations]
        
        plt.figure(figsize=(12, 8))
        
        colors = ['red' if corr < 0 else 'blue' for corr in correlations]
        bars = plt.barh(range(len(descriptors)), correlations, color=colors, alpha=0.7)
        
        plt.yticks(range(len(descriptors)), [desc.replace('_', ' ').title() for desc in descriptors])
        plt.xlabel('Correlation with Bias')
        plt.title('Top 20 Molecular Descriptors Correlated with Bias', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(corr + (0.01 if corr > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'descriptor_bias_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_descriptor_distributions(self, df: pd.DataFrame, descriptors: List[str]) -> None:
        """Create descriptor distribution plots."""
        n_descriptors = len(descriptors)
        n_cols = 4
        n_rows = (n_descriptors + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = [ax for row in axes for ax in (row if isinstance(row, np.ndarray) else [row])]
        
        for i, desc in enumerate(descriptors):
            if i < len(axes) and desc in df.columns:
                # Box plot by bias
                df_clean = df[[desc, 'primary_bias_label']].dropna()
                if not df_clean.empty:
                    sns.boxplot(data=df_clean, x='primary_bias_label', y=desc, ax=axes[i])
                    axes[i].set_title(f'{desc.replace("_", " ").title()}', fontweight='bold')
                    axes[i].tick_params(axis='x', rotation=45)
        
        # Remove empty subplots
        for i in range(len(descriptors), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'descriptor_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_analysis(self, pca_result: np.ndarray, explained_variance: np.ndarray, 
                          bias_labels: pd.Series) -> None:
        """Create PCA analysis plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance plot
        ax1.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), 'bo-')
        ax1.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax1.set_xlabel('Number of Components')
        ax1.set_ylabel('Cumulative Explained Variance')
        ax1.set_title('PCA Explained Variance', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PCA scatter plot (first 2 components)
        if pca_result.shape[1] >= 2:
            scatter = ax2.scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=pd.Categorical(bias_labels).codes, 
                                cmap='viridis', alpha=0.6)
            ax2.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
            ax2.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
            ax2.set_title('PCA Scatter Plot (PC1 vs PC2)', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Bias Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete molecular descriptor analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Complete analysis results
        """
        logger.info("Starting comprehensive molecular descriptor analysis...")
        
        results = {}
        
        # Run all analysis methods
        results['basic_properties'] = self.analyze_basic_properties(df)
        results['bias_correlations'] = self.analyze_descriptor_bias_correlations(df)
        results['distributions'] = self.analyze_descriptor_distributions(df)
        results['clustering'] = self.analyze_descriptor_clustering(df)
        
        logger.info("Molecular descriptor analysis completed successfully!")
        
        return results
