"""
Docking Analysis Module

Analyzes docking affinities and their relationships with bias.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)


class DockingAnalyzer:
    """
    Analyzes docking affinities and their relationships with bias.
    """
    
    def __init__(self, output_dir: str = "results/analysis"):
        """
        Initialize the docking analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def get_docking_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of docking affinity columns.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            List[str]: List of docking affinity column names
        """
        # Docking affinity columns are the receptor names
        docking_columns = []
        for col in df.columns:
            # Skip non-docking columns
            if col in ['ligand_name', 'smiles', 'smiles_duplicate', 'canonical_smiles_standardized']:
                continue
            elif col in ['receptor_family', 'receptor', 'receptor_subtype', 'bias_category', 'bias_pathway', 'reference_ligand', 'assay_1', 'assay_2', 'publication_title', 'author', 'doi', 'pmid', 'year']:
                continue
            elif col in ['primary_bias_label', 'receptor_count']:
                continue
            elif col.startswith('bias_'):
                continue
            elif col in ['molecular_weight', 'logp', 'hba', 'hbd', 'rings', 'tpsa'] or any(col.startswith(prefix) for prefix in ['Max', 'Min', 'qed', 'SPS', 'Mol', 'Num', 'BCUT', 'Avg', 'Balaban', 'Bertz', 'Chi', 'Hall', 'Ipc', 'Kappa', 'Labute', 'PEOE', 'SMR', 'SlogP', 'EState', 'VSA', 'Fraction', 'Heavy', 'NHOH', 'NO', 'NumAliphatic', 'NumAmide', 'NumAromatic', 'NumAtom', 'NumBridge', 'NumHAcceptors', 'NumHDonors', 'NumHetero', 'NumRotatable', 'NumSaturated', 'NumSpiro', 'NumUnspecified', 'Phi', 'Ring', 'fr_']):
                continue
            else:
                docking_columns.append(col)
        
        return docking_columns
    
    def analyze_affinity_distributions(self, df: pd.DataFrame) -> Dict:
        """
        Analyze distributions of docking affinities.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing docking affinity distributions...")
        
        results = {}
        docking_columns = self.get_docking_columns(df)
        
        # Statistical summary for each receptor
        affinity_stats = {}
        for receptor in docking_columns:
            if receptor in df.columns:
                # Filter out default values (-5.0) which represent missing data
                receptor_data = df[receptor][df[receptor] != -5.0]
                
                if len(receptor_data) > 0:
                    affinity_stats[receptor] = {
                        'count': len(receptor_data),
                        'mean': receptor_data.mean(),
                        'std': receptor_data.std(),
                        'min': receptor_data.min(),
                        'max': receptor_data.max(),
                        'median': receptor_data.median(),
                        'q25': receptor_data.quantile(0.25),
                        'q75': receptor_data.quantile(0.75)
                    }
        
        results['affinity_stats'] = affinity_stats
        
        # Overall affinity statistics
        all_affinities = []
        for receptor in docking_columns:
            if receptor in df.columns:
                receptor_data = df[receptor][df[receptor] != -5.0]
                all_affinities.extend(receptor_data.tolist())
        
        if all_affinities:
            results['overall_stats'] = {
                'total_affinities': len(all_affinities),
                'mean': np.mean(all_affinities),
                'std': np.std(all_affinities),
                'min': np.min(all_affinities),
                'max': np.max(all_affinities),
                'median': np.median(all_affinities)
            }
        
        # Create visualizations
        self._plot_affinity_distributions(df, docking_columns)
        
        return results
    
    def analyze_receptor_selectivity(self, df: pd.DataFrame) -> Dict:
        """
        Analyze receptor selectivity patterns.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing receptor selectivity patterns...")
        
        results = {}
        docking_columns = self.get_docking_columns(df)
        
        # Calculate selectivity metrics for each ligand
        selectivity_metrics = []
        
        for idx, row in df.iterrows():
            # Get affinities for this ligand (excluding -5.0 values)
            affinities = []
            receptors = []
            
            for receptor in docking_columns:
                if receptor in df.columns and row[receptor] != -5.0:
                    affinities.append(row[receptor])
                    receptors.append(receptor)
            
            if len(affinities) > 1:
                # Calculate selectivity metrics
                best_affinity = max(affinities)  # Most negative (best binding)
                worst_affinity = min(affinities)  # Least negative (worst binding)
                
                # Handle edge cases for selectivity ratio
                if worst_affinity == 0 or abs(worst_affinity) < 1e-10:
                    selectivity_ratio = 1000.0  # Cap at reasonable value
                else:
                    selectivity_ratio = best_affinity / worst_affinity
                    # Cap extreme values
                    selectivity_ratio = max(-1000.0, min(1000.0, selectivity_ratio))
                
                affinity_range = best_affinity - worst_affinity
                
                selectivity_metrics.append({
                    'ligand_name': row['ligand_name'],
                    'best_affinity': best_affinity,
                    'worst_affinity': worst_affinity,
                    'selectivity_ratio': selectivity_ratio,
                    'affinity_range': affinity_range,
                    'num_receptors': len(affinities)
                })
        
        results['selectivity_metrics'] = selectivity_metrics
        
        # Summary statistics
        if selectivity_metrics:
            selectivity_df = pd.DataFrame(selectivity_metrics)
            results['selectivity_summary'] = {
                'mean_selectivity_ratio': selectivity_df['selectivity_ratio'].mean(),
                'median_selectivity_ratio': selectivity_df['selectivity_ratio'].median(),
                'mean_affinity_range': selectivity_df['affinity_range'].mean(),
                'median_affinity_range': selectivity_df['affinity_range'].median()
            }
        
        # Create visualizations
        self._plot_receptor_selectivity(selectivity_metrics)
        
        return results
    
    def analyze_affinity_bias_relationships(self, df: pd.DataFrame) -> Dict:
        """
        Analyze relationships between docking affinities and bias.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing affinity-bias relationships...")
        
        results = {}
        docking_columns = self.get_docking_columns(df)
        
        # Calculate correlations between affinities and bias
        bias_correlations = {}
        
        for receptor in docking_columns:
            if receptor in df.columns:
                # Filter out default values
                valid_data = df[df[receptor] != -5.0]
                
                if len(valid_data) > 10:  # Need sufficient data
                    # Convert bias to numeric for correlation
                    bias_numeric = pd.Categorical(valid_data['primary_bias_label']).codes
                    
                    # Calculate correlation
                    corr, p_value = stats.pearsonr(valid_data[receptor], bias_numeric)
                    
                    bias_correlations[receptor] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'abs_correlation': abs(corr),
                        'sample_size': len(valid_data)
                    }
        
        # Sort by absolute correlation
        sorted_correlations = sorted(bias_correlations.items(), 
                                   key=lambda x: x[1]['abs_correlation'], reverse=True)
        
        results['bias_correlations'] = dict(sorted_correlations)
        results['top_correlations'] = dict(sorted_correlations[:10])  # Top 10
        
        # Create visualizations
        self._plot_affinity_bias_relationships(df, docking_columns, sorted_correlations[:10])
        
        return results
    
    def analyze_binding_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze binding patterns across receptors.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing binding patterns...")
        
        results = {}
        docking_columns = self.get_docking_columns(df)
        
        # Create affinity matrix
        affinity_matrix = df[docking_columns].copy()
        affinity_matrix = affinity_matrix.replace(-5.0, np.nan)  # Replace default values with NaN
        
        # Calculate binding statistics
        binding_stats = {}
        for receptor in docking_columns:
            if receptor in affinity_matrix.columns:
                receptor_data = affinity_matrix[receptor].dropna()
                
                if len(receptor_data) > 0:
                    binding_stats[receptor] = {
                        'binding_rate': len(receptor_data) / len(df),  # Percentage of ligands that bind
                        'mean_affinity': receptor_data.mean(),
                        'median_affinity': receptor_data.median(),
                        'strong_binders': len(receptor_data[receptor_data < -7.0]),  # Strong binding
                        'weak_binders': len(receptor_data[receptor_data > -5.0])  # Weak binding
                    }
        
        results['binding_stats'] = binding_stats
        
        # Create visualizations
        self._plot_binding_patterns(affinity_matrix, binding_stats)
        
        return results
    
    def _plot_affinity_distributions(self, df: pd.DataFrame, docking_columns: List[str]) -> None:
        """Create affinity distribution plots."""
        # Plot 1: Overall affinity distribution
        all_affinities = []
        for receptor in docking_columns:
            if receptor in df.columns:
                receptor_data = df[receptor][df[receptor] != -5.0]
                all_affinities.extend(receptor_data.tolist())
        
        if all_affinities:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.hist(all_affinities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Binding Affinity (kcal/mol)')
            plt.ylabel('Frequency')
            plt.title('Overall Docking Affinity Distribution', fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Affinity by receptor (top 10)
            receptor_means = []
            receptor_names = []
            
            for receptor in docking_columns[:10]:  # Top 10 receptors
                if receptor in df.columns:
                    receptor_data = df[receptor][df[receptor] != -5.0]
                    if len(receptor_data) > 0:
                        receptor_means.append(receptor_data.mean())
                        receptor_names.append(receptor.replace('_', ' ').title())
            
            plt.subplot(1, 2, 2)
            bars = plt.bar(range(len(receptor_names)), receptor_means, color='lightcoral')
            plt.xlabel('Receptor')
            plt.ylabel('Mean Binding Affinity (kcal/mol)')
            plt.title('Mean Binding Affinity by Receptor', fontweight='bold')
            plt.xticks(range(len(receptor_names)), receptor_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, receptor_means):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{mean_val:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'affinity_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_receptor_selectivity(self, selectivity_metrics: List[Dict]) -> None:
        """Create receptor selectivity plots."""
        if not selectivity_metrics:
            return
        
        selectivity_df = pd.DataFrame(selectivity_metrics)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Selectivity ratio distribution (filter out extreme values)
        selectivity_ratios = selectivity_df['selectivity_ratio']
        # Filter out extreme values for better visualization
        filtered_ratios = selectivity_ratios[(selectivity_ratios >= -100) & (selectivity_ratios <= 100)]
        
        if len(filtered_ratios) > 0:
            ax1.hist(filtered_ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        else:
            ax1.text(0.5, 0.5, 'No data within visualization range', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_xlabel('Selectivity Ratio')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Receptor Selectivity Ratio Distribution', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Affinity range vs number of receptors
        scatter = ax2.scatter(selectivity_df['num_receptors'], selectivity_df['affinity_range'], 
                            alpha=0.6, c=selectivity_df['selectivity_ratio'], cmap='viridis')
        ax2.set_xlabel('Number of Receptors')
        ax2.set_ylabel('Affinity Range (kcal/mol)')
        ax2.set_title('Affinity Range vs Receptor Count', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Selectivity Ratio')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'receptor_selectivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_affinity_bias_relationships(self, df: pd.DataFrame, docking_columns: List[str], 
                                        top_correlations: List[Tuple]) -> None:
        """Create affinity-bias relationship plots."""
        if not top_correlations:
            return
        
        # Plot 1: Top correlations
        receptors = [item[0] for item in top_correlations]
        correlations = [item[1]['correlation'] for item in top_correlations]
        
        plt.figure(figsize=(12, 8))
        
        colors = ['red' if corr < 0 else 'blue' for corr in correlations]
        bars = plt.barh(range(len(receptors)), correlations, color=colors, alpha=0.7)
        
        plt.yticks(range(len(receptors)), [receptor.replace('_', ' ').title() for receptor in receptors])
        plt.xlabel('Correlation with Bias')
        plt.title('Top 10 Receptors: Affinity-Bias Correlations', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add correlation values on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            plt.text(corr + (0.01 if corr > 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', ha='left' if corr > 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'affinity_bias_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_binding_patterns(self, affinity_matrix: pd.DataFrame, binding_stats: Dict) -> None:
        """Create binding pattern plots."""
        # Create heatmap of binding affinities
        plt.figure(figsize=(15, 10))
        
        # Sample data for heatmap (too much data to plot all)
        sample_size = min(100, len(affinity_matrix))
        sample_data = affinity_matrix.sample(n=sample_size, random_state=42)
        
        # Select top receptors by binding rate
        binding_rates = {receptor: stats['binding_rate'] for receptor, stats in binding_stats.items()}
        top_receptors = sorted(binding_rates.items(), key=lambda x: x[1], reverse=True)[:20]
        top_receptor_names = [item[0] for item in top_receptors]
        
        # Create heatmap
        heatmap_data = sample_data[top_receptor_names]
        
        sns.heatmap(heatmap_data, cmap='RdYlBu_r', center=-6, 
                   cbar_kws={'label': 'Binding Affinity (kcal/mol)'})
        
        plt.title('Docking Affinity Heatmap (Sample)', fontsize=14, fontweight='bold')
        plt.xlabel('Receptor')
        plt.ylabel('Ligand (Sample)')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'binding_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete docking analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Complete analysis results
        """
        logger.info("Starting comprehensive docking analysis...")
        
        results = {}
        
        # Run all analysis methods
        results['affinity_distributions'] = self.analyze_affinity_distributions(df)
        results['receptor_selectivity'] = self.analyze_receptor_selectivity(df)
        results['affinity_bias_relationships'] = self.analyze_affinity_bias_relationships(df)
        results['binding_patterns'] = self.analyze_binding_patterns(df)
        
        logger.info("Docking analysis completed successfully!")
        
        return results
