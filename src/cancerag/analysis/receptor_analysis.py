"""
Receptor Analysis Module

Analyzes receptor patterns, families, and coverage in the dataset.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class ReceptorAnalyzer:
    """
    Analyzes receptor patterns, families, and coverage in the dataset.
    """
    
    def __init__(self, output_dir: str = "results/analysis"):
        """
        Initialize the receptor analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_receptor_families(self, df: pd.DataFrame) -> Dict:
        """
        Analyze receptor family patterns.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing receptor family patterns...")
        
        results = {}
        
        # Family distribution
        family_counts = df['receptor_family'].value_counts()
        family_percentages = df['receptor_family'].value_counts(normalize=True) * 100
        
        results['family_counts'] = family_counts.to_dict()
        results['family_percentages'] = family_percentages.to_dict()
        
        # Family bias analysis
        family_bias = df.groupby(['receptor_family', 'primary_bias_label']).size().unstack(fill_value=0)
        family_bias_pct = family_bias.div(family_bias.sum(axis=1), axis=0) * 100
        
        results['family_bias_matrix'] = family_bias.to_dict()
        results['family_bias_percentages'] = family_bias_pct.to_dict()
        
        # Create visualizations
        self._plot_receptor_family_distribution(family_counts, family_percentages)
        self._plot_family_bias_heatmap(family_bias_pct)
        
        return results
    
    def analyze_individual_receptors(self, df: pd.DataFrame) -> Dict:
        """
        Analyze individual receptor patterns.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing individual receptor patterns...")
        
        results = {}
        
        # Receptor distribution
        receptor_counts = df['receptor'].value_counts()
        receptor_percentages = df['receptor'].value_counts(normalize=True) * 100
        
        results['receptor_counts'] = receptor_counts.to_dict()
        results['receptor_percentages'] = receptor_percentages.to_dict()
        
        # Top receptors analysis
        top_receptors = receptor_counts.head(10)
        results['top_receptors'] = top_receptors.to_dict()
        
        # Receptor bias analysis
        receptor_bias = df.groupby(['receptor', 'primary_bias_label']).size().unstack(fill_value=0)
        receptor_bias_pct = receptor_bias.div(receptor_bias.sum(axis=1), axis=0) * 100
        
        results['receptor_bias_matrix'] = receptor_bias.to_dict()
        results['receptor_bias_percentages'] = receptor_bias_pct.to_dict()
        
        # Create visualizations
        self._plot_individual_receptor_distribution(top_receptors)
        self._plot_receptor_bias_heatmap(receptor_bias_pct.head(15))
        
        return results
    
    def analyze_receptor_coverage(self, df: pd.DataFrame) -> Dict:
        """
        Analyze receptor coverage and data quality.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing receptor coverage...")
        
        results = {}
        
        # Coverage statistics
        total_ligands = len(df)
        unique_receptors = df['receptor'].nunique()
        unique_families = df['receptor_family'].nunique()
        
        results['coverage_stats'] = {
            'total_ligands': total_ligands,
            'unique_receptors': unique_receptors,
            'unique_families': unique_families,
            'ligands_per_receptor': total_ligands / unique_receptors,
            'ligands_per_family': total_ligands / unique_families
        }
        
        # Data quality per receptor
        receptor_quality = {}
        for receptor in df['receptor'].unique():
            receptor_data = df[df['receptor'] == receptor]
            
            receptor_quality[receptor] = {
                'ligand_count': len(receptor_data),
                'bias_categories': receptor_data['primary_bias_label'].nunique(),
                'bias_distribution': receptor_data['primary_bias_label'].value_counts().to_dict(),
                'receptor_count_mean': receptor_data['receptor_count'].mean(),
                'receptor_count_std': receptor_data['receptor_count'].std()
            }
        
        results['receptor_quality'] = receptor_quality
        
        # Coverage gaps analysis
        coverage_gaps = self._identify_coverage_gaps(df)
        results['coverage_gaps'] = coverage_gaps
        
        # Create visualizations
        self._plot_receptor_coverage(receptor_quality)
        
        return results
    
    def analyze_receptor_subtypes(self, df: pd.DataFrame) -> Dict:
        """
        Analyze receptor subtype patterns.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Analysis results
        """
        logger.info("Analyzing receptor subtype patterns...")
        
        results = {}
        
        # Subtype distribution
        subtype_counts = df['receptor_subtype'].value_counts()
        results['subtype_counts'] = subtype_counts.to_dict()
        
        # Subtype bias analysis
        subtype_bias = df.groupby(['receptor_subtype', 'primary_bias_label']).size().unstack(fill_value=0)
        subtype_bias_pct = subtype_bias.div(subtype_bias.sum(axis=1), axis=0) * 100
        
        results['subtype_bias_matrix'] = subtype_bias.to_dict()
        results['subtype_bias_percentages'] = subtype_bias_pct.to_dict()
        
        # Subtype coverage analysis
        subtype_coverage = {}
        for subtype in df['receptor_subtype'].unique():
            if pd.notna(subtype):
                subtype_data = df[df['receptor_subtype'] == subtype]
                
                subtype_coverage[subtype] = {
                    'ligand_count': len(subtype_data),
                    'receptor_family': subtype_data['receptor_family'].iloc[0] if len(subtype_data) > 0 else 'Unknown',
                    'bias_categories': subtype_data['primary_bias_label'].nunique(),
                    'bias_distribution': subtype_data['primary_bias_label'].value_counts().to_dict()
                }
        
        results['subtype_coverage'] = subtype_coverage
        
        # Create visualizations
        self._plot_receptor_subtype_analysis(subtype_counts.head(15), subtype_bias_pct.head(15))
        
        return results
    
    def _identify_coverage_gaps(self, df: pd.DataFrame) -> Dict:
        """
        Identify coverage gaps in the dataset.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Coverage gap analysis
        """
        gaps = {}
        
        # Bias coverage gaps
        bias_counts = df['primary_bias_label'].value_counts()
        min_bias_count = bias_counts.min()
        max_bias_count = bias_counts.max()
        
        gaps['bias_imbalance'] = {
            'min_count': min_bias_count,
            'max_count': max_bias_count,
            'imbalance_ratio': max_bias_count / min_bias_count,
            'underrepresented_bias': bias_counts.idxmin(),
            'overrepresented_bias': bias_counts.idxmax()
        }
        
        # Receptor coverage gaps
        receptor_counts = df['receptor'].value_counts()
        low_coverage_receptors = receptor_counts[receptor_counts < 5]  # Less than 5 ligands
        
        gaps['low_coverage_receptors'] = low_coverage_receptors.to_dict()
        
        # Family coverage gaps
        family_counts = df['receptor_family'].value_counts()
        low_coverage_families = family_counts[family_counts < 10]  # Less than 10 ligands
        
        gaps['low_coverage_families'] = low_coverage_families.to_dict()
        
        return gaps
    
    def _plot_receptor_family_distribution(self, family_counts: pd.Series, family_percentages: pd.Series) -> None:
        """Create receptor family distribution plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(family_counts.index, family_counts.values, color=sns.color_palette("husl", len(family_counts)))
        ax1.set_title('Receptor Family Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Receptor Family')
        ax1.set_ylabel('Number of Ligands')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count, pct in zip(bars, family_counts.values, family_percentages.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(family_counts.values, labels=family_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Receptor Family Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'receptor_family_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_family_bias_heatmap(self, family_bias_pct: pd.DataFrame) -> None:
        """Create family bias heatmap."""
        plt.figure(figsize=(12, 8))
        
        sns.heatmap(family_bias_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Bias Distribution by Receptor Family', fontsize=14, fontweight='bold')
        plt.xlabel('Bias Label')
        plt.ylabel('Receptor Family')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'family_bias_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_receptor_distribution(self, top_receptors: pd.Series) -> None:
        """Create individual receptor distribution plot."""
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(range(len(top_receptors)), top_receptors.values, 
                      color=sns.color_palette("husl", len(top_receptors)))
        
        plt.title('Top 10 Receptors by Ligand Count', fontsize=14, fontweight='bold')
        plt.xlabel('Receptor')
        plt.ylabel('Number of Ligands')
        plt.xticks(range(len(top_receptors)), 
                  [receptor.replace(' receptor', '').title() for receptor in top_receptors.index], 
                  rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, top_receptors.values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'individual_receptor_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_receptor_bias_heatmap(self, receptor_bias_pct: pd.DataFrame) -> None:
        """Create receptor bias heatmap."""
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(receptor_bias_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage (%)'})
        
        plt.title('Bias Distribution by Receptor (Top 15)', fontsize=14, fontweight='bold')
        plt.xlabel('Bias Label')
        plt.ylabel('Receptor')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'receptor_bias_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_receptor_coverage(self, receptor_quality: Dict) -> None:
        """Create receptor coverage plots."""
        # Extract data for plotting
        receptors = list(receptor_quality.keys())
        ligand_counts = [receptor_quality[receptor]['ligand_count'] for receptor in receptors]
        bias_categories = [receptor_quality[receptor]['bias_categories'] for receptor in receptors]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Ligand count distribution
        ax1.hist(ligand_counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Number of Ligands per Receptor')
        ax1.set_ylabel('Number of Receptors')
        ax1.set_title('Distribution of Ligand Counts per Receptor', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Bias categories distribution
        ax2.hist(bias_categories, bins=range(1, max(bias_categories) + 2), alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_xlabel('Number of Bias Categories per Receptor')
        ax2.set_ylabel('Number of Receptors')
        ax2.set_title('Distribution of Bias Categories per Receptor', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'receptor_coverage.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_receptor_subtype_analysis(self, subtype_counts: pd.Series, subtype_bias_pct: pd.DataFrame) -> None:
        """Create receptor subtype analysis plots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Subtype distribution
        bars = ax1.bar(range(len(subtype_counts)), subtype_counts.values, 
                      color=sns.color_palette("husl", len(subtype_counts)))
        
        ax1.set_title('Top 15 Receptor Subtypes by Ligand Count', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Receptor Subtype')
        ax1.set_ylabel('Number of Ligands')
        ax1.set_xticks(range(len(subtype_counts)))
        ax1.set_xticklabels([subtype.replace(' receptor', '').title() for subtype in subtype_counts.index], 
                           rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, subtype_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{count}', ha='center', va='bottom')
        
        # Subtype bias heatmap
        sns.heatmap(subtype_bias_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage (%)'}, ax=ax2)
        
        ax2.set_title('Bias Distribution by Receptor Subtype (Top 15)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Bias Label')
        ax2.set_ylabel('Receptor Subtype')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'receptor_subtype_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run complete receptor analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Complete analysis results
        """
        logger.info("Starting comprehensive receptor analysis...")
        
        results = {}
        
        # Run all analysis methods
        results['family_analysis'] = self.analyze_receptor_families(df)
        results['individual_receptors'] = self.analyze_individual_receptors(df)
        results['coverage_analysis'] = self.analyze_receptor_coverage(df)
        results['subtype_analysis'] = self.analyze_receptor_subtypes(df)
        
        logger.info("Receptor analysis completed successfully!")
        
        return results
