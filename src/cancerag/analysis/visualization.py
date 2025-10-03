"""
Visualization Module

Creates comprehensive visualizations for the dataset analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Creates comprehensive visualizations for the dataset analysis.
    """
    
    def __init__(self, output_dir: str = "results/analysis"):
        """
        Initialize the visualization engine.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set up plotly
        pyo.init_notebook_mode(connected=True)
    
    def create_overview_dashboard(self, df: pd.DataFrame) -> None:
        """
        Create an overview dashboard of the dataset.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Creating overview dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Bias Distribution', 'Receptor Family Distribution',
                          'Molecular Weight Distribution', 'LogP Distribution',
                          'Receptor Coverage', 'Bias by Receptor Family'),
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Bias distribution pie chart
        bias_counts = df['primary_bias_label'].value_counts()
        fig.add_trace(
            go.Pie(labels=bias_counts.index, values=bias_counts.values, name="Bias Distribution"),
            row=1, col=1
        )
        
        # 2. Receptor family distribution pie chart
        family_counts = df['receptor_family'].value_counts()
        fig.add_trace(
            go.Pie(labels=family_counts.index, values=family_counts.values, name="Receptor Families"),
            row=1, col=2
        )
        
        # 3. Molecular weight distribution
        fig.add_trace(
            go.Histogram(x=df['molecular_weight'], name="Molecular Weight", nbinsx=30),
            row=2, col=1
        )
        
        # 4. LogP distribution
        fig.add_trace(
            go.Histogram(x=df['logp'], name="LogP", nbinsx=30),
            row=2, col=2
        )
        
        # 5. Receptor coverage
        receptor_counts = df['receptor'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=receptor_counts.index, y=receptor_counts.values, name="Top Receptors"),
            row=3, col=1
        )
        
        # 6. Bias by receptor family
        family_bias = df.groupby(['receptor_family', 'primary_bias_label']).size().unstack(fill_value=0)
        for bias in family_bias.columns:
            fig.add_trace(
                go.Bar(x=family_bias.index, y=family_bias[bias], name=bias),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Dataset Overview Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save as HTML
        fig.write_html(str(self.output_dir / 'overview_dashboard.html'))
        
        # Note: PNG export requires Chrome installation
        # fig.write_image(str(self.output_dir / 'overview_dashboard.png'), width=1200, height=1200)
    
    def create_bias_analysis_plots(self, df: pd.DataFrame) -> None:
        """
        Create bias analysis visualizations.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Creating bias analysis plots...")
        
        # 1. Bias distribution with detailed breakdown
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bias distribution pie chart
        bias_counts = df['primary_bias_label'].value_counts()
        axes[0, 0].pie(bias_counts.values, labels=bias_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Bias Label Distribution', fontweight='bold')
        
        # Bias by receptor family
        family_bias = df.groupby(['receptor_family', 'primary_bias_label']).size().unstack(fill_value=0)
        family_bias_pct = family_bias.div(family_bias.sum(axis=1), axis=0) * 100
        
        sns.heatmap(family_bias_pct, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 1])
        axes[0, 1].set_title('Bias Distribution by Receptor Family (%)', fontweight='bold')
        
        # Bias pathway analysis
        pathway_counts = df['bias_pathway'].value_counts().head(10)
        axes[1, 0].bar(range(len(pathway_counts)), pathway_counts.values)
        axes[1, 0].set_xticks(range(len(pathway_counts)))
        axes[1, 0].set_xticklabels(pathway_counts.index, rotation=45, ha='right')
        axes[1, 0].set_title('Top 10 Bias Pathways', fontweight='bold')
        axes[1, 0].set_ylabel('Count')
        
        # Bias consistency analysis
        ligand_bias_counts = df.groupby('ligand_name')['primary_bias_label'].nunique()
        consistency_data = ligand_bias_counts.value_counts()
        axes[1, 1].bar(consistency_data.index, consistency_data.values)
        axes[1, 1].set_xlabel('Number of Bias Labels per Ligand')
        axes[1, 1].set_ylabel('Number of Ligands')
        axes[1, 1].set_title('Bias Label Consistency', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bias_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_molecular_property_plots(self, df: pd.DataFrame) -> None:
        """
        Create molecular property visualizations.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Creating molecular property plots...")
        
        # Key molecular properties
        properties = ['molecular_weight', 'logp', 'hba', 'hbd', 'rings', 'tpsa']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, prop in enumerate(properties):
            if prop in df.columns:
                # Create box plot by bias
                sns.boxplot(data=df, x='primary_bias_label', y=prop, ax=axes[i])
                axes[i].set_title(f'{prop.replace("_", " ").title()} by Bias', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'molecular_property_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation matrix for molecular properties
        plt.figure(figsize=(10, 8))
        corr_matrix = df[properties].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Molecular Properties Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'molecular_properties_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_docking_analysis_plots(self, df: pd.DataFrame) -> None:
        """
        Create docking analysis visualizations.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Creating docking analysis plots...")
        
        # Get docking columns
        docking_columns = []
        for col in df.columns:
            if col not in ['ligand_name', 'smiles', 'smiles_duplicate', 'canonical_smiles_standardized',
                          'receptor_family', 'receptor', 'receptor_subtype', 'bias_category', 'bias_pathway',
                          'reference_ligand', 'assay_1', 'assay_2', 'publication_title', 'author', 'doi', 'pmid', 'year',
                          'primary_bias_label', 'receptor_count'] and not col.startswith('bias_') and not any(col.startswith(prefix) for prefix in ['Max', 'Min', 'qed', 'SPS', 'Mol', 'Num', 'BCUT', 'Avg', 'Balaban', 'Bertz', 'Chi', 'Hall', 'Ipc', 'Kappa', 'Labute', 'PEOE', 'SMR', 'SlogP', 'EState', 'VSA', 'Fraction', 'Heavy', 'NHOH', 'NO', 'NumAliphatic', 'NumAmide', 'NumAromatic', 'NumAtom', 'NumBridge', 'NumHAcceptors', 'NumHDonors', 'NumHetero', 'NumRotatable', 'NumSaturated', 'NumSpiro', 'NumUnspecified', 'Phi', 'Ring', 'fr_']):
                docking_columns.append(col)
        
        if not docking_columns:
            logger.warning("No docking columns found")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall affinity distribution
        all_affinities = []
        for col in docking_columns:
            if col in df.columns:
                receptor_data = df[col][df[col] != -5.0]
                all_affinities.extend(receptor_data.tolist())
        
        if all_affinities:
            axes[0, 0].hist(all_affinities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Binding Affinity (kcal/mol)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Overall Docking Affinity Distribution', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top receptors by mean affinity
        receptor_means = []
        receptor_names = []
        
        for receptor in docking_columns[:10]:
            if receptor in df.columns:
                receptor_data = df[receptor][df[receptor] != -5.0]
                if len(receptor_data) > 0:
                    receptor_means.append(receptor_data.mean())
                    receptor_names.append(receptor.replace('_', ' ').title())
        
        if receptor_means:
            bars = axes[0, 1].bar(range(len(receptor_names)), receptor_means, color='lightcoral')
            axes[0, 1].set_xlabel('Receptor')
            axes[0, 1].set_ylabel('Mean Binding Affinity (kcal/mol)')
            axes[0, 1].set_title('Mean Binding Affinity by Receptor', fontweight='bold')
            axes[0, 1].set_xticks(range(len(receptor_names)))
            axes[0, 1].set_xticklabels(receptor_names, rotation=45, ha='right')
            
            # Add value labels
            for bar, mean_val in zip(bars, receptor_means):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                               f'{mean_val:.1f}', ha='center', va='bottom')
        
        # 3. Affinity by bias group (for top receptor)
        if docking_columns and docking_columns[0] in df.columns:
            top_receptor = docking_columns[0]
            valid_data = df[df[top_receptor] != -5.0]
            
            if len(valid_data) > 0:
                sns.boxplot(data=valid_data, x='primary_bias_label', y=top_receptor, ax=axes[1, 0])
                axes[1, 0].set_title(f'Binding Affinity by Bias ({top_receptor})', fontweight='bold')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Binding rate by receptor
        binding_rates = []
        receptor_names_binding = []
        
        for receptor in docking_columns[:10]:
            if receptor in df.columns:
                total_ligands = len(df)
                binding_ligands = len(df[df[receptor] != -5.0])
                binding_rate = (binding_ligands / total_ligands) * 100
                binding_rates.append(binding_rate)
                receptor_names_binding.append(receptor.replace('_', ' ').title())
        
        if binding_rates:
            bars = axes[1, 1].bar(range(len(receptor_names_binding)), binding_rates, color='lightgreen')
            axes[1, 1].set_xlabel('Receptor')
            axes[1, 1].set_ylabel('Binding Rate (%)')
            axes[1, 1].set_title('Binding Rate by Receptor', fontweight='bold')
            axes[1, 1].set_xticks(range(len(receptor_names_binding)))
            axes[1, 1].set_xticklabels(receptor_names_binding, rotation=45, ha='right')
            
            # Add value labels
            for bar, rate in zip(bars, binding_rates):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                               f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'docking_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_receptor_analysis_plots(self, df: pd.DataFrame) -> None:
        """
        Create receptor analysis visualizations.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Creating receptor analysis plots...")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Receptor family distribution
        family_counts = df['receptor_family'].value_counts()
        axes[0, 0].pie(family_counts.values, labels=family_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Receptor Family Distribution', fontweight='bold')
        
        # 2. Top receptors by ligand count
        receptor_counts = df['receptor'].value_counts().head(10)
        bars = axes[0, 1].bar(range(len(receptor_counts)), receptor_counts.values, color='lightblue')
        axes[0, 1].set_xlabel('Receptor')
        axes[0, 1].set_ylabel('Number of Ligands')
        axes[0, 1].set_title('Top 10 Receptors by Ligand Count', fontweight='bold')
        axes[0, 1].set_xticks(range(len(receptor_counts)))
        axes[0, 1].set_xticklabels([receptor.replace(' receptor', '').title() for receptor in receptor_counts.index], 
                                  rotation=45, ha='right')
        
        # Add value labels
        for bar, count in zip(bars, receptor_counts.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                           f'{count}', ha='center', va='bottom')
        
        # 3. Receptor coverage analysis
        receptor_quality = {}
        for receptor in df['receptor'].unique():
            receptor_data = df[df['receptor'] == receptor]
            receptor_quality[receptor] = len(receptor_data)
        
        # Plot distribution of ligand counts per receptor
        ligand_counts = list(receptor_quality.values())
        axes[1, 0].hist(ligand_counts, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Number of Ligands per Receptor')
        axes[1, 0].set_ylabel('Number of Receptors')
        axes[1, 0].set_title('Distribution of Ligand Counts per Receptor', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Bias distribution by receptor family
        family_bias = df.groupby(['receptor_family', 'primary_bias_label']).size().unstack(fill_value=0)
        family_bias_pct = family_bias.div(family_bias.sum(axis=1), axis=0) * 100
        
        sns.heatmap(family_bias_pct, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1, 1])
        axes[1, 1].set_title('Bias Distribution by Receptor Family (%)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'receptor_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_statistical_analysis_plots(self, df: pd.DataFrame) -> None:
        """
        Create statistical analysis visualizations.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Creating statistical analysis plots...")
        
        # Key features for analysis
        features = ['molecular_weight', 'logp', 'hba', 'hbd', 'rings', 'tpsa']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            if feature in df.columns:
                # Create violin plot by bias
                sns.violinplot(data=df, x='primary_bias_label', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature.replace("_", " ").title()} Distribution by Bias', fontweight='bold')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_analysis_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create correlation matrix for key features
        plt.figure(figsize=(10, 8))
        corr_matrix = df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Key Features Correlation Matrix', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'key_features_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_plots(self, df: pd.DataFrame) -> None:
        """
        Create interactive plots using Plotly.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Creating interactive plots...")
        
        # 1. Interactive scatter plot: Molecular Weight vs LogP colored by bias
        fig = px.scatter(df, x='molecular_weight', y='logp', color='primary_bias_label',
                        title='Molecular Weight vs LogP by Bias',
                        labels={'molecular_weight': 'Molecular Weight', 'logp': 'LogP'},
                        hover_data=['receptor_family', 'receptor'])
        
        fig.write_html(str(self.output_dir / 'interactive_scatter.html'))
        
        # 2. Interactive box plot: Molecular properties by bias
        fig = px.box(df, x='primary_bias_label', y='molecular_weight',
                    title='Molecular Weight Distribution by Bias')
        
        fig.write_html(str(self.output_dir / 'interactive_boxplot.html'))
        
        # 3. Interactive heatmap: Bias by receptor family
        family_bias = df.groupby(['receptor_family', 'primary_bias_label']).size().unstack(fill_value=0)
        family_bias_pct = family_bias.div(family_bias.sum(axis=1), axis=0) * 100
        
        fig = px.imshow(family_bias_pct.values,
                       labels=dict(x="Bias Label", y="Receptor Family", color="Percentage"),
                       x=family_bias_pct.columns,
                       y=family_bias_pct.index,
                       title='Bias Distribution by Receptor Family (%)')
        
        fig.write_html(str(self.output_dir / 'interactive_heatmap.html'))
    
    def run_visualization(self, df: pd.DataFrame) -> None:
        """
        Run complete visualization suite.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Starting comprehensive visualization...")
        
        # Create all visualizations
        self.create_overview_dashboard(df)
        self.create_bias_analysis_plots(df)
        self.create_molecular_property_plots(df)
        self.create_docking_analysis_plots(df)
        self.create_receptor_analysis_plots(df)
        self.create_statistical_analysis_plots(df)
        self.create_interactive_plots(df)
        
        logger.info("Visualization completed successfully!")
        logger.info(f"All plots saved to: {self.output_dir}")
