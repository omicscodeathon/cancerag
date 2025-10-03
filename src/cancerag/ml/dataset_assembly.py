"""
Dataset Assembly Module for Machine Learning Pipeline

This module combines molecular descriptors, receptor features, and docking results
into a unified dataset for machine learning training and evaluation.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetAssembler:
    """
    Assembles unified dataset for machine learning from multiple data sources.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the dataset assembler.
        
        Args:
            config (Dict): Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.paths = config["paths"]
        
        # Input file paths
        self.ligand_descriptors_path = os.path.join(
            self.paths["processed_data"], "ligands_with_descriptors.csv"
        )
        self.docking_results_path = os.path.join(
            self.paths["reports"], "docking_results", "affinity_comparison.csv"
        )
        self.biasdb_data_path = os.path.join(
            self.paths["raw_data"], "biasdb_data.csv"
        )
        
        # Output file paths
        self.unified_dataset_path = os.path.join(
            self.paths["processed_data"], "unified_ml_dataset.csv"
        )
        self.dataset_summary_path = os.path.join(
            self.paths["processed_data"], "dataset_summary.json"
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.unified_dataset_path), exist_ok=True)
    
    def load_ligand_descriptors(self) -> pd.DataFrame:
        """
        Load molecular descriptors for ligands.
        
        Returns:
            pd.DataFrame: DataFrame containing ligand descriptors
        """
        logger.info("Loading ligand molecular descriptors...")
        
        if not os.path.exists(self.ligand_descriptors_path):
            raise FileNotFoundError(f"Ligand descriptors file not found: {self.ligand_descriptors_path}")
        
        df = pd.read_csv(self.ligand_descriptors_path)
        logger.info(f"Loaded {len(df)} ligands with {len(df.columns)} descriptors")
        
        return df
    
    def load_docking_results(self) -> pd.DataFrame:
        """
        Load docking affinity results.
        
        Returns:
            pd.DataFrame: DataFrame containing docking results
        """
        logger.info("Loading docking affinity results...")
        
        if not os.path.exists(self.docking_results_path):
            raise FileNotFoundError(f"Docking results file not found: {self.docking_results_path}")
        
        df = pd.read_csv(self.docking_results_path)
        logger.info(f"Loaded docking results for {len(df)} ligands across {len([col for col in df.columns if not col.startswith('bias_')])} receptors")
        
        return df
    
    def load_bias_labels(self) -> pd.DataFrame:
        """
        Load bias labels from BiasDB data.
        
        Returns:
            pd.DataFrame: DataFrame containing bias labels
        """
        logger.info("Loading bias labels from BiasDB...")
        
        if not os.path.exists(self.biasdb_data_path):
            logger.warning(f"BiasDB data file not found: {self.biasdb_data_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(self.biasdb_data_path)
        
        # Extract relevant columns for bias labeling
        bias_columns = [
            'ligand_name', 'receptor_subtype', 'bias_category', 
            'bias_pathway', 'reference_ligand'
        ]
        
        # Keep only columns that exist in the dataframe
        available_columns = [col for col in bias_columns if col in df.columns]
        bias_df = df[available_columns].copy()
        
        logger.info(f"Loaded bias labels for {len(bias_df)} ligand-receptor pairs")
        
        return bias_df
    
    def create_bias_labels(self, ligands_df: pd.DataFrame, docking_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create bias labels based on available data.
        
        Args:
            ligands_df (pd.DataFrame): Ligand descriptors with bias information
            docking_df (pd.DataFrame): Docking results
            
        Returns:
            pd.DataFrame: DataFrame with bias labels
        """
        logger.info("Creating bias labels...")
        
        # Start with ligand data that has bias information
        bias_data = []
        
        for _, ligand_row in ligands_df.iterrows():
            ligand_name = ligand_row['ligand_name']
            receptor_subtype = ligand_row.get('receptor_subtype', '')
            bias_category = ligand_row.get('bias_category', '')
            bias_pathway = ligand_row.get('bias_pathway', '')
            
            # Create a simplified bias label
            if bias_category and bias_pathway:
                # Use the bias category as the main label
                bias_label = bias_category
            elif bias_category:
                bias_label = bias_category
            else:
                # If no bias information, mark as 'unknown'
                bias_label = 'unknown'
            
            bias_data.append({
                'ligand_name': ligand_name,
                'receptor_subtype': receptor_subtype,
                'bias_category': bias_category,
                'bias_pathway': bias_pathway,
                'bias_label': bias_label
            })
        
        bias_df = pd.DataFrame(bias_data)
        
        # Create a unified bias label for each ligand
        # For ligands with multiple receptors, use the most common bias type
        ligand_bias_summary = bias_df.groupby('ligand_name')['bias_label'].agg([
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown',
            'count'
        ]).reset_index()
        ligand_bias_summary.columns = ['ligand_name', 'primary_bias_label', 'receptor_count']
        
        logger.info(f"Created bias labels for {len(ligand_bias_summary)} ligands")
        logger.info(f"Bias label distribution: {ligand_bias_summary['primary_bias_label'].value_counts().to_dict()}")
        
        return ligand_bias_summary
    
    def merge_datasets(self, 
                      ligands_df: pd.DataFrame, 
                      docking_df: pd.DataFrame, 
                      bias_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge all datasets into a unified dataset.
        
        Args:
            ligands_df (pd.DataFrame): Ligand descriptors
            docking_df (pd.DataFrame): Docking results
            bias_df (pd.DataFrame): Bias labels
            
        Returns:
            pd.DataFrame: Unified dataset
        """
        logger.info("Merging datasets...")
        
        # Start with ligand descriptors as the base
        unified_df = ligands_df.copy()
        
        # Merge with bias labels
        unified_df = unified_df.merge(
            bias_df, 
            on='ligand_name', 
            how='left'
        )
        
        # Merge with docking results
        unified_df = unified_df.merge(
            docking_df, 
            on='ligand_name', 
            how='left'
        )
        
        logger.info(f"Unified dataset shape: {unified_df.shape}")
        
        return unified_df
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the unified dataset by handling missing values and outliers.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        logger.info("Cleaning dataset...")
        
        original_shape = df.shape
        
        # Handle missing values in bias labels
        df['primary_bias_label'] = df['primary_bias_label'].fillna('unknown')
        
        # Handle missing values in docking results
        # Replace missing docking scores with a default value (e.g., -5.0)
        docking_columns = [col for col in df.columns if not col.startswith(('bias_', 'ligand_name', 'smiles', 'receptor', 'reference', 'assay', 'publication', 'author', 'doi', 'pmid', 'year'))]
        
        for col in docking_columns:
            if col in df.columns:
                df[col] = df[col].fillna(-5.0)  # Default docking score
        
        # Remove rows with too many missing values
        # Keep rows with at least 20% non-null values (more lenient for docking data)
        threshold = len(df.columns) * 0.2
        df = df.dropna(thresh=threshold)
        
        logger.info(f"Dataset cleaning: {original_shape} -> {df.shape}")
        
        return df
    
    def create_feature_groups(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Group features by type for analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict[str, List[str]]: Feature groups
        """
        feature_groups = {
            'molecular_descriptors': [],
            'docking_affinities': [],
            'bias_features': [],
            'metadata': []
        }
        
        for col in df.columns:
            if col in ['ligand_name', 'smiles', 'smiles_duplicate', 'canonical_smiles_standardized']:
                feature_groups['metadata'].append(col)
            elif col.startswith('bias_'):
                feature_groups['bias_features'].append(col)
            elif col in ['receptor_subtype', 'bias_category', 'bias_pathway', 'primary_bias_label', 'receptor_count']:
                feature_groups['metadata'].append(col)
            elif col in ['receptor_family', 'receptor', 'reference_ligand', 'assay_1', 'assay_2', 'publication_title', 'author', 'doi', 'pmid', 'year']:
                feature_groups['metadata'].append(col)
            elif col in ['molecular_weight', 'logp', 'hba', 'hbd', 'rings', 'tpsa'] or col.startswith(('Max', 'Min', 'qed', 'SPS', 'Mol', 'Num', 'BCUT', 'Avg', 'Balaban', 'Bertz', 'Chi', 'Hall', 'Ipc', 'Kappa', 'Labute', 'PEOE', 'SMR', 'SlogP', 'EState', 'VSA', 'Fraction', 'Heavy', 'NHOH', 'NO', 'NumAliphatic', 'NumAmide', 'NumAromatic', 'NumAtom', 'NumBridge', 'NumHAcceptors', 'NumHDonors', 'NumHetero', 'NumRotatable', 'NumSaturated', 'NumSpiro', 'NumUnspecified', 'Phi', 'Ring', 'fr_')):
                feature_groups['molecular_descriptors'].append(col)
            else:
                # Assume it's a docking affinity column
                feature_groups['docking_affinities'].append(col)
        
        return feature_groups
    
    def generate_dataset_summary(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict:
        """
        Generate a summary of the unified dataset.
        
        Args:
            df (pd.DataFrame): Unified dataset
            feature_groups (Dict[str, List[str]]): Feature groups
            
        Returns:
            Dict: Dataset summary
        """
        summary = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'feature_groups': {group: len(features) for group, features in feature_groups.items()},
            'bias_label_distribution': df['primary_bias_label'].value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'receptor_coverage': len([col for col in df.columns if col in feature_groups['docking_affinities']]),
            'molecular_descriptor_count': len(feature_groups['molecular_descriptors']),
            'bias_feature_count': len(feature_groups['bias_features'])
        }
        
        return summary
    
    def save_dataset(self, df: pd.DataFrame, summary: Dict) -> None:
        """
        Save the unified dataset and summary.
        
        Args:
            df (pd.DataFrame): Unified dataset
            summary (Dict): Dataset summary
        """
        logger.info("Saving unified dataset...")
        
        # Save the dataset
        df.to_csv(self.unified_dataset_path, index=False)
        logger.info(f"Saved unified dataset to: {self.unified_dataset_path}")
        
        # Save the summary
        import json
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        summary_serializable = convert_numpy_types(summary)
        with open(self.dataset_summary_path, 'w') as f:
            json.dump(summary_serializable, f, indent=2)
        logger.info(f"Saved dataset summary to: {self.dataset_summary_path}")
    
    def run_assembly(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete dataset assembly process.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: Unified dataset and summary
        """
        logger.info("Starting dataset assembly...")
        
        # Load data
        ligands_df = self.load_ligand_descriptors()
        docking_df = self.load_docking_results()
        biasdb_df = self.load_bias_labels()
        
        # Create bias labels
        bias_df = self.create_bias_labels(ligands_df, docking_df)
        
        # Merge datasets
        unified_df = self.merge_datasets(ligands_df, docking_df, bias_df)
        
        # Clean dataset
        unified_df = self.clean_dataset(unified_df)
        
        # Create feature groups
        feature_groups = self.create_feature_groups(unified_df)
        
        # Generate summary
        summary = self.generate_dataset_summary(unified_df, feature_groups)
        
        # Save results
        self.save_dataset(unified_df, summary)
        
        logger.info("Dataset assembly completed successfully!")
        
        return unified_df, summary


def run_dataset_assembly(config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Main function to run dataset assembly.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Unified dataset and summary
    """
    assembler = DatasetAssembler(config)
    return assembler.run_assembly()


if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run dataset assembly
    dataset, summary = run_dataset_assembly(config)
    
    print("\nDataset Assembly Summary:")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total features: {summary['total_features']}")
    print(f"Feature groups: {summary['feature_groups']}")
    print(f"Bias label distribution: {summary['bias_label_distribution']}")
    print(f"Missing values: {summary['missing_values']} ({summary['missing_percentage']:.2f}%)")
    print(f"Receptor coverage: {summary['receptor_coverage']}")
