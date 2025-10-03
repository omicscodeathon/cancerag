"""
Main Data Analyzer Module

Orchestrates comprehensive data analysis across all analysis modules.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import time

from .bias_analysis import BiasAnalyzer
from .molecular_analysis import MolecularAnalyzer
from .docking_analysis import DockingAnalyzer
from .receptor_analysis import ReceptorAnalyzer
from .statistical_analysis import StatisticalAnalyzer
from .visualization import VisualizationEngine

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Main data analyzer that orchestrates comprehensive analysis across all modules.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data analyzer.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.paths = config["paths"]
        
        # Set up output directory
        self.output_dir = Path(self.paths["reports"]) / "data_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analysis modules
        self.bias_analyzer = BiasAnalyzer(str(self.output_dir))
        self.molecular_analyzer = MolecularAnalyzer(str(self.output_dir))
        self.docking_analyzer = DockingAnalyzer(str(self.output_dir))
        self.receptor_analyzer = ReceptorAnalyzer(str(self.output_dir))
        self.statistical_analyzer = StatisticalAnalyzer(str(self.output_dir))
        self.visualization_engine = VisualizationEngine(str(self.output_dir))
        
        # Results storage
        self.results = {}
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the unified dataset.
        
        Returns:
            pd.DataFrame: Unified dataset
        """
        dataset_path = Path(self.paths["processed_data"]) / "unified_ml_dataset.csv"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Unified dataset not found: {dataset_path}")
        
        logger.info(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
        
        return df
    
    def run_bias_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run bias analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Bias analysis results
        """
        logger.info("Running bias analysis...")
        start_time = time.time()
        
        results = self.bias_analyzer.run_analysis(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Bias analysis completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def run_molecular_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run molecular descriptor analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Molecular analysis results
        """
        logger.info("Running molecular descriptor analysis...")
        start_time = time.time()
        
        results = self.molecular_analyzer.run_analysis(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Molecular analysis completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def run_docking_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run docking analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Docking analysis results
        """
        logger.info("Running docking analysis...")
        start_time = time.time()
        
        results = self.docking_analyzer.run_analysis(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Docking analysis completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def run_receptor_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run receptor analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Receptor analysis results
        """
        logger.info("Running receptor analysis...")
        start_time = time.time()
        
        results = self.receptor_analyzer.run_analysis(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Receptor analysis completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def run_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run statistical analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Statistical analysis results
        """
        logger.info("Running statistical analysis...")
        start_time = time.time()
        
        results = self.statistical_analyzer.run_analysis(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Statistical analysis completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def run_visualization(self, df: pd.DataFrame) -> None:
        """
        Run visualization generation.
        
        Args:
            df (pd.DataFrame): Unified dataset
        """
        logger.info("Running visualization generation...")
        start_time = time.time()
        
        self.visualization_engine.run_visualization(df)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Visualization completed in {elapsed_time:.2f} seconds")
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary report of the analysis.
        
        Args:
            df (pd.DataFrame): Unified dataset
            
        Returns:
            Dict: Summary report
        """
        logger.info("Generating summary report...")
        
        # Basic dataset statistics
        summary = {
            'dataset_overview': {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            },
            'bias_distribution': df['primary_bias_label'].value_counts().to_dict(),
            'receptor_family_distribution': df['receptor_family'].value_counts().to_dict(),
            'receptor_distribution': df['receptor'].value_counts().head(10).to_dict(),
            'molecular_properties': {
                'molecular_weight': {
                    'mean': df['molecular_weight'].mean(),
                    'std': df['molecular_weight'].std(),
                    'min': df['molecular_weight'].min(),
                    'max': df['molecular_weight'].max()
                },
                'logp': {
                    'mean': df['logp'].mean(),
                    'std': df['logp'].std(),
                    'min': df['logp'].min(),
                    'max': df['logp'].max()
                }
            }
        }
        
        # Add analysis results summary
        if self.results:
            summary['analysis_results'] = {
                'bias_analysis': {
                    'class_imbalance_ratio': self.results.get('bias_analysis', {}).get('distribution', {}).get('class_imbalance_ratio', 'N/A'),
                    'bias_consistency': self.results.get('bias_analysis', {}).get('consistency', {}).get('consistency_percentage', 'N/A')
                },
                'molecular_analysis': {
                    'descriptor_count': len(self.molecular_analyzer.get_molecular_descriptors(df)),
                    'top_correlations': len(self.results.get('molecular_analysis', {}).get('bias_correlations', {}).get('top_correlations', {}))
                },
                'docking_analysis': {
                    'receptor_count': len(self.docking_analyzer.get_docking_columns(df)),
                    'affinity_stats': len(self.results.get('docking_analysis', {}).get('affinity_distributions', {}).get('affinity_stats', {}))
                },
                'receptor_analysis': {
                    'unique_receptors': df['receptor'].nunique(),
                    'unique_families': df['receptor_family'].nunique()
                },
                'statistical_analysis': {
                    'correlation_pairs': len(self.results.get('statistical_analysis', {}).get('correlations', {}).get('high_correlation_pairs', [])),
                    'significant_differences': sum(1 for test in self.results.get('statistical_analysis', {}).get('bias_differences', {}).get('statistical_tests', {}).values() 
                                                 if isinstance(test, dict) and test.get('significant_difference', False))
                }
            }
        
        return summary
    
    def save_results(self, results: Dict, df: pd.DataFrame) -> None:
        """
        Save analysis results to files.
        
        Args:
            results (Dict): Analysis results
            df (pd.DataFrame): Dataset for summary generation
        """
        logger.info("Saving analysis results...")
        
        # Save complete results as JSON
        results_file = self.output_dir / 'analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary report
        summary = self.generate_summary_report(df)
        summary_file = self.output_dir / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {self.output_dir}")
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive data analysis.
        
        Returns:
            Dict: Complete analysis results
        """
        logger.info("Starting comprehensive data analysis...")
        start_time = time.time()
        
        # Load dataset
        df = self.load_dataset()
        
        # Run all analyses
        self.results = {}
        
        # 1. Bias Analysis
        self.results['bias_analysis'] = self.run_bias_analysis(df)
        
        # 2. Molecular Descriptor Analysis
        self.results['molecular_analysis'] = self.run_molecular_analysis(df)
        
        # 3. Docking Analysis
        self.results['docking_analysis'] = self.run_docking_analysis(df)
        
        # 4. Receptor Analysis
        self.results['receptor_analysis'] = self.run_receptor_analysis(df)
        
        # 5. Statistical Analysis
        self.results['statistical_analysis'] = self.run_statistical_analysis(df)
        
        # 6. Visualization
        self.run_visualization(df)
        
        # 7. Generate summary report
        self.results['summary'] = self.generate_summary_report(df)
        
        # 8. Save results
        self.save_results(self.results, df)
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Comprehensive data analysis completed in {total_time:.2f} seconds")
        
        return self.results


def run_data_analysis(config: Dict) -> Dict:
    """
    Main function to run comprehensive data analysis.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        Dict: Complete analysis results
    """
    analyzer = DataAnalyzer(config)
    return analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Run comprehensive analysis
    results = run_data_analysis(config)
    
    print("\nData Analysis Summary:")
    print(f"Total analysis time: {results.get('summary', {}).get('analysis_time', 'N/A')}")
    print(f"Results saved to: {config['paths']['reports']}/data_analysis/")
