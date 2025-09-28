#!/usr/bin/env python3
"""
Create sample datasets for fast testing of the docking pipeline.
"""

import pandas as pd
import os

def create_sample_datasets():
    """Create different sample datasets for testing."""
    
    # Load the drug-like ligands
    df = pd.read_csv('data/processed/drug_like_ligands.csv')
    
    print("🎯 CREATING SAMPLE DATASETS FOR FAST TESTING")
    print("=" * 60)
    
    # Option 1: Top 5 receptors (most ligands)
    top5_receptors = df['receptor_subtype'].value_counts().head(5).index.tolist()
    top5_df = df[df['receptor_subtype'].isin(top5_receptors)]
    top5_df.to_csv('data/processed/sample_top5_receptors.csv', index=False)
    print(f"✅ Top 5 receptors: {len(top5_df)} ligands, {len(top5_receptors)} receptors")
    print(f"   Receptors: {', '.join(top5_receptors)}")
    
    # Option 2: Balanced sampling (max 10 per receptor)
    balanced_df = df.groupby('receptor_subtype').apply(
        lambda x: x.sample(min(10, len(x)), random_state=42)
    ).reset_index(drop=True)
    balanced_df.to_csv('data/processed/sample_balanced.csv', index=False)
    print(f"✅ Balanced sampling: {len(balanced_df)} ligands, {balanced_df['receptor_subtype'].nunique()} receptors")
    
    # Option 3: Random subset (100 ligands)
    random_df = df.sample(n=100, random_state=42)
    random_df.to_csv('data/processed/sample_random.csv', index=False)
    print(f"✅ Random subset: {len(random_df)} ligands, {random_df['receptor_subtype'].nunique()} receptors")
    
    # Option 4: Single receptor (D2 - most ligands)
    d2_df = df[df['receptor_subtype'] == 'D2 receptor'].head(20)
    d2_df.to_csv('data/processed/sample_d2_only.csv', index=False)
    print(f"✅ D2 receptor only: {len(d2_df)} ligands, 1 receptor")
    
    print()
    print("📊 COMPUTATIONAL ESTIMATES:")
    print("-" * 40)
    print(f"Full dataset: {len(df)} ligands × {df['receptor_subtype'].nunique()} receptors = {len(df) * df['receptor_subtype'].nunique():,} dockings")
    print(f"Top 5 receptors: {len(top5_df)} ligands × {len(top5_receptors)} receptors = {len(top5_df) * len(top5_receptors):,} dockings")
    print(f"Balanced sampling: {len(balanced_df)} ligands × {balanced_df['receptor_subtype'].nunique()} receptors = {len(balanced_df) * balanced_df['receptor_subtype'].nunique():,} dockings")
    print(f"Random subset: {len(random_df)} ligands × {random_df['receptor_subtype'].nunique()} receptors = {len(random_df) * random_df['receptor_subtype'].nunique():,} dockings")
    print(f"D2 only: {len(d2_df)} ligands × 1 receptor = {len(d2_df):,} dockings")
    
    print()
    print("⏱️  ESTIMATED TIMES (with fast config):")
    print("-" * 40)
    print("Top 5 receptors: ~30-60 minutes")
    print("Balanced sampling: ~20-40 minutes")
    print("Random subset: ~15-30 minutes")
    print("D2 only: ~5-10 minutes")
    
    print()
    print("🚀 TO USE A SAMPLE DATASET:")
    print("-" * 40)
    print("1. Copy the sample file to replace the main dataset:")
    print("   cp data/processed/sample_d2_only.csv data/processed/drug_like_ligands.csv")
    print("2. Or modify the docking script to use the sample file")
    print("3. Use the fast config: configs/config_fast.yaml")

if __name__ == "__main__":
    create_sample_datasets()
