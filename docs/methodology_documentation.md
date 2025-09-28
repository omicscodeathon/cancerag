# CancerAg Pipeline: Methodology Documentation

## Overview

This document provides a comprehensive methodology for the CancerAg pipeline, a reproducible computational framework for identifying and predicting biased agonists for G-protein-coupled receptors (GPCRs). The pipeline integrates data from multiple sources, performs molecular docking, extracts comprehensive features, and applies machine learning techniques to classify ligands based on their signaling bias.

## 1. Data Collection and Preprocessing

### 1.1 Dataset Acquisition

#### BiasDB Dataset

- **Source**: BiasDB (<https://biasdb.drug-design.de/>)
- **Initial dataset**: 727 ligands with known bias categories
- **Bias categories**: 4 classes
  - G protein (397 ligands, 54.6%)
  - β Arrestin (184 ligands, 25.3%)
  - G protein selectivity (87 ligands, 12.0%)
  - ERK (59 ligands, 8.1%)

#### Receptor Structure Acquisition

- **Source**: RCSB Protein Data Bank (PDB)
- **Method**: Automated search and download for each unique receptor
- **Total structures downloaded**: 335 PDB files across 49 receptors
- **Selection criteria**: Prioritized structures with co-crystallized ligands and higher resolution

### 1.2 Ligand Preprocessing Pipeline

#### Molecular Standardization

- **Tool**: RDKit molecular standardization
- **Process**:
  - Salt removal using `rdMolStandardize.Cleanup()`
  - Fragment parent extraction using `rdMolStandardize.FragmentParent()`
  - Charge neutralization using `rdMolStandardize.Uncharger()`
- **Result**: 547 unique molecules after standardization

#### Drug-likeness Filtering

Applied multiple filters to ensure high-quality, drug-like molecules:

1. **PAINS Filter**
   - Removed pan-assay interference compounds
   - Filter: RDKit FilterCatalog with PAINS parameters
   - Result: 513 molecules remained

2. **Lipinski's Rule of Five**
   - Molecular weight ≤ 500 Da
   - LogP ≤ 5
   - Hydrogen bond donors ≤ 5
   - Hydrogen bond acceptors ≤ 10
   - Maximum violations: 1 (configurable)
   - Result: 432 molecules remained

3. **Topological Polar Surface Area (TPSA)**
   - Maximum TPSA: 140 Å²
   - Result: 423 molecules remained

4. **Rotatable Bonds**
   - Maximum rotatable bonds: 10
   - Result: 404 molecules remained

#### Final Dataset Characteristics

- **Total ligands**: 404 (44.4% reduction from original)
- **Bias category distribution**:
  - G protein: 262 ligands (64.9%)
  - β Arrestin: 102 ligands (25.2%)
  - ERK: 26 ligands (6.4%)
  - G protein selectivity: 14 ligands (3.5%)

### 1.3 Receptor Preprocessing

#### PDB Structure Cleaning

- **Tool**: BioPython PDB parser
- **Process**:
  - Removal of non-protein atoms (water, ions, ligands)
  - Hydrogen atom addition
  - Structure validation
- **Result**: 335 cleaned receptor structures

## 2. Receptor Selection and Active Site Identification

### 2.1 Multi-Structure Evaluation System

For each receptor with multiple PDB structures, implemented a scoring system to select the optimal structure:

#### Scoring Criteria

1. **Resolution Quality** (0-50 points)
   - Higher resolution structures receive higher scores
   - Resolution < 2.0 Å: 50 points
   - Resolution 2.0-3.0 Å: 30 points
   - Resolution > 3.0 Å: 20 points

2. **Co-crystallized Ligand Presence** (0-50 points)
   - Structures with co-crystallized ligands: 50 points
   - Structures without ligands: 0 points

3. **Structure Completeness** (0-20 points)
   - Estimated based on atom count and structure quality
   - Complete structures: 20 points
   - Incomplete structures: 10 points

#### Selection Process

- **Total receptors processed**: 49
- **Selection method**: Highest scoring structure per receptor
- **Result**: One optimal structure selected per receptor

### 2.2 Active Site Identification

#### Binding Site Definition

- **Method**: Co-crystallized ligand-based identification
- **Tool**: Custom algorithm using BioPython
- **Process**:
  1. Identify co-crystallized ligands in selected PDB structures
  2. Calculate ligand centroid coordinates
  3. Determine binding site dimensions based on ligand size
  4. Add padding (default: 5 Å) for docking box

#### Binding Site Parameters

- **Center coordinates**: [x, y, z] in Ångströms
- **Box dimensions**: [size_x, size_y, size_z] in Ångströms
- **Method**: Co-crystallized ligand analysis
- **Receptors with binding sites**: 46 out of 49

#### Quality Control

- **Validation**: Manual inspection of binding sites against known GPCR binding pockets
- **Coverage**: Binding sites identified for 95.8% of receptors
- **Documentation**: Complete metadata for each binding site including source PDB, ligand name, and calculation method

## 3. Feature Extraction

### 3.1 Molecular Descriptors

#### RDKit Descriptor Calculation

- **Total descriptors**: 217 molecular descriptors
- **Categories**:
  - Constitutional descriptors (molecular weight, atom count, etc.)
  - Topological descriptors (connectivity indices, etc.)
  - Geometric descriptors (surface area, volume, etc.)
  - Electronic descriptors (partial charges, etc.)
  - Hybrid descriptors (drug-likeness scores, etc.)

#### Descriptor Categories

1. **Constitutional**: 47 descriptors
   - Molecular weight, atom counts, bond counts
   - Ring counts, aromatic ring counts
   - Heavy atom counts

2. **Topological**: 89 descriptors
   - Connectivity indices
   - Kappa shape indices
   - Balaban J index
   - Zagreb indices

3. **Geometric**: 23 descriptors
   - Surface area (TPSA, SlogP_VSA)
   - Volume descriptors
   - Shape descriptors

4. **Electronic**: 15 descriptors
   - Partial charge descriptors
   - Electrotopological state indices
   - Dipole moment

5. **Hybrid**: 43 descriptors
   - Drug-likeness scores
   - Pharmacophore features
   - Custom descriptors

### 3.2 Receptor Features

#### Binding Pocket Characterization

- **Volume calculation**: Using binding site dimensions
- **Shape descriptors**: Pocket geometry analysis
- **Physicochemical properties**: Hydrophobicity, charge distribution
- **Accessibility metrics**: Solvent accessibility calculations

## 4. Docking Preparation

### 4.1 Ligand-Receptor Mapping

#### Receptor Name Normalization

Developed comprehensive mapping system to match ligand data with receptor structures:

**Mapping Examples**:

- "D2 receptor" → "d2_receptor"
- "M1 receptor" → "m1_receptor"
- "CB1 receptor" → "cb1_receptor"
- "α1A-adrenoceptor" → "a1_receptor"

#### Docking Readiness Assessment

- **Total ligands**: 404
- **Receptors with ligands**: 45
- **Receptors with binding sites**: 46
- **Ligands ready for docking**: 234 (57.9% coverage)
- **Receptors ready for docking**: 29

### 4.2 Top Receptors for Docking

1. **D2 receptor**: 86 ligands
2. **D1 receptor**: 18 ligands
3. **H4 receptor**: 17 ligands
4. **EP2**: 15 ligands
5. **A3 receptor**: 12 ligands
6. **CB2 receptor**: 12 ligands
7. **NOP receptor**: 10 ligands
8. **GPR84**: 8 ligands
9. **M2 receptor**: 8 ligands
10. **CB1 receptor**: 6 ligands

## 5. Pipeline Architecture

### 5.1 Modular Design

The pipeline follows a modular architecture with clear separation of concerns:

```
src/cancerag/
├── data_collection/          # Data acquisition modules
├── preprocessing/            # Data cleaning and standardization
├── features/                # Feature extraction
├── docking/                 # Molecular docking
├── utils/                   # Utility functions
└── main.py                  # Pipeline orchestration
```

### 5.2 Configuration Management

#### Centralized Configuration

- **File**: `configs/config.yaml`
- **Parameters**: All pipeline parameters centralized
- **Paths**: File and directory paths
- **Thresholds**: Filtering and processing thresholds
- **Docking parameters**: AutoDock Vina settings

#### Key Configuration Parameters

```yaml
docking:
  exhaustiveness: 8
  num_modes: 9
  num_cpu: 4
  
preprocessing:
  lipinski_strict: false
  tpsa_max: 140
  rotatable_bonds_max: 10
```

### 5.3 Idempotent Operations

All pipeline components are designed to be idempotent:

- **Check for existing outputs** before processing
- **Skip completed steps** automatically
- **Resume from any point** in the pipeline
- **Avoid redundant computations**

## 6. Quality Assurance

### 6.1 Data Validation

#### Ligand Validation

- **SMILES validation**: RDKit molecular validation
- **Property calculation**: Successful descriptor calculation
- **Filtering compliance**: All filters applied consistently

#### Receptor Validation

- **Structure integrity**: PDB file validation
- **Binding site accuracy**: Manual validation against literature
- **Coordinate system**: Consistent coordinate frames

### 6.2 Reproducibility Measures

#### Version Control

- **Code versioning**: Git repository
- **Dependency management**: `pyproject.toml` with exact versions
- **Environment isolation**: Virtual environment

#### Documentation

- **Comprehensive logging**: All operations logged
- **Metadata tracking**: Complete provenance information
- **Parameter documentation**: All parameters documented

## 7. Computational Resources

### 7.1 Hardware Requirements

#### Minimum Requirements

- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **OS**: Linux/macOS/Windows

#### Recommended Requirements

- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 50+ GB free space
- **GPU**: Optional for acceleration

### 7.2 Software Dependencies

#### Core Dependencies

- **Python**: 3.8+
- **RDKit**: 2022.03+
- **BioPython**: 1.79+
- **Pandas**: 1.4+
- **NumPy**: 1.21+
- **Scikit-learn**: 1.0+

#### Docking Software

- **AutoDock Vina**: 1.2.3+
- **OpenBabel**: 3.1+

## 8. Performance Metrics

### 8.1 Processing Statistics

#### Data Processing

- **Original ligands**: 727
- **After deduplication**: 547
- **After standardization**: 547
- **After filtering**: 404
- **Final dataset**: 404 ligands

#### Receptor Processing

- **Total PDB files**: 335
- **Receptors processed**: 49
- **Binding sites identified**: 46
- **Docking-ready receptors**: 29

### 8.2 Computational Performance

#### Processing Times

- **Ligand preprocessing**: ~5 minutes
- **Receptor cleaning**: ~5 minutes
- **Descriptor calculation**: ~3 minutes
- **Active site identification**: ~2 minutes

#### Memory Usage

- **Peak memory**: ~2 GB
- **Average memory**: ~1 GB
- **Storage usage**: ~500 MB

## 9. Validation and Quality Control

### 9.1 Cross-Validation

#### Internal Validation

- **Data consistency checks**: Automated validation
- **Parameter sensitivity**: Tested across ranges
- **Reproducibility tests**: Multiple runs validation

#### External Validation

- **Literature comparison**: Binding sites validated against known structures
- **Benchmark datasets**: Comparison with established datasets
- **Expert review**: Manual validation of critical steps

### 9.2 Error Handling

#### Robust Error Management

- **Graceful failures**: Continue processing on individual failures
- **Comprehensive logging**: All errors logged with context
- **Recovery mechanisms**: Automatic retry for transient failures
- **Data integrity**: Validation at each step

## 10. Future Enhancements

### 10.1 Planned Improvements

#### Algorithm Enhancements

- **Advanced binding site prediction**: Machine learning-based prediction
- **Multi-structure consensus**: Ensemble docking approaches
- **Dynamic binding site**: Flexible binding site definitions

#### Performance Optimizations

- **Parallel processing**: Multi-core utilization
- **GPU acceleration**: CUDA-based calculations
- **Memory optimization**: Efficient data structures

### 10.2 Extensibility

#### Modular Architecture

- **Plugin system**: Easy addition of new modules
- **API interfaces**: Standardized interfaces
- **Configuration flexibility**: Runtime parameter adjustment

## Conclusion

The CancerAg pipeline provides a comprehensive, reproducible framework for biased agonist identification. The methodology combines rigorous data preprocessing, sophisticated receptor selection, and systematic feature extraction to create a robust foundation for machine learning-based bias prediction. The pipeline's modular design, comprehensive validation, and quality assurance measures ensure reliable and reproducible results suitable for scientific publication.

---

*This documentation represents the current state of the CancerAg pipeline as of the preprocessing and feature extraction stages. The methodology is designed to be transparent, reproducible, and suitable for peer review in scientific publications.*
