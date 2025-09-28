# CancerAg Pipeline Sequence Diagram

## Complete Pipeline Flow

```mermaid
sequenceDiagram
    participant User as User
    participant Main as Main Pipeline
    participant BiasDB as BiasDB Retriever
    participant PDB as PDB Retriever
    participant ChEMBL as ChEMBL Retriever
    participant LigandProc as Ligand Preprocessor
    participant ReceptorProc as Receptor Preprocessor
    participant MolDesc as Molecular Descriptors
    participant ActiveSite as Active Site Identifier
    participant Docking as Docking Pipeline
    participant ML as Machine Learning

    User->>Main: Run Pipeline
    Main->>Main: Load Configuration (config.yaml)
    
    Note over Main: STAGE 1: DATA COLLECTION
    Main->>BiasDB: Download biased ligand data
    BiasDB-->>Main: Return biasdb_data.csv
    
    Main->>Main: Extract unique receptors
    Main->>PDB: Download PDB structures for each receptor
    PDB-->>Main: Return PDB files + summary.json
    
    Note over Main: ChEMBL retrieval COMMENTED OUT
    Main->>ChEMBL: [COMMENTED] Fetch unbiased agonists
    ChEMBL-->>Main: [COMMENTED] Return unbiased agonists
    
    Note over Main: STAGE 2: PREPROCESSING
    Main->>LigandProc: Clean and standardize ligands
    LigandProc-->>Main: Return unified_ligands.csv
    
    Main->>ReceptorProc: Clean PDB structures
    ReceptorProc-->>Main: Return cleaned receptors/
    
    Note over Main: STAGE 3: FEATURE EXTRACTION
    Main->>MolDesc: Calculate molecular descriptors
    MolDesc-->>Main: Return ligands_with_descriptors.csv
    
    Main->>ActiveSite: Identify binding sites
    ActiveSite->>ActiveSite: Evaluate PDB structures
    ActiveSite->>ActiveSite: Select best structure per receptor
    ActiveSite->>ActiveSite: Extract binding site from co-crystallized ligand
    ActiveSite-->>Main: Return binding_sites.json + structure_selection_summary.json
    
    Note over Main: STAGE 4: MOLECULAR DOCKING
    Main->>Docking: Run docking pipeline
    Docking->>Docking: Prepare ligands (3D structures)
    Docking->>Docking: Load receptor structures
    Docking->>Docking: Load binding sites
    Docking->>Docking: Run AutoDock Vina
    Docking-->>Main: Return docking results
    
    Note over Main: STAGE 5: MACHINE LEARNING
    Main->>ML: Train models
    ML->>ML: Assemble feature matrix
    ML->>ML: Split data (stratified)
    ML->>ML: Feature selection (Boruta)
    ML->>ML: Train models (RF, XGBoost, CatBoost)
    ML->>ML: Evaluate and select best model
    ML-->>Main: Return trained model + results
    
    Main-->>User: Pipeline Complete
```

## Configuration Flow

```mermaid
graph TD
    A[config.yaml] --> B[Main Pipeline]
    B --> C[Data Collection]
    B --> D[Preprocessing]
    B --> E[Feature Extraction]
    B --> F[Docking]
    B --> G[Machine Learning]
    
    C --> C1[BiasDB Retriever]
    C --> C2[PDB Retriever]
    C --> C3[ChEMBL Retriever - COMMENTED]
    
    D --> D1[Ligand Preprocessor]
    D --> D2[Receptor Preprocessor]
    
    E --> E1[Molecular Descriptors]
    E --> E2[Active Site Identifier]
    
    F --> F1[Docking Pipeline]
    F --> F2[AutoDock Vina]
    
    G --> G1[Feature Selection]
    G --> G2[Model Training]
    G --> G3[Model Evaluation]
```

## Data Flow

```mermaid
graph LR
    A[BiasDB] --> B[biasdb_data.csv]
    C[PDB Database] --> D[PDB Files]
    E[ChEMBL] --> F[Unbiased Agonists - COMMENTED]
    
    B --> G[unified_ligands.csv]
    D --> H[cleaned_receptors/]
    F --> G
    
    G --> I[ligands_with_descriptors.csv]
    H --> J[binding_sites.json]
    
    I --> K[Final Feature Matrix]
    J --> L[Docking Results]
    
    K --> M[Machine Learning Models]
    L --> M
    
    M --> N[Predictions & Results]
```

## Key Components Interaction

### 1. Configuration Management

- **config.yaml**: Central configuration file
- **Paths**: All file paths defined in config
- **Parameters**: Docking, ML, and processing parameters

### 2. Data Collection

- **BiasDB Retriever**: Downloads biased ligand data
- **PDB Retriever**: Downloads receptor structures
- **ChEMBL Retriever**: [COMMENTED] Downloads unbiased agonists

### 3. Preprocessing

- **Ligand Preprocessor**: Standardizes SMILES, removes duplicates
- **Receptor Preprocessor**: Cleans PDB files, removes water/ligands

### 4. Feature Extraction

- **Molecular Descriptors**: Calculates ~200 RDKit descriptors
- **Active Site Identifier**: Selects best structure, identifies binding sites

### 5. Docking

- **Docking Pipeline**: Prepares ligands and receptors
- **AutoDock Vina**: Performs molecular docking
- **Results**: Binding affinity scores

### 6. Machine Learning

- **Feature Assembly**: Combines descriptors + docking scores
- **Model Training**: Multiple algorithms (RF, XGBoost, CatBoost)
- **Evaluation**: Cross-validation and test set evaluation

## File Dependencies

```text
config.yaml
├── data/raw/
│   ├── biasdb_data.csv
│   └── chembl/ (COMMENTED)
├── data/pdb/
│   ├── summary.json
│   └── [receptor_dirs]/
├── data/processed/
│   ├── unified_ligands.csv
│   ├── receptors/
│   ├── ligands_with_descriptors.csv
│   ├── binding_sites.json
│   └── structure_selection_summary.json
└── results/
    ├── docking_results/
    ├── models/
    └── reports/
```

## Current Status

✅ **Completed:**

- BiasDB data collection
- PDB structure retrieval
- Ligand preprocessing
- Receptor preprocessing
- Molecular descriptor calculation
- Enhanced active site identification
- Receptor-ligand mapping

🔄 **In Progress:**

- Molecular docking pipeline
- Machine learning implementation

📝 **Planned:**

- Unbiased agonist addition (when needed)
- Model evaluation and selection
- Results visualization
