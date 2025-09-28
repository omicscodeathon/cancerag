# CancerAg: A Reproducible Pipeline for Biased Agonist Identification

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CancerAg is a computational framework designed to identify and predict biased agonism in G-protein-coupled receptors (GPCRs). This project provides a fully reproducible pipeline that automates data collection, molecular docking, feature engineering, and machine learning to classify ligands based on their signaling pathways.

## ✨ Features

-   **Automated Data Collection:** Gathers data from BiasDB, PDB, and ChEMBL to build a comprehensive dataset.
-   **Dynamic Active Site Identification:** Uses co-crystallized ligands to accurately define docking sites, avoiding generic coordinates.
-   **Robust Feature Engineering:** Calculates over 200 molecular descriptors for ligands and characterizes receptor binding pockets.
-   **Integrated Molecular Docking:** Utilizes AutoDock Vina to predict ligand binding affinities.
-   **Hybrid Machine Learning Approach:** Employs both unsupervised clustering to discover natural data separations and supervised learning for predictive classification.
-   **Advanced Model Training:** Implements powerful feature selection with Boruta and trains multiple state-of-the-art models (XGBoost, CatBoost, RandomForest).
-   **Reproducibility:** The entire pipeline is configurable and scriptable, ensuring consistent results.

## 🏗️ Pipeline Architecture

The project is organized into four main stages, flowing from raw data collection to final model evaluation.

```mermaid
graph TD
    subgraph A[1. Data Collection]
        A1[Download BiasDB Data] --> A2{Get Unique Receptors};
        A2 --> A3[Download PDB Structures];
        A2 --> A4[Download ChEMBL Agonists];
    end

    subgraph B[2. Preprocessing & Feature Extraction]
        A3 --> B1[Clean Receptors];
        A4 --> B2[Clean Ligands];
        A1 --> B2;
        B1 --> B3[Identify Active Sites];
        B1 --> B4[Calculate Receptor Descriptors];
        B2 --> B5[Calculate Ligand Descriptors];
    end

    subgraph C[3. Molecular Docking]
        B3 --> C1{Define Docking Box};
        B2 --> C2[Prepare Ligands for Docking];
        C1 & C2 --> C3[Run AutoDock Vina];
        C3 --> C4[Extract Binding Affinity];
    end

    subgraph D[4. Machine Learning]
        B4 & B5 & C4 --> D1[Assemble Feature Matrix];
        D1 --> D2{Unsupervised Clustering};
        D2 --> D3[Analyze Clusters];
        D1 --> D4{Supervised Classification};
        D4 --> D5[Stratified Split];
        D5 --> D6[Boruta Feature Selection];
        D6 --> D7[Train & Evaluate Models];
        D7 --> D8[Select Best Model];
    end

    A --> B --> C --> D;
```

##  workflows

### Data Collection Workflow

This sequence diagram illustrates how the pipeline gathers data from external databases.

```mermaid
sequenceDiagram
    participant P as Pipeline
    participant BDB as BiasDB
    participant PDB as Protein Data Bank
    participant CBL as ChEMBL

    P->>BDB: Request all biased ligand data
    BDB-->>P: Return JSON data
    P->>P: Parse data, save as biasdb_data.csv
    P->>P: Extract unique receptor IDs

    loop for each receptor
        P->>PDB: Query for 3D structures
        PDB-->>P: Return PDB files
        P->>CBL: Query for known agonists
        CBL-->>P: Return agonist molecules
    end
```

### Machine Learning Workflow

This diagram shows the steps involved in the machine learning phase, from feature assembly to model deployment.

```mermaid
graph TD
    Start((Start)) --> A[Assemble Final Feature Matrix];
    A --> B{Split Data};
    B -- Stratified Train/Test Split --> C[Training Set];
    B -- Stratified Train/Test Split --> D[Test Set];
    
    C --> E[Boruta Feature Selection];
    E --> F{Train Models};
    F -- Logistic Regression --> G[Evaluate Model 1];
    F -- Random Forest --> H[Evaluate Model 2];
    F -- XGBoost --> I[Evaluate Model 3];
    F -- CatBoost --> J[Evaluate Model 4];

    subgraph "Model Evaluation"
        G & H & I & J --> K[Compare Performance on Validation Set];
    end

    K --> L[Select Best Model];
    L & D --> M[Final Evaluation on Test Set];
    M --> End((End));

```

## 🚀 Getting Started

### Prerequisites

-   Python 3.10+
-   [UV](https://github.com/astral-sh/uv) package manager
-   [AutoDock Vina](http://vina.scripps.edu/download.html)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/cancerag.git
    cd cancerag
    ```

2.  **Install Python dependencies using UV:**
    ```bash
    uv pip install -r requirements.txt
    ```

3.  **Install AutoDock Vina:**
    Follow the instructions on the [official website](http://vina.scripps.edu/download.html) to install Vina and ensure it is available in your system's PATH.

### Configuration

All pipeline parameters are controlled from the `configs/config.yaml` file. Before running, you can adjust settings such as file paths, docking exhaustiveness, and machine learning model parameters.

### Usage

To run the entire pipeline, execute the main script:

```bash
python src/cancerag/main.py
```

## 📁 Project Structure

```
.
├── configs/
│   └── config.yaml         # Pipeline configuration
├── data/
│   ├── raw/                # Raw downloaded data (BiasDB, ChEMBL)
│   ├── interim/            # Intermediate data files
│   ├── processed/          # Final processed data for ML
│   └── pdb/                # Downloaded PDB files
├── results/
│   ├── figures/            # Plots and visualizations
│   ├── models/             # Saved trained models
│   └── reports/            # Generated reports
├── src/
│   └── cancerag/
│       ├── data_collection/ # Scripts for downloading data
│       ├── preprocessing/   # Data cleaning and preparation scripts
│       ├── features/        # Feature extraction scripts
│       ├── docking/         # Docking-related scripts
│       ├── ml/              # Machine learning models and training
│       └── main.py          # Main pipeline execution script
├── tests/                  # Unit and integration tests
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
