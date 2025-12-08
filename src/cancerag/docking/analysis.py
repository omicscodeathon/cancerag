import numpy as np
import pandas as pd


def _parse_log_file(log_file: str) -> list:
    """Extracts binding affinities from a Vina log file."""
    affinities = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                if line.strip().startswith(
                    ("1", "2", "3", "4", "5", "6", "7", "8", "9")
                ):
                    parts = line.split()
                    if len(parts) >= 2:
                        affinities.append(float(parts[1]))
    except (FileNotFoundError, ValueError) as e:
        print(f"  - WARNING: Could not parse log file {log_file}: {e}")
    return affinities


def parse_docking_results(raw_results: list) -> list:
    """
    Parses the raw output from the docking runner.

    Args:
        raw_results (list): The list of result dictionaries from the runner.

    Returns:
        list: A sorted list of parsed results with key information.
    """
    parsed_results = []
    for result in raw_results:
        if not result["success"]:
            continue

        affinities = _parse_log_file(result["log_file"])
        if not affinities:
            continue

        parsed_results.append(
            {
                "ligand_name": result["ligand_name"],
                "best_affinity": min(affinities),
                "mean_affinity": np.mean(affinities),
                "all_affinities": affinities,
                "out_file": result["out_file"],
            }
        )

    # Sort by best binding affinity (most negative is best)
    parsed_results.sort(key=lambda x: x["best_affinity"])
    return parsed_results


def compare_receptor_affinities(all_parsed_results: dict) -> pd.DataFrame:
    """
    Compares binding affinities across all receptors to find biased ligands.

    Args:
        all_parsed_results (dict): Parsed results, keyed by receptor name.

    Returns:
        pd.DataFrame: A dataframe comparing affinities and calculating bias scores.
    """
    ligand_data = {}
    receptor_names = list(all_parsed_results.keys())

    for receptor, results in all_parsed_results.items():
        for result in results:
            ligand_name = result["ligand_name"]
            if ligand_name not in ligand_data:
                ligand_data[ligand_name] = {r: None for r in receptor_names}
            ligand_data[ligand_name][receptor] = result["best_affinity"]

    df = pd.DataFrame.from_dict(ligand_data, orient="index")
    df.index.name = "ligand_name"

    # Calculate bias scores (difference in affinity)
    if len(receptor_names) >= 2:
        for i, r1 in enumerate(receptor_names):
            for r2 in receptor_names[i + 1 :]:
                bias_col = f"bias_{r1}_vs_{r2}"
                df[bias_col] = df[r1] - df[r2]

    return df
