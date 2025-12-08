import logging
import os
import sys

import pandas as pd
import requests

from cancerag.utils.network import (
    NetworkRetrier,
    NetworkRetrySettings,
    create_retry_session,
)

logger = logging.getLogger(__name__)


def download_biasdb_data(
    output_path: str, network_config: dict | None = None
) -> pd.DataFrame:
    """
    Fetches the complete dataset from the BiasDB web server and saves it as a CSV.
    This function is idempotent - it will skip download if the file already exists.

    Args:
        output_path (str): The file path to save the downloaded data.
        network_config (dict | None): Optional network retry configuration.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the structured BiasDB data.
    """
    # Check if file already exists (idempotent behavior)
    if os.path.exists(output_path):
        logger.info(
            f"BiasDB data already exists at {output_path}. Loading existing data..."
        )
        try:
            df = pd.read_csv(output_path)
            logger.info(
                f"Successfully loaded {len(df)} records from existing BiasDB data."
            )
            return df
        except Exception as e:
            logger.warning(
                f"Could not load existing BiasDB data: {e}. Re-downloading..."
            )

    url = "https://biasdb.drug-design.de/data_0/query?user_query=default_query"
    logger.info(f"Fetching data from BiasDB URL: {url}")

    retry_settings = NetworkRetrySettings.from_config(network_config)
    retrier = NetworkRetrier(retry_settings, logger=logger)
    session = create_retry_session(retry_settings, allowed_methods=["GET"])

    def _fetch_biasdb() -> list[dict]:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        return response.json()

    try:
        raw_data = retrier.run(
            "BiasDB download",
            _fetch_biasdb,
            (requests.exceptions.RequestException,),
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from BiasDB after retries: {e}")
        sys.exit(1)
    except ValueError:
        logger.error("Failed to decode JSON from BiasDB response.")
        sys.exit(1)

    # Define the comprehensive column headers based on analysis of the data structure
    headers = [
        "ligand_name",
        "smiles",
        "smiles_duplicate",
        "receptor_family",
        "receptor",
        "receptor_subtype",
        "bias_category",
        "bias_pathway",
        "reference_ligand",
        "assay_1",
        "assay_2",
        "publication_title",
        "author",
        "doi",
        "pmid",
        "year",
        "molecular_weight",
        "logp",
        "hba",
        "hbd",
        "rings",
        "tpsa",
    ]

    # The data is a direct list in the JSON response
    data_list = raw_data

    if not data_list:
        logger.warning("BiasDB query returned no data.")
        return pd.DataFrame(columns=headers)

    df = pd.DataFrame(data_list, columns=headers)

    # Save the data to the specified CSV file
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved BiasDB data to {output_path}")
    except IOError as e:
        logger.error(f"Failed to write BiasDB data to {output_path}: {e}")
        sys.exit(1)

    return df


if __name__ == "__main__":
    # Example of how to run this module directly
    import os

    # Ensure the output directory exists
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file and run the downloader
    output_file = os.path.join(output_dir, "biasdb_data.csv")
    download_biasdb_data(output_file)
