from chembl_webresource_client.new_client import new_client
import pandas as pd
import logging
from tqdm import tqdm
import time
import random
import os
from datetime import datetime
import requests.exceptions


def setup_logging():
    """
    Configure comprehensive logging with both console and file outputs.

    Returns:
        logging.Logger: Configured logger object
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Generate unique log filename with timestamp
    log_filename = f'logs/chembl_data_retrieval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

def get_targets(logger, query="estrogen receptor", max_retries=3):
    """
    Retrieve targets from ChEMBL with robust error handling and retry mechanism.

    Args:
        logger (logging.Logger): Logging object
        query (str): Search query for targets
        max_retries (int): Maximum number of retry attempts

    Returns:
        list: List of target dictionaries
    """
    for attempt in range(max_retries):
        try:
            target_query = new_client.target
            targets = list(target_query.filter(target_synonym__icontains=query))

            if not targets:
                logger.warning(f"No targets found for query: {query}")
                return []

            logger.info(f"Found {len(targets)} targets matching '{query}'")
            return targets

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached. Unable to fetch targets.")
                return []

def get_bioactivity_data(logger, target_chembl_id, max_retries=3):
    """
    Retrieve bioactivity data with comprehensive error handling and data cleaning.

    Args:
        logger (logging.Logger): Logging object
        target_chembl_id (str): ChEMBL target identifier
        max_retries (int): Maximum number of retry attempts

    Returns:
        pd.DataFrame: Cleaned bioactivity data
    """
    for attempt in range(max_retries):
        try:
            activity_query = new_client.activity
            activities = list(activity_query.filter(
                target_chembl_id=target_chembl_id
            ).only(
                ['molecule_chembl_id', 'canonical_smiles',
                 'standard_type', 'standard_value', 'standard_units']
            ))

            if not activities:
                logger.warning(f"No bioactivity data for target {target_chembl_id}")
                return pd.DataFrame()

            # Convert to DataFrame and clean
            activity_data = pd.DataFrame(activities)
            activity_data.dropna(subset=['canonical_smiles'], inplace=True)

            logger.info(f"Retrieved {len(activity_data)} bioactivity records")
            return activity_data

        except Exception as e:
            logger.error(f"Error retrieving bioactivity data on attempt {attempt + 1}: {e}")

            if attempt < max_retries - 1:
                wait_time = random.uniform(1, 5)  # Random wait to prevent rate limiting
                logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to retrieve data for target {target_chembl_id}")
                return pd.DataFrame()

def automate_estrogen_receptor_data(
    query="estrogen receptor",
    output_dir="data",
    max_targets=None
):
    """
    Comprehensive data retrieval process with progress tracking and error resilience.

    Args:
        query (str): Search query for targets
        output_dir (str): Directory to save output files
        max_targets (int, optional): Limit number of targets processed

    Returns:
        pd.DataFrame: Combined bioactivity data
    """
    # Setup logging
    logger = setup_logging()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Retrieve targets
    targets = get_targets(logger, query)

    if max_targets:
        targets = targets[:max_targets]

    # Prepare progress tracking
    all_data = []

    # Use tqdm for progress visualization
    for target in tqdm(targets, desc="Processing Targets", unit="target"):
        target_name = target['pref_name']
        target_chembl_id = target['target_chembl_id']

        logger.info(f"Processing Target: {target_name} (ChEMBL ID: {target_chembl_id})")

        # Retrieve bioactivity data
        bioactivity_data = get_bioactivity_data(logger, target_chembl_id)

        if not bioactivity_data.empty:
            bioactivity_data['Target Name'] = target_name
            bioactivity_data['Target ChEMBL ID'] = target_chembl_id
            all_data.append(bioactivity_data)

        # Optional: Small pause to prevent overwhelming the server
        time.sleep(random.uniform(0.5, 2))

    # Combine data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

        # Generate unique filename
        output_filename = os.path.join(
            output_dir,
            f"estrogen_receptor_bioactivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        # Save data
        combined_data.to_csv(output_filename, index=False)
        logger.info(f"Data saved to {output_filename}")

        return combined_data
    else:
        logger.warning("No data retrieved")
        return pd.DataFrame()

def main():
    """
    Main execution function with error handling.
    """
    try:
        combined_data = automate_estrogen_receptor_data(max_targets=50)  # Limit for testing

        if not combined_data.empty:
            print(f"Total records retrieved: {len(combined_data)}")
            print(combined_data.head())
        else:
            print("No data retrieved.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
