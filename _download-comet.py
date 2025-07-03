import argparse
import comet_ml
import json
import logging
from pathlib import Path

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR = Path("_distributions")

def download_assets(experiment, output_dir: Path):
    """
    Downloads specific assets from a Comet experiment and saves them to the
    specified directory.

    Args:
        experiment: An authenticated Comet.ml experiment object.
        output_dir (str): The path to the directory where assets will be saved.
    """
    # List of asset filenames to download
    asset_names = ["reward-to-go_.json", "final_reward_.json"]
    asset_list = experiment.get_asset_list()
    found_assets = {asset['fileName']: asset for asset in asset_list}

    for asset_name in asset_names:
        logging.info(f"Searching for asset: {asset_name}")
        if asset_name in found_assets:
            asset_info = found_assets[asset_name]
            try:
                # Download the asset content
                asset_data = experiment.get_asset(asset_info['assetId'], return_type="json")
                output_filename = output_dir / asset_name
                
                # Save the downloaded data to a JSON file
                with open(output_filename, 'w') as f:
                    json.dump(asset_data, f, indent=4)
                logging.info(f"Successfully downloaded and saved asset to: {output_filename}")

            except Exception as e:
                logging.error(f"Failed to download or save asset '{asset_name}'. Details: {e}")
        else:
            logging.warning(f"Asset '{asset_name}' not found in experiment '{experiment.id}'.")

def download_metrics(experiment, output_dir: Path):
    """
    Downloads specific metrics from a Comet experiment and saves them to JSON
    files in the specified directory.

    Args:
        experiment: An authenticated Comet.ml experiment object.
        output_dir (str): The path to the directory where metrics will be saved.
    """
    # List of metric names to download
    metric_names = ["reward mean", "reward std"]

    for metric_name in metric_names:
        logging.info(f"Fetching metric: {metric_name}")
        try:
            # Retrieve all data points for the given metric
            metric_data = experiment.get_metrics(metric_name)
            if not metric_data:
                logging.warning(f"Metric '{metric_name}' not found or has no data.")
                continue

            # Sanitize the metric name to create a valid filename
            safe_filename = metric_name.replace(" ", "_") + ".json"
            output_filename = output_dir / safe_filename

            # Save the metric data to a JSON file
            with open(output_filename, 'w') as f:
                json.dump(metric_data, f, indent=4)
            logging.info(f"Successfully downloaded and saved metric to: {output_filename}")

        except Exception as e:
            logging.error(f"Failed to download metric '{metric_name}'. Details: {e}")

def download_parameters(experiment, output_dir: Path):
    """
    Downloads the hyperparameters from a Comet experiment and saves them to a
    JSON file.

    Args:
        experiment: An authenticated Comet.ml experiment object.
        output_dir (str): The path to the directory where the file will be saved.
    """
    logging.info("Fetching hyperparameters...")
    try:
        # Retrieve a summary of all hyperparameters
        params = experiment.get_parameters_summary()
        if not params:
            logging.warning("No hyperparameters found for this experiment.")
            return

        # Convert the list of parameter dictionaries to a more readable single dictionary
        hyperparameters = {param['name']: param['valueCurrent'] for param in params}

        output_filename = output_dir / "hyperparameters.json"

        # Save the hyperparameters to a JSON file
        with open(output_filename, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        logging.info(f"Successfully downloaded and saved hyperparameters to: {output_filename}")

    except Exception as e:
        logging.error(f"Failed to download hyperparameters. Details: {e}")

def download_experiment_data(experiment_id: str):
    """
    Connects to Comet.ml, fetches a specific experiment, and downloads
    pre-defined assets and metrics into a dedicated folder.

    Args:
        experiment_id (str): The unique key of the Comet.ml experiment.
    """
    logging.info("Initializing Comet API...")
    try:
        # It's recommended to have your API key set in your environment variables
        api = comet_ml.API()
    except Exception as e:
        logging.error(f"Could not initialize Comet API. Is your API key set correctly? Details: {e}")
        return

    logging.info(f"Fetching experiment with ID: {experiment_id}")
    try:
        experiment = api.get_experiment_by_key(experiment_id)
    except Exception:
        logging.error(f"Experiment with ID '{experiment_id}' not found.")
        return

    # Create a directory named after the experiment ID to store all files
    DATA_DIR = Path("_distributions")
    output_dir = DATA_DIR / experiment_id
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    

    # Download the specified assets, metrics, and parameters
    download_assets(experiment, output_dir)
    download_metrics(experiment, output_dir)
    download_parameters(experiment, output_dir)

    logging.info("\n--- Download process finished! ---")
    logging.info(f"All available files have been saved in the directory: '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download key assets (reward-to-go, final_reward) and metrics (reward mean, reward std) from a Comet.ml experiment."
    )
    parser.add_argument(
        "experiment_id", 
        type=str, 
        help="The ID (key) of the Comet.ml experiment to download data from."
    )
    args = parser.parse_args()
    
    download_experiment_data(args.experiment_id)