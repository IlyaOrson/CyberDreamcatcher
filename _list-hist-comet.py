import comet_ml
import argparse
import sys
from collections import Counter

def list_available_histograms(experiment_key: str):
    """
    Connects to Comet.ml, finds an experiment by its key, and lists
    all unique histogram names logged to it.

    Args:
        experiment_key (str): The unique key of the Comet experiment.
    """
    try:
        # 1. Authenticate and initialize the Comet API
        print("Authenticating with Comet.ml...")
        comet_ml.login()
        api = comet_ml.API()
        print("Authentication successful.")

        # 2. Get the experiment using its key
        print(f"Fetching experiment with key: {experiment_key}")
        experiment = api.get_experiment_by_key(experiment_key)

        if not experiment:
            print(f"Error: Experiment with key '{experiment_key}' not found.")
            sys.exit(1)

        print(f"Successfully retrieved experiment: {experiment.name} ({experiment.workspace}/{experiment.project_name})")

        # 3. Get all histogram assets from the experiment
        print("Searching for histogram assets...")
        asset_list = experiment.get_asset_list(asset_type="histogram_combined_3d")

        if not asset_list:
            print("\nNo histogram assets were found in this experiment.")
            return

        # 4. Collect all histogram names
        # The name you log with experiment.log_histogram_3d() becomes the 'fileName'
        all_names = [asset.get("fileName") for asset in asset_list if asset.get("fileName")]
        
        if not all_names:
            print("\nFound histogram assets, but they have no names. This is unusual.")
            return

        # Use a set for uniqueness, then sort for consistent output
        unique_names = sorted(list(set(all_names)))
        
        # Optional: Count how many times each was logged
        name_counts = Counter(all_names)

        # 5. Display the results
        print("\n✅ Available Histograms:")
        print("-----------------------")
        for name in unique_names:
            count = name_counts[name]
            print(f"  - {name} (logged {count} times)")
        
        print("\nTip: Use one of these names with the plotting script.")

    except comet_ml.exceptions.CometRestApiException as e:
        print(f"\nAn error occurred communicating with the Comet API: {e}")
        print("Please check your API key, experiment key, and that the workspace/project is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List all available histogram names from a Comet.ml experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        help="The unique key of the experiment (found in the experiment URL)."
    )
    args = parser.parse_args()
    
    list_available_histograms(args.experiment)