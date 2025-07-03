import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def plot_heatmap_distribution(
    json_file, colormap, log_scale, y_tick_freq, title
):
    """
    Loads Comet distribution data and creates a 2D Histogram (Heatmap).
    """
    # --- 1. Load and Preprocess Data ---
    print("Loading and preprocessing data...")
    try:
        with open(json_file, "r") as f:
            data_dict = json.load(f)
        histograms_data = data_dict["histograms"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Error: Could not read or parse file '{json_file}'.\nDetails: {e}")
        return

    # Create a list of dictionaries to build the DataFrame
    records = []
    for step_data in histograms_data:
        for bin_index, count in step_data["histogram"]["index_values"]:
            records.append({"step": step_data["step"], "value": bin_index, "count": count})

    if not records:
        print("Error: No data points found in the JSON file.")
        return
        
    df = pd.DataFrame(records)
    print(f"Successfully created DataFrame with {len(df)} records.")

    # --- 2. Pivot Data for the Heatmap ---
    # We need to transform the data into a grid:
    # Rows = Steps, Columns = Bin Values, Cell Content = Counts
    print("Pivoting data for heatmap...")
    heatmap_data = df.pivot_table(
        index='step',
        columns='value',
        values='count',
        fill_value=0  # Fill missing combinations with 0 count
    )
    
    # Sort the index (steps) to ensure chronological order
    heatmap_data.sort_index(inplace=True)

    # --- 3. Create the Heatmap Plot ---
    # Set up the figure size
    plt.figure(figsize=(16, 10))

    # Use a logarithmic color scale if requested. This is great for data
    # where some counts are vastly larger than others.
    norm = LogNorm() if log_scale else None

    # Create the heatmap using seaborn
    ax = sns.heatmap(
        heatmap_data,
        cmap=colormap,
        norm=norm,
        cbar_kws={'label': 'Frequency (Count)'} # Label for the color bar
    )

    # --- 4. Style and Finalize the Plot ---
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("Bin Index (Represents Reward-to-Go)", fontsize=12)
    ax.set_ylabel("Training Step", fontsize=12)
    
    # Intelligently set y-axis ticks to prevent overcrowding
    step_labels = heatmap_data.index
    tick_positions = np.arange(len(step_labels))
    
    # Only show a label for every Nth step
    ax.set_yticks(tick_positions[::y_tick_freq] + 0.5) # Add 0.5 to center ticks
    ax.set_yticklabels(step_labels[::y_tick_freq])
    
    plt.xticks(rotation=45) # Rotate x-axis labels slightly for better fit
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Comet ML distribution data as a 2D heatmap.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("json_file", type=str, help="Path to the JSON file from Comet ML.")
    parser.add_argument("--colormap", type=str, default="viridis", help="Matplotlib colormap. 'viridis', 'plasma', 'inferno' are good choices.")
    parser.add_argument("--log_scale", action="store_true", help="Use a logarithmic scale for color intensity. Good for high-contrast data.")
    parser.add_argument("--y_tick_freq", type=int, default=5, help="Show a y-axis tick label for every N steps.")
    parser.add_argument("--title", type=str, default="Heatmap of Reward-to-Go Distribution over Training Steps", help="Main title for the plot.")

    args = parser.parse_args()
    plot_heatmap_distribution(args.json_file, args.colormap, args.log_scale, args.y_tick_freq, args.title)