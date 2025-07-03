import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import seaborn as sns
from pathlib import Path

def process_file(json_path, smooth_window):
    """Reads a JSON file, processes it to get mean and std, and returns a smoothed DataFrame."""
    try:
        df = pd.read_json(json_path)
    except Exception as e:
        print(f"Error reading {json_path.name}: {e}")
        return None

    if 'type' in df.columns:
        df = df.drop(columns=["type"])

    try:
        agg_df = df.explode(["x", "y"]).astype({'x': float, 'y': float}).groupby('x')['y'].agg(['mean', 'std']).reset_index()
    except (KeyError, ValueError):
        print(f"Warning: Could not process {json_path.name}. Ensure it has 'x' and 'y' lists.")
        return None

    agg_df = agg_df.set_index('x')
    if len(agg_df.index) > 1:
        steps = np.diff(agg_df.index.sort_values())
        min_step = steps[steps > 0].min() if len(steps[steps > 0]) > 0 else 1
        full_index = pd.Index(np.arange(agg_df.index.min(), agg_df.index.max() + min_step, min_step), name='x')
        agg_df = agg_df.reindex(full_index).interpolate(method='linear')
    
    agg_df = agg_df.reset_index()
    agg_df['std'] = agg_df['std'].fillna(0)

    if smooth_window > 1:
        agg_df['mean'] = agg_df['mean'].rolling(window=smooth_window, min_periods=1, center=True).mean()
        agg_df['std'] = agg_df['std'].rolling(window=smooth_window, min_periods=1, center=True).mean()
    
    return agg_df

def main():
    parser = argparse.ArgumentParser(
        description='Plot mean rewards from a directory of JSON files against a baseline for publication.'
    )
    parser.add_argument('data_dir', type=str,
                        help='Path to the directory containing the JSON files.')
    parser.add_argument('--smooth', type=int, default=10,
                        help='Window size for moving average smoothing.')
    parser.add_argument('--show-2std', action='store_true',
                        help='Show the +-2 standard deviation shadow.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: Directory not found at {data_dir}")
        return

    baseline_path = data_dir / 'Scenario2.json'
    if not baseline_path.is_file():
        print(f"Error: Baseline file 'Scenario2.json' not found in {data_dir}")
        return

    print("Processing baseline file...")
    baseline_df = process_file(baseline_path, args.smooth)
    if baseline_df is None:
        print("Could not process baseline file. Aborting.")
        return

    plot_order = [
        'Scenario2 - User2 User4.json', 'Scenario2 + User5 User6.json',
        'Scenario2 - User4.json',       'Scenario2 + User5.json',
        'Scenario2 - User2.json',       'Scenario2 + User6.json'
    ]

    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", 8)
    baseline_color = palette[0]

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True, sharey=False)
    axes = axes.flatten()

    for i, filename in enumerate(plot_order):
        ax = axes[i]
        file_path = data_dir / filename
        comp_color = palette[i + 1]

        if not file_path.is_file():
            print(f"Warning: {filename} not found, skipping plot.")
            ax.set_visible(False)
            continue
        
        print(f"Processing {filename}...")
        comp_df = process_file(file_path, args.smooth)

        if comp_df is not None:
            # Plot baseline
            ax.plot(baseline_df['x'], baseline_df['mean'], label='Scenario 2', color=baseline_color, linewidth=2.5)
            ax.fill_between(baseline_df['x'], baseline_df['mean'] - baseline_df['std'], baseline_df['mean'] + baseline_df['std'], alpha=0.2, color=baseline_color)
            if args.show_2std:
                ax.fill_between(baseline_df['x'], baseline_df['mean'] - 2 * baseline_df['std'], baseline_df['mean'] + 2 * baseline_df['std'], alpha=0.1, color=baseline_color)

            # Plot comparison
            title = file_path.stem.replace('Scenario2', '').replace('+', ' Add').replace('-', ' Remove').strip()
            ax.plot(comp_df['x'], comp_df['mean'], label=title, color=comp_color, linewidth=2.5)
            ax.fill_between(comp_df['x'], comp_df['mean'] - comp_df['std'], comp_df['mean'] + comp_df['std'], alpha=0.2, color=comp_color)
            if args.show_2std:
                ax.fill_between(comp_df['x'], comp_df['mean'] - 2 * comp_df['std'], comp_df['mean'] + 2 * comp_df['std'], alpha=0.1, color=comp_color)
            
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=12, frameon=True)
            ax.tick_params(axis='both', which='major', labelsize=12)

    fig.supxlabel('Step', fontsize=18, fontweight='bold')
    fig.supylabel('Reward', fontsize=18, fontweight='bold')

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.97])
    output_filename = data_dir / "publication_comparison_grid.pdf"
    plt.savefig(output_filename, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Publication-quality plot saved to {output_filename}")
    plt.show()

if __name__ == '__main__':
    main()