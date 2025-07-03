import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def plot_distributions(csv_path: Path, plot_type: str | None, clip_quantile: float, timestep_focus: str):
    """
    Reads performance data from a CSV and plots reward-to-go distributions.

    Args:
        csv_path (Path): The path to the input CSV file.
        plot_type (str | None): The type of plot to generate. If None, a default is chosen.
        clip_quantile (float): The lower quantile for clipping in 'violin-clipped' mode.
        timestep_focus (str): Focus on the 'first', 'last', or 'all' timesteps.
    """
    df = pd.read_csv(csv_path)

    # Determine the actual plot type if not specified
    if plot_type is None:
        if timestep_focus == 'first':
            plot_type = 'boxenplot'
        elif timestep_focus == 'last':
            plot_type = 'violin'
        else:  # 'all'
            plot_type = 'boxplot'

    title_suffix = ''
    if timestep_focus == 'first':
        step = df['timestep'].min()
        df = df[df['timestep'] == step]
        title_suffix = f' (First Timestep: {step})'
    elif timestep_focus == 'last':
        step = df['timestep'].max()
        df = df[df['timestep'] == step]
        title_suffix = f' (Last Timestep: {step})'

    y_var, y_label = 'reward_to_go', 'Reward-to-Go'
    title = f'{plot_type.replace("-", " ").title()} of Reward-to-Go vs. Timestep{title_suffix}'

    if plot_type == 'violin-clipped':
        clip_val = df['reward_to_go'].quantile(clip_quantile / 100)
        df['reward_to_go_clipped'] = df['reward_to_go'].clip(lower=clip_val)
        y_var = 'reward_to_go_clipped'
        y_label = f'Reward-to-Go (Clipped below {clip_quantile}th Quantile)'
        title = f'Clipped Violin Plot of Reward-to-Go vs. Timestep{title_suffix}'

    # Select the plotting function
    plot_funcs = {
        'violin': sns.violinplot,
        'violin-clipped': sns.violinplot,
        'boxplot': sns.boxplot,
        'boxenplot': sns.boxenplot,
        'stripplot': sns.stripplot
    }
    plot_func = plot_funcs.get(plot_type, sns.violinplot)

    plt.figure(figsize=(14, 8))
    plot_kwargs = {'data': df, 'x': 'timestep', 'y': y_var, 'hue': 'Policy'}
    if plot_type in ['violin', 'violin-clipped']:
        plot_kwargs.update({'split': True, 'inner': 'quart', 'linewidth': 1})
    elif plot_type == 'stripplot':
        plot_kwargs.update({'alpha': 0.7, 'jitter': True})

    plot_func(**plot_kwargs)

    sns.despine(left=True)
    plt.title(title)
    plt.xlabel('Timestep')
    plt.ylabel(y_label)

    output_path = csv_path.with_suffix('.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

def main():
    """Main function to parse arguments and run the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot reward-to-go distributions from a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "csv_path", type=Path,
        help="Path to the CSV file containing the performance data."
    )
    parser.add_argument(
        "--plot-type", type=str, default=None, 
        choices=["violin", "violin-clipped", "boxplot", "boxenplot", "stripplot"],
        help="Type of plot to generate. If not provided, a smart default is chosen."
    )
    parser.add_argument(
        "--clip-quantile", type=float, default=5.0,
        help="Lower quantile for clipping when using --plot-type violin-clipped."
    )
    parser.add_argument(
        "--timestep-focus", type=str, default='first', choices=["first", "last", "all"],
        help="Focus the plot on the first, last, or all timesteps."
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        print(f"Error: File not found at {args.csv_path}")
        return

    plot_distributions(args.csv_path, args.plot_type, args.clip_quantile, args.timestep_focus)

if __name__ == "__main__":
    main()
