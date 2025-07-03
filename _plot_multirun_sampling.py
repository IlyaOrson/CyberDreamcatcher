import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def get_scenario_from_hydra_overrides(run_dir: Path) -> str:
    """Extracts the scenario name from the Hydra overrides file."""
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.is_file():
        return "Unknown"
    overrides = OmegaConf.load(overrides_path)
    for override in overrides:
        if override.startswith("scenario="):
            return override.split("=")[1]
    return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Plot rewards-to-go from a Hydra multirun directory."
    )
    parser.add_argument(
        "multirun_path",
        type=Path,
        help="Path to the Hydra multirun directory (e.g., 'multirun/YYYY-MM-DD/HH-MM-SS').",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="violin",
        choices=["violin", "violin-clipped", "boxplot", "boxenplot", "stripplot"],
        help="Type of plot to generate.",
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=0.02,
        help="Quantile to clip the y-axis at for 'violin-clipped' plot type.",
    )
    parser.add_argument(
        "--timestep-focus",
        type=str,
        default="last",
        choices=["first", "last"],
        help="Focus on the first or last timestep for the plot.",
    )
    parser.add_argument(
        "--seaborn-style",
        type=str,
        default="whitegrid",
        help="Seaborn plot style.",
    )
    parser.add_argument(
        "--seaborn-palette",
        type=str,
        default="muted",
        help="Seaborn color palette.",
    )
    parser.add_argument(
        "--seaborn-context",
        type=str,
        default="talk",
        choices=["paper", "notebook", "talk", "poster"],
        help="Seaborn plot context.",
    )
    args = parser.parse_args()

    if not args.multirun_path.is_dir():
        print(f"Error: Directory not found at {args.multirun_path}")
        return

    csv_files = list(args.multirun_path.glob("*/rewards_to_go.csv"))
    if not csv_files:
        print(
            f"Error: No 'rewards_to_go.csv' files found in subdirectories of {args.multirun_path}"
        )
        return

    all_dfs = []
    for csv_file in csv_files:
        run_dir = csv_file.parent
        scenario_name = get_scenario_from_hydra_overrides(run_dir)
        df = pd.read_csv(csv_file)
        df["Scenario"] = scenario_name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Relabel policy for clarity in legend
    combined_df["Policy"] = combined_df["Policy"].replace(
        {"Trained": "Trained on Scenario 2"}
    )

    # Custom scenario label formatting for the plot
    def format_scenario_label(label):
        # Replace underscores with spaces first
        label = label.replace("_", " ")
        # If the label contains both 'Scenario2' and 'User', remove 'Scenario2'
        if "Scenario2" in label and "User" in label:
            return label.replace("Scenario2", "").strip()
        return label

    combined_df["Scenario"] = combined_df["Scenario"].apply(format_scenario_label)

    y_label = "Reward-to-Go"
    if args.timestep_focus == "first":
        focused_df = combined_df[
            combined_df["timestep"] == combined_df["timestep"].min()
        ]
    else:  # last
        focused_df = combined_df[
            combined_df["timestep"] == combined_df["timestep"].max()
        ]
        y_label = "Final Reward"

    # Plotting
    sns.set_theme(
        style=args.seaborn_style,
        palette=args.seaborn_palette,
        context=args.seaborn_context,
    )
    fig, ax = plt.subplots(figsize=(16, 9))

    plot_params = {
        "data": focused_df,
        "x": "Scenario",
        "y": "reward_to_go",
        "hue": "Policy",
        "ax": ax,
    }

    if args.plot_type == "violin":
        sns.violinplot(**plot_params, split=True)
    elif args.plot_type == "violin-clipped":
        sns.violinplot(**plot_params, split=True)
        min_val = focused_df["reward_to_go"].quantile(args.clip_quantile)
        max_val = focused_df["reward_to_go"].quantile(1 - args.clip_quantile)
        ax.set_ylim(min_val, max_val)
    elif args.plot_type == "boxplot":
        sns.boxplot(**plot_params)
    elif args.plot_type == "boxenplot":
        sns.boxenplot(**plot_params)
    elif args.plot_type == "stripplot":
        sns.stripplot(**plot_params, dodge=True)

    # Final plot adjustments for publication
    ax.set_xlabel("Scenario")
    ax.set_ylabel(y_label)
    plt.xticks(rotation=0)  # Horizontal x-axis labels

    # Move legend to top, outside the plot
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.get_legend().remove()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(focused_df["Policy"].unique()),
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for top legend

    # Save the plot
    output_filename = args.multirun_path / f"comparison_plot_{args.timestep_focus}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")


if __name__ == "__main__":
    main()
