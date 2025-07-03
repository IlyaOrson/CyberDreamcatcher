from pathlib import Path
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = Path("_distributions")

def load_histograms_to_dataframe(json_path: Path) -> pd.DataFrame:
    """
    Loads Comet ML v2 histogram data and processes it into a tidy DataFrame.
    """
    print(f"Loading and processing data from: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    processed_data = []
    for record in data["histograms"]:
        training_step = record["step"]
        hist_meta = record["histogram"]

        start = hist_meta["start"]
        step_param = hist_meta["step"]
        offset = hist_meta.get("offset", 0)

        for index, count in hist_meta["index_values"]:
            value = start * (step_param ** (index + offset))
            for _ in range(count):
                processed_data.append({"step": training_step, "value": value})

    if not processed_data:
        print("Warning: No data was processed. The resulting DataFrame is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(processed_data)
    df["step"] = pd.Categorical(
        df["step"], ordered=True, categories=sorted(df["step"].unique())
    )

    print(f"Successfully created DataFrame with {len(df)} observations.")
    print(f"Unique steps found: {df['step'].unique().tolist()}")
    return df


def create_distribution_plot(
    df: pd.DataFrame,
    plot_type: str,
    xlabel: str,
    ylabel: str,
    output_file: str = None,
    palette: str = "viridis",
    context: str = "talk",
    fig_size: tuple[int, int] = (10, 6),
    dpi: int = 300,
    clip_to_zero: bool = False,
    density_cap_value: float = None,
    use_colorbar: bool = False,
    use_custom_x_ticks: bool = False,
):
    """Generates and saves a publication-quality plot of the evolving distribution."""
    if df.empty:
        print("Cannot generate plot: DataFrame is empty.")
        return

    print(f"Generating plot of type: '{plot_type}' with context '{context}'...")
    sns.set_theme(context=context, style="whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    if plot_type == "kde-overlap":
        kde_kwargs = {
            "data": df,
            "x": "value",
            "hue": "step",
            "fill": True,
            "alpha": 0.3,
            "common_norm": False,
            "palette": palette,
            "ax": ax,
        }
        if clip_to_zero:
            kde_kwargs["clip"] = (-float("inf"), 0)

        sns.kdeplot(**kde_kwargs)

        if density_cap_value is not None:
            ax.set_ylim(bottom=0, top=density_cap_value)

        if use_colorbar:
            # Remove original legend and add a custom colorbar at the top
            if ax.get_legend():
                ax.get_legend().remove()

            min_step = df["step"].cat.categories.min()
            max_step = df["step"].cat.categories.max()
            norm = plt.Normalize(min_step, max_step)
            sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
            sm.set_array([])

            # Create a slimmer, more compact colorbar at the bottom
            cbar = fig.colorbar(
                sm,
                ax=ax,
                orientation="horizontal",
                location="bottom",
                pad=0.1,  # Slightly more padding for bottom placement
                shrink=0.8,
                aspect=40,
                extend="both",
            )
            
            # Make the colorbar thinner
            cbar.ax.tick_params(size=0, labelsize='small', pad=2)  # Smaller font size and padding for labels
            cbar.outline.set_visible(False)  # Hide the colorbar frame
            
            # Adjust colorbar position and size (moved to bottom)
            cbar.ax.set_position([0.1, 0.05, 0.8, 0.02])  # [left, bottom, width, height]
            
            # Set custom ticks and labels for the extremes
            cbar.set_ticks([min_step, max_step])
            cbar.set_ticklabels(["Unoptimised", "Optimised"])
            
            # Tight layout with minimal padding, accounting for bottom colorbar
            fig.tight_layout(pad=0.1, rect=[0, 0.08, 1, 0.95])  # Adjusted rect for bottom colorbar
        else:
            fig.tight_layout(pad=0.2)

        if use_custom_x_ticks:
            xmin, xmax = ax.get_xlim()
            # Adjust label positions for a balanced look
            tick_positions = [xmin + (xmax - xmin) * 0.3, xmin + (xmax - xmin) * 0.7]
            tick_labels = ["Operational server Impact", "Healthy operation server"]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize='medium')
            ax.set_xlabel("", labelpad=5)  # Remove x-label and add small padding
            ax.set_yticks([])  # Remove y-axis ticks
            ax.grid(False)
            sns.despine(ax=ax, left=True, bottom=True, trim=True)
            # Remove y-label completely for final reward plot
            ax.set_ylabel("")
        else:
            ax.set_xlabel(xlabel or "Value")
            ax.set_ylabel(ylabel or "Density")

    elif plot_type == "facet":
        plt.close(fig)  # Close the unused figure for facet plot
        g = sns.FacetGrid(
            df,
            col="step",
            hue="step",
            col_wrap=min(4, len(df.step.unique())),
            sharex=True,
            sharey=False,
            palette=palette,
        )
        g.map(sns.histplot, "value", kde=True)
        g.set_axis_labels(xlabel or "Value", ylabel or "Count / Density")
        g.set_titles("Step {col_name}")
        plt.tight_layout(pad=0.2)

    if output_file:
        print(f"Saving plot to {output_file} (DPI: {dpi})...")
        # Use tight layout with minimal padding
        plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
        # Save with minimal padding and tight layout
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05, dpi=dpi)


def create_mean_reward_plot(
    experiment_id: str,
    output_file: str,
    context: str = "talk",
    fig_size: tuple[int, int] = (10, 6),
    dpi: int = 300,
    line_color: tuple[str, int] = ("muted", 4),
):
    """Generates a plot of mean reward over steps with 1 and 2 std dev confidence intervals."""
    print("\n--- Generating Mean Reward Progress Plot ---")
    mean_path = DATA_DIR / experiment_id / "reward_mean.json"
    std_path = DATA_DIR / experiment_id / "reward_std.json"

    def _load_metric(path):
        """Load metric data, handling both list and dict formats."""
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {int(item["step"]): float(item["metricValue"]) for item in data}
        elif isinstance(data, dict):
            return {int(k): float(v) for k, v in data.items()}
        raise TypeError(f"Unsupported data format in {path}: {type(data)}")

    try:
        mean_data = _load_metric(mean_path)
        std_data = _load_metric(std_path)
    except (FileNotFoundError, TypeError, KeyError, ValueError) as e:
        print(f"Warning: Could not load or parse metric data: {e}. Skipping mean reward plot.")
        return

    df = pd.DataFrame({"step": list(mean_data.keys()), "mean": list(mean_data.values())})
    std_df = pd.DataFrame({"step": list(std_data.keys()), "std": list(std_data.values())})
    df = pd.merge(df, std_df, on="step").sort_values("step")

    if df.empty:
        print("Warning: Metric data is empty after processing. Skipping mean reward plot.")
        return

    # calculate max possible value for clipping
    max_reward = df["mean"].max()
    min_std = df["std"].min()
    clip_max = min(max_reward + min_std / 2, 0)

    df["std_1_upper"] = (df["mean"] + df["std"]).clip(upper=clip_max)
    df["std_1_lower"] = df["mean"] - df["std"]
    df["std_2_upper"] = (df["mean"] + (2 * df["std"])).clip(upper=clip_max)
    df["std_2_lower"] = df["mean"] - (2 * df["std"])

    sns.set_theme(context=context, style="whitegrid")
    fig, ax = plt.subplots(figsize=fig_size)

    # Get color from palette
    palette_name, color_idx = line_color
    color = sns.color_palette(palette_name)[color_idx]
    
    # Plot 2-std dev shadow (lighter) - using same color as line but more transparent
    ax.fill_between(
        df["step"],
        df["std_2_lower"],
        df["std_2_upper"],
        color=color,
        alpha=0.15,  # Slightly more transparent than before
        label="±2 std dev",
    )
    # Plot 1-std dev shadow (darker) - using same color as line but semi-transparent
    ax.fill_between(
        df["step"],
        df["std_1_lower"],
        df["std_1_upper"],
        color=color,
        alpha=0.3,  # Slightly more transparent than before
        label="±1 std dev",
    )
    
    # Plot the mean line with the specified color
    sns.lineplot(data=df, x="step", y="mean", ax=ax, color=color, linewidth=2.5, label="Mean Reward")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    
    # Save the figure
    print(f"Saving mean reward plot to {output_file} (DPI: {dpi})...")
    plt.savefig(output_file, bbox_inches="tight", dpi=dpi)
    
    # Close the figure to free memory
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot evolving distributions from a Comet ML JSON histogram file for publication.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- File and Plotting Arguments ---
    plot_group = parser.add_argument_group("Plotting Parameters")
    plot_group.add_argument(
        "experiment_id",
        type=str,
        help="The Comet ML experiment ID. The script expects to find 'reward-to-go_.json' inside a directory with this name.",
    )
    plot_group.add_argument(
        "--plot-type",
        type=str,
        default="kde-overlap",
        choices=["kde-overlap", "facet"],
        help="Type of plot to generate (default: kde-overlap).\n"
        " - kde-overlap: Overlapping density plots for each step.\n"
        " - facet: A grid of histograms, one for each step.",
    )
    plot_group.add_argument(
        "--xlabel", type=str, default="Reward-to-Go", help="Label for the X-axis."
    )
    plot_group.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Label for the Y-axis (uses sensible defaults if not set).",
    )

    plot_group.add_argument(
        "--palette",
        type=str,
        default="plasma",
        help="Seaborn color palette for the main plot (e.g., 'viridis', 'plasma', 'crest').",
    )
    plot_group.add_argument(
        "--context",
        type=str,
        default="talk",
        choices=["paper", "notebook", "talk", "poster"],
        help="The plotting context to use, affecting font sizes and line widths (default: talk).",
    )
    plot_group.add_argument(
        "--fig-size",
        type=str,
        default="10,6",
        help="Figure size as 'width,height' in inches (e.g., '10,6').",
    )
    plot_group.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="The resolution of the output image in dots per inch (default: 300).",
    )
    plot_group.add_argument(
        "--no-cap-density",
        dest="cap_density",
        action="store_false",
        help="Disable the default capping of the main KDE plot's y-axis at 0.6.",
    )

    # --- Sampling Arguments ---
    sampling_group = parser.add_argument_group("Distribution Sampling")
    sampling_group.add_argument(
        "--plot-single-step",
        type=int,
        default=None,
        metavar="STEP",
        help="Plot only a single, specific step, ignoring other sampling flags.",
    )
    sampling_group.add_argument(
        "--sample-every",
        type=int,
        default=4,
        metavar="N",
        help="Plot every N-th step (default: 4). Ignored if --plot-single-step is used.",
    )
    sampling_group.add_argument(
        "--no-first-last",
        action="store_true",
        help="Disable the default behavior of always including the first and last steps when sampling. Ignored if --plot-single-step is used.",
    )

    # --- Plot Styling Arguments ---
    style_group = parser.add_argument_group("Plot Styling")
    style_group.add_argument(
        "--line-color",
        type=str,
        default="muted,4",
        help="Line color as 'palette,index' (e.g., 'muted,4' or 'viridis,2'). Default: 'muted,4'.",
    )

    # --- Scaling Arguments ---
    scaling_group = parser.add_argument_group("Data Scaling")
    scaling_group.add_argument(
        "--no-scale-per-step",
        dest="scale_per_step",
        action="store_false",
        help="If set, disables the default behavior of scaling each step's distribution to match the recorded mean/std dev.",
    )

    args = parser.parse_args()

    try:
        fig_size_tuple = tuple(map(int, args.fig_size.split(',')))

        # 1. Generate mean reward plot
        mean_reward_output_file = DATA_DIR / args.experiment_id / "mean_reward.png"
        # Parse line color argument
        try:
            palette_name, color_idx = args.line_color.split(',')
            line_color = (palette_name.strip(), int(color_idx))
        except (ValueError, AttributeError):
            print(f"Warning: Invalid line color format '{args.line_color}'. Using default ('muted', 4).")
            line_color = ("muted", 4)
            
        create_mean_reward_plot(
            experiment_id=args.experiment_id,
            output_file=mean_reward_output_file,
            context=args.context,
            fig_size=fig_size_tuple,
            dpi=args.dpi,
            line_color=line_color,
        )

        # 2. Load and process main distribution data
        json_path = DATA_DIR / args.experiment_id / "reward-to-go_.json"
        dataframe = load_histograms_to_dataframe(json_path)
        if dataframe.empty:
            print("Main reward data file was empty. Skipping distribution plots.")
            return

        # 3. Determine which steps to plot
        all_steps = dataframe["step"].cat.categories.tolist()
        steps_to_plot = all_steps

        if args.plot_single_step is not None:
            if args.plot_single_step not in all_steps:
                raise ValueError(
                    f"Step {args.plot_single_step} not found. Available steps: {all_steps}"
                )
            steps_to_plot = [args.plot_single_step]
            print(f"\nPlotting single specified step: {steps_to_plot}")

        elif args.sample_every > 1 and len(all_steps) > 2:
            steps_to_plot_set = set(all_steps[:: args.sample_every])
            if not args.no_first_last:
                steps_to_plot_set.add(all_steps[0])
                steps_to_plot_set.add(all_steps[-1])

            steps_to_plot = sorted(list(steps_to_plot_set))
            print(
                f"\nSampling enabled. Plotting {len(steps_to_plot)} of {len(all_steps)} total steps: {steps_to_plot}"
            )

        # 4. Determine and apply scaling
        if args.scale_per_step:
            print("\n--- Scaling Data Per Step ---")

            def load_metric_data(path):
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {int(k): float(v) for k, v in data.items()}
                if isinstance(data, list):
                    return {int(item["step"]): float(item["metricValue"]) for item in data}
                raise TypeError(f"Unsupported data format in {path}: {type(data)}")

            mean_path = DATA_DIR / args.experiment_id / "reward_mean.json"
            std_path = DATA_DIR / args.experiment_id / "reward_std.json"

            try:
                reward_means = load_metric_data(mean_path)
                reward_stds = load_metric_data(std_path)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Could not find scaling data file: {e.filename}. Ensure 'reward_mean.json' and 'reward_std.json' are present."
                )

            def scale_group(group):
                step = int(group.name)
                target_mean = reward_means.get(step)
                target_std = reward_stds.get(step)
                if target_mean is None or target_std is None:
                    if step in steps_to_plot:
                        print(f"Warning: Scaling data not found for step {step}. It will not be scaled.")
                    return group
                current_mean = group.mean()
                current_std = group.std()
                scale_factor = target_std / current_std if current_std > 0 else 1.0
                shift = target_mean - (current_mean * scale_factor)
                if step in steps_to_plot:
                    print(f"  - Step {step}: Applied scale={scale_factor:.4e}, shift={shift:.4e}")
                return (group * scale_factor) + shift

            dataframe["value"] = dataframe.groupby("step", observed=False)["value"].transform(scale_group)
            print("---------------------------")

        # 5. Filter dataframe to only include the steps we want to plot
        if len(steps_to_plot) < len(all_steps):
            print(f"\n--- Filtering for steps: {steps_to_plot} ---")
            dataframe = dataframe[dataframe["step"].isin(steps_to_plot)].copy()
            dataframe["step"] = dataframe["step"].cat.remove_unused_categories()
            print(f"Dataframe now has {len(dataframe)} observations.")
            print("---------------------------------")

        # 6. If scaling is active, discard positive values for plotting
        if args.scale_per_step:
            print("\n--- Discarding positive values for plotting ---")
            if not dataframe.empty:
                original_counts = dataframe.groupby("step", observed=True).size()
                neg_dataframe = dataframe[dataframe["value"] < 0].copy()
                final_counts = neg_dataframe.groupby("step", observed=True).size().reindex(original_counts.index, fill_value=0)
                
                stats = pd.DataFrame({"original": original_counts, "final": final_counts})
                stats["dropped"] = stats["original"] - stats["final"]
                stats["percentage_dropped"] = (stats["dropped"] / stats["original"]) * 100

                for step, row in stats.iterrows():
                    print(f"  - Step {step}: Dropped {row['dropped']} of {row['original']} points ({row['percentage_dropped']:.2f}%) because they were >= 0.")
                
                dataframe = neg_dataframe
                if not dataframe.empty:
                    dataframe["step"] = dataframe["step"].cat.remove_unused_categories()
                else:
                    print("Warning: All data points were filtered out. Plot will be empty.")
            else:
                print("Dataframe is empty before filtering, skipping.")
            print("---------------------------------------------")

        # 7. Create and save the main plot
        output_file = DATA_DIR / args.experiment_id / "reward-to-go.png"
        create_distribution_plot(
            df=dataframe,
            plot_type=args.plot_type,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            output_file=output_file,
            palette=args.palette,
            context=args.context,
            fig_size=fig_size_tuple,
            dpi=args.dpi,
            clip_to_zero=args.scale_per_step,
            density_cap_value=0.6 if args.plot_type == "kde-overlap" and args.cap_density else None,
        )

        # 8. Plot and save final reward distribution if available
        final_reward_json_path = DATA_DIR / args.experiment_id / "final_reward_.json"
        if final_reward_json_path.exists():
            print(f"\n--- Found Final Reward Data at {final_reward_json_path} ---")
            final_reward_df = load_histograms_to_dataframe(final_reward_json_path)

            if not final_reward_df.empty:
                final_reward_output_file = DATA_DIR / args.experiment_id / "final_reward.png"
                create_distribution_plot(
                    df=final_reward_df,
                    plot_type="kde-overlap",
                    xlabel="Final Reward",
                    ylabel="Density",
                    output_file=final_reward_output_file,
                    palette="Spectral",
                    context=args.context,
                    fig_size=fig_size_tuple,
                    dpi=args.dpi,
                    clip_to_zero=False,  # Never scale or clip this plot
                    density_cap_value=None,  # Never cap the final reward plot
                    use_colorbar=True,
                    use_custom_x_ticks=True,
                )
            else:
                print("Final reward data file was empty. Skipping plot.")
        else:
            print("\n--- No final_reward_.json found. Skipping final reward plot. ---")

    except (FileNotFoundError, ValueError, KeyError, IndexError) as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    if plt.get_fignums():
        print("Displaying plots...")
        plt.show()
    else:
        print("No plots were generated.")


if __name__ == "__main__":
    main()
