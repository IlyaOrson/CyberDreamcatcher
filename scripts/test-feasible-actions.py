"""
Test the success rate of all feasible actions in the environment.

This script initializes a GraphEnv, iterates through all feasible actions,
and executes each action multiple times to determine its success rate.
The results are then plotted as a heatmap.

Example Usage:
    python scripts/test-feasible-actions.py

MAIN FINDING:

IMAGE_TO_VALID_DECOYS = {
    "Gateway": ["DecoyApache", "DecoyHarakaSMPT", "DecoyTomcat", "DecoyVsftpd"],
    "Velociraptor_Server": ["DecoyApache", "DecoyHarakaSMPT", "DecoyTomcat", "DecoyVsftpd"],
    "OP_Server": ["DecoyApache", "DecoyHarakaSMPT", "DecoyTomcat", "DecoyVsftpd"],
    "Internal": ["DecoyFemitter"],
    "windows_user_host1": ["DecoyApache", "DecoySmss", "DecoySvchost", "DecoyTomcat"],
    "windows_user_host2": ["DecoyApache", "DecoyFemitter", "DecoySSHD", "DecoyTomcat"],
    "linux_user_host1": ["DecoySSHD", "DecoyVsftpd"],
    "linux_user_host2": ["DecoyVsftpd"],
}
"""

import logging
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from cyberdreamcatcher.env import GraphEnv

# Configure logger
LOGGER = logging.getLogger(__name__)


def plot_action_success_heatmap(
    success_matrix, host_names, action_names, output_path, show=False, block=False
):
    """Plots a heatmap of action success rates."""
    fig, ax = plt.subplots(figsize=(14, 12))
    # Set vmin=0 and vmax=1 for the color scale.
    im = ax.imshow(success_matrix, vmin=0, vmax=1, cmap="viridis")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(action_names)))
    ax.set_yticks(np.arange(len(host_names)))
    ax.set_xticklabels(action_names)
    ax.set_yticklabels(host_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate")

    ax.set_title("Action Success Rates")
    fig.tight_layout()
    if show:
        plt.show(block=block)
    fig.savefig(output_path)
    LOGGER.info(f"Saved action success rate heatmap to {output_path}")


def print_success_report(success_counts, num_trials, output_path=None):
    """Prints a human-readable report of action success rates using rich.table."""
    report = defaultdict(list)
    for (host, action), count in success_counts.items():
        if count / num_trials > 0:
            report[action].append(host or "Global")

    table = Table(title="Action Success Report")
    table.add_column("Action", justify="left", style="cyan", no_wrap=True)
    table.add_column("Successfully Executed On", justify="left", style="magenta")

    for action, hosts in sorted(report.items()):
        table.add_row(action, ", ".join(sorted(hosts)))

    console = Console()
    console.print(table)

    if output_path:
        with open(output_path, "w") as f:
            file_console = Console(file=f)
            file_console.print(table)
        LOGGER.info(f"Saved action success report to {output_path}")


@dataclass
class Cfg:
    """Configuration for the GraphEnv."""

    scenario: str = "Scenario2"
    seed: int = 42
    num_trials: int = 3


# Registering the Config class with the expected name 'cfg'.
cs = ConfigStore.instance()
cs.store(name="config", node=Cfg)


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: Cfg) -> None:
    """Main function to test feasible actions and plot success rates."""
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    LOGGER.info(f"Output directory: {output_dir}")

    env = GraphEnv(scenario=cfg.scenario)

    num_trials = cfg.num_trials
    success_counts = defaultdict(int)

    # Get feasible actions and sort them for consistent ordering
    feasible_actions = sorted(env.feasible_actions, key=lambda x: (x[0] or "", x[1]))

    LOGGER.info(
        f"Testing {len(feasible_actions)} feasible actions, {num_trials} trials each..."
    )

    for host_name, action_name in tqdm(feasible_actions, desc="Testing Actions"):
        for _ in range(num_trials):
            env.reset()
            action_tuple = env.action_name_to_idx(host_name, action_name)
            env.step(action_tuple)
            if env.previous_action.success:
                success_counts[(host_name, action_name)] += 1

    print_success_report(
        success_counts, num_trials, output_path=output_dir / "action_success_report.txt"
    )

    # Create a matrix to hold success rates
    host_names = ["Global"] + env.host_names
    action_names = env.action_names
    host_map = {name: i for i, name in enumerate(host_names)}
    action_map = {name: i for i, name in enumerate(action_names)}

    success_matrix = np.full((len(host_names), len(action_names)), -1.0)

    for (host_name, action_name), count in success_counts.items():
        rate = count / num_trials
        action_idx = action_map[action_name]
        # Determine the host index, using 'Global' for None
        host_idx = host_map[host_name or "Global"]
        success_matrix[host_idx, action_idx] = rate

    plot_action_success_heatmap(
        success_matrix=success_matrix,
        host_names=host_names,
        action_names=action_names,
        output_path=output_dir / "action_success_rates.png",
        show=True,
        block=True,
    )


if __name__ == "__main__":
    main()
