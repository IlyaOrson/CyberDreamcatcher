import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging


import hydra
from hydra.core.config_store import ConfigStore
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns

from cyberdreamcatcher.utils import (
    load_trained_weights,
    long_format_dataframe,
    downsample_dataframe,
)
from cyberdreamcatcher.sampler import EpisodeSampler
from cyberdreamcatcher.plots import plot_joyplot

# sns.set_theme(style="ticks")
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Disable specific loggers
logging.getLogger("CybORGLog-Process").setLevel(logging.CRITICAL)
logging.getLogger("cyberdreamcatcher.utils").setLevel(logging.CRITICAL)


@dataclass
class Cfg:
    policy_weights: Optional[str] = None
    scenario: Optional[str] = "Scenario2"
    seed: int = 31415
    episode_length: int = 30
    num_episodes: int = 1000
    num_jobs: int = -1


# Registering the Config class with the expected name 'args'.
# https://hydra.cc/docs/tutorials/structured_config/minimal_example/
cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)


@hydra.main(version_base=None, config_name="args", config_path=None)
def main(cfg: Cfg):
    # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
    print(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")

    if cfg.policy_weights is None:
        print("No policy given, random weights will be used.")
        run_name = f"Performance_random_on_{cfg.scenario}_seed_{cfg.seed}"
        policy_weights = None
    else:
        policy_weights, trained_scenario = load_trained_weights(cfg.policy_weights)

        run_name = f"Performance_on_{cfg.scenario}_trained_on_{trained_scenario}_seed_{cfg.seed}"

    log_dir = Path(output_dir) / run_name

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in cfg.items()])),
    )

    print(f"Plotting performance on scenario {cfg.scenario}.")

    # Initialize with number of parallel jobs (-1 uses all available cores)
    sampler = EpisodeSampler(
        cfg.seed,
        cfg.scenario,
        cfg.episode_length,
        policy_weights=policy_weights,
        num_jobs=cfg.num_jobs,
    )

    # Sample episodes in parallel
    stacked_rewards_to_go = sampler.sample_episodes(num_episodes=cfg.num_episodes)

    writer.add_histogram(
        "distributions/reward-to-go",
        stacked_rewards_to_go[:, 0],
    )
    writer.add_histogram(
        "distributions/final-reward",
        stacked_rewards_to_go[:, -1],
    )

    df = long_format_dataframe(stacked_rewards_to_go)

    data_filename = Path(output_dir) / "rewards_to_go.csv"
    df.to_csv(data_filename, index=False)

    df = downsample_dataframe(df, step=5)
    plot_joyplot(df)

    plot_filename = Path(output_dir) / "joyplot.png"
    plt.savefig(
        plot_filename,
        dpi=300,
        bbox_inches="tight",
        # pad_inches=0.1,
    )
    print(f"Saved figure at {plot_filename}")


main()
