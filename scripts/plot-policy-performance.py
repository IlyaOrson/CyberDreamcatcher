import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging


import hydra
from hydra.core.config_store import ConfigStore
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    assert (
        cfg.policy_weights or cfg.scenario
    ), "Please provide either 'scenario' or 'policy_weights'."

    policy_weights = None
    scenario = cfg.scenario
    if cfg.policy_weights:
        policy_weights, trained_scenario = load_trained_weights(cfg.policy_weights)
        print(f"Loaded policy trained on {trained_scenario}.")
        if trained_scenario != cfg.scenario:
            print("Overloading provided scenario.")
            scenario = trained_scenario
    print(f"Plotting performance on scenario {scenario}.")

    dfs = []
    random_sampler = EpisodeSampler(
        cfg.seed,
        cfg.scenario,
        cfg.episode_length,
        policy_weights=None,
        num_jobs=cfg.num_jobs,
    )
    random_stacked_rewards_to_go = random_sampler.sample_episodes(
        num_episodes=cfg.num_episodes
    )
    df_long = long_format_dataframe(random_stacked_rewards_to_go)
    df_long["Policy"] = "Random"
    dfs.append(df_long)

    if cfg.policy_weights:
        loaded_sampler = EpisodeSampler(
            cfg.seed,
            cfg.scenario,
            cfg.episode_length,
            policy_weights=policy_weights,
            num_jobs=cfg.num_jobs,
        )
        loaded_stacked_rewards_to_go = loaded_sampler.sample_episodes(
            num_episodes=cfg.num_episodes
        )
        df_long = long_format_dataframe(loaded_stacked_rewards_to_go)
        df_long["Policy"] = "Trained"
        dfs.append(df_long)

    df = pd.concat(dfs)

    data_filename = Path(output_dir) / "rewards_to_go.csv"
    df.to_csv(data_filename, index=False)
    print(f"Saved results in {data_filename}")

    df = downsample_dataframe(df, steps=[0,20,25,27,29])
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
