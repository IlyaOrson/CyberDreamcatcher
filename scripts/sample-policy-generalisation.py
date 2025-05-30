from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
import logging

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
import pandas as pd

from cyberdreamcatcher.utils import (
    load_trained_weights,
    long_format_dataframe,
)
from cyberdreamcatcher.sampler import EpisodeSampler


# Disable specific loggers
logging.getLogger("CybORGLog-Process").setLevel(logging.CRITICAL)
logging.getLogger("cyberdreamcatcher.utils").setLevel(logging.CRITICAL)


@dataclass
class Cfg:
    local_policies: Optional[List[str]] = None
    policy_weights: Optional[str] = None
    latent_node_dim: int = 8
    actor_heads: int = 3
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
    print(f"Working directory : {Path.cwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")

    assert cfg.local_policies, "Please provide the paths to trained local policies."

    run_name = f"Generalisation_random_policy_seed_{cfg.seed}"
    foreign_policy_weights = None
    if cfg.policy_weights is None:
        print("No policy given, random weights will be used.")
    else:
        policy_path = Path(cfg.policy_weights)
        assert policy_path.is_file()

        policy_dir = policy_path.parent
        logged_cfg_path = policy_dir / ".hydra" / "config.yaml"
        assert logged_cfg_path.is_file()
        logged_cfg = OmegaConf.load(logged_cfg_path)
        print("Configuration used to train loaded policy.")
        print(OmegaConf.to_yaml(logged_cfg))
        trained_scenario = logged_cfg.scenario
        foreign_policy_weights = torch.load(policy_path, weights_only=True)

    foreign_results = {}  # scenario --> loaded policy performance samples
    local_results = {}  # scenario --> locally trained policy performance samples
    for local_policy_path in cfg.local_policies:
        policy_weights, trained_scenario = load_trained_weights(local_policy_path)

        specialised_policy_sampler = EpisodeSampler(
            cfg.seed,
            trained_scenario,
            cfg.episode_length,
            policy_weights=policy_weights,
            latent_node_dim=logged_cfg.latent_node_dim,
            actor_heads=logged_cfg.actor_heads,
            num_jobs=cfg.num_jobs,
        )
        stacked_rewards_to_go, _ = specialised_policy_sampler.sample_episodes(
            num_episodes=cfg.num_episodes
        )
        local_results[trained_scenario] = stacked_rewards_to_go

        # Generalization samples
        foreign_policy_sampler = EpisodeSampler(
            cfg.seed,
            trained_scenario,
            cfg.episode_length,
            policy_weights=foreign_policy_weights,
            latent_node_dim=logged_cfg.latent_node_dim,
            actor_heads=logged_cfg.actor_heads,
            num_jobs=cfg.num_jobs,
        )
        stacked_rewards_to_go, _ = foreign_policy_sampler.sample_episodes(
            num_episodes=cfg.num_episodes
        )
        foreign_results[trained_scenario] = stacked_rewards_to_go

    print(f"Generalisation_policy_trained_on_{trained_scenario}_seed_{cfg.seed}")

    dfs = []
    for scenario, reward_array in local_results.items():
        df_long = long_format_dataframe(reward_array)
        df_long["Scenario"] = scenario
        df_long["Policy"] = "Local"
        dfs.append(df_long)
    for scenario, reward_array in foreign_results.items():
        df_long = long_format_dataframe(reward_array)
        df_long["Scenario"] = scenario
        df_long["Policy"] = "Foreign"
        dfs.append(df_long)

    df = pd.concat(dfs)

    data_filename = Path(output_dir) / "rewards_to_go.csv"
    df.to_csv(data_filename, index=False)
    print(f"Stored results in {data_filename}")


main()
