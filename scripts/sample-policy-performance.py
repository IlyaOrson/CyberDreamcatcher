from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging


import hydra
from hydra.core.config_store import ConfigStore
import pandas as pd

from cyberdreamcatcher.utils import (
    load_trained_weights,
    long_format_dataframe,
)
from cyberdreamcatcher.sampler import EpisodeSampler


# Disable specific loggers
logging.getLogger("CybORGLog-Process").setLevel(logging.CRITICAL)


@dataclass
class Cfg:
    scenario: Optional[str] = None
    policy_weights: Optional[str] = None
    latent_node_dim: int = 8
    actor_heads: int = 3
    seed: int = 0
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

    assert (
        cfg.policy_weights or cfg.scenario
    ), "Please provide either 'scenario' or 'policy_weights'."

    policy_weights = None
    if cfg.policy_weights:
        policy_weights, logged_cfg = load_trained_weights(
            cfg.policy_weights, weights_only=False
        )
        if logged_cfg:
            # If any of the logged policy parameters are different from the provided config,
            # print a warning and use the logged config parameters.
            for key in ["scenario", "latent_node_dim", "actor_heads"]:
                provided_value = getattr(cfg, key)
                logged_value = getattr(logged_cfg, key)
                if provided_value and provided_value != logged_value:
                    print(
                        f"Warning: {key} in logged config ({logged_value}) "
                        f"differs from provided {key} ({provided_value}). "
                        f"Using logged {key}."
                    )
                    setattr(cfg, key, logged_value)

    dfs = []
    random_sampler = EpisodeSampler(
        cfg.seed,
        cfg.scenario,
        cfg.episode_length,
        latent_node_dim=cfg.latent_node_dim,
        actor_heads=cfg.actor_heads,
        policy_weights=None,
        num_jobs=cfg.num_jobs,
    )
    random_stacked_rewards_to_go, _ = random_sampler.sample_episodes(
        num_episodes=cfg.num_episodes
    )
    df_long = long_format_dataframe(random_stacked_rewards_to_go)
    df_long["Policy"] = "Random"
    dfs.append(df_long)

    if policy_weights:
        loaded_sampler = EpisodeSampler(
            cfg.seed,
            cfg.scenario,
            cfg.episode_length,
            latent_node_dim=cfg.latent_node_dim,
            actor_heads=cfg.actor_heads,
            policy_weights=policy_weights,
            num_jobs=cfg.num_jobs,
        )
        loaded_stacked_rewards_to_go, _ = loaded_sampler.sample_episodes(
            num_episodes=cfg.num_episodes
        )
        df_long = long_format_dataframe(loaded_stacked_rewards_to_go)
        df_long["Policy"] = "Trained"
        dfs.append(df_long)

    df = pd.concat(dfs)

    data_filename = Path(output_dir) / "rewards_to_go.csv"
    df.to_csv(data_filename, index=False)
    print(f"Saved results in {data_filename}")


main()
