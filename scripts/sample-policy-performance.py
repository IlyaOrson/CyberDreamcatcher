from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging


import hydra
from hydra.core.config_store import ConfigStore
import pandas as pd

from cyberdreamcatcher.utils import (
    get_policy_weights_and_config,
    long_format_dataframe,
    set_all_seeds,
)
from cyberdreamcatcher.sampler import EpisodeSampler


# Disable specific loggers
logging.getLogger("CybORGLog-Process").setLevel(logging.CRITICAL)


@dataclass
class Cfg:
    scenario: Optional[str] = None
    policy_weights: Optional[str] = None
    comet_experiment_key: Optional[str] = None
    comet_model_name: Optional[str] = None
    comet_model_step: Optional[int] = None
    latent_node_dim: int = 8
    actor_heads: int = 3
    seed: Optional[int] = None
    episode_length: int = 30
    num_episodes: int = 1000
    num_jobs: int = -1
    use_single_seed: bool = False


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

    policy_weights, logged_cfg = get_policy_weights_and_config(cfg, output_dir)

    assert (
        policy_weights is not None or cfg.scenario
    ), "Please provide either 'scenario', 'policy_weights', or 'comet_experiment_key'."

    if logged_cfg:
        # If any of the logged policy parameters are different from the provided config,
        # print a warning and use the logged config parameters.
        # The keys in the logged config might be prefixed (e.g., 'policy_latent_node_dim')
        keys_to_check = {
            "scenario": ["scenario"],
            "latent_node_dim": ["latent_node_dim", "policy_latent_node_dim"],
            "actor_heads": ["actor_heads", "policy_actor_heads"],
        }
        for key, logged_key_options in keys_to_check.items():
            logged_value = None
            for logged_key in logged_key_options:
                if hasattr(logged_cfg, logged_key):
                    logged_value = getattr(logged_cfg, logged_key)
                    break

            if logged_value is None:
                continue

            provided_value = getattr(cfg, key)

            # The values from comet can be strings, need to cast them
            if isinstance(provided_value, int):
                logged_value = int(logged_value)

            if provided_value and provided_value != logged_value:
                warning_msg = (
                    f"Warning: {key} in logged config ({logged_value}) "
                    f"differs from provided {key} ({provided_value}). "
                )
                if key == "scenario":
                    warning_msg += "Using provided scenario."
                    print(warning_msg)
                else:
                    warning_msg += f"Using logged {key}."
                    print(warning_msg)
                    setattr(cfg, key, logged_value)

        # For reproducibility, it's crucial to use the same seed as the training run.
        # Prioritize the seed from the logged configuration.
        logged_seed = getattr(logged_cfg, "seed", None)
        if logged_seed is not None:
            logged_seed = int(logged_seed)
            if cfg.seed is not None and logged_seed != cfg.seed:
                print(
                    f"Warning: Overriding provided seed ({cfg.seed}) with seed from "
                    f"logged config ({logged_seed}) for reproducibility."
                )
            cfg.seed = logged_seed

        if cfg.seed is None:
            raise ValueError(
                "No seed was found in the logged config and no seed was provided "
                "via the command line. Please provide a seed for reproducible results."
            )

    # Set all seeds for reproducibility
    set_all_seeds(cfg.seed)

    dfs = []
    if policy_weights:
        weights_to_load = policy_weights
        if isinstance(policy_weights, dict) and "model_state_dict" in policy_weights:
            print("Loading model state dict from checkpoint.")
            weights_to_load = policy_weights["model_state_dict"]

        loaded_sampler = EpisodeSampler(
            cfg.seed,
            cfg.scenario,
            cfg.episode_length,
            latent_node_dim=cfg.latent_node_dim,
            actor_heads=cfg.actor_heads,
            policy_weights=weights_to_load,
            num_jobs=cfg.num_jobs,
            use_single_seed=cfg.use_single_seed,
        )
        loaded_stacked_rewards_to_go, _ = loaded_sampler.sample_episodes(
            num_episodes=cfg.num_episodes
        )
        df_long = long_format_dataframe(loaded_stacked_rewards_to_go)
        df_long["Policy"] = "Trained"
        dfs.append(df_long)

    random_sampler = EpisodeSampler(
        cfg.seed,
        cfg.scenario,
        cfg.episode_length,
        latent_node_dim=cfg.latent_node_dim,
        actor_heads=cfg.actor_heads,
        policy_weights=None,
        num_jobs=cfg.num_jobs,
        use_single_seed=cfg.use_single_seed,
    )
    random_stacked_rewards_to_go, _ = random_sampler.sample_episodes(
        num_episodes=cfg.num_episodes
    )
    df_long = long_format_dataframe(random_stacked_rewards_to_go)
    df_long["Policy"] = "Random"
    dfs.append(df_long)

    df = pd.concat(dfs)

    data_filename = Path(output_dir) / "rewards_to_go.csv"
    df.to_csv(data_filename, index=False)
    print(f"Saved results in {data_filename}")


main()
