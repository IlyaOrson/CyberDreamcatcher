import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter

from cyberdreamcatcher.sampler import EpisodeSampler


# Disable specific loggers
logging.getLogger("CybORGLog-Process").setLevel(logging.CRITICAL)
logging.getLogger("cyberdreamcatcher.utils").setLevel(logging.CRITICAL)


@dataclass
class Cfg:
    policy_weights: Optional[str] = None
    scenario: Optional[str] = "Scenario2"
    seed: int = 31415
    episode_length: int = 30
    num_episodes: int = 100
    num_jobs: int = 1


# TODO adapt to generalization plot
# if not cfg.scenario:
#     scenarios_dir = Path.cwd() / "scenarios"
#     scenarios = [file.name for file in scenarios_dir.iterdir() if file.is_file()]
# else:
#     scenarios = [cfg.scenario]

# for scenario in scenarios:
#     pass


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
        policy_path = Path(cfg.policy_weights)
        assert policy_path.is_file()

        policy_dir = policy_path.parent
        logged_cfg_path = policy_dir / ".hydra" / "config.yaml"
        assert logged_cfg_path.is_file()
        logged_cfg = OmegaConf.load(logged_cfg_path)
        print("Configuration used to train loaded policy.")
        print(OmegaConf.to_yaml(logged_cfg))
        trained_scenario = logged_cfg.scenario
        policy_weights = torch.load(policy_path, weights_only=True)

        run_name = f"Performance_on_{cfg.scenario}_trained_on_{trained_scenario}_seed_{cfg.seed}"

    log_dir = Path(output_dir) / run_name

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in cfg.items()])),
    )

    # Initialize with number of parallel jobs (-1 uses all available cores)
    sampler = EpisodeSampler(
        cfg.seed,
        cfg.scenario,
        cfg.episode_length,
        policy_weights=policy_weights,
        writer=writer,
        num_jobs=cfg.num_jobs,
    )

    # Sample episodes in parallel
    reward_mean, reward_std = sampler.sample_episodes(num_episodes=cfg.num_episodes)

    #TODOviolin plots!

main()
