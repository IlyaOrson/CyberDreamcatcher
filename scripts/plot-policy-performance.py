import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
import torch
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.sampler import EpisodeSampler


@dataclass
class Config:
    policy_path: Optional[str] = None
    scenario: Optional[str] = None
    seed: int = 31415
    episode_length: int = 30
    num_episodes: int = 100

# https://hydra.cc/docs/tutorials/structured_config/minimal_example/
cs = ConfigStore.instance()
# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
    print(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")

    if cfg.policy_path is None:
        print("No policy given, random weights will be used.")
    else:
        # TODO print details used to train the policy weights
        print(f"Loading policy from {cfg.policy_path}")

    if not cfg.scenario:
        scenarios_dir = Path.cwd() / "scenarios"
        scenarios = [file.name for file in scenarios_dir.iterdir() if file.is_file()]
    else:
        scenarios = [cfg.scenario]

    for scenario in scenarios:
        
        run_name = f"PPO_{scenario}_seed_{cfg.seed}"
        log_dir = Path(output_dir) / run_name

        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in cfg.items()])),
        )

        env = GraphWrapper(scenario=scenario, max_steps=cfg.episode_length, render_mode=None)
        policy = Police(env, latent_node_dim=env.host_embedding_size)

        # load trained policy
        if cfg.policy_path:
            file_path = Path(cfg.policy_path)
            policy.load_state_dict(torch.load(file_path))

        # NOTE Call model.eval() to set dropout and batch normalization layers
        # to evaluation mode before running inference.
        # Failing to do this will yield inconsistent inference results.
        policy.eval()

        sampler = EpisodeSampler(env, policy, seed=cfg.seed, writer=writer)
        sampler.sample_episodes(cfg.num_episodes, counter=-1)

main()
