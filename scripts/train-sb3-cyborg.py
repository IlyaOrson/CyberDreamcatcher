import os
from pathlib import Path
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from stable_baselines3 import PPO # A2C

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

from blueskynet.utils import get_scenario


@dataclass
class Config:
    scenario: str = "Scenario2_-_User2_User4"  # "Scenario2_+_User5_User6"
    max_episode_steps: int = 30
    total_policy_steps: int = 1_000_000  # 1_000_000 produces competitive results
    progress_bar: bool = True
    policy_device: str = "cpu"
    policy_verbosity: int = 1

# https://hydra.cc/docs/tutorials/structured_config/minimal_example/
cs = ConfigStore.instance()

# Registering the Config class with the name 'config'.
cs.store(name="config", node=Config)

@hydra.main(version_base=None, config_name="config")
def script(cfg: Config) -> None:
    # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
    print(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")

    # scenario_path = get_scenario(name="Scenario2", from_cyborg=True)
    scenario_path = get_scenario(name=cfg.scenario, from_cyborg=False)
    cyborg = CybORG(scenario_path, "sim", agents={'Red': RedMeanderAgent})
    env = ChallengeWrapper(agent_name="Blue", env=cyborg, max_steps=cfg.max_episode_steps)

    model = PPO("MlpPolicy", env, verbose=cfg.policy_verbosity, device=cfg.policy_device)
    model.learn(total_timesteps=cfg.total_policy_steps, progress_bar=cfg.progress_bar)

    # store trained policy
    file_path = Path(output_dir) / "trained_model"
    model.save(file_path)

    # Example of policy inference

    # num_steps = 10
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(num_steps):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     print(10*"-")


script()
