from unittest.mock import patch
from functools import partial
from pathlib import Path
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
import numpy as np
from stable_baselines3 import PPO

from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper, BlueTableWrapper

from cyberdreamcatcher.utils import get_scenario


# Remove compromised information from flat vector observation.
def create_vector(self, success, see_activity=True, see_compromised=True):
    table = self._create_blue_table(success)._rows

    proto_vector = []
    for row in table:
        # Activity
        if see_activity:
            activity = row[3]
            if activity == "None":
                value = [0, 0]
            elif activity == "Scan":
                value = [1, 0]
            elif activity == "Exploit":
                value = [1, 1]
            else:
                raise ValueError("Table had invalid Access Level")
            proto_vector.extend(value)

        # Compromised
        if see_compromised:
            compromised = row[4]
            if compromised == "No":
                value = [0, 0]
            elif compromised == "Unknown":
                value = [1, 0]
            elif compromised == "User":
                value = [0, 1]
            elif compromised == "Privileged":
                value = [1, 1]
            else:
                raise ValueError("Table had invalid Access Level")
            proto_vector.extend(value)

    # test if patch is being used
    # raise RuntimeError(f'Flat observation length: {len(proto_vector)}')

    return np.array(proto_vector)


@dataclass
class Cfg:
    # "Scenario2_-_User2_User4"  # "Scenario2_+_User5_User6"
    scenario: str = "Scenario2"
    max_episode_steps: int = 30
    total_policy_steps: int = 1_000_000  # 1_000_000 produces competitive results
    progress_bar: bool = True
    policy_device: str = "cpu"
    policy_verbosity: int = 1
    see_activity: bool = True
    see_compromised: bool = True


# Registering the Config class with the expected name 'args'.
# https://hydra.cc/docs/tutorials/structured_config/minimal_example/
cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)


@hydra.main(version_base=None, config_name="hydra", config_path="conf")
def main(cfg: Cfg) -> None:
    # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
    print(f"Working directory : {Path.cwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")
    print(f"Observation with activity info: {cfg.see_activity}")
    print(f"Observation with compromised info: {cfg.see_compromised}")

    def create_environment():
        scenario_path = get_scenario(name=cfg.scenario, from_cyborg=False)
        cyborg = CybORG(scenario_path, "sim", agents={"Red": RedMeanderAgent})
        env = ChallengeWrapper(
            agent_name="Blue", env=cyborg, max_steps=cfg.max_episode_steps
        )
        return env

    def train_model(env):
        model = PPO(
            "MlpPolicy",
            env,
            verbose=cfg.policy_verbosity,
            device=cfg.policy_device,
            # tensorboard_log=output_dir,
        )
        model.learn(
            total_timesteps=cfg.total_policy_steps, progress_bar=cfg.progress_bar
        )

        # store trained policy
        file_path = Path(output_dir) / "trained_model"
        model.save(file_path)

        return model

    if cfg.see_activity and cfg.see_compromised:
        # Use the original version (stateful)
        env = create_environment()
        model = train_model(env)
    else:
        # Use the patched version (stateless)
        with patch.object(
            BlueTableWrapper,
            "_create_vector",
            side_effect=partial(
                create_vector,
                see_activity=cfg.see_activity,
                see_compromised=cfg.see_compromised,
            ),
            autospec=True,
        ):
            env = create_environment()
            model = train_model(env)

    # Example of policy inference
    # num_steps = 10
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(num_steps):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     print(10*"-")


if __name__ == "__main__":
    main()
