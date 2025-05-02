import os
from dataclasses import dataclass
from typing import Optional
import logging

import hydra
from hydra.core.config_store import ConfigStore
from rich import inspect
from rich.console import Console
from rich.rule import Rule
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.logging import RichHandler
import torch
# import matplotlib.pyplot as plt

from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.utils import load_trained_weights
# from cyberdreamcatcher.plots import plot_action_probabilities, plot_observation_encoded


LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    policy_weights: Optional[str] = None
    scenario: Optional[str] = "Scenario2"
    seed: int = 0
    episode_length: int = 30
    quiet: bool = False
    log_level: str = "INFO"
    track_history: bool = True


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
            print("Will ignore the provided scenario.")
            scenario = trained_scenario
    print(f"Plotting performance on scenario {scenario}.")

    console = Console(quiet=cfg.quiet)
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])

    # plt.show(block=False)

    scenario = None
    # scenario = "Scenario2"
    # scenario = "Scenario2_-_User2_User4"
    # scenario = "Scenario2_+_User5_User6"
    env = GraphEnv(scenario=scenario, track_history=cfg.track_history)

    obs, info = env.reset()
    # env.render()

    # console.print(Rule("InitialObservation", style="bold red"))
    # pprint(info["observation"])

    if env.track_history:
        console.print(Rule("TrueTable", style="bold red"))
        console.print(info["true_table"])
        console.print(Rule("BlueTable", style="bold red"))
        console.print(info["blue_table"])
        console.print(Rule("RedTable", style="bold red"))
        console.print(info["red_table"])

    if cfg.policy_weights:
        policy = Police(env)
        policy.load_state_dict(policy_weights)

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        # TimeRemainingColumn(),
        console=console,
    ) as progress:
        for step in progress.track(range(cfg.episode_length), description="Running steps..."):
            if cfg.policy_weights:
                action, log_prob, entropy, value = policy(obs)
                # visualise action probability distribution
                # plot_action_probabilities(env, policy, obs)
            else:
                action = torch.tensor(env.action_space.sample())

            obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            # plot_observation_encoded(env, obs, show=True)

            if env.track_history:

                console.print(Rule(f"Step {step}", style="bold red"))
                console.print(Rule("Action", style="bold red"))
                inspect(info["action"], console=console)
                console.print(Rule("Observation", style="bold red"))
                console.print(info["observation"])

                console.print(Rule("Hosts", style="bold red"))
                console.print(info["hosts"])
                console.print(Rule("Connections", style="bold red"))
                console.print(info["connections"])

                console.print(Rule("TrueTable", style="bold red"))
                console.print(info["true_table"])

                console.print(Rule("Last Blue Action", style="bold red"))
                inspect(env.cyborg.get_last_action(agent="Blue"), console=console)
                console.print(Rule("BlueTable", style="bold red"))
                console.print(info["blue_table"])

                console.print(Rule("Last Red Action", style="bold red"))
                inspect(env.cyborg.get_last_action(agent="Red"), console=console)

                console.print(Rule("RedTable", style="bold red"))
                console.print(info["red_table"])
                console.print(Rule("Red Observation", style="bold red"))
                console.print(info["red_obs"])

                console.print(Rule("Reward", style="bold red"))
                console.print(env.cyborg.get_rewards())

    # plt.show()


main()
