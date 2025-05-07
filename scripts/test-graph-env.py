from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

import hydra
from hydra.core.config_store import ConfigStore
from rich import inspect
from rich.console import Console
from rich.rule import Rule
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.logging import RichHandler
import torch
import pandas as pd
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
    render_mode: Optional[str] = None


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

    scenario = cfg.scenario
    if cfg.policy_weights and Path(cfg.policy_weights).exists():
        policy_weights, trained_scenario = load_trained_weights(cfg.policy_weights)
        print(f"Loaded policy trained on {trained_scenario}.")
        if trained_scenario != cfg.scenario:
            print("Will ignore the provided scenario.")
            scenario = trained_scenario
    print(f"Plotting performance on scenario {scenario}.")

    console = Console(quiet=cfg.quiet)
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])

    # plt.show(block=False)

    # scenario = None
    # scenario = "Scenario2"
    # scenario = "Scenario2_-_User2_User4"
    # scenario = "Scenario2_+_User5_User6"
    env = GraphEnv(
        scenario=scenario, track_history=cfg.track_history, render_mode=cfg.render_mode
    )

    obs, info = env.reset()
    # env.render()

    # console.print(Rule("InitialObservation", style="bold red"))
    # pprint(info["observation"])

    if env.track_history:
        console.print(Rule("True Table", style="bold red"))
        console.print(info["true_table"])
        console.print(Rule("Blue Table", style="bold red"))
        console.print(info["blue_table"])
        console.print(Rule("Red Table", style="bold red"))
        console.print(info["red_table"])

    if cfg.policy_weights:
        policy = Police(env)
        if Path(cfg.policy_weights).exists():
            policy.load_state_dict(torch.load(cfg.policy_weights))
        else:
            console.print(
                f"Policy weights not found at {cfg.policy_weights}, using random weights instead."
            )

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        # TimeRemainingColumn(),
        console=console,
    ) as progress:
        for step in progress.track(
            range(cfg.episode_length), description="Running steps..."
        ):
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
                inspect(info["prev_action"], console=console)

                console.print(Rule("Blue Observation", style="bold red"))
                console.print(info["cyborg_result"]["observation"])
                # console.print(info["blue_obs"])

                console.print(Rule("Hosts Observed", style="bold red"))
                console.print(info["hosts_obs"])
                console.print(Rule("Connections Observed", style="bold red"))
                console.print(info["connections_obs"])
                console.print(Rule("Exploited Hosts", style="bold red"))
                console.print(info["exploited_hosts"])
                console.print(Rule("Malware Hosts", style="bold red"))
                console.print(info["malware_hosts"])
                console.print(Rule("Encoded Observation", style="bold red"))
                # console.print(info["encoded_observation"].x)
                df = pd.DataFrame(info["encoded_observation"].x)
                df.columns = env.NodeFeatures._fields
                df.index = env.host_names
                console.print(df)
                console.print(Rule("Encoded Edges", style="bold red"))
                console.print(info["encoded_observation"].edge_index)
                console.print(Rule("Encoded Edge Weights", style="bold red"))
                console.print(torch.squeeze((info["encoded_observation"].edge_attr)))
                console.print(Rule("Encoded Global Attributes", style="bold red"))
                console.print(info["encoded_observation"].global_attr)

                console.print(Rule("True Table", style="bold red"))
                console.print(info["true_table"])

                console.print(Rule("Last Blue Action", style="bold red"))
                inspect(env.cyborg.get_last_action(agent="Blue"), console=console)
                console.print(Rule("Blue Table", style="bold red"))
                console.print(info["blue_table"])

                console.print(Rule("Last Red Action", style="bold red"))
                inspect(env.cyborg.get_last_action(agent="Red"), console=console)

                console.print(Rule("Red Table", style="bold red"))
                console.print(info["red_table"])
                console.print(Rule("Red Observation", style="bold red"))
                console.print(info["red_obs"])

                console.print(Rule("Reward", style="bold red"))
                console.print(env.cyborg.get_rewards())

    # plt.show()


main()
