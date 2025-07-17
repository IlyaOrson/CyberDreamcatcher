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
import matplotlib.pyplot as plt

from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.utils import (
    get_policy_weights_and_config,
)
from cyberdreamcatcher.plots import (
    plot_attention_graph,
    plot_feasible_connections,
    plot_action_probabilities,
)  # , plot_observation_encoded


LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    use_policy: bool = True
    policy_weights: Optional[str] = None
    comet_experiment_key: Optional[str] = None
    comet_model_name: Optional[str] = None
    comet_model_step: Optional[int] = None
    latent_node_dim: int = 5
    actor_heads: int = 2
    num_layers: int = 3
    share_weights: bool = False
    residual: bool = True
    scenario: str = "Scenario2"
    seed: int = 0
    episode_length: int = 30
    failed_action_penalty: float = -0.05
    quiet: bool = False
    progress_bar: bool = True
    log_level: str = "INFO"
    track_history: bool = True
    render_mode: Optional[str] = None
    plot_action_probabilities: bool = False
    plot_attention: bool = False
    # Only plot attention for Restore/Remove actions
    plot_only_restore_remove: bool = False


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

    scenario = cfg.scenario
    latent_node_dim = cfg.latent_node_dim
    actor_heads = cfg.actor_heads

    if logged_cfg:
        # Logged policy parameters should match the provided config parameters
        if logged_cfg.scenario != scenario:
            LOGGER.info(
                f"Logged scenario {logged_cfg.scenario} is different from provided scenario {scenario}."
            )

    LOGGER.info(f"Simulating scenario {scenario}.")

    console = Console(quiet=cfg.quiet)
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])

    # plt.show(block=False)

    # scenario = None
    # scenario = "Scenario2"
    # scenario = "Scenario2_-_User2_User4"
    # scenario = "Scenario2_+_User5_User6"
    env = GraphEnv(
        scenario=scenario,
        track_history=cfg.track_history,
        render_mode=cfg.render_mode,
        failed_action_penalty=cfg.failed_action_penalty,
    )
    # if cfg.plot_attention:
    #     with plt.xkcd():
    #         plot_feasible_connections(env)
    obs, info = env.reset()
    # env.render()

    policy = None
    if cfg.use_policy:
        policy = Police(
            env,
            latent_node_dim=latent_node_dim,
            actor_heads=actor_heads,
            num_layers=cfg.num_layers,
            share_weights=cfg.share_weights,
            residual=cfg.residual,
        )
        if policy_weights is not None:
            if (
                isinstance(policy_weights, dict)
                and "model_state_dict" in policy_weights
            ):
                LOGGER.info(
                    f"CometML checkpoint found. Loading 'model_state_dict' with epoch {policy_weights['epoch']} and reward mean {policy_weights['reward_mean']}"
                )
                policy.load_state_dict(policy_weights["model_state_dict"])
            else:
                policy.load_state_dict(policy_weights)
            policy.eval()
        else:
            LOGGER.warning("Policy weights not provided. Using random policy.")
    else:
        LOGGER.warning("Policy not used. Using random actions.")

    # console.print(Rule("InitialObservation", style="yellow"))
    # pprint(info["observation"])
    if env.track_history:
        console.print(Rule("True Table", style="yellow"))
        console.print(info["true_table"])
        console.print(Rule("Blue Table", style="yellow"))
        console.print(info["blue_table"])
        console.print(Rule("Red Table", style="yellow"))
        console.print(info["red_table"])

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        # TimeRemainingColumn(),
        console=console,
        disable=not cfg.progress_bar,
    ) as progress:
        rewards = []
        for step in progress.track(
            range(cfg.episode_length), description="Running steps..."
        ):
            if policy:
                if cfg.plot_action_probabilities:
                    plot_action_probabilities(env, policy, obs, show=True, block=True)
                report = policy(obs, return_attention_weights=cfg.plot_attention)
                action = report.action

                if cfg.plot_attention and report.attention is not None:
                    # Get action name and check if we should plot based on configuration
                    action_name = env.action_to_name(action)
                    should_plot = True

                    if cfg.plot_only_restore_remove:
                        should_plot = action_name[1] in ["Restore", "Remove"]
                        if not should_plot:
                            LOGGER.debug(
                                f"Skipping attention plot for action (plot_only_restore_remove=True): {action_name}"
                            )

                    if should_plot:
                        # with plt.xkcd():
                        plot_attention_graph(
                            env, report.attention, action, show=True, block=True
                        )
            else:
                action = torch.tensor(env.action_space.sample())

            obs, reward, terminated, truncated, info = env.step(action)
            # env.render()
            # plot_observation_encoded(env, obs, show=True)

            rewards.append(reward)

            if env.track_history:
                console.print(Rule(f"STEP {step}", style="bold red"))

                console.print(Rule("Action", style="yellow"))
                # inspect(info["prev_action"], console=console)
                console.print(env.previous_action)

                console.print(Rule("Blue Observation", style="yellow"))
                console.print(info["cyborg_result"]["observation"])
                # console.print(info["blue_obs"])

                console.print(Rule("Hosts Observed", style="yellow"))
                console.print(info["hosts_obs"])
                console.print(Rule("Connections Observed", style="yellow"))
                console.print(info["connections_obs"])
                console.print(Rule("Exploited Hosts", style="yellow"))
                console.print(info["exploited_hosts"])
                console.print(Rule("Malware Hosts", style="yellow"))
                console.print(info["malware_hosts"])
                console.print(Rule("Encoded Observation", style="yellow"))
                # console.print(info["encoded_observation"].x)
                df = pd.DataFrame(info["encoded_observation"].x)
                df.columns = env.NodeFeatures._fields
                df.index = env.host_names
                console.print(df)
                console.print(Rule("Encoded Edges", style="yellow"))
                console.print(info["encoded_observation"].edge_index)
                if info["encoded_observation"].edge_attr:
                    console.print(Rule("Encoded Edge Weights", style="yellow"))
                    console.print(info["encoded_observation"].edge_attr.T)
                if getattr(info["encoded_observation"], "global_attr", None):
                    console.print(Rule("Encoded Global Attributes", style="yellow"))
                    console.print(info["encoded_observation"].global_attr)

                console.print(Rule("True Table", style="yellow"))
                console.print(info["true_table"])

                console.print(Rule("Last Blue Action", style="yellow"))
                inspect(env.cyborg.get_last_action(agent="Blue"), console=console)
                console.print(Rule("Blue Table", style="yellow"))
                console.print(info["blue_table"])

                console.print(Rule("Last Red Action", style="yellow"))
                inspect(env.cyborg.get_last_action(agent="Red"), console=console)

                console.print(Rule("Red Table", style="yellow"))
                console.print(info["red_table"])
                console.print(Rule("Red Observation", style="yellow"))
                console.print(info["red_obs"])

                console.print(Rule("Reward", style="yellow"))
                console.print(env.cyborg.get_rewards())
                if env.previous_action.success == 0:
                    console.print(f"Failed action penalty: {env.failed_action_penalty}")

    # plt.show()
    console.print(Rule("Failed Actions", style="purple"))
    console.print(env.failed_actions)

    console.print(Rule("Total Reward", style="purple"))
    console.print(rewards)
    console.print(sum(rewards))


main()
