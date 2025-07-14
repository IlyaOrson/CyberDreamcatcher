from pathlib import Path
import logging
import pickle
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import comet_ml
from comet_ml.integration.pytorch import watch, log_model
from rich.logging import RichHandler
from tqdm import tqdm
import nevergrad as ng
import numpy as np
import torch

from cyberdreamcatcher.utils import (
    set_all_seeds,
    state_dict_to_vector,
    vector_to_state_dict,
    load_trained_weights,
    count_parameters,
)
from cyberdreamcatcher.sampler import EpisodeSampler
from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police

LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    scenario: str = "Scenario2"
    episode_length: int = 30
    failed_action_penalty: float = -0.05
    batch_size_episodes: int = 200
    seed: int = 0
    log_level: str = "INFO"
    log_comet: bool = True

    # Policy
    policy_weights: Optional[str] = None
    latent_node_dim: int = 15
    actor_heads: int = 3

    # Nevergrad settings
    budget: int = 500
    use_mean_reward: bool = True
    optimizer: str = "TwoPointsDE"


class NevergradTrainer:
    def __init__(self, env, policy, conf, output_dir):
        self.env = env
        self.policy = policy
        self.conf = conf
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        self.experiment = None
        if conf.log_comet:
            try:
                comet_ml.login()
                self.experiment = comet_ml.start(
                    project_name="cyberdreamcatcher",
                )
                self.experiment.set_name(f"nevergrad_{conf.optimizer}_seed_{conf.seed}")
                self.experiment.log_parameters(
                    OmegaConf.to_container(conf, resolve=True)
                )
                watch(self.policy, log_step_interval=10)
            except Exception as e:
                LOGGER.warning(f"CometML initialization failed: {e}")
                self.experiment = None

    def evaluate_parameters(self, params_vector: np.ndarray) -> list[float]:
        state_dict = vector_to_state_dict(params_vector, self.policy.state_dict())

        sampler = EpisodeSampler(
            seed=self.conf.seed,
            scenario=self.conf.scenario,
            episode_length=self.conf.episode_length,
            latent_node_dim=self.conf.latent_node_dim,
            actor_heads=self.conf.actor_heads,
            policy_weights=state_dict,
            num_jobs=-1,  # Use all available cores for evaluation
        )

        rewards_to_go, _ = sampler.sample_episodes(self.conf.batch_size_episodes)
        episode_rewards = rewards_to_go[:, 0].tolist()
        return episode_rewards

    def optimize(self):
        initial_state_dict = self.policy.state_dict()
        initial_params_vector = state_dict_to_vector(initial_state_dict)
        param_dim = len(initial_params_vector)
        LOGGER.info(f"Policy parameter dimension: {param_dim}")
        if self.experiment:
            self.experiment.log_other("policy_parameter_dimension", param_dim)

        parametrization = ng.p.Array(init=initial_params_vector).set_name("policy_params")
        optimizer = ng.optimizers.registry[self.conf.optimizer](
            parametrization=parametrization, budget=self.conf.budget
        )
        optimizer.suggest(initial_params_vector)

        LOGGER.info("Starting Nevergrad optimization loop...")
        pbar = tqdm(total=self.conf.budget, desc="Optimizing")
        best_known_reward = -np.inf

        for step in range(self.conf.budget):
            candidate = optimizer.ask()
            episode_rewards = self.evaluate_parameters(candidate.value)

            if self.conf.use_mean_reward:
                mean_reward = np.mean(episode_rewards)
                optimizer.tell(candidate, -mean_reward)
            else:
                for reward in episode_rewards:
                    optimizer.tell(candidate, -reward)
            
            batch_best_reward = np.max(episode_rewards)
            if batch_best_reward > best_known_reward:
                best_known_reward = batch_best_reward

            if self.experiment:
                self.experiment.log_metrics({
                    "batch_reward_mean": np.mean(episode_rewards),
                    "batch_reward_std": np.std(episode_rewards),
                    "best_reward_so_far": best_known_reward,
                }, step=step)

            pbar.update(1)
            pbar.set_postfix({"Best Reward": f"{best_known_reward:.4f}"})

        pbar.close()

        recommendation = optimizer.provide_recommendation()
        best_params_vector = recommendation.value
        LOGGER.info(f"Optimization finished. Best observed reward: {best_known_reward:.3f}")

        best_state_dict = vector_to_state_dict(best_params_vector, initial_state_dict)
        self.policy.load_state_dict(best_state_dict)

        if self.experiment:
            self.experiment.log_metric("final_best_reward", best_known_reward)
            log_model(self.experiment, self.policy, "best_policy")

            self.experiment.end()

        return self.policy


cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)

@hydra.main(version_base=None, config_name="hydra", config_path="conf")
def main(cfg: Cfg) -> None:
    set_all_seeds(cfg.seed)
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
    LOGGER.info("Starting Nevergrad GNN Training")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

    assert cfg.policy_weights or cfg.scenario, "Please provide either 'scenario' or 'policy_weights'."

    scenario = cfg.scenario
    policy_weights = None
    if cfg.policy_weights and Path(cfg.policy_weights).exists():
        policy_weights, trained_scenario = load_trained_weights(cfg.policy_weights)
        LOGGER.info(f"Found policy trained on {trained_scenario}.")
        if trained_scenario != cfg.scenario:
            LOGGER.warning(f"Will ignore the provided scenario {cfg.scenario}.")
            scenario = trained_scenario

    env = GraphEnv(
        scenario=scenario,
        max_steps=cfg.episode_length,
        failed_action_penalty=cfg.failed_action_penalty,
    )
    policy = Police(
        env,
        latent_node_dim=cfg.latent_node_dim,
        actor_heads=cfg.actor_heads,
    )

    if policy_weights:
        policy.load_state_dict(policy_weights)

    trainer = NevergradTrainer(env, policy, cfg, output_dir=output_dir)

    LOGGER.info("Starting optimization")
    LOGGER.info(f"Policy parameters: {count_parameters(policy)}")
    best_policy = trainer.optimize()

    file_path = output_dir / "best_policy.pt"
    LOGGER.info(f"Saving final best policy to {file_path}")
    torch.save(best_policy.state_dict(), file_path)

    LOGGER.info("Training finished successfully!")


if __name__ == "__main__":
    main()
