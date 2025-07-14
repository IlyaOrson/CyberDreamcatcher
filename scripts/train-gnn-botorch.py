from pathlib import Path
import logging
import time
from dataclasses import dataclass
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import comet_ml
from comet_ml.integration.pytorch import log_model, watch
from rich.logging import RichHandler
import torch
import numpy as np
from botorch.models.gp_regression import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize

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
    batch_size_episodes: int = 100
    seed: int = 0
    log_level: str = "INFO"
    log_comet: bool = True

    # Policy
    policy_weights: Optional[str] = None
    latent_node_dim: int = 15
    actor_heads: int = 3

    # BoTorch settings
    num_initial_points: int = 20
    budget: int = 1000
    bounds_min: float = -2.0
    bounds_max: float = 2.0


class BoTorchTrainer:
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
                self.experiment.set_name(f"botorch_seed_{conf.seed}")
                self.experiment.log_parameters(
                    OmegaConf.to_container(conf, resolve=True)
                )
                watch(self.policy, log_step_interval=10)
            except Exception as e:
                LOGGER.warning(f"CometML initialization failed: {e}")
                self.experiment = None

    def evaluate_parameters(self, params_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a tensor of parameter sets."""
        # Ensure params_vector is a tensor
        if isinstance(params_vector, np.ndarray):
            params_vector = torch.from_numpy(params_vector).to(self.device, dtype=torch.float64)

        # Ensure params_vector is 2D
        if params_vector.ndim == 1:
            params_vector = params_vector.unsqueeze(0)

        rewards = []
        # BoTorch evaluates candidates sequentially in a batch
        for params in params_vector:
            # Ensure params is a tensor before processing
            if not isinstance(params, torch.Tensor):
                params = torch.from_numpy(params).to(self.device)
            state_dict = vector_to_state_dict(params.cpu().numpy(), self.policy.state_dict())
            
            sampler = EpisodeSampler(
                seed=self.conf.seed,
                scenario=self.conf.scenario,
                episode_length=self.conf.episode_length,
                latent_node_dim=self.conf.latent_node_dim,
                actor_heads=self.conf.actor_heads,
                policy_weights=state_dict,
                num_jobs=-1, # Use all available cores for evaluation
            )

            # For BoTorch, we typically want a single, high-quality estimate for the given parameters.
            # So we average the rewards from a batch of episodes.
            rewards_to_go, _ = sampler.sample_episodes(self.conf.batch_size_episodes)
            mean_reward = np.mean(rewards_to_go[:, 0])
            rewards.append(mean_reward)
        
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float64).unsqueeze(1)
        # Return rewards and zero observation noise
        return rewards_tensor, torch.zeros_like(rewards_tensor)

    def optimize(self):
        run_start_time = time.time()
        initial_state_dict = self.policy.state_dict()
        param_dim = len(state_dict_to_vector(initial_state_dict))
        LOGGER.info(f"Policy parameter dimension: {param_dim}")
        if self.experiment:
            self.experiment.log_other("policy_parameter_dimension", param_dim)

        bounds = torch.tensor(
            [[self.conf.bounds_min] * param_dim, [self.conf.bounds_max] * param_dim],
            dtype=torch.float64,
            device=self.device,
        )

        LOGGER.info("Generating initial random points...")
        initial_x = (
            torch.rand(self.conf.num_initial_points, param_dim, device=self.device, dtype=torch.float64)
            * (bounds[1] - bounds[0])
            + bounds[0]
        )

        train_x = initial_x
        train_y = torch.empty(self.conf.num_initial_points, 1, device=self.device, dtype=torch.float64)
        train_y_var = torch.empty(self.conf.num_initial_points, 1, device=self.device, dtype=torch.float64)

        for i in range(self.conf.num_initial_points):
            params_vector = train_x[i].cpu().numpy()
            mean, var = self.evaluate_parameters(params_vector)
            train_y[i] = mean
            train_y_var[i] = var

        LOGGER.info("--- Starting BoTorch Optimization Loop ---")
        for iteration in range(self.conf.budget):
            iter_start_time = time.time()
            train_x_normalized = normalize(train_x, bounds)

            model = SingleTaskGP(train_x_normalized, train_y, train_Yvar=train_y_var)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            best_f = train_y.max().item()
            acq_func = LogExpectedImprovement(model, best_f=best_f)

            new_x_normalized, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.0] * param_dim, [1.0] * param_dim], device=self.device, dtype=torch.float64),
                q=1, num_restarts=10, raw_samples=1024,
            )

            new_x_tensor = unnormalize(new_x_normalized.detach(), bounds=bounds)
            new_params_vector = new_x_tensor.squeeze(0).cpu().numpy()
            new_y, new_y_var = self.evaluate_parameters(new_params_vector)

            train_x = torch.cat([train_x, new_x_tensor])
            train_y = torch.cat([train_y, new_y.view(1, -1)])
            train_y_var = torch.cat([train_y_var, new_y_var.view(1, -1)])

            iter_time = time.time() - iter_start_time
            current_best_reward = train_y.max().item()
            LOGGER.info(f"Iteration {iteration+1}/{self.conf.budget} finished in {iter_time:.2f}s. Best reward so far: {current_best_reward:.3f}")
            if self.experiment:
                self.experiment.log_metrics({
                    "acquisition_value": acq_value.item(),
                    "candidate_reward": new_y.item(),
                    "best_reward_so_far": current_best_reward,
                    "iteration_time": iter_time
                }, step=iteration)

        best_idx = train_y.argmax()
        best_reward = train_y[best_idx].item()
        best_params_tensor = unnormalize(train_x_normalized[best_idx], bounds)
        best_params_vector = best_params_tensor.cpu().numpy()

        LOGGER.info("--- Optimization Finished ---")
        LOGGER.info(f"Total time: {time.time() - run_start_time:.2f}s")
        LOGGER.info(f"Best reward found: {best_reward:.4f}")

        best_state_dict = vector_to_state_dict(best_params_vector, initial_state_dict)
        self.policy.load_state_dict(best_state_dict)

        if self.experiment:
            self.experiment.log_metric("final_best_reward", best_reward)
            log_model(self.experiment, self.policy, "best_policy")
            self.experiment.end()

        return self.policy

cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)

@hydra.main(version_base=None, config_name="hydra", config_path="conf")
def main(cfg: Cfg) -> None:
    set_all_seeds(cfg.seed)
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
    LOGGER.info("Starting BoTorch GNN Training")
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

    trainer = BoTorchTrainer(env, policy, cfg, output_dir=output_dir)

    LOGGER.info("Starting optimization")
    LOGGER.info(f"Policy parameters: {count_parameters(policy)}")
    best_policy = trainer.optimize()

    file_path = output_dir / "best_policy.pt"
    LOGGER.info(f"Saving final best policy to {file_path}")
    torch.save(best_policy.state_dict(), file_path)

    LOGGER.info("Training finished successfully!")

if __name__ == "__main__":
    main()
