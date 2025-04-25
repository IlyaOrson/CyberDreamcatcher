from pathlib import Path
import logging
import time
from dataclasses import dataclass, field
# import warnings

import hydra
import torch
import numpy as np
import botorch
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
)
from cyberdreamcatcher.sampler import EpisodeSampler
from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police

LOGGER = logging.getLogger(__name__)

# Filter the specific BoTorch InputDataWarning about unit cube scaling
# warnings.filterwarnings("ignore", category=botorch.exceptions.warnings.InputDataWarning)

@dataclass
class Cfg:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_level: str = "INFO"
    scenario: str = "Scenario2"
    episode_length: int = 30
    num_jobs: int = -1
    policy_kwargs: dict = field(default_factory=dict)
    n_initial_points: int = 20
    budget: int = 200
    num_episodes_sample: int = 100
    bounds_min: float = -2.0
    bounds_max: float = 2.0


def evaluate_parameters(
    parameters_vector: np.ndarray,
    policy_ref_state_dict: dict,
    sampler: EpisodeSampler,
    cfg: Cfg,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluates a set of policy parameters using the EpisodeSampler."""
    # Convert parameter vector back to state dict
    current_state_dict = vector_to_state_dict(parameters_vector, policy_ref_state_dict)

    # Set the policy weights on the sampler instance
    sampler.policy_weights = current_state_dict

    # Sample episodes using the updated policy weights stored in the sampler
    batch_rewards_to_go, _ = sampler.sample_episodes(
        num_episodes=cfg.num_episodes_sample
    )

    total_rewards = batch_rewards_to_go[:, 0]
    # Convert mean and variance to tensors on the correct device
    mean_reward = torch.tensor(np.mean(total_rewards), dtype=torch.float64, device=device)
    variance_reward = torch.tensor(
        max(np.var(total_rewards), 1e-6) if len(total_rewards) > 1 else 1e-6,
        dtype=torch.float64,
        device=device,
    )

    LOGGER.info(
        f"Evaluated params via Sampler. Mean Reward: {mean_reward.item():.3f}, Variance: {variance_reward.item():.3f}"
    )

    return mean_reward, variance_reward


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="args", node=Cfg)

@hydra.main(config_path="conf", config_name="hydra", version_base=None)
def train(cfg: Cfg):
    run_start_time = time.time()
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.basicConfig(level=cfg.log_level)
    LOGGER.info("Starting BoTorch GNN Training")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(
        f"Config used (simplified): Scenario={cfg.scenario}, Episodes={cfg.episode_length}, Samples={cfg.num_episodes_sample}"
    )

    set_all_seeds(cfg.seed)
    device = torch.device(cfg.device)

    env_template = GraphWrapper(scenario=cfg.scenario, max_steps=cfg.episode_length)
    policy_template = Police(env_template, **cfg.policy_kwargs).to(device)
    initial_state_dict = policy_template.state_dict()
    initial_params_vector = state_dict_to_vector(initial_state_dict)
    param_dim = len(initial_params_vector)
    LOGGER.info(f"Policy has {param_dim} parameters.")
    del env_template, policy_template

    sampler = EpisodeSampler(
        seed=cfg.seed,
        scenario=cfg.scenario,
        episode_length=cfg.episode_length,
        num_jobs=cfg.num_jobs,
    )

    bounds = torch.tensor(
        [[cfg.bounds_min] * param_dim, [cfg.bounds_max] * param_dim],
        dtype=torch.float64,
        device=device,
    )

    initial_params_tensor = torch.tensor(
        initial_params_vector, dtype=torch.float64, device=device
    )

    train_x = initial_params_tensor + (
        torch.rand(cfg.n_initial_points, param_dim, dtype=torch.float64, device=device)
        * (bounds[1] - bounds[0])
        + bounds[0]
    )

    # Normalize initial points to [0, 1]^d
    train_x_normalized = normalize(train_x, bounds)
    train_x_normalized.clamp_(0.0, 1.0) # Clamp to handle potential float precision issues

    train_y = []
    train_y_var = []

    LOGGER.info(f"Generating {cfg.n_initial_points} initial evaluation points...")
    # Evaluate initial points using their *original* scale
    for i, params_tensor in enumerate(train_x):
        LOGGER.info(f"Evaluating initial point {i+1}/{cfg.n_initial_points}")
        mean, var = evaluate_parameters(
            params_tensor.detach().cpu().numpy(), # Pass original scale params
            initial_state_dict,
            sampler,
            cfg,
            device,
        )
        train_y.append(mean)
        train_y_var.append(var)

    # Use torch.stack to combine the list of scalar tensors into a 1D tensor
    train_y = torch.stack(train_y).unsqueeze(-1)
    train_y_var = torch.stack(train_y_var).unsqueeze(-1)

    train_y = train_y.to(device)
    train_y_var = train_y_var.to(device)

    LOGGER.info(f"Initial Mean Reward: {train_y.mean():.3f}")

    num_iterations = cfg.budget - cfg.n_initial_points
    if num_iterations <= 0:
        LOGGER.warning(
            "Budget <= n_initial_points. No optimization iterations will run."
        )

    for iteration in range(1, num_iterations + 1):
        iter_start_time = time.time()
        LOGGER.info(f"--- BoTorch Iteration {iteration}/{num_iterations} ---")

        # --- Fit SingleTaskGP with observed variance ---
        # Ensure variance is non-negative and has a minimum floor for stability
        train_y_var_safe = train_y_var.clamp_min(1e-6)

        gp = SingleTaskGP(train_X=train_x_normalized, train_Y=train_y, train_Yvar=train_y_var_safe)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        gp.to(device)
        mll.to(device)
        try:
            fit_gpytorch_mll(mll)
            LOGGER.info("SingleTaskGP fitted with observed variance.")
        except botorch.exceptions.ModelFittingError as e:
            LOGGER.error(f"GP model fitting failed: {e}")
            LOGGER.warning("Skipping iteration due to model fitting error.")
            continue
        except Exception as e: # Catch potential lower-level GPyTorch errors
            LOGGER.error(f"Unexpected error fitting GP: {e}")
            LOGGER.warning("Skipping iteration due to unexpected model fitting error.")
            continue

        # --- Define Acquisition Function using the fitted GP ---
        best_observed_value = train_y.max().item()
        acq_func = LogExpectedImprovement(
            model=gp, # Use the single GP model
            best_f=best_observed_value,
            maximize=True,
        )

        try:
            candidate, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor([[0.0] * param_dim, [1.0] * param_dim], dtype=torch.float64, device=device),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
        except Exception as e:
            LOGGER.error(f"Acquisition function optimization failed: {e}")
            LOGGER.warning("Skipping iteration due to acquisition optimization error.")
            continue

        # Unnormalize the candidate back to the original parameter space
        new_x_tensor = unnormalize(candidate.detach(), bounds)

        new_x_vector = new_x_tensor.squeeze(0).cpu().numpy()
        LOGGER.info(f"Evaluating candidate point {iteration}")
        new_y, new_y_var = evaluate_parameters(
            new_x_vector,
            initial_state_dict, # Pass template state dict
            sampler,            # Pass sampler instance
            cfg,                # Pass config
            device              # Pass device for tensor creation
        )

        new_x_normalized = normalize(new_x_tensor, bounds)
        new_x_normalized.clamp_(0.0, 1.0) # Clamp the new point before concatenation
        train_x_normalized = torch.cat([train_x_normalized, new_x_normalized])
        # Keep the clamp after concatenation too, for extra safety
        train_x_normalized.clamp_(0.0, 1.0) # Clamp again after adding new point

        train_y = torch.cat([train_y, new_y.view(1, -1)])
        train_y_var = torch.cat([train_y_var, new_y_var.view(1, -1)])

        train_y = train_y.to(device)
        train_y_var = train_y_var.to(device)

        iter_time = time.time() - iter_start_time
        current_best_reward = train_y.max().item()
        LOGGER.info(
            f"Iteration {iteration} finished in {iter_time:.2f}s. Acq value: {acq_value.item():.3f}. Candidate reward: {new_y.item():.3f}. Best reward so far: {current_best_reward:.3f}"
        )

    best_idx = train_y.argmax()
    best_reward = train_y[best_idx].item()
    best_params_tensor = unnormalize(train_x_normalized[best_idx], bounds)
    best_params_vector = best_params_tensor.cpu().numpy()

    LOGGER.info("--- Optimization Finished ---")
    LOGGER.info(f"Total time: {time.time() - run_start_time:.2f}s")
    LOGGER.info(f"Total evaluations: {cfg.budget}")
    LOGGER.info(f"Best reward found: {best_reward:.4f}")

    try:
        LOGGER.info(f"Saving best policy state dict to {output_dir}/best_policy.pt")
        best_state_dict = vector_to_state_dict(best_params_vector, initial_state_dict)
        torch.save(best_state_dict, output_dir / "best_policy.pt")

        torch.save(best_params_tensor, output_dir / "best_params_tensor.pt")
        np.save(output_dir / "best_params_vector.npy", best_params_vector)

        torch.save(
            {
                "train_x": train_x_normalized.cpu(), # Save normalized X
                "train_y": train_y.cpu(),
                "train_y_var": train_y_var.cpu(),
            },
            output_dir / "training_data.pt",
        )
    except Exception as e:
        LOGGER.error(f"Failed to save final results: {e}")


if __name__ == "__main__":
    train()
