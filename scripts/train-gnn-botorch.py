import os
from pathlib import Path
import logging
import time
from dataclasses import dataclass, field
# import warnings

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import comet_ml
from comet_ml.integration.pytorch import log_model
from dotenv import load_dotenv
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
)
from cyberdreamcatcher.sampler import EpisodeSampler
from cyberdreamcatcher.env import GraphEnv
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
    policy_kwargs: dict = field(default_factory=lambda: {"latent_node_dim": 3})
    num_initial_points: int = 20
    budget: int = 200
    num_episodes_sample: int = 100
    bounds_min: float = -2.0
    bounds_max: float = 2.0
    log_comet: bool = True


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
    mean_reward = torch.tensor(
        np.mean(total_rewards), dtype=torch.float64, device=device
    )
    variance_reward = torch.tensor(
        max(np.var(total_rewards), 1e-6) if len(total_rewards) > 1 else 1e-6,
        dtype=torch.float64,
        device=device,
    )

    LOGGER.info(
        f"Evaluated params via Sampler. Mean Reward: {mean_reward.item():.3f}, Variance: {variance_reward.item():.3f}"
    )

    return mean_reward, variance_reward


cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)


@hydra.main(config_path="conf", config_name="hydra", version_base=None)
def train(cfg: Cfg):
    run_start_time = time.time()
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
    LOGGER.info("Starting BoTorch GNN Training")
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Using device: {cfg.device}")

    experiment = None
    if cfg.log_comet:
        try:
            # --- Comet ML Setup ---
            load_dotenv()
            experiment = comet_ml.Experiment(
                api_key=os.getenv("COMET_API_KEY"),
                project_name=os.getenv("COMET_PROJECT_NAME"),
                auto_param_logging=False,
                auto_metric_logging=False,
            )
            experiment.set_name(f"botorch_seed_{cfg.seed}")
            experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))
            experiment.log_html(f"<p>Output Directory: {output_dir}</p>")
            # --- End Comet ML Setup ---
        except Exception as e:
            LOGGER.warning(f"CometML initialization failed: {e}")
            experiment = None

    LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

    set_all_seeds(cfg.seed)
    device = torch.device(cfg.device)

    env_template = GraphEnv(scenario=cfg.scenario, max_steps=cfg.episode_length)
    policy_template = Police(env_template, **cfg.policy_kwargs).to(device)
    initial_state_dict = policy_template.state_dict()
    initial_params_vector = state_dict_to_vector(initial_state_dict)
    param_dim = len(initial_params_vector)
    LOGGER.info(f"Policy parameter dimension: {param_dim}")
    if experiment is not None:
        experiment.log_other("policy_parameter_dimension", param_dim)
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

    initial_x_tensor = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
        cfg.num_initial_points, param_dim, device=device
    )
    initial_y = []
    initial_y_var = []

    for i in range(cfg.num_initial_points):
        params_vector = initial_x_tensor[i].cpu().numpy()
        LOGGER.info(f"Evaluating initial point {i+1}/{cfg.num_initial_points}")
        start_eval_time = time.time()
        y, y_var = evaluate_parameters(
            params_vector, initial_state_dict, sampler, cfg, device
        )
        eval_time = time.time() - start_eval_time
        initial_y.append(y)
        initial_y_var.append(y_var)
        if experiment is not None:
            experiment.log_metric("initial_point_reward", y.item(), step=i)
            experiment.log_metric("initial_point_variance", y_var.item(), step=i)
            experiment.log_metric("initial_point_eval_time", eval_time, step=i)

    train_x = initial_x_tensor
    train_y = torch.stack(initial_y).unsqueeze(-1)  # Shape [n_initial, 1]
    train_y_var = torch.stack(initial_y_var).unsqueeze(-1)  # Shape [n_initial, 1]

    LOGGER.info("Starting Bayesian Optimization loop...")

    for iteration in range(cfg.budget):
        iter_start_time = time.time()
        if experiment:
            experiment.set_step(
                cfg.num_initial_points + iteration
            )  # Set step correctly

        train_x_normalized = normalize(train_x, bounds)
        train_x_normalized.clamp_(0.0, 1.0)

        LOGGER.info(f"Iteration {iteration+1}/{cfg.budget}: Fitting GP model...")
        try:
            model = SingleTaskGP(train_x_normalized, train_y, train_Yvar=train_y_var)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            LOGGER.info("GP model fitted successfully.")
            if experiment is not None:
                try:
                    experiment.log_metric(
                        "gp_lengthscale",
                        model.covar_module.base_kernel.lengthscale.item(),
                    )
                    experiment.log_metric(
                        "gp_outputscale", model.covar_module.outputscale.item()
                    )
                    experiment.log_metric("gp_noise", model.likelihood.noise.item())
                except AttributeError as e:
                    LOGGER.warning(f"Could not log GP hyperparameter: {e}")

        except Exception as e:
            LOGGER.error(f"GP model fitting failed: {e}")

        best_observed_value = train_y.max().item()
        acq_func = LogExpectedImprovement(
            model=model,
            best_f=best_observed_value,
            maximize=True,
        )

        try:
            candidate, acq_value = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.tensor(
                    [[0.0] * param_dim, [1.0] * param_dim],
                    dtype=torch.float64,
                    device=device,
                ),
                q=1,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            LOGGER.info("Acquisition function optimized.")
        except Exception as e:
            LOGGER.error(f"Acquisition function optimization failed: {e}")
            LOGGER.warning("Skipping iteration due to acquisition optimization error.")
            continue

        new_x_tensor = unnormalize(candidate.detach(), bounds)

        new_x_vector = new_x_tensor.squeeze(0).cpu().numpy()
        LOGGER.info(f"Evaluating candidate point {iteration+1}")
        new_y, new_y_var = evaluate_parameters(
            new_x_vector, initial_state_dict, sampler, cfg, device
        )

        new_x_normalized = normalize(new_x_tensor, bounds)
        new_x_normalized.clamp_(0.0, 1.0)
        train_x_normalized = torch.cat([train_x_normalized, new_x_normalized])
        train_x_normalized.clamp_(0.0, 1.0)

        train_x = torch.cat([train_x, new_x_tensor])
        train_y = torch.cat([train_y, new_y.view(1, -1)])
        train_y_var = torch.cat([train_y_var, new_y_var.view(1, -1)])

        train_y = train_y.to(device)
        train_y_var = train_y_var.to(device)

        iter_time = time.time() - iter_start_time
        current_best_reward = train_y.max().item()
        LOGGER.info(
            f"Iteration {iteration+1} finished in {iter_time:.2f}s. Acq value: {acq_value.item():.3f}. Candidate reward: {new_y.item():.3f}. Best reward so far: {current_best_reward:.3f}"
        )
        if experiment is not None:
            experiment.log_metric("acquisition_value", acq_value.item())
            experiment.log_metric("candidate_reward", new_y.item())
            experiment.log_metric("candidate_reward_variance", new_y_var.item())
            experiment.log_metric(
                "best_reward_so_far", current_best_reward
            )  # Renamed for clarity
            experiment.log_metric("iteration_time", iter_time)

    best_idx = train_y.argmax()
    best_reward = train_y[best_idx].item()
    best_params_tensor = unnormalize(
        train_x_normalized[best_idx], bounds
    )  # Use normalized x for lookup
    best_params_vector = best_params_tensor.cpu().numpy()

    LOGGER.info("--- Optimization Finished ---")
    LOGGER.info(f"Total time: {time.time() - run_start_time:.2f}s")
    LOGGER.info(
        f"Total evaluations: {cfg.num_initial_points + cfg.budget}"
    )  # Corrected total evaluations
    LOGGER.info(f"Best reward found: {best_reward:.4f}")
    if experiment is not None:
        final_step = (
            cfg.num_initial_points + cfg.budget
        )  # Define final step for summary metrics
        experiment.log_metric("final_best_reward", best_reward, step=final_step)
        experiment.log_metric(
            "total_runtime", time.time() - run_start_time, step=final_step
        )
        experiment.log_other("final_best_parameter_index", best_idx.item())

    try:
        LOGGER.info(f"Saving best policy state dict to {output_dir}/best_policy.pt")
        best_state_dict = vector_to_state_dict(best_params_vector, initial_state_dict)
        best_policy_path = output_dir / "best_policy.pt"
        torch.save(best_state_dict, best_policy_path)

        best_params_tensor_path = output_dir / "best_params_tensor.pt"
        torch.save(best_params_tensor, best_params_tensor_path)

        best_params_vector_path = output_dir / "best_params_vector.npy"
        np.save(best_params_vector_path, best_params_vector)

        training_data_path = output_dir / "training_data.pt"
        torch.save(
            {
                "train_x_normalized": train_x_normalized.cpu(),  # Save normalized for consistency
                "train_y": train_y.cpu(),
                "train_y_var": train_y_var.cpu(),
            },
            training_data_path,
        )
        if experiment is not None:
            LOGGER.info("Logging final assets to Comet...")
            experiment.log_asset(best_policy_path)
            experiment.log_asset(best_params_tensor_path)
            experiment.log_asset(best_params_vector_path)
            experiment.log_asset(training_data_path)

            LOGGER.info("Logging final policy model to Comet...")
            temp_policy = Police(
                env_template, **cfg.policy_kwargs
            )  # Create a policy instance
            temp_policy.load_state_dict(best_state_dict)  # Load the best weights
            log_model(experiment, temp_policy, "BestPolicy")

    except Exception as e:
        LOGGER.error(f"Failed to save or log final results: {e}")

    if experiment is not None:
        LOGGER.info("Ending Comet experiment.")
        experiment.end()

    LOGGER.info("BoTorch training finished successfully!")


if __name__ == "__main__":
    train()
