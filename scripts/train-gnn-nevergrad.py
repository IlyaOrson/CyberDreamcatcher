import os
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import comet_ml

from tqdm import tqdm
import nevergrad as ng
import numpy as np
import torch

from cyberdreamcatcher.utils import (
    set_all_seeds,
    state_dict_to_vector,
    vector_to_state_dict,
)
from cyberdreamcatcher.sampler import EpisodeSampler
from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police


@dataclass
class Cfg:
    seed: int = 0
    scenario: str = "Scenario2"
    episode_length: int = 30
    batch_size_episodes: int = 200
    num_jobs: int = -1
    # Nevergrad settings
    budget: int = 500  # Total number of 'ask' calls (parameter sets evaluated)
    use_mean_reward: bool = True
    optimizer: str = "TwoPointsDE"  # Or "TBPSA", "CMA", "PSO", "NG", etc.
    init_policy_path: str | None = None
    latent_node_dim: int = 4
    actor_heads: int = 1
    log_comet: bool = True


# --- Hydra Setup ---
cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)


@hydra.main(version_base=None, config_name="hydra", config_path="conf")
def main(cfg: Cfg) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {output_dir}")

    experiment = None  # Initialize experiment to None
    if cfg.log_comet:
        try:
            experiment = comet_ml.Experiment(
                project_name="cyberdreamcatcher",
                auto_param_logging=False,
                auto_metric_logging=False,
            )
            experiment.set_name(f"nevergrad_{cfg.optimizer}_seed_{cfg.seed}")
            # Use OmegaConf for consistency
            experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))
            experiment.log_html(f"<p>Output Directory: {output_dir}</p>")
        except Exception as e:
            print(f"WARNING: CometML initialization failed: {e}")
            experiment = None  # Ensure it's None on failure

    print("Configuration:")
    print(cfg)

    set_all_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Policy and Parameterization ---
    env = GraphEnv(scenario=cfg.scenario, max_steps=cfg.episode_length)
    # Create a dummy policy instance to get the state dict structure
    # It won't be used for sampling directly in the main thread
    policy_structure_provider = Police(
        env, latent_node_dim=cfg.latent_node_dim, actor_heads=cfg.actor_heads
    ).to(device)

    # Load initial weights if specified into the structure provider policy
    if cfg.init_policy_path:
        print(f"Loading initial policy weights from: {cfg.init_policy_path}")
        policy_structure_provider.load_state_dict(
            torch.load(cfg.init_policy_path, map_location=device)
        )
        policy_structure_provider.eval()  # Set to eval mode

    initial_state_dict = policy_structure_provider.state_dict()
    initial_params_vector = state_dict_to_vector(initial_state_dict)
    param_dim = len(initial_params_vector)
    print(f"Policy parameter dimension: {param_dim}")
    if experiment is not None:
        experiment.log_other("policy_parameter_dimension", param_dim)

    # Define the search space for nevergrad
    # Use the initial parameters as a starting point hint
    parametrization = ng.p.Array(init=initial_params_vector).set_name("policy_params")

    # --- Setup Optimizer ---
    # Adjust budget based on 'tell' calls
    optimizer = ng.optimizers.registry[cfg.optimizer](
        parametrization=parametrization, budget=cfg.budget
    )
    # Suggest initial point (optional but good practice)
    optimizer.suggest(initial_params_vector)

    # --- Prepare Sampler ---
    # Instantiate sampler once to reuse workers if backend allows
    sampler = EpisodeSampler(
        seed=cfg.seed,
        scenario=cfg.scenario,
        episode_length=cfg.episode_length,
        num_jobs=cfg.num_jobs,
        latent_node_dim=cfg.latent_node_dim,  # Added
        actor_heads=cfg.actor_heads,  # Added
        # policy_weights are set dynamically per 'ask' call
    )

    # --- Run Optimization using Ask/Tell ---
    print(f"Starting Nevergrad optimization with ask/tell interface...")
    print(f"Budget: {cfg.budget} 'ask' calls, {cfg.batch_size_episodes} episodes/ask.")

    pbar = tqdm(total=cfg.budget, desc="Nevergrad Optimization (Ask calls)")
    best_known_reward = float("-inf")  # Track best reward found so far
    step = 0

    for i in range(cfg.budget):
        if experiment is not None:
            experiment.set_step(step)
        candidate = optimizer.ask()
        params_vector = candidate.value

        # 1. Convert numpy vector back to state_dict
        current_state_dict = vector_to_state_dict(
            params_vector,
            initial_state_dict,  # Use initial_state_dict for structure
        )

        # 2. Update sampler's policy weights for this evaluation batch
        sampler.policy_weights = current_state_dict

        # 3. Sample episodes using the candidate parameters
        #    This leverages EpisodeSampler's internal parallelism

        # sample_episodes returns stacked_rewards_to_go, stacked_log_probs
        batch_rewards_to_go, _ = sampler.sample_episodes(cfg.batch_size_episodes)

        # 4. Extract total reward (reward-to-go at step 0) for each episode
        # Shape: (batch_size_episodes, len_episode)
        # The final score for a blue agent is the cumulative reward received by the agent over the course of the scenario run. https://github.com/cage-challenge/cage-challenge-2
        episode_rewards = batch_rewards_to_go[:, 0]  # reward-to-go at step 0
        # episode_rewards = batch_rewards_to_go[:, -1]  # reward at last step

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # 5. Tell Nevergrad the result
        #    Rewards in CybORG are penalties (negative), so negate for minimization
        if cfg.use_mean_reward:
            optimizer.tell(candidate, -mean_reward)
            # Log to Comet
            if experiment is not None:
                experiment.log_metric("reward_mean", mean_reward, step=step)
                experiment.log_metric("reward_std", std_reward, step=step)
                # Log the running best reward found so far
                if mean_reward > best_known_reward:
                    best_known_reward = mean_reward
                    experiment.log_metric(
                        "best_reward_so_far", best_known_reward, step=step
                    )

        else:
            # Tell optimizer each result individually
            # Note: check if this is appropriate for the chosen optimizer
            for reward_value in episode_rewards:
                optimizer.tell(candidate, -reward_value)

            # Track best reward observed across *all* individual episodes
            current_batch_best_reward = np.max(episode_rewards)
            if current_batch_best_reward > best_known_reward:
                best_known_reward = current_batch_best_reward
            # Log to Comet (log mean/std/best of the batch and running best)
            if experiment is not None:
                experiment.log_metric(
                    "batch_reward_mean", np.mean(episode_rewards), step=step
                )
                experiment.log_metric(
                    "batch_reward_std", np.std(episode_rewards), step=step
                )
                experiment.log_metric(
                    "best_reward_so_far", best_known_reward, step=step
                )

        pbar.update(1)
        pbar.set_postfix(
            {"Best Reward": f"{best_known_reward:.4f}"}
        )  # Use tracked best_known_reward
        step += 1

    pbar.close()

    # --- Get Recommendation and Save Results ---
    recommendation = optimizer.provide_recommendation()

    best_params_vector = recommendation.value
    # Note: recommendation.loss might be the loss of the specific point recommended,
    # or an average depending on the optimizer. best_known_loss tracked manually might be more intuitive.
    print(f"\nOptimization finished.")
    print(
        f"Optimizer recommended point loss: {recommendation.loss if recommendation.loss is not None else 'N/A'}"
    )
    print(f"Empirically best reward observed during run: {best_known_reward:.3f}")

    # Convert best params back to state_dict
    best_state_dict = vector_to_state_dict(best_params_vector, initial_state_dict)

    # Save the best policy state_dict
    best_policy_path = output_dir / "best_policy.pt"
    torch.save(best_state_dict, best_policy_path)
    print(f"Saved best policy weights to: {best_policy_path}")
    if experiment is not None:
        experiment.log_asset(best_policy_path)  # Log asset

    # Save the full optimizer state if needed for checkpointing (optional)
    optimizer_state_path = output_dir / "optimizer_state.pkl"
    with open(optimizer_state_path, "wb") as f:
        pickle.dump(optimizer.state, f)
    print(f"Saved optimizer state to: {optimizer_state_path}")
    if experiment is not None:
        experiment.log_asset(optimizer_state_path)  # Log asset

    print("\nVoila!")

    # --- Evaluate the final recommended policy ---
    print("\nEvaluating final recommended policy...")
    sampler.policy_weights = best_state_dict
    final_rewards_to_go, _ = sampler.sample_episodes(
        num_episodes=cfg.batch_size_episodes * 2
    )
    final_total_rewards = final_rewards_to_go[:, 0]
    final_mean_reward = np.mean(final_total_rewards)
    final_std_reward = np.std(final_total_rewards)
    print(
        f"Final Policy Mean Reward: {final_mean_reward:.4f} +/- {final_std_reward:.4f}"
    )
    if experiment is not None:
        experiment.log_metric("final_policy_mean_reward", final_mean_reward)
        experiment.log_metric("final_policy_std_reward", final_std_reward)
        experiment.log_histogram_3d(
            final_total_rewards, name="final_policy_reward_distribution"
        )
        # Log the empirically best reward found during the optimization run
        experiment.log_metric("optimization_best_reward_overall", best_known_reward)

    # You might want to log the actual best reward obtained *during* optimization too
    # if Nevergrad provides it easily, e.g., through recommendation.loss
    best_loss_from_optimizer = recommendation.loss
    if best_loss_from_optimizer is not None:
        print(
            f"Best loss reported by optimizer for recommendation: {best_loss_from_optimizer}"
        )
        # Remember loss is negative reward
        if experiment is not None:
            experiment.log_metric(
                "recommended_policy_optimization_loss", best_loss_from_optimizer
            )
            experiment.log_metric(
                "recommended_policy_optimization_reward", -best_loss_from_optimizer
            )

    if experiment is not None:
        experiment.end()


if __name__ == "__main__":
    main()
