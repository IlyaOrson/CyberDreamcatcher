import os
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import torch
import nevergrad as ng
from tqdm import tqdm
import hydra
from hydra.core.config_store import ConfigStore

from cyberdreamcatcher.utils import (
    set_all_seeds,
    state_dict_to_vector,
    vector_to_state_dict,
)
from cyberdreamcatcher.sampler import EpisodeSampler
from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police


@dataclass
class Cfg:
    seed: int = 0
    scenario: str = "Scenario2"
    episode_length: int = 30
    num_episodes_sample: int = 100
    num_jobs: int = -1
    # Nevergrad settings
    budget: int = 500  # Total number of 'ask' calls (parameter sets evaluated)
    optimizer_name: str = "TBPSA"  # Or "OnePlusOne", "CMA", etc.
    init_policy_path: str | None = None  # Optional path to load initial policy weights
    # Policy settings (if needed, e.g., hidden dims)
    policy_kwargs: dict = field(default_factory=dict)


# --- Hydra Setup ---
cs = ConfigStore.instance()
cs.store(name="config", node=Cfg)


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg: Cfg) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {output_dir}")
    print("Configuration:")
    print(cfg)

    set_all_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize Policy and Parameterization ---
    env = GraphWrapper(scenario=cfg.scenario, max_steps=cfg.episode_length)
    # Create a dummy policy instance to get the state dict structure
    # It won't be used for sampling directly in the main thread
    policy_structure_provider = Police(env, **cfg.policy_kwargs).to(device)

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

    # Define the search space for nevergrad
    # Use the initial parameters as a starting point hint
    parametrization = ng.p.Array(init=initial_params_vector).set_name("policy_params")

    # --- Setup Optimizer ---
    # Adjust budget based on 'tell' calls
    optimizer = ng.optimizers.registry[cfg.optimizer_name](
        parametrization=parametrization, budget=cfg.budget * cfg.num_episodes_sample
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
        # policy_weights are set dynamically per 'ask' call
    )

    # --- Run Optimization using Ask/Tell ---
    print(f"Starting Nevergrad optimization with ask/tell interface...")
    print(
        f"Budget: {cfg.budget} 'ask' calls, {cfg.num_episodes_sample} episodes/ask, {optimizer.budget} total 'tell' calls."
    )

    pbar = tqdm(total=cfg.budget, desc="Nevergrad Optimization (Ask calls)")
    best_known_reward = float("-inf")  # Track best reward found so far

    for i in range(cfg.budget):
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
        batch_rewards_to_go, _ = sampler.sample_episodes(cfg.num_episodes_sample)

        # 4. Extract total reward (reward-to-go at step 0) for each episode
        # Shape: (num_episodes_sample, len_episode)
        # The final score for a blue agent is the cumulative reward received by the agent over the course of the scenario run. https://github.com/cage-challenge/cage-challenge-2
        episode_rewards = batch_rewards_to_go[:, 0]  # reward-to-go at step 0
        # episode_rewards = batch_rewards_to_go[:, -1]  # reward at last step

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        # 5. Tell Nevergrad the result for EACH episode
        for reward_value in episode_rewards:
            # Rewards in CybORG are penalties (negative), so negate for minimization
            optimizer.tell(candidate, -reward_value)

        # Track best reward observed across all tells
        current_best_reward = np.max(episode_rewards)
        if current_best_reward > best_known_reward:
            best_known_reward = current_best_reward

        # Save a checkpoint of the best policy so far
        state_dict = vector_to_state_dict(params_vector, initial_state_dict)
        policy_path = output_dir / f"best_policy_{i}.pt"
        torch.save(state_dict, policy_path)

        # Save the full optimizer state if needed for checkpointing
        optimizer_state_path = output_dir / f"optimizer_state_{i}.pkl"
        with open(optimizer_state_path, "wb") as f:
            pickle.dump(optimizer, f)  # Save the whole optimizer instance

        pbar.set_postfix(
            {
                "best_reward": f"{best_known_reward:.3f}",
                "mean_reward": f"{mean_reward:.3f}",
                "std_reward": f"{std_reward:.3f}",
            }
        )


        pbar.update(1)  # Update progress bar after processing one 'ask' candidate

    pbar.close()

    # --- Get Recommendation and Save Results ---
    recommendation = optimizer.provide_recommendation()  # Get the final best candidate

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

    # Save the full optimizer state if needed for checkpointing (optional)
    optimizer_state_path = output_dir / "optimizer_state.pkl"
    with open(optimizer_state_path, "wb") as f:
        pickle.dump(optimizer, f)  # Save the whole optimizer instance
    print(f"Saved full optimizer state to: {optimizer_state_path}")

    print("\nVoila!")


if __name__ == "__main__":
    main()

# NOTE: Nevergrad optimizer selection

# TBPSA (Tree-structured Parzen Algorithm):
# Why: Explicitly designed for noisy problems and handles high parallelism (num_workers) well. This seems like a very strong match given that your objective function relies on sampled episode returns, which are inherently noisy.
# Consideration: Might be computationally slightly more intensive than simpler methods per evaluation.

# PSO (Particle Swarm Optimization):
# Why: Known for its robustness and good performance on continuous problems. It also scales well with high parallelism. A solid, well-understood choice.
# Consideration: Can sometimes converge prematurely on complex landscapes, but generally reliable. also scales well with high parallelism. A solid, well-understood choice.
