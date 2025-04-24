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
    num_episodes_sample: int = 200
    num_jobs: int = -1
    # Nevergrad settings
    budget: int = 500  # Total number of 'ask' calls (parameter sets evaluated)
    use_mean_reward: bool = True  # Policy settings (if needed, e.g., hidden dims)
    optimizer: str = "TwoPointsDE"  # Or "TBPSA", "CMA", "PSO", "NG", etc.
    init_policy_path: str | None = None  # Optional path    
    policy_kwargs: dict = field(default_factory=dict)


# --- Hydra Setup ---
cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)


@hydra.main(version_base=None, config_name="hydra", config_path="conf")
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
        # policy_weights are set dynamically per 'ask' call
    )

    # --- Run Optimization using Ask/Tell ---
    print(f"Starting Nevergrad optimization with ask/tell interface...")
    print(f"Budget: {cfg.budget} 'ask' calls, {cfg.num_episodes_sample} episodes/ask.")

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

        # 5. Tell Nevergrad the result
        #    Rewards in CybORG are penalties (negative), so negate for minimization
        if cfg.use_mean_reward:
            optimizer.tell(candidate, -mean_reward)
            # Track best reward observed across
            if mean_reward > best_known_reward:
                best_known_reward = mean_reward
        else:
            # Tell optimizer each result individually
            # Note: check if this is appropriate for the chosen optimizer
            for reward_value in episode_rewards:
                optimizer.tell(candidate, -reward_value)

            # Track best reward observed across
            current_best_reward = np.max(episode_rewards)
            if current_best_reward > best_known_reward:
                best_known_reward = current_best_reward

        if i % 10 == 0:
            # Save a checkpoint of the best policy so far
            state_dict = vector_to_state_dict(params_vector, initial_state_dict)
            policy_path = output_dir / f"best_policy_{i}.pt"
            torch.save(state_dict, policy_path)

            # Save the full optimizer state if needed for checkpointing
            optimizer_state_path = output_dir / f"optimizer_state_{i}.pkl"
            with open(optimizer_state_path, "wb") as file:
                pickle.dump(optimizer, file)  # Save the whole optimizer instance

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
# https://facebookresearch.github.io/nevergrad/optimization.html#choosing-an-optimizer

# Start with CMA. It's a powerful, well-regarded default for this kind of problem.
# Also try TwoPointsDE (or another DE variant). It's often competitive and sometimes better, especially with high noise.
# If seeking potentially higher performance (and willing to explore), investigate the NGOpt family.
# Regardless of the algorithm, provide the mean of your 1000 noisy samples as the objective function value returned to the optimizer.

# Key Problem Characteristics:

# Blackbox: No gradient information available directly from the RL environment/policy interaction.
# Dimensionality: ~500 - 1000 parameters. This is moderate-to-high dimensionality for blackbox optimization.
# Noise: High noise, inherent in RL reward signals (stochastic environments, exploration, initial conditions). You get 1000 noisy samples per parameter evaluation.
# Evaluation Output: Can provide mean/variance or raw samples.

# Analysis & Recommendations:

# CMA (Covariance Matrix Adaptation Evolution Strategy):

# Why: CMA-ES is often a very strong baseline for continuous optimization problems up to several hundred or even a thousand dimensions. It adapts the covariance matrix of its search distribution, allowing it to learn correlations between parameters and effectively navigate complex fitness landscapes. It's known for its relative robustness to noise, especially when using population averaging inherent in the algorithm.
# Suitability: Excellent fit. Handles dimensionality, reasonable noise robustness. It implicitly averages over its population, smoothing out some noise effects.

# TwoPointsDE or other DE Variants (Differential Evolution):

# Why: DE is a population-based algorithm that relies on differences between population members to create new candidate solutions. This reliance on ranking and differences often makes it quite robust to noise. It scales reasonably well with dimensions. TwoPointsDE is often recommended by the Nevergrad developers as a strong DE variant.
# Suitability: Very good fit. Handles dimensionality, good noise robustness. Simpler concept than CMA but often very effective.

# NGOpt Family (e.g., NGOpt, DiagonalCMA depending on Nevergrad version/availability):

# Why: These often represent more recent or advanced Estimation of Distribution Algorithms (EDAs) or Natural Evolution Strategies (NES). They are specifically designed for optimizing noisy functions and can sometimes outperform CMA or DE, especially if tuned correctly. DiagonalCMA could be relevant if you suspect low parameter correlation.
# Suitability: Potentially excellent, possibly state-of-the-art within Nevergrad for this task. Might require more understanding of their specific mechanisms or slight tuning.

# TBPSA (Tree-structured Parzen Estimator Bayesian Sampling Algorithm):

# Why: This is a sequential model-based optimization (SMBO) method, similar to Bayesian Optimization but using Parzen estimators instead of Gaussian Processes typically. It's known to work well in hyperparameter optimization (like Optuna uses) and can handle conditional parameters. It models the probability of good vs. bad points.
# Suitability: Moderate fit. It can handle noise, but its performance compared to ES methods at 500-1000 dimensions for direct policy search is less consistently documented than CMA/DE/NES. Model building can add overhead.

# PSO (Particle Swarm Optimization):

# Why: Another population-based algorithm inspired by social behavior. Particles 'fly' through the search space, influenced by their own best position and the swarm's best position.
# Suitability: Moderate fit. Can work well, but sometimes prone to premature convergence compared to DE or CMA. Noise robustness is decent due to population averaging effects.

# Regarding Evaluation Output:

# Mean and Variance: Most standard Nevergrad optimizers (CMA, DE, PSO, NGOpt) primarily use the mean (or a single scalar fitness value) returned by the function evaluation. They don't explicitly use the variance information.

# Bayesian Optimization (BO): If you were considering BO (which might struggle at 1000 dimensions without specific techniques like random embeddings or specialized kernels), providing the variance could be very useful to inform the noise level (alpha parameter) in its Gaussian Process model. However, given the dimensionality and common practices in RL policy search, CMA/DE/NGOpt are often preferred over standard BO.

# Raw Noisy Evaluations: Standard Nevergrad optimizers typically expect one objective value per function call (representing one parameter vector evaluation). They are not designed to directly process the 1000 raw samples. You would almost certainly calculate the mean (or perhaps a robust statistic like the median, or a Conditional Value at Risk - CVaR) from your 1000 samples and return that single scalar value to the optimizer.
