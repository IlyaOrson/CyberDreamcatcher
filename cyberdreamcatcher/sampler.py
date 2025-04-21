import numpy as np
from tqdm import trange, tqdm

import torch
from joblib import Parallel, delayed

from cyberdreamcatcher.utils import set_all_seeds
from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police


def collect_rewards_log_probs(env, policy, seed):
    """Compute a single episode given a policy and track useful quantities for learning."""

    set_all_seeds(seed)

    # define initial conditions
    obs, info = env.reset(seed=seed)

    log_probs = []
    rewards = []
    done = False
    while not done:
        action, log_prob, entropy, value = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        done = terminated or truncated

    # rewards to go per timestep
    # (no discount because episodes have fixed length)
    rewards_to_go = np.flip(np.cumsum(np.flip(np.array(rewards))))

    return rewards_to_go, log_probs


def collect_trajectory(env, policy, seed):
    """
    Compute a single episode using the provided policy and environment.
    Results need to be serializable (e.g., CPU tensors, dicts, etc.).
    """

    set_all_seeds(seed)

    # NOTE Call model.eval() to set dropout and batch normalization layers
    # to evaluation mode before running inference.
    # Failing to do this will yield inconsistent inference results.
    if hasattr(policy, "eval"):
        policy.eval()

    # Reset the environment
    # Ensure obs is serializable. If it's a complex object or GPU tensor,
    # it might need conversion before appending/returning.
    obs, info = env.reset(seed=seed)

    all_obs = []
    all_actions = []
    all_rewards = []
    done = False

    # Disable gradient calculations during trajectory rollout
    with torch.no_grad():
        while not done:
            # Store the current observation *before* taking the step.
            # Handle potential complex observation types (e.g., graph data).
            # If obs is a torch tensor, move to CPU. If custom object, ensure pickleable.
            # Assuming obs is directly appendable for now. Needs verification based on GraphWrapper.
            processed_obs = obs  # Placeholder: Add conversion if needed (e.g., obs.cpu(), obs.to_dict())
            all_obs.append(processed_obs)

            # Get action from the policy
            action, log_prob, entropy, value = policy(obs)

            # Convert action tensor to a basic type or CPU tensor for storage/serialization
            # Assuming action_tensor is a single value tensor. Adjust if multi-dimensional.
            all_actions.append(action)

            # Step the environment
            obs, reward, terminated, truncated, info = env.step(
                action
            )  # Use the processed action

            # Store reward
            all_rewards.append(reward)

            # Check termination conditions
            done = terminated or truncated

    # Return the raw trajectory components
    return all_obs, all_actions, all_rewards


class EpisodeSampler:
    def __init__(
        self,
        seed,
        scenario,
        episode_length,
        policy_weights=None,
        num_jobs=1,
    ):
        self.seed = seed
        self.scenario = scenario
        self.episode_length = episode_length

        self.policy_weights = policy_weights
        self.num_jobs = num_jobs  # Number of parallel jobs (-1 means use all cores)

        set_all_seeds(self.seed)

    def sample_episodes(self, num_episodes):
        """
        Executes multiple episodes in parallel under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        def _collect_rewards_log_probs(seed, scenario, episode_length, policy_weights):
            "Create an independent environment and policy"
            env = GraphWrapper(
                scenario=scenario,
                max_steps=episode_length,
                render_mode=None,
            )
            policy = Police(env, latent_node_dim=env.host_embedding_size)

            # load trained policy
            if policy_weights:
                policy.load_state_dict(policy_weights)

            return collect_rewards_log_probs(env, policy, seed)

        # Run episodes in parallel
        # unordered because there is no need to track seeds <--> episodes
        parallel_generator = Parallel(
            n_jobs=self.num_jobs, return_as="generator_unordered"
        )(
            delayed(_collect_rewards_log_probs)(
                self.seed + i, self.scenario, self.episode_length, self.policy_weights
            )
            for i in range(num_episodes)
        )
        batch_rewards_to_go = [
            _
            for _ in tqdm(
                parallel_generator,
                total=num_episodes,
                desc="Collecting rewards and log probabilities",
            )
        ]

        stacked_rewards_to_go = np.vstack([t[0] for t in batch_rewards_to_go])
        stacked_log_probs = torch.stack(
            [torch.stack(t[1]) for t in batch_rewards_to_go]
        )

        return stacked_rewards_to_go, stacked_log_probs  # a row per episode

    def sample_trajectories(self, num_episodes):
        """
        Executes multiple episodes in parallel under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        def _collect_trajectory(seed, scenario, episode_length, policy_weights):
            "Create an independent environment and policy"
            env = GraphWrapper(
                scenario=scenario,
                max_steps=episode_length,
                render_mode=None,
            )
            policy = Police(env, latent_node_dim=env.host_embedding_size)

            # load trained policy
            if policy_weights:
                policy.load_state_dict(policy_weights)

            return collect_trajectory(env, policy, seed)

        # Run episodes in parallel
        # ordered so that seeds are consecutive
        # if seeds <--> episodes are tracked we can use the same seed
        # in the main process for log_prob recalculation to match the worker

        # NOTE  Sampling actions in workers, recomputing log_probs on the main thread
        #       is the correct pattern for vanilla PyTorch + joblib/multiprocessing...
        #       However, this approach only works if:
        #       - The policy in the main process is in the exact same state as the workers
        #         at sampling time (weights, random state, dropout/batchnorm, etc.).
        #       - The action sampling is deterministic given the observation and random seed
        #         (or you store the random seed used for each action and replay it).
        #       - If there is any mismatch, the recomputed log_prob may not match the action
        #         that was actually taken, leading to incorrect gradients.
        parallel_generator = Parallel(n_jobs=self.num_jobs, return_as="generator")(
            delayed(_collect_trajectory)(
                self.seed + i, self.scenario, self.episode_length, self.policy_weights
            )
            for i in range(num_episodes)
        )
        batch_trajectories = [
            _
            for _ in tqdm(
                parallel_generator,
                total=num_episodes,
                desc="Collecting trajectories",
            )
        ]
        return batch_trajectories  # a row per episode
