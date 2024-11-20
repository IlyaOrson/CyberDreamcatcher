import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from joblib import Parallel, delayed

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police


EPS = np.finfo(np.float32).eps.item()

# Set unique seeds
# random.seed(episode_seed)
# torch.manual_seed(episode_seed)
# np.random.seed(episode_seed)


def run_episode(env, policy, seed):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # random.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)

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

    return rewards_to_go


class EpisodeSampler:
    def __init__(
        self,
        seed,
        scenario,
        episode_length,
        policy_weights=None,
        writer=None,
        num_jobs=1,
    ):
        self.seed = seed
        self.scenario = scenario
        self.episode_length = episode_length

        self.policy_weights = policy_weights
        self.writer = writer
        self.num_jobs = num_jobs  # Number of parallel jobs (-1 means use all cores)

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def sample_episodes(self, num_episodes, counter=None):
        """
        Executes multiple episodes in parallel under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """
        # Generate unique seeds for each episode
        seeds = [self.seed + i for i in range(num_episodes)]

        envs = [
            GraphWrapper(
                scenario=self.scenario,
                max_steps=self.episode_length,
                render_mode=None,
            )
            for _ in range(num_episodes)
        ]

        # from itertools import repeat
        # repeat(policy, times=num_episodes)

        # model_copy = type(mymodel)() # get a new instance
        # model_copy.load_state_dict(mymodel.state_dict()) # copy weights and stuff

        policy = Police(envs[0], latent_node_dim=envs[0].host_embedding_size)

        # load trained policy
        if self.policy_weights:
            policy.load_state_dict(self.policy_weights)

            # NOTE Call model.eval() to set dropout and batch normalization layers
            # to evaluation mode before running inference.
            # Failing to do this will yield inconsistent inference results.
            policy.eval()

        policies = [deepcopy(policy) for _ in range(num_episodes)]

        # Run episodes in parallel
        batch_rewards_to_go = Parallel(n_jobs=self.num_jobs, verbose=10)(
            delayed(run_episode)(env, policy, seed)
            for env, policy, seed in tqdm(
                zip(envs, policies, seeds), desc="Sampling episodes", total=num_episodes
            )
        )

        if self.writer:
            stacked_rewards_to_go = np.vstack(batch_rewards_to_go)
            if counter:
                self.writer.add_histogram(
                    "distributions/reward-to-go",
                    stacked_rewards_to_go[:, 0],
                    global_step=counter,
                )
                self.writer.add_histogram(
                    "distributions/final-reward",
                    stacked_rewards_to_go[:, -1],
                    global_step=counter,
                )
            else:
                self.writer.add_histogram(
                    "distributions/reward-to-go",
                    stacked_rewards_to_go[:, 0],
                )
                self.writer.add_histogram(
                    "distributions/final-reward",
                    stacked_rewards_to_go[:, -1],
                )

        reward_mean = np.mean(batch_rewards_to_go, axis=0)
        reward_std = np.std(batch_rewards_to_go, axis=0)

        return reward_mean, reward_std
