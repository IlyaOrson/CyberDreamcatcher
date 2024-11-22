import random
import numpy as np
from tqdm import trange

import torch
from joblib import Parallel, delayed

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police


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
        num_jobs=1,
    ):
        self.seed = seed
        self.scenario = scenario
        self.episode_length = episode_length

        self.policy_weights = policy_weights
        self.num_jobs = num_jobs  # Number of parallel jobs (-1 means use all cores)

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def sample_episodes(self, num_episodes):
        """
        Executes multiple episodes in parallel under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        def _run_episode(seed, scenario, episode_length, policy_weights):
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

                # NOTE Call model.eval() to set dropout and batch normalization layers
                # to evaluation mode before running inference.
                # Failing to do this will yield inconsistent inference results.
                policy.eval()

            return run_episode(env, policy, seed)

        # Run episodes in parallel
        batch_rewards_to_go = Parallel(n_jobs=self.num_jobs, verbose=10)(
            delayed(_run_episode)(
                self.seed + i, self.scenario, self.episode_length, self.policy_weights
            )
            for i in trange(num_episodes, desc="Sampling episodes")
        )

        stacked_rewards_to_go = np.vstack(batch_rewards_to_go)

        return stacked_rewards_to_go  # a row per episode
