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

