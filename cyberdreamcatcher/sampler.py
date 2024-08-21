import random

import numpy as np
from tqdm import trange
import torch


EPS = np.finfo(np.float32).eps.item()


class EpisodeSampler:
    def __init__(self, env, policy, seed=0, writer=None) -> None:
        self.env = env
        self.policy = policy

        self.seed = seed
        self.writer = writer

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def run_episode(self):
        """Compute a single episode given a policy and track useful quantities for learning."""

        # define initial conditions
        obs, info = self.env.reset(seed=self.seed)

        log_probs = []
        rewards = []
        done = False
        while not done:
            action, log_prob, entropy, value = self.policy(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            done = terminated or truncated

        # rewards to go per timestep
        # (no discount because episodes have fixed length)

        rewards_to_go = np.flip(np.cumsum(np.flip(np.array(rewards))))

        return rewards_to_go

    def sample_episodes(self, num_episodes, counter=None):
        """
        Executes multiple episodes under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        batch_rewards_to_go = [None for _ in range(num_episodes)]

        for epi in trange(num_episodes, desc="Sampling episodes"):
            rewards_to_go = self.run_episode()
            batch_rewards_to_go[epi] = rewards_to_go

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
                    # global_step=counter,
                )
                self.writer.add_histogram(
                    "distributions/final-reward",
                    stacked_rewards_to_go[:, -1],
                    # global_step=counter,
                )

        reward_mean = np.mean(batch_rewards_to_go)
        reward_std = np.std(batch_rewards_to_go)

        # for epi in range(num_episodes):
        #     reward_to_go_baselined = (batch_rewards_to_go[epi] - reward_mean) / (
        #         reward_std + EPS
        #     )

        return reward_mean, reward_std
