# adapted from https://github.com/OptiMaL-PSE-Lab/REINFORCE-PSE

from dataclasses import dataclass

import numpy as np
from tqdm import trange
import torch


EPS = np.finfo(np.float32).eps.item()


@dataclass
class Conf:
    scenario: str
    episode_length: int
    num_episodes_sample: int
    seed: int
    learning_rate: float
    optimizer_iterations: int


class REINFORCE:
    def __init__(self, env, policy, conf) -> None:
        self.env = env
        self.policy = policy
        self.conf = conf

        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    def run_episode(self):
        """Compute a single episode given a policy and track useful quantities for learning."""

        # define initial conditions
        obs, info = self.env.reset(seed=self.conf.seed)

        sum_log_probs = 0.0
        sum_reward = 0.0
        done = False
        while not done:
            action, log_prob = self.policy(obs)

            obs, reward, terminated, truncated, info = self.env.step(action)

            sum_log_probs += log_prob
            sum_reward += reward

            done = terminated or truncated

        return sum_reward, sum_log_probs

    def sample_episodes(self):
        """
        Executes multiple episodes under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        num_episodes = self.conf.num_episodes_sample
        rewards = [None for _ in range(num_episodes)]
        sum_log_probs = [None for _ in range(num_episodes)]

        for epi in trange(num_episodes, desc="Sampling episodes"):
            reward, sum_log_prob = self.run_episode()
            rewards[epi] = reward
            sum_log_probs[epi] = sum_log_prob

        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)

        log_prob_R = 0.0
        for epi in reversed(range(num_episodes)):
            baselined_reward = (rewards[epi] - reward_mean) / (reward_std + EPS)
            # invert signs to maximize reward
            log_prob_R -= sum_log_probs[epi] * baselined_reward

        mean_log_prob_R = log_prob_R / num_episodes
        return mean_log_prob_R, reward_mean, reward_std

    def learn(self):
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.conf.learning_rate)

        pbar = trange(self.conf.optimizer_iterations, desc="Optimizer iteration")
        for _ in pbar:
            mean_log_prob_R, reward_mean, reward_std = self.sample_episodes()

            mean_log_prob_R.backward()
            optimizer.step()
            optimizer.zero_grad()

            pbar.write(f"Roll-out mean reward: {reward_mean:.3} +- {reward_std:.2}")

        return policy.state_dict()


if __name__ == "__main__":
    from blueskynet.env import GraphWrapper
    from blueskynet.policy import Police

    conf = Conf(
        scenario="Scenario2_-_User2_User4",  # "Scenario2_+_User5_User6"
        episode_length=30,
        num_episodes_sample=10,
        seed=0,
        learning_rate=1e-4,
        optimizer_iterations=100,
    )

    env = GraphWrapper(scenario=conf.scenario, max_steps=conf.episode_length)

    policy = Police(env, latent_node_dim=env.host_embedding_size)

    trainer = REINFORCE(env, policy, conf)

    # test sampling
    mean_log_prob_R, reward_mean, reward_std = trainer.sample_episodes()

    params_dict = trainer.learn()

    # store trained policy
    # torch.save(policy.state_dict(), policy_path)

    print("Voila!")
