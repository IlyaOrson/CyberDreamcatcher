# adapted from https://github.com/OptiMaL-PSE-Lab/REINFORCE-PSE

import numpy as np
from tqdm import trange

import torch


EPS = np.finfo(np.float32).eps.item()


def episode_reinforce(env, policy, seed=0):
    """Compute a single episode given a policy and track useful quantities for learning."""

    # define initial conditions
    obs, info = env.reset(seed=seed)

    sum_log_probs = 0.0
    sum_reward = 0.0
    done = False
    while not done:

        # logits = self.policy(state)
        # action, log_prob = sample_actions(logits)
        action, log_prob = policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        sum_log_probs += log_prob
        sum_reward += reward

        done = terminated or truncated

    return sum_reward, sum_log_probs

def sample_episodes_reinforce(env, policy, num_episodes, seed):
    """
    Executes multiple episodes under the current stochastic policy,
    gets an average of the reward and the summed log probabilities
    and use them to form the baselined loss function to optimize.
    """

    rewards = [None for _ in range(num_episodes)]
    sum_log_probs = [None for _ in range(num_episodes)]

    for epi in trange(num_episodes, desc="Sampling episodes"):
        reward, sum_log_prob = episode_reinforce(env, policy, seed=seed)
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


if __name__=="__main__":

    from blueskynet.env import GraphWrapper
    from blueskynet.policy import Police

    scenario = "Scenario2_-_User2_User4"
    # scenario = "Scenario2_+_User5_User6"
    env = GraphWrapper(scenario=scenario, max_steps=30)

    policy = Police(env, latent_node_dim=env.host_embedding_size)

    seed = 0
    size_episode_batch = 100
    learning_rate = 1e-1
    optimizer_iterations = 50

    # test sampling
    # mean_log_prob_R, reward_mean, reward_std = sample_episodes_reinforce(env, policy, size_episode_batch, seed)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    pbar = trange(optimizer_iterations, desc="Optimizer iteration")
    for iteration in pbar:

        mean_log_prob_R, reward_mean, reward_std = sample_episodes_reinforce(env, policy, size_episode_batch, seed)

        optimizer.zero_grad()
        mean_log_prob_R.backward()
        optimizer.step()

        pbar.write(
            f"Roll-out mean reward: {reward_mean:.3} +- {reward_std:.2}"
        )

    # store trained policy
    # torch.save(policy.state_dict(), policy_path)

    print("Voila!")
