# adapted from https://github.com/OptiMaL-PSE-Lab/REINFORCE-PSE

import random
from dataclasses import dataclass
# from itertools import accumulate

import numpy as np
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter


EPS = np.finfo(np.float32).eps.item()


@dataclass
class Cfg:
    scenario: str = "Scenario2_-_User2_User4"  # "Scenario2_+_User5_User6"
    episode_length: int = 30
    num_episodes_sample: int = 1000
    seed: int = 0
    learning_rate: float = 1e-2
    optimizer_iterations: int = 300


class REINFORCE:
    def __init__(self, env, policy, conf, log_dir=None) -> None:
        self.env = env
        self.policy = policy
        self.conf = conf
        self.log_dir = log_dir

        self.optimizer_step = 0
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in self.conf.items()])),
        )

        random.seed(conf.seed)
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

    def run_episode(self):
        """Compute a single episode given a policy and track useful quantities for learning."""

        # define initial conditions
        obs, info = self.env.reset(seed=self.conf.seed)

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
        # pure python version
        # rewards_to_go = list(reversed(list(accumulate(reversed(rewards)))))
        rewards_to_go = np.flip(np.cumsum(np.flip(np.array(rewards))))

        return rewards_to_go, log_probs

    def sample_episodes(self, counter=None):
        """
        Executes multiple episodes under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        num_episodes = self.conf.num_episodes_sample
        batch_rewards_to_go = [None for _ in range(num_episodes)]
        batch_log_probs = [None for _ in range(num_episodes)]

        for epi in trange(num_episodes, desc="Sampling episodes"):
            rewards_to_go, log_probs = self.run_episode()
            batch_rewards_to_go[epi] = rewards_to_go
            batch_log_probs[epi] = log_probs

        if counter:
            stacked_rewards_to_go = np.vstack(batch_rewards_to_go)
            self.writer.add_histogram(
                "reward-to-go distribution",
                stacked_rewards_to_go[:, 0],
                global_step=counter,
            )
            self.writer.add_histogram(
                "final reward distribution",
                stacked_rewards_to_go[:, -1],
                global_step=counter,
            )

        reward_mean = np.mean(batch_rewards_to_go)
        reward_std = np.std(batch_rewards_to_go)

        log_prob_R = 0.0
        for epi in range(num_episodes):
            # TODO is normalizing a valid baseline?
            reward_to_go_baselined = (batch_rewards_to_go[epi] - reward_mean) / (
                reward_std + EPS
            )

            # invert signs to maximize reward
            log_prob_R -= torch.sum(
                torch.mul(
                    torch.stack(batch_log_probs[epi]),
                    torch.tensor(reward_to_go_baselined),
                )
            )
            for log_prob, reward_to_go in zip(
                batch_log_probs[epi], reward_to_go_baselined
            ):
                log_prob_R -= log_prob * reward_to_go

        mean_log_prob_R = log_prob_R / num_episodes
        return mean_log_prob_R, reward_mean, reward_std

    def learn(self):
        optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.conf.learning_rate
        )

        pbar = trange(self.conf.optimizer_iterations, desc="Optimizer iteration")
        for it in pbar:
            mean_log_prob_R, reward_mean, reward_std = self.sample_episodes(counter=it)

            mean_log_prob_R.backward()
            optimizer.step()
            optimizer.zero_grad()

            self.writer.add_scalar("mean reward", reward_mean, global_step=it)
            self.writer.add_scalar("mean std", reward_std, global_step=it)

            pbar.write(f"Roll-out mean reward: {reward_mean:.3} +- {reward_std:.2}")

            if it % 20 == 0:
                file_path = Path(self.log_dir) / f"trained_params_iter_{it}.pt"
                torch.save(self.policy.state_dict(), file_path)

        return self.policy.state_dict()


if __name__ == "__main__":
    import os
    from pathlib import Path

    import hydra
    from hydra.core.config_store import ConfigStore

    from cyberdreamcatcher.env import GraphWrapper
    from cyberdreamcatcher.policy import Police

    # Registering the Config class with the expected name 'args'.
    # https://hydra.cc/docs/tutorials/structured_config/minimal_example/
    cs = ConfigStore.instance()
    cs.store(name="args", node=Cfg)

    @hydra.main(version_base=None, config_name="hydra", config_path="conf")
    def main(cfg: Cfg) -> None:
        # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
        print(f"Working directory : {os.getcwd()}")
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(f"Output directory  : {output_dir}")

        env = GraphWrapper(scenario=cfg.scenario, max_steps=cfg.episode_length)

        policy = Police(env, latent_node_dim=env.host_embedding_size)

        trainer = REINFORCE(env, policy, cfg, log_dir=output_dir)

        params_dict = trainer.learn()

        # store trained policy
        file_path = Path(output_dir) / "trained_params.pt"
        torch.save(policy.state_dict(), file_path)

        trainer.writer.close()

        # policy.load_state_dict(params_dict)
        # policy.load_state_dict(torch.load(file_path))
        # NOTE Call model.eval() to set dropout and batch normalization layers
        # to evaluation mode before running inference.
        # Failing to do this will yield inconsistent inference results.
        # policy.eval()

        print("Voila!")

    main()
