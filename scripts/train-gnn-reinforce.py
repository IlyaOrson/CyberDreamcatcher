from dataclasses import dataclass

import numpy as np
from tqdm import trange
import torch
from torch.utils.tensorboard import SummaryWriter

from cyberdreamcatcher.utils import set_all_seeds
from cyberdreamcatcher.sampler import collect_rewards_log_probs

EPS = np.finfo(np.float32).eps.item()


@dataclass
class Cfg:
    scenario: str = "Scenario2"  # "Scenario2_+_User5_User6"
    episode_length: int = 30
    num_episodes_sample: int = 500
    seed: int = 0
    learning_rate: float = 1e-2
    optimizer_iterations: int = 200
    latent_node_dim: int = 3
    normalize_advantage: bool = False


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

        set_all_seeds(conf.seed)

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
            rewards_to_go, log_probs = collect_rewards_log_probs(
                self.env, self.policy, self.conf.seed
            )
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

        rewards_to_go = np.stack(batch_rewards_to_go)
        log_probs = torch.stack(batch_log_probs)

        reward_mean = np.mean(rewards_to_go[:,0])
        reward_std = np.std(rewards_to_go[:,0])

        # Calculate the time-step-dependent baseline (mean across episodes for each time step)
        # Shape: (episode_length,)
        baselines_t = np.mean(rewards_to_go, axis=0)

        # Calculate advantages A_t = R_t - b_t using broadcasting
        # NumPy automatically subtracts the 1D baselines_t from each row of rewards_to_go
        # Shape: (num_episodes, episode_length)
        advantages = rewards_to_go - baselines_t

        # A'_{i,t} = (A_{i,t} - mean_A) / (std_A + EPS)
        if self.conf.normalize_advantage:
            advantages_mean = np.mean(advantages)
            advantages_std = np.std(advantages)
            advantages = (advantages - advantages_mean) / (advantages_std + EPS)

        # invert signs to maximize reward
        log_prob_R = - torch.sum(torch.mul(log_probs, torch.tensor(advantages)))

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

            pbar.write(f"Roll-out mean reward: {reward_mean:.3f} +- {reward_std:.2f}")
            pbar.set_postfix(
                {
                    "reward mean": f"{reward_mean:.3f}",
                    "reward std": f"{reward_std:.2f}",
                }
            )

            if it % 10 == 0:
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

        policy = Police(env, latent_node_dim=cfg.latent_node_dim)

        trainer = REINFORCE(env, policy, cfg, log_dir=output_dir)

        params_dict = trainer.learn()

        # store trained policy
        file_path = Path(output_dir) / "trained_params.pt"
        # torch.save(params_dict, file_path)
        torch.save(policy.state_dict(), file_path)

        trainer.writer.close()

        print("Voila!")

    main()
