from pathlib import Path
from dataclasses import dataclass

import numpy as np
from tqdm import trange, tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import hydra
from hydra.core.config_store import ConfigStore

from cyberdreamcatcher.utils import set_all_seeds
from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.sampler import EpisodeSampler

EPS = np.finfo(np.float32).eps.item()


@dataclass
class Cfg:
    scenario: str = "Scenario2_-_User2_User4"
    episode_length: int = 30
    num_episodes_sample: int = 1000
    seed: int = 0
    learning_rate: float = 1e-2
    optimizer_iterations: int = 300
    num_jobs: int = -1


class REINFORCEParallel:
    def __init__(self, env, policy, conf, log_dir=None):
        self.env = env
        self.policy = policy
        self.conf = conf
        self.log_dir = log_dir
        self.optimizer_step = 0
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % (
                "\n".join(
                    [f"|{key}|{value}|" for key, value in self.conf.__dict__.items()]
                )
            ),
        )
        set_all_seeds(conf.seed)

    def sample_episodes(self, counter=None):
        """Executes multiple episodes in parallel under the current stochastic policy.
        Trajectories are sampled in parallel, but log_probs are recomputed in the main process.
        """
        num_episodes = self.conf.num_episodes_sample
        sampler = EpisodeSampler(
            seed=self.conf.seed,
            scenario=self.conf.scenario,
            episode_length=self.conf.episode_length,
            policy_weights=self.policy.state_dict(),
            num_jobs=self.conf.num_jobs,
        )
        batch_trajectories = sampler.sample_trajectories(num_episodes)
        batch_rewards_to_go = []
        batch_log_probs = []
        pbar = tqdm(
            total=len(batch_trajectories) * self.conf.episode_length,
            desc="Processing (obs, action) pairs",
        )

        # Set policy to evaluation mode before computing log_probs
        self.policy.eval()

        for i, episode in enumerate(batch_trajectories):
            set_all_seeds(self.conf.seed + i)

            obs_seq, actions_seq, rewards_seq = episode
            # Compute rewards-to-go
            rewards_to_go = np.flip(np.cumsum(np.flip(np.array(rewards_seq))))
            batch_rewards_to_go.append(rewards_to_go)
            # Recompute log_probs for each (obs, action) using the CURRENT policy
            log_probs = []
            for obs, action in zip(obs_seq, actions_seq):
                # Forward pass through policy
                report = self.policy(obs, action=action)
                log_probs.append(report.log_prob)
                pbar.update(1)
            batch_log_probs.append(log_probs)
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
        num_episodes = len(batch_rewards_to_go)
        for epi in range(num_episodes):
            reward_to_go_baselined = (batch_rewards_to_go[epi] - reward_mean) / (
                reward_std + EPS
            )
            for log_prob, reward_to_go in zip(
                batch_log_probs[epi], reward_to_go_baselined
            ):
                log_prob_R -= log_prob * reward_to_go
        mean_log_prob_R = log_prob_R / num_episodes
        return mean_log_prob_R, reward_mean, reward_std

    def learn(self):

        self.policy.train()

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


# Registering the Config class with the expected name 'args'.
cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)


@hydra.main(version_base=None, config_name="hydra", config_path="conf")
def main(cfg: Cfg) -> None:
    import os

    print(f"Working directory : {os.getcwd()}")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : {output_dir}")
    env = GraphWrapper(scenario=cfg.scenario, max_steps=cfg.episode_length)
    policy = Police(env, latent_node_dim=env.host_embedding_size)
    trainer = REINFORCEParallel(env, policy, cfg, log_dir=output_dir)
    params_dict = trainer.learn()
    # store trained policy
    file_path = Path(output_dir) / "trained_params.pt"
    torch.save(policy.state_dict(), file_path)
    trainer.writer.close()
    print("Voila!")


if __name__ == "__main__":
    main()
