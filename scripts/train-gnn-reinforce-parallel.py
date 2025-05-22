from pathlib import Path
from dataclasses import dataclass
import logging
import gc

import numpy as np
from tqdm import trange, tqdm
import torch
import comet_ml
from comet_ml.integration.pytorch import log_model
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rich.logging import RichHandler

from cyberdreamcatcher.utils import set_all_seeds
from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.sampler import EpisodeSampler

EPS = np.finfo(np.float32).eps.item()

LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    scenario: str = "Scenario2"
    episode_length: int = 30
    batch_size_episodes: int = 1000
    seed: int = 0
    learning_rate: float = 1e-2
    optimizer_iterations: int = 300
    num_jobs: int = -1
    normalize_advantage: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    latent_node_dim: int = 3
    log_comet: bool = True
    log_level: str = "INFO"


class REINFORCEParallel:
    def __init__(self, env, policy, conf, output_dir):
        self.env = env
        self.policy = policy
        self.policy.to(conf.device)

        self.conf = conf
        self.output_dir = output_dir
        self.optimizer_step = 0

        self.experiment = None
        if conf.log_comet:
            try:
                self.experiment = comet_ml.Experiment(
                    project_name="cyberdreamcatcher",
                    auto_param_logging=False,
                    auto_metric_logging=False,
                )
                self.experiment.set_name(f"reinforce_parallel_seed_{conf.seed}")
                self.experiment.log_parameters(
                    OmegaConf.to_container(conf, resolve=True)
                )
                self.experiment.log_html(f"<p>Output Directory: {self.output_dir}</p>")
            except Exception as e:
                LOGGER.warning(f"CometML initialization failed: {e}")
                self.experiment = None

        set_all_seeds(conf.seed)

    def sample_episodes(self, counter=None):
        num_episodes = self.conf.batch_size_episodes
        policy_weights = {k: v.cpu() for k, v in self.policy.state_dict().items()}
        sampler = EpisodeSampler(
            seed=self.conf.seed,
            scenario=self.conf.scenario,
            episode_length=self.conf.episode_length,
            policy_weights=policy_weights,
            num_jobs=self.conf.num_jobs,
        )
        batch_trajectories = sampler.sample_trajectories(num_episodes)
        batch_rewards_to_go = []
        batch_log_probs = []
        pbar = tqdm(
            total=num_episodes,
            desc="Processing trajectories",
            leave=False,
        )

        self.policy.eval()

        # TODO: use batch processing on GPU (requires calling policy on batch of observations to get only logits)
        for _, episode in enumerate(batch_trajectories):
            obs_seq, actions_seq, rewards_seq, log_probs_seq = episode
            rewards_to_go = np.flip(np.cumsum(np.flip(np.array(rewards_seq))))
            batch_rewards_to_go.append(rewards_to_go)
            log_probs = []
            for obs, action in zip(obs_seq, actions_seq):
                obs = obs.to(self.conf.device)
                action = action.to(self.conf.device)
                report = self.policy(obs, action=action)
                log_probs.append(report.log_prob)
            assert all(torch.isclose(p, l) for (p, l) in zip(log_probs_seq, log_probs))
            batch_log_probs.append(torch.stack(log_probs))
            pbar.update(1)

        pbar.close()

        rewards_to_go = np.stack(batch_rewards_to_go)
        log_probs = torch.stack(batch_log_probs)

        total_rewards = rewards_to_go[:, 0]
        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)

        baselines_t = np.mean(rewards_to_go, axis=0)

        advantages = rewards_to_go - baselines_t

        if self.conf.normalize_advantage:
            advantages_mean = np.mean(advantages)
            advantages_std = np.std(advantages)
            advantages = (advantages - advantages_mean) / (advantages_std + EPS)

        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=log_probs.device
        )
        log_prob_R = -torch.sum(torch.mul(log_probs, advantages_tensor))

        mean_log_prob_R = log_prob_R / num_episodes

        if counter is not None and self.experiment:
            self.experiment.log_metric("reward mean", reward_mean, step=counter)
            self.experiment.log_metric("reward std", reward_std, step=counter)
            self.experiment.log_histogram_3d(
                total_rewards, name="total reward distribution", step=counter
            )
            self.experiment.log_histogram_3d(
                rewards_to_go[:, -1], name="final reward distribution", step=counter
            )

        return mean_log_prob_R, reward_mean, reward_std

    def learn(self):
        self.policy.train()

        optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.conf.learning_rate
        )
        pbar = trange(self.conf.optimizer_iterations, desc="Optimizer iteration")
        for it in pbar:
            mean_log_prob_R, reward_mean, reward_std = self.sample_episodes(counter=it)
            mean_log_prob_R.backward()
            optimizer.step()
            optimizer.zero_grad()

            if self.experiment:
                self.experiment.log_metric("loss", mean_log_prob_R.item(), step=it)
                total_norm = 0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                self.experiment.log_metric("gradient_norm", total_norm, step=it)

            del mean_log_prob_R

            if it % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Safely get and format the loss value
            loss_val = (
                pbar.postfix.get("loss", "N/A")
                if isinstance(pbar.postfix, dict)
                else "N/A"
            )
            loss_str = (
                f"{loss_val:.3f}"
                if isinstance(loss_val, (int, float))
                else str(loss_val)
            )
            pbar.set_postfix({"reward mean": f"{reward_mean:.3f}", "loss": loss_str})
            pbar.write(f"Roll-out mean reward: {reward_mean:.3f} +- {reward_std:.2f}")

            if it % 100 == 0:
                file_path = Path(self.output_dir) / f"policy_step_{it}.pt"
                torch.save(self.policy.state_dict(), file_path)
                if self.experiment:
                    self.experiment.log_asset(
                        file_path, file_name=f"policy_step_{it}.pt"
                    )


cs = ConfigStore.instance()
cs.store(name="args", node=Cfg)


@hydra.main(version_base=None, config_name="hydra", config_path="conf")
def main(cfg: Cfg) -> None:
    import os

    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
    LOGGER.info("Starting Parallel REINFORCE GNN Training")
    LOGGER.info(f"Working directory : {os.getcwd()}")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    LOGGER.info(f"Output directory  : {output_dir}")
    LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

    env = GraphEnv(scenario=cfg.scenario, max_steps=cfg.episode_length)
    policy = Police(env, latent_node_dim=cfg.latent_node_dim)
    trainer = REINFORCEParallel(env, policy, cfg, output_dir=output_dir)

    LOGGER.info("Starting training loop")
    trainer.learn()

    file_path = Path(output_dir) / "trained_params.pt"
    LOGGER.info(f"Saving final policy to {file_path}")
    torch.save(policy.state_dict(), file_path)

    if trainer.experiment is not None:
        trainer.experiment.log_asset(file_path)
        log_model(trainer.experiment, policy, "Policy")
        trainer.experiment.end()

    LOGGER.info("Training finished successfully!")


if __name__ == "__main__":
    raise NotImplementedError(
        """Parallel REINFORCE works but it is not faster than the serial version
        because the policy does not work over batches of observations to take advantage of the GPU...
        """
    )
    main()
