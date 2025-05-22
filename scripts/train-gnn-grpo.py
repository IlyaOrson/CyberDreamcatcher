from dataclasses import dataclass
from pathlib import Path
import logging
import gc

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

import comet_ml
from comet_ml.integration.pytorch import log_model
from rich.logging import RichHandler

import numpy as np
from tqdm import trange
import torch

from cyberdreamcatcher.utils import set_all_seeds
from cyberdreamcatcher.sampler import run_episode
from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police

EPS = np.finfo(np.float32).eps.item()

LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    scenario: str = "Scenario2"
    episode_length: int = 30
    batch_size_episodes: int = 500  # Corresponds to M in GRPO (batch size)
    seed: int = 0
    learning_rate: float = 1e-2
    optimizer_iterations: int = 200
    latent_node_dim: int = 3
    log_comet: bool = True
    log_level: str = "INFO"


class GRPO:
    """Group Relative Policy Optimization Trainer"""

    def __init__(self, env, policy, conf: Cfg, output_dir) -> None:
        self.env = env
        self.policy = policy
        self.conf = conf
        self.output_dir = output_dir
        self.device = next(policy.parameters()).device

        self.optimizer_step = 0

        self.experiment = None
        if conf.log_comet:
            try:
                self.experiment = comet_ml.Experiment(
                    project_name="cyberdreamcatcher",
                    auto_param_logging=False,
                    auto_metric_logging=False,
                )
                self.experiment.set_name(f"grpo_seed_{conf.seed}")
                # Log parameters using OmegaConf
                self.experiment.log_parameters(
                    OmegaConf.to_container(conf, resolve=True)
                )
                self.experiment.log_html(f"<p>Output Directory: {self.output_dir}</p>")
            except Exception as e:
                LOGGER.warning(f"CometML initialization failed: {e}")
                self.experiment = None

        set_all_seeds(conf.seed)

    def sample_episodes(self, counter=None):
        """
        Executes multiple episodes, calculates GRPO advantages based on
        normalized total rewards, and computes the policy loss.
        """
        num_episodes = self.conf.batch_size_episodes
        batch_total_rewards = np.zeros(num_episodes)
        batch_log_probs = [None for _ in range(num_episodes)]

        for epi in trange(num_episodes, desc="Sampling episodes for GRPO"):
            # Vary seed per episode to ensure diversity in sampling
            episode_seed = self.conf.seed + self.optimizer_step * num_episodes + epi
            try:
                episode_data = run_episode(
                    self.env,
                    self.policy,
                    seed=episode_seed,
                    device=self.device,
                )
                batch_total_rewards[epi] = episode_data["total_reward"]
                # Ensure log_probs is a tensor on the correct device
                batch_log_probs[epi] = episode_data["log_probs"].to(self.device)
            except Exception as e:
                print(f"Warning: Episode {epi} failed: {e}")
                # Keep log_probs as None for failed episodes

        # Filter out failed episodes (where log_probs is None)
        valid_indices = [i for i, lp in enumerate(batch_log_probs) if lp is not None]
        if not valid_indices:
            print("Warning: No valid episodes collected in this batch.")
            # Return zero loss and stats if no valid episodes
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0, 0.0

        valid_rewards = batch_total_rewards[valid_indices]
        valid_log_probs = [batch_log_probs[i] for i in valid_indices]

        # Calculate mean and std of total rewards across the valid episodes in the batch
        reward_mean = np.mean(valid_rewards)
        reward_std = np.std(valid_rewards)

        # Log metrics if Comet is enabled
        if counter is not None and self.experiment:
            self.experiment.log_metric("reward_mean", reward_mean, step=counter)
            self.experiment.log_metric("reward_std", reward_std, step=counter)
            self.experiment.log_histogram_3d(
                valid_rewards, name="total_reward distribution", step=counter
            )

        # Calculate GRPO advantages: A_j = (r_j - mu) / (sigma + eps)
        advantages = (valid_rewards - reward_mean) / (reward_std + EPS)
        advantages_tensor = torch.tensor(
            advantages, dtype=torch.float32, device=self.device
        )

        # Calculate policy loss for each valid episode
        policy_losses = []
        for i in range(len(valid_log_probs)):
            # Sum log probs for the episode and multiply by episode advantage
            episode_log_prob_sum = torch.sum(valid_log_probs[i])
            policy_losses.append(-episode_log_prob_sum * advantages_tensor[i])

        # Average loss over the valid episodes in the batch
        mean_policy_loss = torch.mean(torch.stack(policy_losses))

        return mean_policy_loss, reward_mean, reward_std

    def learn(self):
        optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.conf.learning_rate
        )

        pbar = trange(self.conf.optimizer_iterations, desc="Optimizer iteration")
        for it in pbar:
            self.optimizer_step = it
            # Get GRPO loss and stats
            mean_policy_loss, reward_mean, reward_std = self.sample_episodes(counter=it)

            # Optimize (only if loss requires grad, i.e., valid episodes were found)
            if mean_policy_loss.requires_grad:
                optimizer.zero_grad()
                mean_policy_loss.backward()
                optimizer.step()
                loss_val = mean_policy_loss.item()
            else:
                loss_val = mean_policy_loss.item()

            # Logging
            if self.experiment:
                self.experiment.log_metric("loss", loss_val, step=it)
                # Gradient norm calculation (only if backward was called)
                if mean_policy_loss.requires_grad:
                    total_norm = 0
                    for p in self.policy.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm**0.5
                    self.experiment.log_metric("gradient_norm", total_norm, step=it)
                else:
                    self.experiment.log_metric("gradient_norm", 0.0, step=it)

            # Explicitly delete loss tensor
            del mean_policy_loss

            # Periodically run garbage collection
            if it % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.write(
                f"Iteration {it}: Mean Reward: {reward_mean:.3f} ± {reward_std:.2f}, Loss: {loss_val:.4f}"
            )
            pbar.set_postfix(
                {
                    "reward_mean": f"{reward_mean:.3f}",
                    "reward_std": f"{reward_std:.2f}",
                    "loss": f"{loss_val:.4f}",
                }
            )

            # Save model periodically
            if it > 0 and it % 100 == 0:
                file_path = Path(self.output_dir) / f"policy_step_{it}.pt"
                torch.save(self.policy.state_dict(), file_path)
                # Log checkpoint as asset
                if self.experiment:
                    self.experiment.log_asset(
                        file_path, file_name=f"policy_step_{it}.pt"
                    )

        # Save final model
        final_params = self.policy.state_dict()
        file_path = Path(self.output_dir) / "trained_params_final.pt"
        torch.save(final_params, file_path)

        return final_params


# Hydra setup
cs = ConfigStore.instance()
cs.store(name="config", node=Cfg)


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg: Cfg) -> None:
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
    LOGGER.info("Starting GRPO GNN Training")
    LOGGER.info(f"Working directory : {os.getcwd()}")
    # Hydra automatically creates an output directory based on config overrides
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    LOGGER.info(f"Output directory  : {output_dir}")
    LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

    # Setup environment
    env = GraphEnv(scenario=cfg.scenario, max_steps=cfg.episode_length)

    # Setup device and policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")
    policy = Police(env, latent_node_dim=cfg.latent_node_dim).to(device)

    # Instantiate GRPO trainer
    trainer = GRPO(env, policy, cfg, output_dir)

    # Run training
    LOGGER.info("Starting GRPO training loop...")
    trainer.learn()

    # Save final policy (redundant as learn saves it, but keep for clarity)
    final_policy_path = output_dir / "trained_params_final.pt"
    if not final_policy_path.exists():
        LOGGER.info(f"Saving final policy to {final_policy_path}")
        torch.save(policy.state_dict(), final_policy_path)
    else:
        LOGGER.info(f"Final policy already saved at {final_policy_path}")

    # Log final policy and end experiment
    if trainer.experiment is not None:
        LOGGER.info("Logging final policy asset and model to Comet...")
        trainer.experiment.log_asset(final_policy_path)
        log_model(trainer.experiment, policy, "Policy")
        LOGGER.info("Ending Comet experiment.")
        trainer.experiment.end()

    LOGGER.info("Training finished successfully. Voila!")


if __name__ == "__main__":
    raise NotImplementedError("WIP: GRPO is not implemented yet.")
    main()
