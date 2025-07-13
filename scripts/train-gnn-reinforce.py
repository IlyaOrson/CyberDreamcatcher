from dataclasses import dataclass
from pathlib import Path
import gc
import logging
from typing import Optional

from rich.logging import RichHandler
import comet_ml
from comet_ml.integration.pytorch import watch, log_model
from tqdm import trange
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cyberdreamcatcher.utils import (
    set_all_seeds,
    gradient_norm,
    count_parameters,
    load_trained_weights,
)
from cyberdreamcatcher.sampler import collect_rewards_log_probs
from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police

EPS = np.finfo(np.float32).eps.item()

LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    scenario: str = "Scenario2"
    episode_length: int = 30
    failed_action_penalty: float = -0.05
    batch_size_episodes: int = 600
    seed: int = 0
    learning_rate: float = 1e-2
    optimizer_iterations: int = 500
    grad_clipping: float = 5
    normalize_advantage: bool = True
    entropy_coef: float = 0.01

    policy_weights: Optional[str] = None
    latent_node_dim: int = 15
    actor_heads: int = 3

    log_comet: bool = True
    log_freq: int = 20
    log_level: str = "INFO"

    # Learning rate scheduler
    use_scheduler: bool = False
    scheduler_mode: str = "max"  # 'max' because we monitor reward
    scheduler_factor: float = 0.8
    scheduler_patience: int = 30
    scheduler_threshold: float = 0.01
    scheduler_threshold_mode: str = (
        # "abs"  # 'abs' --> improvement = new_metric > best_metric + threshold
        "rel"  # 'rel' --> improvement = new_metric > best_metric * (1 + threshold)
    )
    scheduler_cooldown: int = 10
    scheduler_min_lr: float = 1e-4
    scheduler_eps: float = 1e-8


class REINFORCE:
    def __init__(self, env, policy, conf, output_dir) -> None:
        self.env = env
        self.policy = policy
        self.conf = conf
        self.output_dir = output_dir

        self.experiment = None
        if conf.log_comet:
            try:
                comet_ml.login()
                self.experiment = comet_ml.start(
                    project_name="cyberdreamcatcher",
                )
                self.experiment.set_name(f"reinforce_seed_{conf.seed}")
                # self.experiment.add_tags(["reinforce"])
                self.experiment.log_parameters(
                    OmegaConf.to_container(conf, resolve=True)
                )
                self.experiment.log_parameters(
                    {
                        "host_encoding": env.NodeFeatures._fields,
                        "edge_encoding": env.EdgeFeatures._fields
                        if env.EdgeFeatures
                        else None,
                        "global_encoding": env.GlobalFeatures._fields
                        if env.GlobalFeatures
                        else None,
                        "host_encoding_dim": env.host_encoding_dim,
                        "edge_encoding_dim": env.edge_encoding_dim,
                        "global_encoding_dim": env.global_encoding_dim,
                    },
                    prefix="env",
                )
                self.experiment.log_parameter(
                    "policy_parameters", count_parameters(self.policy)
                )
                watch(self.policy, log_step_interval=self.conf.log_freq)
            except Exception as e:
                LOGGER.warning(f"CometML initialization failed: {e}")
                self.experiment = None

        self.optimizer_step = 0

    def sample_episodes(self, counter=None):
        """
        Executes multiple episodes under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        num_episodes = self.conf.batch_size_episodes
        batch_rewards_to_go = [None for _ in range(num_episodes)]
        batch_log_probs = [None for _ in range(num_episodes)]
        batch_entropies = [None for _ in range(num_episodes)]

        pbar = trange(
            num_episodes,
            desc=f"Optimizer step {counter} - Collecting rewards-to-go and log probabilities",
            leave=False,
        )
        for i in pbar:
            seed = self.conf.seed + self.optimizer_step * num_episodes + i
            rewards_to_go, log_probs, entropies = collect_rewards_log_probs(
                self.env, self.policy, seed
            )
            batch_rewards_to_go[i] = rewards_to_go
            batch_log_probs[i] = log_probs
            batch_entropies[i] = entropies

        return batch_rewards_to_go, batch_log_probs, batch_entropies

    def learn(self):
        optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.conf.learning_rate
        )

        scheduler = None
        if self.conf.use_scheduler:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.conf.scheduler_mode,
                factor=self.conf.scheduler_factor,
                patience=self.conf.scheduler_patience,
                threshold=self.conf.scheduler_threshold,
                threshold_mode=self.conf.scheduler_threshold_mode,
                cooldown=self.conf.scheduler_cooldown,
                min_lr=self.conf.scheduler_min_lr,
                eps=self.conf.scheduler_eps,
            )

        pbar = trange(self.conf.optimizer_iterations, desc="Optimizer iteration")
        for it, _ in enumerate(pbar):
            # sample a batch of episodes
            (batch_rewards_to_go, batch_log_probs, batch_entropies) = self.sample_episodes(
                counter=it
            )

            # Unpack batch and compute statistics
            rewards_to_go = torch.tensor(np.array(batch_rewards_to_go), dtype=torch.float32)
            log_probs = torch.stack(batch_log_probs)
            entropies = torch.stack(batch_entropies)

            reward_mean = rewards_to_go[:, 0].mean().item()
            reward_std = rewards_to_go[:, 0].std().item()
            # entropy_mean = entropies.mean().item()

            # Compute advantage
            if self.conf.normalize_advantage:
                # Reshape rewards to (batch_size, episode_length)
                # It's already in this shape from the sampler
                # Calculate mean and std per timestep
                mean_per_step = rewards_to_go.mean(dim=0, keepdim=True)
                std_per_step = rewards_to_go.std(dim=0, keepdim=True)
                # Normalize and get advantage
                advantage = (rewards_to_go - mean_per_step) / (std_per_step + EPS)
            else:
                advantage = rewards_to_go

            if self.experiment and (it % self.conf.log_freq == 0):
                # Log rewards-to-go distribution at the initial and final timesteps
                self.experiment.log_histogram_3d(
                    rewards_to_go[:, 0].tolist(),
                    name="rewards_to_go_t0",
                    step=it,
                )
                self.experiment.log_histogram_3d(
                    rewards_to_go[:, -1].tolist(),
                    name="rewards_to_go_t_final",
                    step=it,
                )

            # Flatten tensors for loss calculation
            advantage = advantage.view(-1)
            log_probs = log_probs.view(-1)
            entropies = entropies.view(-1)

            # Compute loss and update policy
            # loss is the negative of the objective function
            policy_loss = (log_probs * advantage).mean()
            entropy_loss = entropies.mean()
            loss = -(policy_loss + self.conf.entropy_coef * entropy_loss)

            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent them from exploding
            if self.conf.grad_clipping:
                clip_grad_norm_(self.policy.parameters(), self.conf.grad_clipping)

            optimizer.step()
            self.optimizer_step += 1

            if self.conf.use_scheduler:
                scheduler.step(reward_mean)

            if self.experiment:
                self.experiment.log_metric("reward_mean", reward_mean, step=it)
                self.experiment.log_metric("reward_std", reward_std, step=it)
                self.experiment.log_metric("loss", loss.item(), step=it)
                self.experiment.log_metric("policy_loss", policy_loss.item(), step=it)
                self.experiment.log_metric("entropy_loss", entropy_loss.item(), step=it)
                self.experiment.log_metric(
                    "grad_norm",
                    gradient_norm(self.policy.parameters()),
                    step=it,
                )
                if self.conf.use_scheduler:
                    self.experiment.log_metric(
                        "learning_rate", scheduler._last_lr[0], step=it
                    )

            # Periodically run garbage collection
            if it % self.conf.log_freq == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.write(f"Roll-out reward: {reward_mean:.3f} +- {reward_std:.2f}")
            pbar.set_postfix(
                {
                    "reward mean": f"{reward_mean:.3f}",
                    "reward std": f"{reward_std:.2f}",
                }
            )

            if it % self.conf.log_freq == 0:
                file_path = self.output_dir / f"policy_step_{it}.pt"
                torch.save(self.policy.state_dict(), file_path)
                if self.experiment:
                    # self.experiment.log_asset(
                    #     file_path, file_name=f"policy_step_{it}.pt"
                    # )
                    model_checkpoint = {
                        "epoch": it,
                        "model_state_dict": self.policy.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "reward_mean": reward_mean,
                    }
                    log_model(
                        experiment=self.experiment,
                        model=model_checkpoint,
                        model_name=f"policy",
                        metadata={
                            "reward_mean": reward_mean,
                        },
                    )
            if self.experiment:
                self.experiment.log_epoch_end(it)

        if self.experiment:
            log_model(
                experiment=self.experiment, model=self.policy, model_name="policy"
            )
            self.experiment.end()

        return


if __name__ == "__main__":
    # Registering the Config class with the expected name 'args'.
    # https://hydra.cc/docs/tutorials/structured_config/minimal_example/
    cs = ConfigStore.instance()
    cs.store(name="args", node=Cfg)

    @hydra.main(version_base=None, config_name="hydra", config_path="conf")
    def main(cfg: Cfg) -> None:
        set_all_seeds(cfg.seed)
        logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
        LOGGER.info("Starting REINFORCE GNN Training")
        # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
        LOGGER.info(f"Working directory : {Path.cwd()}")
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        LOGGER.info(f"Output directory  : {output_dir}")
        LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

        assert (
            cfg.policy_weights or cfg.scenario
        ), "Please provide either 'scenario' or 'policy_weights'."

        scenario = cfg.scenario
        policy_weights = None
        if cfg.policy_weights and Path(cfg.policy_weights).exists():
            policy_weights, trained_scenario = load_trained_weights(cfg.policy_weights)
            LOGGER.info(f"Found policy trained on {trained_scenario}.")
            if trained_scenario != cfg.scenario:
                LOGGER.warning(f"Will ignore the provided scenario {cfg.scenario}.")
                scenario = trained_scenario

        env = GraphEnv(
            scenario=scenario,
            max_steps=cfg.episode_length,
            failed_action_penalty=cfg.failed_action_penalty,
        )
        policy = Police(
            env,
            latent_node_dim=cfg.latent_node_dim,
            actor_heads=cfg.actor_heads,
        )

        if policy_weights:
            policy.load_state_dict(policy_weights)

        trainer = REINFORCE(env, policy, cfg, output_dir=output_dir)

        LOGGER.info("Starting training loop")
        LOGGER.info(f"Policy parameters: {count_parameters(policy)}")
        trainer.learn()

        # store trained policy
        file_path = Path(output_dir) / "trained_params.pt"
        LOGGER.info(f"Saving final policy to {file_path}")
        torch.save(policy.state_dict(), file_path)

        LOGGER.info("Training finished successfully!")

    main()
