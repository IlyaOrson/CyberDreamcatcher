from dataclasses import dataclass
from pathlib import Path
import gc
import logging

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

from cyberdreamcatcher.utils import set_all_seeds, gradient_norm, count_parameters
from cyberdreamcatcher.sampler import collect_rewards_log_probs

EPS = np.finfo(np.float32).eps.item()

LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    scenario: str = "Scenario2"
    episode_length: int = 30
    batch_size_episodes: int = 500
    seed: int = 0
    learning_rate: float = 7e-3
    optimizer_iterations: int = 500
    grad_clipping: float = 5
    normalize_advantage: bool = True

    latent_node_dim: int = 5
    actor_heads: int = 5

    log_comet: bool = True
    log_freq: int = 20
    log_level: str = "INFO"

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_mode: str = "max"  # 'max' because we monitor reward
    scheduler_factor: float = 0.8
    scheduler_patience: int = 50
    scheduler_threshold: float = 0.1
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
                        "edge_encoding": env.EdgeFeatures._fields,
                        "global_encoding": env.GlobalFeatures._fields,
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

        set_all_seeds(conf.seed)

    def sample_episodes(self, counter=None):
        """
        Executes multiple episodes under the current stochastic policy,
        gets an average of the reward and the summed log probabilities
        and use them to form the baselined loss function to optimize.
        """

        num_episodes = self.conf.batch_size_episodes
        batch_rewards_to_go = [None for _ in range(num_episodes)]
        batch_log_probs = [None for _ in range(num_episodes)]

        for epi in trange(num_episodes, desc="Sampling episodes"):
            rewards_to_go, log_probs = collect_rewards_log_probs(
                self.env, self.policy, self.conf.seed
            )
            batch_rewards_to_go[epi] = rewards_to_go
            batch_log_probs[epi] = log_probs

        rewards_to_go = np.stack(batch_rewards_to_go)
        log_probs = torch.stack(batch_log_probs)

        reward_mean = np.mean(rewards_to_go[:, 0])
        reward_std = np.std(rewards_to_go[:, 0])

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
        log_prob_R = -torch.sum(torch.mul(log_probs, torch.tensor(advantages)))

        mean_log_prob_R = log_prob_R / num_episodes

        if counter and self.experiment:
            if counter in range(0, self.conf.optimizer_iterations, self.conf.log_freq):
                self.experiment.log_histogram_3d(
                    rewards_to_go[:, 0], name="reward-to-go", step=counter
                )
                self.experiment.log_histogram_3d(
                    rewards_to_go[:, -1], name="final reward", step=counter
                )
            self.experiment.log_metric("reward mean", reward_mean, step=counter)
            self.experiment.log_metric("reward std", reward_std, step=counter)

        return mean_log_prob_R, reward_mean, reward_std

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
                verbose=True,
            )

        pbar = trange(self.conf.optimizer_iterations, desc="Optimizer iteration")
        for it in pbar:
            optimizer.zero_grad()

            mean_log_prob_R, reward_mean, reward_std = self.sample_episodes(counter=it)

            mean_log_prob_R.backward()

            # Apply gradient clipping if enabled
            if self.conf.grad_clipping > 0:
                clip_grad_norm_(
                    self.policy.parameters(),
                    max_norm=self.conf.grad_clipping,
                    norm_type=2,
                    error_if_nonfinite=True,
                )

            optimizer.step()

            if scheduler is not None:
                scheduler.step(reward_mean)  # Step the scheduler with the reward

            # Log loss and gradient norm if Comet is enabled
            if self.experiment:
                loss_val = mean_log_prob_R.item()
                self.experiment.log_metric("loss", loss_val, step=it)
                if scheduler is not None:
                    # Log current learning rate
                    current_lr = scheduler.get_last_lr()[-1]
                    self.experiment.log_metric(
                        "current_learning_rate", current_lr, step=it
                    )

                grad_norm = gradient_norm(self.policy)
                self.experiment.log_metric("gradient_norm", grad_norm, step=it)

            # Explicitly delete loss and related tensors to potentially help GC
            del mean_log_prob_R

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
    import hydra
    from hydra.core.config_store import ConfigStore

    from cyberdreamcatcher.env import GraphEnv
    from cyberdreamcatcher.policy import Police

    # Registering the Config class with the expected name 'args'.
    # https://hydra.cc/docs/tutorials/structured_config/minimal_example/
    cs = ConfigStore.instance()
    cs.store(name="args", node=Cfg)

    @hydra.main(version_base=None, config_name="hydra", config_path="conf")
    def main(cfg: Cfg) -> None:
        logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
        LOGGER.info("Starting REINFORCE GNN Training")
        # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
        LOGGER.info(f"Working directory : {Path.cwd()}")
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        LOGGER.info(f"Output directory  : {output_dir}")
        LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

        env = GraphEnv(scenario=cfg.scenario, max_steps=cfg.episode_length)
        policy = Police(
            env, latent_node_dim=cfg.latent_node_dim, actor_heads=cfg.actor_heads
        )
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
