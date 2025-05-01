from dataclasses import dataclass
from pathlib import Path
import gc
import logging

from dotenv import load_dotenv
from rich.logging import RichHandler
import comet_ml
from comet_ml.integration.pytorch import log_model
from tqdm import trange
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import numpy as np
import torch

from cyberdreamcatcher.utils import set_all_seeds
from cyberdreamcatcher.sampler import collect_rewards_log_probs

EPS = np.finfo(np.float32).eps.item()

LOGGER = logging.getLogger(__name__)


@dataclass
class Cfg:
    scenario: str = "Scenario2"
    episode_length: int = 30
    num_episodes_sample: int = 1000
    seed: int = 0
    learning_rate: float = 1e-3
    optimizer_iterations: int = 300
    latent_node_dim: int = 4
    actor_heads: int = 1
    normalize_advantage: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_comet: bool = True
    log_level: str = "INFO"


class REINFORCE:
    def __init__(self, env, policy, conf, output_dir) -> None:
        self.env = env
        self.policy = policy
        self.conf = conf
        self.output_dir = output_dir

        self.experiment = None
        if conf.log_comet:
            try:
                # --- Comet ML Setup ---
                load_dotenv()
                self.experiment = comet_ml.Experiment(
                    api_key=os.getenv("COMET_API_KEY"),
                    project_name=os.getenv("COMET_PROJECT_NAME"),
                    auto_param_logging=False,
                    auto_metric_logging=False,
                )
                self.experiment.set_name(f"reinforce_seed_{conf.seed}")
                self.experiment.log_parameters(
                    OmegaConf.to_container(conf, resolve=True)
                )
                # --- End Comet ML Setup ---
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

        num_episodes = self.conf.num_episodes_sample
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
        optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.conf.learning_rate
        )

        pbar = trange(self.conf.optimizer_iterations, desc="Optimizer iteration")
        for it in pbar:
            optimizer.zero_grad()

            mean_log_prob_R, reward_mean, reward_std = self.sample_episodes(counter=it)

            mean_log_prob_R.backward()
            optimizer.step()

            # Log loss and gradient norm if Comet is enabled
            if self.experiment:
                loss_val = mean_log_prob_R.item()
                self.experiment.log_metric("loss", loss_val, step=it)

                total_norm = 0
                for p in self.policy.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm**0.5
                self.experiment.log_metric("gradient_norm", total_norm, step=it)

            # Explicitly delete loss and related tensors to potentially help GC
            del mean_log_prob_R

            # Periodically run garbage collection
            if it % 10 == 0:
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

            if it % 10 == 0:
                file_path = self.output_dir / f"policy_step_{it}.pt"
                torch.save(self.policy.state_dict(), file_path)
                if self.experiment is not None:
                    self.experiment.log_asset(
                        file_path, file_name=f"policy_step_{it}.pt"
                    )

        if self.experiment is not None:
            self.experiment.end()
            log_model(self.experiment, self.policy, "Policy")

        return


if __name__ == "__main__":
    import os
    from pathlib import Path

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
        LOGGER.info(f"Working directory : {os.getcwd()}")
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        LOGGER.info(f"Output directory  : {output_dir}")
        LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

        env = GraphEnv(scenario=cfg.scenario, max_steps=cfg.episode_length)
        policy = Police(
            env, latent_node_dim=cfg.latent_node_dim, actor_heads=cfg.actor_heads
        )
        trainer = REINFORCE(env, policy, cfg, output_dir=output_dir)

        LOGGER.info("Starting training loop")
        trainer.learn()

        # store trained policy
        file_path = Path(output_dir) / "trained_params.pt"
        LOGGER.info(f"Saving final policy to {file_path}")
        torch.save(policy.state_dict(), file_path)

        LOGGER.info("Training finished successfully!")

    main()
