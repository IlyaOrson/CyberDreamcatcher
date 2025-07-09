# adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler
import comet_ml
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_, get_total_norm
from torch_geometric.data import Batch

from cyberdreamcatcher.env import GraphEnv
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.utils import (
    set_all_seeds,
    gradient_norm,
    count_parameters,
    load_trained_weights,
)

LOGGER = logging.getLogger(__name__)
EPS = np.finfo(np.float32).eps.item()


@dataclass
class Cfg:
    # Environment settings
    scenario: str = "Scenario2"
    episode_length: int = 30
    seed: int = 1

    # Training settings
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 30
    anneal_lr: bool = True
    weight_decay: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 5
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # PPO target KL divergence (for PPO-Penalty)
    # Reward normalization
    norm_rewards: bool = True
    # Policy settings
    policy_weights: Optional[str] = None
    latent_node_dim: int = 8
    actor_heads: int = 3
    train_critic: bool = True

    # Logging
    log_comet: bool = True
    log_freq: int = 20
    log_level: str = "INFO"

    # Reward sampling
    num_ep_reward_sample: int = 100
    sample_frequency: int = 100


class PPO:
    def __init__(self, env, policy, conf, output_dir) -> None:
        self.env = env
        self.policy = policy
        self.conf = conf
        self.output_dir = output_dir

        # Ensure policy is in training mode and has critic enabled
        self.policy.train()
        assert (
            self.policy.train_critic
        ), "Policy critic must be enabled for PPO training"

        # Setup optimizer
        self.optimizer = optim.AdamW(
            policy.parameters(), lr=conf.learning_rate, weight_decay=conf.weight_decay
        )
        self.device = next(policy.parameters()).device

        # Track training progress
        self.global_step = 0
        self.start_time = time.time()

        # Set seeds
        set_all_seeds(conf.seed)

        # Setup Comet ML
        self.experiment = None
        if conf.log_comet:
            try:
                # Initialize Comet ML
                self.experiment = comet_ml.Experiment(project_name="cyberdreamcatcher")
                self.experiment.set_name(f"ppo_seed_{conf.seed}")

                # Log configuration parameters
                config_dict = OmegaConf.to_container(
                    conf, resolve=True, enum_to_str=True
                )
                if isinstance(config_dict, dict):
                    self.experiment.log_parameters(config_dict)

                # Log environment parameters
                env_params = {
                    "host_encoding": str(env.NodeFeatures._fields),
                    "host_encoding_dim": env.host_encoding_dim,
                    "edge_encoding_dim": env.edge_encoding_dim
                    if hasattr(env, "edge_encoding_dim")
                    else None,
                    "global_encoding_dim": env.global_encoding_dim
                    if hasattr(env, "global_encoding_dim")
                    else None,
                }
                if hasattr(env, "EdgeFeatures") and env.EdgeFeatures is not None:
                    env_params["edge_encoding"] = str(env.EdgeFeatures._fields)
                if hasattr(env, "GlobalFeatures") and env.GlobalFeatures is not None:
                    env_params["global_encoding"] = str(env.GlobalFeatures._fields)

                self.experiment.log_parameters(env_params, prefix="env")

                # Log model parameters count
                self.experiment.log_parameter(
                    "policy_parameters", count_parameters(self.policy)
                )

                # Only call watch if it's available (requires newer Comet ML version)
                if hasattr(self.experiment, "watch"):
                    self.experiment.watch(self.policy, log_freq=conf.log_freq)
                else:
                    LOGGER.info(
                        "Comet ML watch() not available - model gradients and parameters won't be logged"
                    )

                LOGGER.info("Comet ML initialized successfully")

            except Exception as e:
                LOGGER.warning(f"CometML initialization failed: {e}")
                self.experiment = None

    def collect_experience(self):
        """Collect trajectories using the current policy."""

        obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        log_probs_buffer = []
        values_buffer = []
        dones_buffer = []
        episode_returns = []

        # Reset environment
        obs, _ = self.env.reset(seed=self.conf.seed + self.global_step)

        # Track episode statistics
        episode_return = 0

        # Collect data for episode_length
        for _ in range(self.conf.episode_length):
            # Move observation to device if it's a PyTorch tensor

            obs = obs.to(self.device)

            # Get action from policy
            with torch.no_grad():
                report = self.policy(obs)
                action = report.action.detach()
                log_prob = report.log_prob.detach()
                value = report.value.detach()

            # Take step in environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Update episode statistics
            episode_return += reward

            # Store transition
            obs_buffer.append(obs)
            actions_buffer.append(action.to(self.device))
            rewards_buffer.append(reward)
            log_probs_buffer.append(log_prob)
            values_buffer.append(value)
            dones_buffer.append(done)

            # Update observation
            obs = next_obs

            # Handle episode end
            if done:
                # Log episode statistics
                episode_returns.append(episode_return)

                # Reset environment and episode tracking
                obs, _ = self.env.reset()
                episode_return = 0

        # Calculate advantages and returns
        with torch.no_grad():
            # Bootstrap value if not done
            obs = obs.to(self.device)
            police_report = self.policy(obs)
            next_value = police_report.value.detach()

            # Convert to tensors
            rewards_tensor = torch.tensor(
                rewards_buffer, device=self.device, dtype=torch.float32
            )
            values_tensor = torch.stack(values_buffer)
            dones_tensor = torch.tensor(
                dones_buffer, device=self.device, dtype=torch.bool
            )

            advantages, returns = self.compute_advantages(
                rewards_tensor, values_tensor, dones_tensor, next_value
            )

        # Batch observations if they are PyTorch Geometric Data objects
        obs_batch = Batch.from_data_list(obs_buffer)

        # Stack other tensors
        actions_tensor = torch.stack(actions_buffer)
        log_probs_tensor = torch.stack(log_probs_buffer)

        # Calculate episode statistics
        mean_episode_return = np.mean(episode_returns)

        return (
            obs_batch,
            actions_tensor,
            log_probs_tensor,
            values_tensor,
            returns,
            advantages,
            mean_episode_return,
        )

    def compute_advantages(
        self, rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95
    ):
        """Compute advantages using GAE.

        Args:
            rewards: Tensor of shape [episode_length]
            values: Tensor of shape [episode_length]
            dones: Tensor of shape [episode_length] indicating episode boundaries
            next_value: Tensor of shape [1] with value of next state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            advantages: Tensor of advantages
            returns: Tensor of returns
        """
        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae_lam = 0

        # Ensure next_value is a scalar
        if torch.is_tensor(next_value):
            next_value = next_value.item()

        # Initialize the last advantage
        next_non_terminal = 1.0 - dones[-1].float()
        delta = rewards[-1] + gamma * next_value * next_non_terminal - values[-1]
        advantages[-1] = last_gae_lam = delta

        # Iterate backwards through the trajectory
        for t in reversed(range(len(rewards) - 1)):
            # dones[t] is True if state s_{t+1} (resulting from action a_t in s_t) is terminal.
            # If s_{t+1} is terminal, then effectively V(s_{t+1}) is 0 for TD and future advantages are not propagated.
            s_t_plus_1_is_non_terminal = 1.0 - dones[t].float()

            # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            # values[t+1] is V(s_{t+1})
            delta = (
                rewards[t]
                + gamma * values[t + 1] * s_t_plus_1_is_non_terminal
                - values[t]
            )

            # A_t = delta_t + gamma * lambda * A_{t+1} * (1 - done_t)
            # last_gae_lam is A_{t+1} from previous iteration (for state s_{t+1})
            advantages[t] = last_gae_lam = (
                delta + gamma * gae_lambda * s_t_plus_1_is_non_terminal * last_gae_lam
            )

        # Calculate returns
        returns = advantages + values

        # Normalize advantages
        if self.conf.norm_adv and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def learn(self):
        """Train the policy using PPO with fixed-length episodes."""
        # Training metrics
        episode_returns = []
        start_time = time.time()

        # Training loop\
        num_iterations = self.conf.total_timesteps // self.conf.batch_size
        for iteration in range(1, num_iterations + 1):
            # Anneal learning rate if needed
            if self.conf.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / num_iterations
                lr_now = frac * self.conf.learning_rate
                self.optimizer.param_groups[0]["lr"] = lr_now

            # Collect experience
            (
                observations,
                actions,
                old_log_probs,
                values,
                returns,
                advantages,
                mean_episode_return,
            ) = self.collect_experience()

            # Track episode statistics
            episode_returns.append(mean_episode_return)

            # PPO optimization
            clip_fracs = []
            approx_kls = []
            pg_losses = []
            v_losses = []
            entropy_losses = []

            # Flatten the batch for minibatch updates
            batch_size = len(observations)
            indices = torch.randperm(batch_size, device=self.device)

            for epoch in range(self.conf.update_epochs):
                # Early stopping based on KL divergence
                if (
                    approx_kls
                    and self.conf.target_kl is not None
                    and np.mean(approx_kls) > self.conf.target_kl
                ):
                    LOGGER.info(
                        f"Early stopping at epoch {epoch} due to reaching max KL: {np.mean(approx_kls):.3f}"
                    )
                    break

                # Reshuffle for each epoch
                indices = torch.randperm(batch_size, device=self.device)

                # Process minibatches
                minibatch_size = self.conf.batch_size // self.conf.num_minibatches
                for start in range(0, batch_size, minibatch_size):
                    end = min(start + minibatch_size, batch_size)
                    if start >= batch_size:
                        continue

                    mb_inds = indices[start:end]

                    # Get minibatch (Handle PyG Batch objects)
                    data_list = [observations.get_example(i) for i in mb_inds.tolist()]
                    # TODO make policy work over batches
                    # mb_obs = Batch.from_data_list(data_list).to(self.device)

                    mb_actions = actions[mb_inds].to(self.device)
                    mb_old_log_probs = old_log_probs[mb_inds].to(self.device)
                    mb_advantages = advantages[mb_inds].detach().to(self.device)
                    mb_returns = returns[mb_inds].detach().to(self.device)
                    mb_old_values = values[mb_inds].detach().to(self.device)

                    # Get new action probabilities and value

                    # TODO make policy work over batches
                    # policy_output = self.policy(mb_obs, mb_actions)
                    # _, new_log_probs, entropy, new_values = policy_output

                    # --- sequentially for now ---
                    new_log_probs = []
                    entropy = []
                    new_values = []

                    for obs, action in zip(data_list, mb_actions):
                        police_report = self.policy(obs, action)
                        _, log_prob, ent, value = police_report

                        new_log_probs.append(log_prob)
                        entropy.append(ent)
                        new_values.append(value)

                    new_log_probs = torch.stack(new_log_probs)
                    entropies = torch.stack(entropy)
                    new_values = torch.stack(new_values)
                    # --- sequentially for now ---

                    # Calculate ratio (pi_theta / pi_theta_old)
                    log_ratio = new_log_probs - mb_old_log_probs
                    ratio = log_ratio.exp()

                    # Calculate approx KL for early stopping
                    with torch.no_grad():
                        # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        approx_kls.append(approx_kl.item())
                        clip_fracs += [
                            ((ratio - 1.0).abs() > self.conf.clip_coef)
                            .float()
                            .mean()
                            .item()
                        ]

                    # Policy loss with clipping
                    ratio = torch.clamp(ratio, 0.25, 4.0)  # Additional safety clamp
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1.0 - self.conf.clip_coef, 1.0 + self.conf.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss with clipping
                    if self.conf.clip_vloss:
                        v_loss_unclipped = (new_values - mb_returns) ** 2
                        v_clipped = mb_old_values + torch.clamp(
                            new_values - mb_old_values,
                            -self.conf.clip_coef,
                            self.conf.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                    # Entropy bonus
                    entropy_loss = entropies.mean()

                    # Store losses for logging
                    pg_losses.append(pg_loss.item())
                    v_losses.append(v_loss.item())
                    entropy_losses.append(entropy_loss.item())

                    # Total loss
                    loss = (
                        pg_loss
                        - self.conf.ent_coef * entropy_loss
                        + v_loss * self.conf.vf_coef
                    )

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    if self.conf.max_grad_norm > 0:
                        clip_grad_norm_(
                            self.policy.parameters(), self.conf.max_grad_norm
                        )

                    self.optimizer.step()

            # Increment global step counter once per iteration
            self.global_step += self.conf.batch_size

            # Logging and checkpointing
            if iteration % self.conf.log_freq == 0:
                # Calculate metrics
                with torch.no_grad():
                    # Calculate explained variance
                    y_pred = values.detach().cpu().numpy()
                    y_true = returns.detach().cpu().numpy()
                    var_y = np.var(y_true)
                    explained_var = (
                        np.nan
                        if var_y == 0
                        else 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
                    )

                    # Calculate statistics
                    clip_frac = np.mean(clip_fracs) if clip_fracs else 0
                    avg_kl = np.mean(approx_kls) if approx_kls else 0
                    avg_pg_loss = np.mean(pg_losses) if pg_losses else 0
                    avg_v_loss = np.mean(v_losses) if v_losses else 0
                    avg_entropy = np.mean(entropy_losses) if entropy_losses else 0

                    # Episode statistics
                    mean_episode_return = (
                        np.mean(episode_returns[-self.conf.log_freq :])
                        if episode_returns
                        else 0
                    )

                    # Calculate SPS (steps per second)
                    sps = int(self.global_step / (time.time() - start_time))

                # Log metrics to Comet ML
                if self.experiment:
                    metrics = {
                        "losses/total_loss": avg_pg_loss
                        + self.conf.vf_coef * avg_v_loss
                        - self.conf.ent_coef * avg_entropy,
                        "losses/policy_loss": avg_pg_loss,
                        "losses/value_loss": avg_v_loss,
                        "losses/entropy": avg_entropy,
                        "metrics/explained_variance": explained_var,
                        "metrics/clip_fraction": clip_frac,
                        "metrics/approx_kl": avg_kl,
                        "metrics/gradient_norm": get_total_norm(
                            self.policy.parameters()
                        ),
                        "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "charts/SPS": sps,
                        "charts/mean_episode_return": mean_episode_return,
                    }
                    self.experiment.log_metrics(metrics, step=self.global_step)

                # Log to console
                LOGGER.info(
                    f"Iteration {iteration} (Step {self.global_step}):\n"
                    f"  Loss: {avg_pg_loss + self.conf.vf_coef * avg_v_loss - self.conf.ent_coef * avg_entropy:.3f}\n"
                    f"  Policy Loss: {avg_pg_loss:.3f}, Value Loss: {avg_v_loss:.3f}, Entropy: {avg_entropy:.3f}\n"
                    f"  Explained Var: {explained_var:.3f}, Clip Frac: {clip_frac:.3f}, KL: {avg_kl:.3f}\n"
                    f"  Mean Return: {mean_episode_return:.2f}, SPS: {sps}"
                )

                # Save checkpoint
                if (
                    iteration % (self.conf.log_freq * 5) == 0
                ):  # Save every 5 log intervals
                    checkpoint_path = (
                        self.output_dir / f"policy_step_{self.global_step}.pt"
                    )
                    torch.save(
                        {
                            "iteration": iteration,
                            "global_step": self.global_step,
                            "model_state_dict": self.policy.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "loss": avg_pg_loss
                            + self.conf.vf_coef * avg_v_loss
                            - self.conf.ent_coef * avg_entropy,
                            "config": self.conf,
                        },
                        checkpoint_path,
                    )

                    if self.experiment:
                        self.experiment.log_asset(
                            str(checkpoint_path),
                            file_name=f"policy_step_{self.global_step}.pt",
                        )

                    LOGGER.info(f"Saved checkpoint to {checkpoint_path}")

        # Save final model
        final_path = self.output_dir / "final_policy.pt"
        torch.save(self.policy.state_dict(), final_path)
        if self.experiment:
            self.experiment.log_asset(str(final_path), file_name="final_policy.pt")

        if self.experiment:
            self.experiment.end()


@hydra.main(version_base=None, config_name="hydra", config_path="conf")
def main(cfg: Cfg) -> None:
    # Set up logging
    logging.basicConfig(level=cfg.log_level, handlers=[RichHandler()])
    LOGGER.info("Starting PPO GNN Training")
    LOGGER.info(f"Working directory: {Path.cwd()}")
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    LOGGER.info(f"Output directory: {output_dir}")
    LOGGER.info(f"Config used: {OmegaConf.to_yaml(cfg)}")

    # Validate config
    assert (
        cfg.policy_weights or cfg.scenario
    ), "Please provide either 'scenario' or 'policy_weights'."

    # Load policy weights if provided
    scenario = cfg.scenario
    policy_weights = None
    if cfg.policy_weights and Path(cfg.policy_weights).exists():
        policy_weights, trained_scenario = load_trained_weights(cfg.policy_weights)
        LOGGER.info(f"Found policy trained on {trained_scenario}.")
        if trained_scenario != cfg.scenario:
            LOGGER.warning(f"Will ignore the provided scenario {cfg.scenario}.")
            scenario = trained_scenario

    # Create environment and policy
    env = GraphEnv(scenario=scenario, max_steps=cfg.episode_length)
    policy = Police(
        env,
        latent_node_dim=cfg.latent_node_dim,
        actor_heads=cfg.actor_heads,
        train_critic=True,  # Enable critic for PPO
    )

    # Load weights if provided
    if policy_weights:
        policy.load_state_dict(policy_weights)

    # Create and train agent
    agent = PPO(env, policy, cfg, output_dir)

    LOGGER.info("Starting training loop")
    LOGGER.info(f"Policy parameters: {count_parameters(policy)}")
    agent.learn()

    # Save final policy
    file_path = output_dir / "trained_params.pt"
    LOGGER.info(f"Saving final policy to {file_path}")
    torch.save(policy.state_dict(), file_path)

    LOGGER.info("Training finished successfully!")


if __name__ == "__main__":
    # FIXME: PPO stagnates
    # raise NotImplementedError("PPO is not working yet. See #20 in the GitHub repo.")

    # Registering the Config class with the expected name 'args'.
    # https://hydra.cc/docs/tutorials/structured_config/minimal_example/
    cs = ConfigStore.instance()
    cs.store(name="args", node=Cfg)

    main()
