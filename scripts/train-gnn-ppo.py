# adapted from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from tqdm import trange
import numpy as np
import hydra
from hydra.core.config_store import ConfigStore

# import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

# from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police
from cyberdreamcatcher.sampler import EpisodeSampler


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    # the name of this experiment
    seed: int = 1
    # seed of the experiment
    torch_deterministic: bool = True
    # if toggled, `torch.backends.cudnn.deterministic=False`
    cuda: bool = True
    # if toggled, cuda will be enabled by default
    capture_video: bool = False
    # whether to capture videos of the agent performances (check out `videos` folder)

    # Graph wrapper stuff
    scenario: str = "Scenario2_-_User2_User4"
    # how many episodes to sample with a fixed policy to estimate the reward distribution
    num_ep_reward_sample: int = 100
    # how many opt steps to wait between reward sampling
    sample_frequency: int = 100

    # Algorithm specific arguments

    # the id of the environment
    # env_id: str = "CartPole-v1"
    # total timesteps of the experiments
    total_timesteps: int = 30000  # 500000
    # the learning rate of the optimizer
    learning_rate: float = 2.5e-4
    # the number of parallel game environments
    num_envs: int = 1  # 4 NOTE gym vectorized environments do not work without compatibility to gym observations
    # the number of steps to run in each environment per policy rollout
    num_steps: int = 30
    # Toggle learning rate annealing for policy and value networks
    anneal_lr: bool = True
    # the discount factor gamma
    gamma: float = 0.99
    # the lambda for the general advantage estimation
    gae_lambda: float = 0.95
    # the number of mini-batches
    num_minibatches: int = 4
    # the K epochs to update the policy
    update_epochs: int = 4
    # Toggles advantages normalization
    norm_adv: bool = True
    # the surrogate clipping coefficient
    clip_coef: float = 0.2
    # Toggles whether or not to use a clipped loss for the value function, as per the paper.
    clip_vloss: bool = True
    # coefficient of the entropy
    ent_coef: float = 0.01
    # coefficient of the value function
    vf_coef: float = 0.5
    # the maximum norm for the gradient clipping
    max_grad_norm: float = 0.5
    # the target KL divergence threshold
    target_kl: Optional[float] = None

    # to be filled in runtime
    # the batch size (computed in runtime)
    batch_size: int = 0
    # the mini-batch size (computed in runtime)
    minibatch_size: int = 0
    # the number of iterations (computed in runtime)
    num_iterations: int = 0


# def make_env(env_id, idx, capture_video, run_name):
#     def thunk():
#         if capture_video and idx == 0:
#             env = gym.make(env_id, render_mode="rgb_array")
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         else:
#             env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         return env

#     return thunk


# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


# class Agent(nn.Module):
#     def __init__(self, envs):
#         super().__init__()
#         self.critic = nn.Sequential(
#             layer_init(
#                 nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
#             ),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 1), std=1.0),
#         )
#         self.actor = nn.Sequential(
#             layer_init(
#                 nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)
#             ),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, 64)),
#             nn.Tanh(),
#             layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
#         )

#     def get_value(self, x):
#         return self.critic(x)

#     def get_action_and_value(self, x, action=None):
#         logits = self.actor(x)
#         probs = Categorical(logits=logits)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action), probs.entropy(), self.critic(x)


if __name__ == "__main__":
    # https://hydra.cc/docs/tutorials/structured_config/minimal_example/
    cs = ConfigStore.instance()
    cs.store(name="config", node=Args)

    @hydra.main(version_base=None, config_name="config")
    def main(args: Args) -> None:
        # https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/
        print(f"Working directory : {os.getcwd()}")
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(f"Output directory  : {output_dir}")

        # args = tyro.cli(Args)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
        run_name = f"PPO_{args.scenario}__{args.seed}"
        log_dir = Path(output_dir) / run_name

        # if args.track:
        #     import wandb
        #     wandb.init(
        #         project=args.wandb_project_name,
        #         entity=args.wandb_entity,
        #         sync_tensorboard=True,
        #         config=vars(args),
        #         name=run_name,
        #         monitor_gym=True,
        #         save_code=True,
        #     )
        writer = SummaryWriter(log_dir=log_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        )

        # env setup
        # envs = gym.vector.SyncVectorEnv(
        #     [
        #         make_env(args.env_id, i, args.capture_video, run_name)
        #         for i in range(args.num_envs)
        #     ],
        # )

        env = GraphWrapper(scenario=args.scenario, max_steps=args.num_steps)
        agent = Police(env, train_critic=True, latent_node_dim=env.host_embedding_size).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        sampler = EpisodeSampler(env, agent, seed=args.seed, writer=writer)
        sampler.sample_episodes(args.num_ep_reward_sample, counter=0)

        # ALGO Logic: Storage setup
        # obs = torch.zeros(
        #     (args.num_steps, args.num_envs) + envs.single_observation_space.shape
        # ).to(device)
        # actions = torch.zeros(
        #     (args.num_steps, args.num_envs) + envs.single_action_space.shape
        # ).to(device)
        # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        # rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        # dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        # values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        obs = [None for _ in range(args.num_steps)]
        actions = [None for _ in range(args.num_steps)]
        logprobs = torch.zeros(args.num_steps).to(device)
        rewards = torch.zeros(args.num_steps).to(device)
        dones = torch.zeros(args.num_steps).to(device)
        # dones = torch.full((args.num_steps,), False).to(device)
        values = torch.zeros(args.num_steps).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = env.reset(seed=args.seed)
        # next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        pbar = trange(1, args.num_iterations + 1, desc="Optimizer iteration")
        for iteration in pbar:
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                # obs.append(next_obs)
                # dones.append(next_done)

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # action, logprob, _, value = agent.get_action_and_value(next_obs)
                    action, logprob, _, value = agent(next_obs)
                    values[step] = value.flatten()
                    # values.append(value)

                actions[step] = action
                logprobs[step] = logprob
                # actions.append(action)
                # logprobs.append(logprob)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = env.step(
                    # action.cpu().numpy()
                    action
                )
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                # rewards.append(torch.tensor(reward).to(device).view(-1))
                next_done = torch.Tensor(np.array(next_done)).to(device)
                # next_obs, next_done = (
                #     torch.Tensor(next_obs).to(device),
                #     torch.Tensor(next_done).to(device),
                # )

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step}, episodic_return={info['episode']['r']}"
                            )
                            writer.add_scalar(
                                "charts/episodic_return",
                                info["episode"]["r"],
                                global_step,
                            )
                            writer.add_scalar(
                                "charts/episodic_length",
                                info["episode"]["l"],
                                global_step,
                            )

            # bootstrap value if not done
            with torch.no_grad():
                _, next_value = agent.get_action_logits(next_obs)
                # next_value = agent.critic(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten the batch
            b_obs = obs
            b_logprobs = logprobs
            b_actions = actions
            b_advantages = advantages
            b_returns = returns
            b_values = values
            # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            # b_logprobs = logprobs.reshape(-1)
            # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            # b_advantages = advantages.reshape(-1)
            # b_returns = returns.reshape(-1)
            # b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    newlogprob, entropies, newvalue = [], [], []
                    for idx in mb_inds:
                        _action, action_log_prob, entropy, value = agent(b_obs[idx], b_actions[idx])
                        newlogprob.append(action_log_prob)
                        entropies.append(entropy)
                        newvalue.append(value)

                    # _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    #     b_obs[mb_inds], b_actions.long()[mb_inds]
                    # )
                    logratio = torch.stack(newlogprob) - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    # newvalue = newvalue.view(-1)
                    newvalue = torch.stack(newvalue)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = torch.stack(entropies).mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            writer.add_scalar("losses/loss", loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            pbar.write("-"*20)
            pbar.write(f"loss: {loss:.3}")
            pbar.write(f"value_loss: {v_loss:.3}")
            pbar.write(f"policy_loss: {pg_loss:.3}")
            pbar.write(f"entropy_loss: {entropy_loss:.3}")

            # sample reward distribution
            if iteration % args.sample_frequency == 0:
                sampler.sample_episodes(args.num_ep_reward_sample, counter=iteration)

        # envs.close()
        writer.close()

        # store trained policy
        file_path = Path(output_dir) / "trained_params.pt"
        torch.save(agent.state_dict(), file_path)
        print(f"Trained agent weights stored in {file_path}")

    main()
