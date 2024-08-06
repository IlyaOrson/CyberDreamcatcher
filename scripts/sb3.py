# TODO adapt observation space to one supported by sb3 api

from stable_baselines3 import PPO

from cyberdreamcatcher.env import GraphWrapper
from cyberdreamcatcher.policy import Police

scenario = "Scenario2_-_User2_User4"
# scenario = "Scenario2_+_User5_User6"
env = GraphWrapper(scenario=scenario)

policy = Police(env, latent_node_dim=env.host_embedding_size)


def sb3_policy_constructor(
    observation_space,
    action_space,
    lr_schedule,
    **policy_kwargs,
):
    return policy


# model = PPO("MultiInputPolicy", env, verbose=1, device="cpu")
model = PPO(sb3_policy_constructor, env, verbose=1, device="cpu")

num_steps = 10
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(num_steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(10 * "-")
