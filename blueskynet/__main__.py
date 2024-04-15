from rich.pretty import pprint

from blueskynet.env import GraphWrapper
from blueskynet.policy import ConditionalPolice, Police

# scenario = None
scenario = "Scenario2_-_User2_User4"
# scenario = "Scenario2_+_User5_User6"
env = GraphWrapper(scenario=scenario)

print("Observation:")
pprint(env.get_raw_observation())

print(env.get_true_table())
print(env.get_blue_table())

print("Voila!")

obs, info = env.reset()
env.render()

for step in range(50):

    # action = env.action_space.sample()

    policy = Police(env, latent_node_dim=env.host_embedding_size)
    # policy = ConditionalPolice(env, latent_node_dim=env.host_embedding_size)
    action, log_prob = policy(obs)

    print(f"host_name = {env.host_names[action[0]]}")
    print(f"action_name = {env.action_names[action[1]]}")

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    print(env.get_true_table())
    print(env.get_blue_table())

# TODO adapt to sb3 api
# from stable_baselines3 import PPO
# model = PPO("MultiInputPolicy", env, verbose=1, device="cpu")

# num_steps = 10
# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(num_steps):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     print(10*"-")
